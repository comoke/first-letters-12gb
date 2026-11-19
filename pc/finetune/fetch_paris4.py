"""fetch_paris4.py <w03|w09|w06|w00> <out root> [--labels-only] [--window i0:i1,j0:j1] [--max-chunks N] [--every N]

Fetches one Scroll 1 (PHercParis4) segment for the fine-tune demo (docs/finetune-protocol-2026-11-17.md) and writes it in the
layout v8306lib.src_1447 reads:
  <out>/<tag>/labels/inklabels.png, labels/mask.png   uint8 0/255, channel 10 of the organisers' ink_9um aligned labels
  <out>/<tag>/layers/00.tif .. 23.tif                  uint8, the 9.6 um S3 level-2 surface volume, planes 7..102 (96 planes)
                                                       mean-pooled in groups of 4 (00 = planes 7-10, source order)
  <out>/<tag>/crop.json                                crop, sizes, sources, plane range, pooling, notes
Steps:
  1. label + supervision-mask zarrs (HF bucket, blosc chunks (21,128,128)) -> <out>/<tag>/raw_labels/{ink,mask}/<i>.<j>
     (compressed bytes as served; '<i>.<j>.none' marks a 404 = all-zero chunk). HF requests paced (< 3000 per 5 min), resumable.
  2. supervision-mask bbox on the level-2 canvas; crop = bbox + 128 px, clipped to the canvas.
  3. S3 level-2 chunks (raw uint8 (109,128,128), C order, uncompressed) that intersect the crop AND lie within 128 px of a
     supervised pixel: one byte range per chunk = planes 7..102 -> <out>/<tag>/raw_vol/<i>.<j>.u8 (96x128x128), resumable,
     retried with backoff. Chunks farther than 128 px from the mask are never fetched and stay 0 in the layers (src_1447 only
     admits tiles fully inside the mask and reads at most CTX = 128 px of context around them).
  4. pool + write layers / labels / crop.json.
Ends with 'FETCH_DONE <tag> <bytes fetched this run>' or 'FETCH_FAIL <tag> <why>'.
Test runs: --window restricts the label chunks considered (level-2 chunk rows i0:i1, cols j0:j1, half-open) and so the crop;
--max-chunks caps the volume chunks (the ones nearest the mask centroid are taken).  --labels-only stops after step 2 and
reports the volume chunks / bytes a full run would fetch.
Deps: numpy, numcodecs, tifffile, opencv (all in fl-claim-a7:final).  Network only; never executes anything downloaded."""
import argparse, json, os, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np, numcodecs, tifffile, cv2

SEG = {'w03': '20231106155351', 'w09': '20230929220926', 'w06': '20231210121321', 'w00': '20231016151002'}   # pair 1: w03 -> w09; pair 2 (amendment 1): w06 -> w00
S3 = 'https://vesuvius-challenge-open-data.s3.amazonaws.com'
VOL = 'surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr'
HF = 'https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices'
CH, NZ, Z0, NP, POOL, MARGIN, LCH = 128, 109, 7, 96, 4, 128, 10    # chunk px, planes, first plane, planes kept, pool, margin, label ch
PLANE = CH * CH
BLOSC = numcodecs.Blosc()
STATS = dict(bytes=0, hf=0, s3=0, retries=0)
_lk = threading.Lock(); _last = [0.0]


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def hf_pace():                                     # HF bucket: 3000 resolves / 300 s -> one request per 0.11 s (2,700 / 5 min)
    with _lk:
        w = _last[0] + 0.11 - time.time()
        if w > 0: time.sleep(w)
        _last[0] = time.time()


def fetch(url, rng=None, hf=False, tries=12):
    """GET (optionally a byte range). Returns bytes, or None on 404. Retries 429 / 5xx / network errors with backoff."""
    for t in range(tries):
        if hf: hf_pace()
        req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}'} if rng else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            with _lk: STATS['bytes'] += len(b); STATS['hf' if hf else 's3'] += 1
            return b
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            wait = 60 if e.code == 429 else min(120, 2 ** t)
        except Exception:
            wait = min(120, 2 ** t)
        with _lk: STATS['retries'] += 1
        time.sleep(wait)
    raise RuntimeError(f'gave up after {tries} tries: {url}')


def put(path, b):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f: f.write(b)
    os.replace(tmp, path)


def label_chunks(tag, kind, keys, out, every):
    """fetch label zarr chunks (resumable); returns {(i,j): ch10 uint8 (128,128)} for the keys."""
    case = f'phercparis4-{tag}'; base = f'{HF}/{case}/{case}_{"inklabels" if kind == "ink" else "supervision_mask"}.zarr/0'
    d = f'{out}/raw_labels/{kind}'; os.makedirs(d, exist_ok=True)
    def one(k):
        p = f'{d}/{k[0]}.{k[1]}'
        if not (os.path.exists(p) or os.path.exists(p + '.none')):
            b = fetch(f'{base}/0.{k[0]}.{k[1]}', hf=True)
            if b is None: put(p + '.none', b'')
            else: put(p, b)
        if os.path.exists(p + '.none'): return k, None
        a = np.frombuffer(BLOSC.decode(open(p, 'rb').read()), np.uint8).reshape(21, CH, CH)
        return k, a[LCH].copy()
    res = {}
    with ThreadPoolExecutor(8) as ex:
        for n, (k, a) in enumerate(ex.map(one, keys), 1):
            res[k] = a
            if n % every == 0 or n == len(keys): log(f'labels {kind} {n}/{len(keys)} (HF req {STATS["hf"]}, retries {STATS["retries"]})')
    return res


def assemble(chunks, shape, box):
    """paste ch10 chunks into a (box) crop of the canvas; box = (y0, y1, x0, x1)"""
    y0, y1, x0, x1 = box; out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for (i, j), a in chunks.items():
        if a is None: continue
        cy0, cx0 = i * CH, j * CH; cy1, cx1 = min(shape[0], cy0 + CH), min(shape[1], cx0 + CH)
        sy0, sy1, sx0, sx1 = max(cy0, y0), min(cy1, y1), max(cx0, x0), min(cx1, x1)
        if sy0 >= sy1 or sx0 >= sx1: continue
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0 - cy0:sy1 - cy0, sx0 - cx0:sx1 - cx0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag', choices=sorted(SEG)); ap.add_argument('out')
    ap.add_argument('--labels-only', action='store_true'); ap.add_argument('--window'); ap.add_argument('--max-chunks', type=int)
    ap.add_argument('--every', type=int, default=500)
    a = ap.parse_args(); tag, sid = a.tag, SEG[a.tag]; out = os.path.join(a.out, tag); os.makedirs(out, exist_ok=True)
    T0 = time.time()
    vbase = f'{S3}/PHercParis4/segments/{sid}/{VOL}/2'
    case = f'phercparis4-{tag}'
    za_v = json.loads(fetch(f'{vbase}/.zarray')); za_m = json.loads(fetch(f'{HF}/{case}/{case}_supervision_mask.zarr/0/.zarray', hf=True))
    za_l = json.loads(fetch(f'{HF}/{case}/{case}_inklabels.zarr/0/.zarray', hf=True))
    shape = tuple(za_v['shape'][1:])
    assert za_v['shape'][0] == NZ and za_v['chunks'] == [NZ, CH, CH] and za_v['compressor'] is None and za_v['dtype'] == '|u1', za_v
    assert tuple(za_m['shape'][1:]) == shape == tuple(za_l['shape'][1:]) and za_m['chunks'] == [21, CH, CH], (za_m['shape'], shape)
    ni, nj = -(-shape[0] // CH), -(-shape[1] // CH)
    i0, i1, j0, j1 = 0, ni, 0, nj
    if a.window:
        (i0, i1), (j0, j1) = [map(int, s.split(':')) for s in a.window.split(',')]
    keys = [(i, j) for i in range(i0, i1) for j in range(j0, j1)]
    log(f'{tag} {sid}: canvas {shape}, label chunks {len(keys)} (window {i0}:{i1},{j0}:{j1})')

    # 1-2. labels, bbox, crop
    M = label_chunks(tag, 'mask', keys, out, a.every)
    win = (i0 * CH, min(shape[0], i1 * CH), j0 * CH, min(shape[1], j1 * CH))
    mwin = assemble(M, shape, win) > 0
    if not mwin.any(): print(f'FETCH_FAIL {tag} no supervised pixel in window', flush=True); sys.exit(2)
    ys, xs = np.nonzero(mwin)
    bbox = [int(ys.min() + win[0]), int(ys.max() + 1 + win[0]), int(xs.min() + win[2]), int(xs.max() + 1 + win[2])]
    box = (max(0, bbox[0] - MARGIN), min(shape[0], bbox[1] + MARGIN), max(0, bbox[2] - MARGIN), min(shape[1], bbox[3] + MARGIN))
    if a.window:                                               # test runs keep the crop inside the window
        box = (max(box[0], win[0]), min(box[1], win[1]), max(box[2], win[2]), min(box[3], win[3]))
    # chunks needed: intersect the crop and lie within MARGIN px of a supervised pixel
    near = cv2.dilate(mwin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * MARGIN + 1, 2 * MARGIN + 1))) > 0
    need = []
    for i in range(box[0] // CH, -(-box[1] // CH)):
        for j in range(box[2] // CH, -(-box[3] // CH)):
            sy0, sy1 = max(i * CH, box[0]) - win[0], min((i + 1) * CH, box[1]) - win[0]
            sx0, sx1 = max(j * CH, box[2]) - win[2], min((j + 1) * CH, box[3]) - win[2]
            if sy1 > sy0 and sx1 > sx0 and near[sy0:sy1, sx0:sx1].any(): need.append((i, j))
    n_box = (-(-box[1] // CH) - box[0] // CH) * (-(-box[3] // CH) - box[2] // CH)
    log(f'bbox {bbox}, crop {box} ({box[1] - box[0]} x {box[3] - box[2]} px), volume chunks needed {len(need)} of {n_box} in the crop '
        f'-> {len(need) * NP * PLANE / 1e9:.2f} GB')
    est = dict(tag=tag, segment=sid, canvas=list(shape), mask_bbox_yyxx=bbox, crop_yyxx=list(box), crop_chunks=n_box,
               needed_chunks=len(need), needed_GB=round(len(need) * NP * PLANE / 1e9, 3), mask_px=int(mwin.sum()))
    json.dump(est, open(f'{out}/estimate.json', 'w'), indent=1)
    if a.labels_only:
        L = label_chunks(tag, 'ink', keys, out, a.every)       # fetch the ink chunks too, so the full run only reads the disk
        print(f'FETCH_DONE {tag} {STATS["bytes"]} (labels only; {json.dumps(est)})', flush=True); return

    # 3. volume chunks (planes 7..102), nearest the mask centroid first when capped
    if a.max_chunks:
        cy, cx = ys.mean() + win[0], xs.mean() + win[2]
        need = sorted(need, key=lambda k: (k[0] * CH + 64 - cy) ** 2 + (k[1] * CH + 64 - cx) ** 2)[:a.max_chunks]
    vd = f'{out}/raw_vol'; os.makedirs(vd, exist_ok=True)
    rng = (Z0 * PLANE, (Z0 + NP) * PLANE - 1)
    def vone(k):
        p = f'{vd}/{k[0]}.{k[1]}.u8'
        if os.path.exists(p) or os.path.exists(p + '.none'): return k
        b = fetch(f'{vbase}/0/{k[0]}/{k[1]}', rng=rng)
        if b is None: put(p + '.none', b''); return k
        if len(b) != NP * PLANE: raise RuntimeError(f'chunk {k}: got {len(b)} B, expected {NP * PLANE} (range ignored?)')
        put(p, b); return k
    with ThreadPoolExecutor(16) as ex:
        for n, _ in enumerate(ex.map(vone, need), 1):
            if n % a.every == 0 or n == len(need):
                log(f'volume {n}/{len(need)} chunks, {STATS["bytes"] / 1e9:.2f} GB this run, {(time.time() - T0) / 60:.1f} min')

    # 4. pool into a disk-backed (24, h, w) array, then write layers + labels
    L = label_chunks(tag, 'ink', keys, out, a.every)
    h, w = box[1] - box[0], box[3] - box[2]; nl = NP // POOL
    os.makedirs(f'{out}/layers', exist_ok=True); os.makedirs(f'{out}/labels', exist_ok=True)
    mm_path = f'{out}/pooled.tmp.u8'
    mm = np.memmap(mm_path, np.uint8, 'w+', shape=(nl, h, w))
    missing = 0
    for (i, j) in need:
        p = f'{vd}/{i}.{j}.u8'
        if not os.path.exists(p): missing += 1; continue
        v = np.fromfile(p, np.uint8).reshape(NP, CH, CH).astype(np.uint16)
        pooled = ((v.reshape(nl, POOL, CH, CH).sum(1) + POOL // 2) // POOL).astype(np.uint8)
        cy0, cx0 = i * CH, j * CH
        sy0, sy1, sx0, sx1 = max(cy0, box[0]), min(cy0 + CH, box[1], shape[0]), max(cx0, box[2]), min(cx0 + CH, box[3], shape[1])
        mm[:, sy0 - box[0]:sy1 - box[0], sx0 - box[2]:sx1 - box[2]] = pooled[:, sy0 - cy0:sy1 - cy0, sx0 - cx0:sx1 - cx0]
    mm.flush()
    for l in range(nl): tifffile.imwrite(f'{out}/layers/{l:02d}.tif', np.asarray(mm[l]))
    del mm; os.remove(mm_path)
    msk = assemble(M, shape, box); ink = assemble(L, shape, box)
    cv2.imwrite(f'{out}/labels/mask.png', (msk > 0).astype(np.uint8) * 255)
    cv2.imwrite(f'{out}/labels/inklabels.png', (ink > 0).astype(np.uint8) * 255)
    vol_bytes = sum(os.path.getsize(f'{vd}/{i}.{j}.u8') for i, j in need if os.path.exists(f'{vd}/{i}.{j}.u8'))
    info = dict(est, test=bool(a.window or a.max_chunks), window=[i0, i1, j0, j1], margin_px=MARGIN,
                fetched_chunks=len(need) - missing, empty_404_chunks=sum(os.path.exists(f'{vd}/{i}.{j}.u8.none') for i, j in need),
                volume_bytes_on_disk=vol_bytes, bytes_fetched_this_run=STATS['bytes'], requests=dict(hf=STATS['hf'], s3=STATS['s3'],
                retries=STATS['retries']), layer_shape=[h, w], layers=nl, dtype='uint8',
                source_volume=f'{S3}/PHercParis4/segments/{sid}/{VOL}', source_level=2, um=9.6,
                source_labels=f'{HF}/{case}/', label_channel=LCH, label_values='0/255 (1 = ink in the source), mask 0/255',
                planes=[Z0, Z0 + NP - 1], plane_byte_range_per_chunk=list(rng), pooling=f'mean of {POOL} consecutive planes, rounded; '
                'layer k = planes 7+4k .. 10+4k (00 = 7-10, 23 = 99-102), source order', centre_plane=54.5,
                not_fetched='volume chunks farther than margin_px from any supervised pixel are not fetched and are 0 in layers',
                ink_frac_in_mask=round(float((ink[msk > 0] > 0).mean()), 4) if (msk > 0).any() else None,
                mask_frac=round(float((msk > 0).mean()), 4),
                note='v8306lib.src_1447 reads layers 23 -> 0 (reversed). Which order the model needs is decided later by the '
                     'zero-shot layer-order check on A (protocol step 1); these files are in source order.',
                minutes=round((time.time() - T0) / 60, 2))
    json.dump(info, open(f'{out}/crop.json', 'w'), indent=1)
    print(f'FETCH_DONE {tag} {STATS["bytes"]}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        tag = sys.argv[1] if len(sys.argv) > 1 else '?'
        print(f'FETCH_FAIL {tag} {type(e).__name__}: {e}', flush=True); sys.exit(1)
