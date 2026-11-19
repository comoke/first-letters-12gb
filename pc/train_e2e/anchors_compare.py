"""anchors_compare.py <rebuilt.npz> <shipped.npz> <out.json> : compare a rebuilt anchor pack with the shipped one.

Reports: file sha256 of both; per zip member (one .npy per key) size, CRC, payload sha256 and zip date_time; per key dtype,
shape, number of differing elements and max abs diff (info compared as parsed JSON, key by key).  Because np.savez stamps each
member with the wall-clock time, file bytes can differ while every array is identical; to test that, the rebuilt zip is
rewritten with the shipped members' date_time (nothing else changed) and its sha256 is reported as sha256_ts_normalised.
"""
import sys, json, hashlib, zipfile, io
import numpy as np


def sha(b): return hashlib.sha256(b).hexdigest()


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def members(p):
    z = zipfile.ZipFile(p)
    return {i.filename: dict(size=i.file_size, crc=f'{i.CRC:08x}', date_time=list(i.date_time), compress=i.compress_type,
                             payload_sha256=sha(z.read(i.filename))) for i in z.infolist()}


def ts_normalised_sha(reb, shp):
    """patch the 4-byte DOS time/date fields of the rebuilt zip (local headers + central directory) to the shipped ones"""
    zs = {i.filename: i for i in zipfile.ZipFile(shp).infolist()}
    b = bytearray(open(reb, 'rb').read())
    for i in zipfile.ZipFile(reb).infolist():
        s = zs.get(i.filename)
        if s is None: return None
        dt = s.date_time
        dos_t = (dt[3] << 11) | (dt[4] << 5) | (dt[5] // 2); dos_d = ((dt[0] - 1980) << 9) | (dt[1] << 5) | dt[2]
        o = i.header_offset; assert b[o:o + 4] == b'PK\x03\x04'
        b[o + 10:o + 12] = dos_t.to_bytes(2, 'little'); b[o + 12:o + 14] = dos_d.to_bytes(2, 'little')
    # central directory entries
    o = 0
    while True:
        o = b.find(b'PK\x01\x02', o)
        if o < 0: break
        n = int.from_bytes(b[o + 28:o + 30], 'little'); name = bytes(b[o + 46:o + 46 + n]).decode()
        dt = zs[name].date_time
        dos_t = (dt[3] << 11) | (dt[4] << 5) | (dt[5] // 2); dos_d = ((dt[0] - 1980) << 9) | (dt[1] << 5) | dt[2]
        b[o + 12:o + 14] = dos_t.to_bytes(2, 'little'); b[o + 14:o + 16] = dos_d.to_bytes(2, 'little'); o += 46
    return sha(bytes(b))


def main(reb, shp, out):
    R, S = np.load(reb), np.load(shp)
    rep = dict(rebuilt=reb, shipped=shp, sha256_rebuilt=fsha(reb), sha256_shipped=fsha(shp), keys={})
    rep['identical_file'] = rep['sha256_rebuilt'] == rep['sha256_shipped']
    rep['members_rebuilt'], rep['members_shipped'] = members(reb), members(shp)
    rep['keys_only_rebuilt'] = sorted(set(R.files) - set(S.files)); rep['keys_only_shipped'] = sorted(set(S.files) - set(R.files))
    for k in sorted(set(R.files) & set(S.files)):
        a, s = R[k], S[k]; d = dict(dtype=[str(a.dtype), str(s.dtype)], shape=[list(a.shape), list(s.shape)])
        if k == 'info':
            ja, js = json.loads(str(a)), json.loads(str(s))
            d['info_diff'] = {kk: [ja.get(kk), js.get(kk)] for kk in sorted(set(ja) | set(js)) if ja.get(kk) != js.get(kk)}
            d['equal'] = a.dtype == s.dtype and str(a) == str(s)
        elif a.shape == s.shape:
            if a.dtype.kind in 'fiub' and s.dtype.kind in 'fiub':      # chunked over axis 0 (keeps a 250 M px img under 4 GB)
                nd, mx = 0, 0.0
                for i in range(0, max(1, len(a)) if a.ndim else 1, 64):
                    x = (a[i:i + 64] if a.ndim else a).astype(np.float64); y = (s[i:i + 64] if s.ndim else s).astype(np.float64)
                    nd += int((x != y).sum()); mx = max(mx, float(np.abs(x - y).max()) if x.size else 0.0)
                d['n_diff'] = nd; d['n'] = int(a.size); d['max_abs_diff'] = mx
            else:
                d['n_diff'] = int((a != s).sum()); d['n'] = int(a.size)
            d['equal'] = a.dtype == s.dtype and d['n_diff'] == 0
        else:
            d['equal'] = False
        rep['keys'][k] = d
    rep['all_keys_equal'] = all(v['equal'] for v in rep['keys'].values()) and not rep['keys_only_rebuilt'] and not rep['keys_only_shipped']
    rep['member_payloads_equal'] = {m: rep['members_rebuilt'].get(m, {}).get('payload_sha256') == v['payload_sha256']
                                    for m, v in rep['members_shipped'].items()}
    rep['sha256_ts_normalised'] = ts_normalised_sha(reb, shp)
    rep['identical_after_ts_normalisation'] = rep['sha256_ts_normalised'] == rep['sha256_shipped']
    json.dump(rep, open(out, 'w'), indent=1)
    print(json.dumps({k: rep[k] for k in ('sha256_rebuilt', 'sha256_shipped', 'identical_file', 'all_keys_equal',
                                          'identical_after_ts_normalisation')}))
    for k, v in rep['keys'].items(): print(' ', k, json.dumps(v)[:400])


if __name__ == '__main__':
    main(*sys.argv[1:4])
