# hostguard.sh : shared host-safety check for the GPU chains (source it; the caller defines ts/log).
# Under WSL, VRAM is charged to Windows commit (~8.7 GB per 8 GiB), and a full C: crashed the WSL VM on 2026-11-14.
#   hg_check : one sample -> HG_ROW (c_free_mb,win_free_commit_mb,win_commit_total_mb) and HG_WHY (empty = OK)
#   - C: floor MIN_C_MB (default 25600: keep 25 GB free on the SSD that holds the pagefile)
#   - low free commit halts only if the commit limit is at its ceiling (RAM + pagefile max) or stays low for 4 checks
#     without the limit rising: free commit dips for a moment while Windows grows the pagefile
#   - commit unreadable 5 checks in a row -> halt (fail closed)
#   - drops the WSL page cache every 10 checks and before judging low commit (returns ~2.5 GB of commit per drop)
# Outside WSL (no powershell.exe) the guard is a no-op: hg_check always passes. Native Linux is untested.
if ! command -v powershell.exe >/dev/null 2>&1; then
  HG_ROW=NA,NA,NA; HG_WHY=; hg_check() { HG_ROW=NA,NA,NA; HG_WHY=; }; return 0 2>/dev/null || true
fi
MIN_C_MB=${MIN_C_MB:-25600}; MIN_COMMIT_MB=${MIN_COMMIT_MB:-512}
wincommit() { timeout 25 powershell.exe -NoProfile -Command '$o=Get-CimInstance Win32_OperatingSystem; "{0},{1}" -f [int]($o.FreeVirtualMemory/1024),[int]($o.TotalVirtualMemorySize/1024)' 2>/dev/null | tr -d '\r'; }
dropcache() { timeout 30 /mnt/c/Windows/System32/wsl.exe -d "${WSL_DISTRO_NAME:-Ubuntu}" -u root -e sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches' >/dev/null 2>&1; }
COMMIT_MAX_MB=$(timeout 30 powershell.exe -NoProfile -Command '$r=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1MB; $p=(Get-CimInstance Win32_PageFileSetting | Measure-Object MaximumSize -Sum).Sum; [int]($r+$p)' 2>/dev/null | tr -d '\r')
case "$COMMIT_MAX_MB" in ''|*[!0-9]*) COMMIT_MAX_MB=0;; esac      # unknown -> any low reading counts as "at ceiling" (fail closed)
HG_NA=0; HG_LOW=0; HG_TOT0=0; HG_I=0; HG_ROW=; HG_WHY=
hg_check() {
  local cmb wc tot
  HG_WHY=""; HG_I=$((HG_I + 1)); [ $((HG_I % 10)) = 0 ] && dropcache
  cmb=$(df -m /mnt/c | awk 'NR==2 {print $4}'); wc=$(wincommit)
  case "${wc%%,*}" in [0-9]*) [ "${wc%%,*}" -lt $((MIN_COMMIT_MB + 1024)) ] && { dropcache; sleep 5; wc=$(wincommit); };; esac
  HG_ROW="$cmb,${wc:-NA,NA}"
  [ -n "$cmb" ] && [ "$cmb" -lt $MIN_C_MB ] && HG_WHY="low C: space (${cmb} MB free < $MIN_C_MB)"
  case "${wc%%,*}" in
    ""|*[!0-9]*) HG_NA=$((HG_NA + 1)); [ $HG_NA -ge 5 ] && HG_WHY="Windows commit unreadable 5 checks in a row (fail closed)";;
    *) HG_NA=0; tot=${wc##*,}
       if [ "${wc%%,*}" -lt $MIN_COMMIT_MB ]; then
         if [ "$tot" -le "$HG_TOT0" ]; then HG_LOW=$((HG_LOW + 1)); else HG_LOW=1; fi
         if [ "$tot" -ge $((COMMIT_MAX_MB - 256)) ]; then HG_WHY="low Windows commit (${wc%%,*} MB free, limit $tot MB at its ceiling)"
         elif [ $HG_LOW -ge 4 ]; then HG_WHY="low Windows commit (${wc%%,*} MB free for 4 checks, limit stuck at $tot MB)"; fi
       else HG_LOW=0; fi
       HG_TOT0=$tot;;
  esac
  return 0
}
