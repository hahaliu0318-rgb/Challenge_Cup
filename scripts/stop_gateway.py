"""Stop only a gateway whose PID, /proc start time and script path all match."""
import json
import os
from pathlib import Path
import signal
import time
ROOT=Path(__file__).resolve().parents[1]

def identity_matches(row):
    pid=int(row['pid'])
    if pid<=1: return False
    proc=Path('/proc')/str(pid)
    if not proc.exists(): return False
    ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    cmd=(proc/'cmdline').read_bytes().split(b'\0')
    expected=str(ROOT/'scripts/run_gateway.py')
    return ticks==str(row['start_ticks']) and row['root']==str(ROOT) and row['script']==expected and os.fsencode(expected) in cmd

def main():
    file=ROOT/'runtime/state/gateway.json'
    if not file.exists(): print('No owned gateway record.'); return
    row=json.loads(file.read_text(encoding='utf-8'))
    if not identity_matches(row): raise SystemExit('PID identity mismatch or stale record. No process stopped; inspect the record manually.')
    os.kill(row['pid'],signal.SIGTERM)
    for _ in range(60):
        if not Path(f"/proc/{row['pid']}").exists(): print('Gateway stopped.'); return
        time.sleep(1)
    raise SystemExit('Graceful shutdown still pending; no SIGKILL issued. Inspect gateway/worker logs.')

if __name__=='__main__': main()
