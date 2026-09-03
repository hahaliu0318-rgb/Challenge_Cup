"""Foreground gateway with PID/start-time identity for safe stop."""
import json
import os
from pathlib import Path
import socket
import sys
import uvicorn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'router'))
from app.config import load_config

def start_ticks(pid):
    # comm can contain spaces; fields after the closing parenthesis start at field 3.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]

def main():
    cfg=load_config()
    host=cfg['server']['host']; port=int(cfg['server']['port'])
    if host!='127.0.0.1': raise SystemExit('This release only permits 127.0.0.1 binding.')
    with socket.socket() as probe: probe.bind((host,port))
    record=ROOT/'runtime/state/gateway.json'
    record.parent.mkdir(parents=True,exist_ok=True)
    identity={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'root':str(ROOT),'script':str(Path(__file__).resolve())}
    with record.open('x',encoding='utf-8') as stream: json.dump(identity,stream)
    try:
        uvicorn.run('app.main:app',host=host,port=port,workers=1)
    finally:
        if record.exists() and json.loads(record.read_text())==identity: record.unlink()

if __name__=='__main__': main()
