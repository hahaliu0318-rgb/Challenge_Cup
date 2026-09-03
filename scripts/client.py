"""Standard-library API client. Run sample commands on the inference server."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TERMINAL={'succeeded','failed','cancelled'}

def api(base, endpoint, method='GET', payload=None, timeout=30):
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request(base.rstrip('/')+endpoint,data=data,method=method,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode("utf-8",errors="replace")}') from exc

def sample_request(name, root=ROOT):
    catalog=json.loads((root/'examples/catalog.json').read_text(encoding='utf-8'))
    record=next((s for s in catalog if s['id']==name),None)
    if record is None: raise ValueError(f'Unknown sample: {name}')
    request=dict(record['request'])
    paths=request['image'] if isinstance(request['image'],list) else [request['image']]
    resolved=[str((root/v).resolve()) if not Path(v).is_absolute() else v for v in paths]
    missing=[v for v in resolved if not Path(v).is_file()]
    if missing: raise ValueError('Sample image missing; follow examples/README.md: '+str(missing))
    request['image']=resolved if isinstance(request['image'],list) else resolved[0]
    return request

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['samples','health','workers','preview','submit','infer','get','watch','cancel'])
    p.add_argument('--base-url',default='http://127.0.0.1:7860')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--sample')
    group.add_argument('--request',type=Path)
    p.add_argument('--job-id')
    p.add_argument('--poll-seconds',type=float,default=2)
    p.add_argument('--max-wait',type=float,default=0,help='Stop watching only; does not cancel server job. 0 waits indefinitely.')
    p.add_argument('--timeout',type=float,default=30)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=None
    try:
        if args.command=='samples':
            result=json.loads((ROOT/'examples/catalog.json').read_text(encoding='utf-8'))
        elif args.command in {'health','workers'}:
            result=api(args.base_url,'/healthz' if args.command=='health' else '/v1/workers',timeout=args.timeout)
        elif args.command in {'preview','submit','infer'}:
            if not (args.sample or args.request): p.error('Use --sample or --request')
            payload=sample_request(args.sample) if args.sample else json.loads(args.request.read_text(encoding='utf-8-sig'))
            endpoint={'preview':'/v1/routes/preview','submit':'/v1/jobs','infer':'/v1/infer?trace=true'}[args.command]
            result=api(args.base_url,endpoint,'POST',payload,timeout=max(args.timeout,1800) if args.command=='infer' else args.timeout)
        else:
            if not args.job_id: p.error('--job-id is required')
            # Prevent a supplied ID from changing the endpoint path.
            import uuid
            job=str(uuid.UUID(args.job_id))
            endpoint='/v1/jobs/'+job
            if args.command=='watch':
                start=time.monotonic()
                while True:
                    result=api(args.base_url,endpoint,timeout=args.timeout)
                    print(json.dumps({'job_id':job,'status':result['status'],'route_worker':result.get('route_worker')},ensure_ascii=False),flush=True)
                    if result['status'] in TERMINAL: break
                    if args.max_wait>0 and time.monotonic()-start>=args.max_wait:
                        print('Stopped watching; server job remains active. Reuse the job ID.',flush=True)
                        break
                    time.sleep(max(0.2,args.poll_seconds))
            else:
                result=api(args.base_url,endpoint,'DELETE' if args.command=='cancel' else 'GET',timeout=args.timeout)
        rendered=json.dumps(result,ensure_ascii=False,indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x',encoding='utf-8') as stream: stream.write(rendered+'\n')
        return 1 if isinstance(result,dict) and result.get('status') in {'failed','cancelled'} else 0
    except (OSError,ValueError,RuntimeError) as exc:
        print(f'ERROR: {exc}')
        return 2

if __name__=='__main__': raise SystemExit(main())
