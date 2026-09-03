"""Read-only deployment checks. Does not load weights or initialize CUDA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'router'))
from app.config import load_config

def weights_present(root):
    root=Path(root)
    indices=list(root.glob('*.index.json'))
    shard_names=set()
    for path in indices:
        obj=json.loads(path.read_text(encoding='utf-8'))
        shard_names.update(obj.get('weight_map',{}).values())
    if shard_names:
        return all((root/name).is_file() and (root/name).stat().st_size>1024 for name in shard_names)
    return any(f.stat().st_size>1024 for pattern in ('*.safetensors','pytorch_model*.bin') for f in root.glob(pattern))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--imports',action='store_true'); args=p.parse_args()
    config=load_config(); checks=[]
    def check(name,ok): checks.append({'check':name,'ok':bool(ok)})
    for name, model in config['models'].items():
        if not model.get('enabled',True): continue
        check(name+' Python executable',os.path.isfile(model['python']) and os.access(model['python'],os.X_OK))
        paths=[model['worker_script'],model['working_dir'],*model.get('required_paths',[])]
        for path in paths: check(name+' '+path,Path(path).exists() and os.access(path,os.R_OK))
        for key,value in model.get('env',{}).items():
            if key in {'QWEN_MODEL_PATH','QWEN_ADAPTER_PATH','GEOLLAVA_MODEL_PATH','ZOOM_MODEL_PATH','ZOOM_SEARCH_MODEL_PATH'}:
                check(name+' weights '+key,weights_present(value))
                file=Path(value)/'config.json'
                if file.exists():
                    vision=json.loads(file.read_text(encoding='utf-8')).get('mm_vision_tower')
                    if vision:
                        check(name+' local vision weights',Path(vision).is_dir() and weights_present(vision))
                        if name=='zoomsearch': check('detail vision path contains siglip','siglip' in vision.lower())
        if args.imports:
            modules={'qwen':'torch, transformers, peft, qwen_vl_utils','geollava':'torch, transformers, flash_attn, longva','zoomsearch':'torch, transformers, llava'}[name]
            env={**os.environ,**model.get('env',{}),'CUDA_VISIBLE_DEVICES':''}
            result=subprocess.run([model['python'],'-c','import '+modules],cwd=model['working_dir'],env=env,capture_output=True,text=True,timeout=90)
            check(name+' CPU-only import smoke',result.returncode==0)
            if result.returncode: print(result.stderr[-2000:],file=sys.stderr)
    for path in config['runtime']['allowed_image_roots']: check('image root '+path,Path(path).is_dir())
    check('loopback binding',config['server']['host']=='127.0.0.1')
    print(json.dumps({'checks':checks,'status':'passed' if all(c['ok'] for c in checks) else 'failed'},ensure_ascii=False,indent=2))
    return 0 if all(c['ok'] for c in checks) else 1

if __name__=='__main__': raise SystemExit(main())
