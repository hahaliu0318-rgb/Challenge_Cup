"""Generate portable, private configuration without loading models."""
import argparse
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def gpu_list(value, count):
    result = [int(x.strip()) for x in value.split(',')]
    if len(result) != count or min(result) < 0 or len(set(result)) != count:
        raise ValueError(f'Expected {count} distinct non-negative GPU indices: {value}')
    return [result]

def build(asset_root, env_root, external_root, general, global_, detail, root=ROOT):
    a, e, x, r = map(lambda p: Path(p).resolve(), (asset_root, env_root, external_root, root))
    rt = r / 'runtime'
    q = r / 'integrations/qwen'
    g = r / 'integrations/geollava'
    gp = rt / 'model_views/geollava8k'
    dp = rt / 'model_views/llava-onevision-qwen2-7b-ov'
    common = {'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'HF_HOME':str(a/'hf-cache')}
    def worker(name, python, cwd, candidates, memory, env, required, start, infer):
        return dict(display_name=name, python=str(python), worker_script='', working_dir=str(cwd), gpu_candidates=candidates, min_free_mib=memory, env={**common, **{k:str(v) for k,v in env.items()}}, required_paths=[str(v) for v in required], startup_timeout_sec=start, inference_timeout_sec=infer)
    models = {
      'qwen':worker('General image and paired-image route',e/'general/bin/python',q,general,[10240],
        {'QWEN_MODEL_PATH':a/'general_base','QWEN_ADAPTER_PATH':a/'general_adapter','QWEN_UTILS_DIR':q,'QWEN_MAX_PIXELS':1204224},
        [a/'general_base/config.json',a/'general_adapter/adapter_config.json',a/'general_adapter/adapter_model.safetensors',q/'qwen25vl_utils.py'],600,600),
      'geollava':worker('High-resolution global route',e/'global/bin/python',g,global_,[22528,22528],
        {'GEOLLAVA_MODEL_PATH':gp,'GEOLLAVA_SCRIPTS_DIR':g,'GEOLLAVA_MAX_MEMORY_GIB':21,'GEOLLAVA_ATTENTION':'flash_attention_2','GEOLLAVA_CONV_TEMPLATE':'vicuna_v1','PYTHONPATH':x/'geollava8k/longva'},
        [gp/'config.json',g/'xlrs_lite_eval.py',x/'geollava8k/longva/longva/model/builder.py'],900,900),
      'zoomsearch':worker('High-resolution detail route',e/'detail/bin/python',x/'zoomsearch',detail,[23552,9728],
        {'ZOOM_MODEL_PATH':dp,'ZOOM_SEARCH_MODEL_PATH':a/'retrieval_model','ZOOM_CODE_ROOT':x/'zoomsearch','ZOOM_LLAVA_ROOT':x/'llava-next','ZOOM_RUNTIME_DIR':rt/'zoom_runtime','ZOOM_MAX_NEW_TOKENS':1024,'ZOOM_MAX_STEP':10,'ZOOM_MAX_DEPTH':50,'ZOOM_BIAS_VALUE':0.3},
        [dp/'config.json',a/'retrieval_model/config.json',x/'zoomsearch/vlm/config.py',rt/'vision/siglip-so400m-patch14-384/config.json'],1200,1800),
      'change_agent':{'enabled':False,'phase':2,'display_name':'Reserved route','disabled_reason':'Not part of the phase-one release.'},
    }
    for key in ('qwen','geollava','zoomsearch'):
        models[key]['worker_script'] = str(r/'router/workers'/f'{key}_worker.py')
    config = {'models_config':'models.yaml','server':{'host':'127.0.0.1','port':7860,'default_trace':True},'runtime':{
        'state_dir':str(rt/'state'),'log_dir':str(rt/'logs'),'jobs_db':str(rt/'state/jobs.sqlite3'),'job_retention_days':7,'worker_idle_ttl_sec':900,'gpu_poll_interval_sec':15,'queue_poll_interval_sec':2,'low_resolution_max_edge':1024,'max_image_file_size_bytes':1500000000,'max_image_pixels':500000000,'allowed_image_roots':[str(r/'examples'),str(rt)]}}
    return config, {'models':models}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--asset-root',required=True)
    p.add_argument('--env-root',required=True)
    p.add_argument('--external-root',default=str(ROOT/'external'))
    p.add_argument('--general-gpus',default='0')
    p.add_argument('--global-gpus',default='0,1')
    p.add_argument('--detail-gpus',default='1,2')
    args=p.parse_args()
    config, models = build(args.asset_root,args.env_root,args.external_root,gpu_list(args.general_gpus,1),gpu_list(args.global_gpus,2),gpu_list(args.detail_gpus,2))
    dest=ROOT/'router/config'
    paths=[dest/'router.yaml',dest/'models.yaml']
    if any(path.exists() for path in paths):
        raise SystemExit('Local configuration exists; edit it explicitly. No files overwritten.')
    dest.mkdir(parents=True,exist_ok=True)
    for path,payload in zip(paths,[config,models]):
        path.write_text(yaml.safe_dump(payload,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(json.dumps({'config':str(paths[0]),'models':str(paths[1]),'next':'prepare_model_views.py, then preflight.py'},ensure_ascii=False))

if __name__=='__main__': main()
