"""Create local config + symlink views; never copy/modify original weights."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def create_view(source, target, vision):
    source, target, vision=Path(source).resolve(),Path(target),Path(vision).absolute()
    if target.exists() or target.is_symlink():
        raise ValueError(f'Refusing to overwrite existing model view: {target}')
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    if not (vision/'config.json').is_file():
        raise ValueError(f'Vision dependency missing: {vision}')
    target.mkdir(parents=True)
    for f in source.iterdir():
        if f.name != 'config.json':
            (target/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
    cfg['mm_vision_tower']=str(vision)
    (target/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--asset-root',required=True)
    args=p.parse_args()
    asset=Path(args.asset_root).resolve()
    target=ROOT/'runtime/model_views'
    vision=ROOT/'runtime/vision/siglip-so400m-patch14-384'
    # Validate everything before creating anything. These views are private and Linux-only.
    for name in ('global_model','detail_model','global_vision','detail_vision'):
        if not (asset/name/'config.json').is_file():
            raise SystemExit(f'Missing asset: {name}/config.json')
    if any(p.exists() or p.is_symlink() for p in [vision,target/'geollava8k',target/'llava-onevision-qwen2-7b-ov']):
        raise SystemExit('Views already exist; inspect them manually. Nothing overwritten.')
    vision.parent.mkdir(parents=True,exist_ok=True)
    vision.symlink_to(asset/'detail_vision',target_is_directory=True)
    create_view(asset/'global_model',target/'geollava8k',asset/'global_vision')
    create_view(asset/'detail_model',target/'llava-onevision-qwen2-7b-ov',vision)
    print('Created two model views; weight files remain at their original locations.')

if __name__=='__main__': main()
