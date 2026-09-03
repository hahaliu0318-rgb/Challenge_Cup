"""Check the explicitly scoped release tree; --strict also gates final submission."""
import argparse
import hashlib
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIRS={'router','scripts','tests','docs','assets','environments','examples','integrations','training','results'}
ROOT_FILES={'README.md','.gitignore','.gitattributes','RELEASE.json'}
SKIP={'__pycache__','.pytest_cache','.git'}
BAD_EXT={'.safetensors','.pt','.pth','.bin','.ckpt','.arrow','.parquet','.sqlite3','.log','.pem','.key'}

def release_files(root=ROOT):
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if any(part in SKIP for part in rel.parts): continue
        if rel.as_posix() in {'router/config/router.yaml','router/config/models.yaml'}: continue
        if rel.parts[0] not in DIRS and rel.as_posix() not in ROOT_FILES: continue
        if path.is_symlink(): raise ValueError('Symlink not allowed in public package: '+rel.as_posix())
        if path.is_file(): yield path

def inspect(root=ROOT,strict=False):
    errors=[]; warnings=[]; files=list(release_files(root))
    bad_paths=['/data/'+x for x in ('lb','wjh','zxw')]
    sensitive=[re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),re.compile(r'(?i)(?:password|passwd|api_key)\s*[:=]\s*[\"\x27][^\"\x27\s]{4,}[\"\x27]')]
    for file in files:
        rel=file.relative_to(root).as_posix()
        if file.suffix in BAD_EXT or file.name.startswith('.env'): errors.append('Forbidden file: '+rel)
        if file.stat().st_size>20*1024*1024: errors.append('File exceeds 20 MiB: '+rel)
        if file.suffix in {'.py','.md','.sh','.json','.yaml','.yml','.txt'}:
            text=file.read_text(encoding='utf-8')
            if any(value in text for value in bad_paths): errors.append('Private server path: '+rel)
            if any(regex.search(text) for regex in sensitive): errors.append('Potential credential: '+rel)
        if file.suffix=='.json': json.loads(file.read_text(encoding='utf-8'))
        if file.suffix=='.py': compile(file.read_text(encoding='utf-8'),str(file),'exec')
    for row in json.loads((root/'examples/images_manifest.json').read_text(encoding='utf-8')):
        path=root/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']: errors.append('Example hash mismatch: '+row['file'])
    manifest=json.loads((root/'assets/models_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['assets']:
        fields=('platform','url','access','model_revision')
        if item.get('kind')=='official_dependency_reference': fields+=('config_sha256',)
        elif item.get('kind')=='project_adapter': fields+=('file_sha256',)
        else: fields+=('archive_sha256',)
        if any(not item.get(k) for k in fields): warnings.append('Incomplete download entry: '+item['id'])
        if item.get('url') and not item['url'].startswith('https://'): errors.append('Download URL must use HTTPS: '+item['id'])
    release=json.loads((root/'RELEASE.json').read_text(encoding='utf-8'))
    for key,value in release['acceptance'].items():
        if value is not True: warnings.append('Pending acceptance: '+key)
    warnings.append('Automated scanning is not proof of licensing or absence of all sensitive information; perform manual review.')
    failed=bool(errors or (strict and len(warnings)>1))
    return {'status':'failed' if failed else 'draft_checks_passed','file_count':len(files),'bytes':sum(p.stat().st_size for p in files),'errors':errors,'pending':warnings,'final_submission_ready':not errors and len(warnings)==1}

def main():
    p=argparse.ArgumentParser();p.add_argument('--strict',action='store_true');args=p.parse_args()
    try: result=inspect(strict=args.strict)
    except (OSError,ValueError,SyntaxError) as exc: print(str(exc));return 1
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if result['status']=='failed' else 0

if __name__=='__main__': raise SystemExit(main())
