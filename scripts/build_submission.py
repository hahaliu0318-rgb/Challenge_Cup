"""Build only allowlisted files, never the research workspace or runtime assets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from check_release import ROOT,inspect,release_files

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--final',action='store_true');args=p.parse_args()
    report=inspect(strict=args.final)
    if report['status']=='failed': raise SystemExit(json.dumps(report,ensure_ascii=False,indent=2))
    if not args.final and 'DRAFT' not in args.output.name.upper(): raise SystemExit('Draft filename must contain DRAFT; final requires --final and completed acceptance.')
    out=args.output.resolve()
    if out.is_relative_to(ROOT): raise SystemExit('Write the archive outside the repository directory.')
    out.parent.mkdir(parents=True,exist_ok=True)
    hashes=[]
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for file in release_files():
            data=file.read_bytes();rel=file.relative_to(ROOT).as_posix()
            info=zipfile.ZipInfo('Challenge_Cup/'+rel,date_time=(2026,9,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16
            z.writestr(info,data)
            hashes.append(hashlib.sha256(data).hexdigest()+'  '+rel)
        z.writestr('Challenge_Cup/SHA256SUMS.txt','\n'.join(hashes)+'\n')
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix+'.sha256').write_text(digest+'  '+out.name+'\n',encoding='utf-8')
    print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':digest,'draft':not args.final,'final_submission_ready':report['final_submission_ready']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
