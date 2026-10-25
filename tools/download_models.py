"""Explicit, pinned, hash-verified model downloads. Importing does no I/O."""
import argparse,hashlib,json,os,re,tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request,urlopen
DEFAULT_MANIFEST=Path(__file__).resolve().parents[1]/'model_manifest.json'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def checked_file(row,root):
    if not re.fullmatch(r'[0-9a-fA-F]{40}',row.get('revision','')):raise ValueError('Missing immutable revision')
    if not re.fullmatch(r'[0-9a-fA-F]{64}',row.get('sha256') or ''):raise ValueError('Missing SHA256 for '+row.get('filename','?'))
    if not isinstance(row.get('size_bytes'),int) or row['size_bytes']<=0:raise ValueError('Missing exact byte size')
    repo=row.get('repo_id','');filename=row.get('filename','')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+',repo) or not filename or '..' in Path(filename).parts or filename.startswith(('/','\\')):raise ValueError('Invalid HF source')
    # Manifest destinations use POSIX separators on every host. Validate raw
    # input before native Path renders it with Windows backslashes.
    raw=row.get('destination','')
    if not isinstance(raw,str) or not raw or '\\' in raw or ':' in raw or '\x00' in raw:
        raise ValueError('Invalid destination')
    relative=PurePosixPath(raw)
    if relative.is_absolute() or any(part in ('','.', '..') for part in raw.split('/')):
        raise ValueError('Invalid destination')
    target=(root/Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):raise ValueError('Destination escapes models directory')
    url=f'https://huggingface.co/{repo}/resolve/{row["revision"]}/{quote(filename,safe="/")}'
    return target,url

def verify(path,row):
    return path.is_file() and path.stat().st_size==row['size_bytes'] and digest(path)==row['sha256'].lower()

def install(row,root,*,verify_only=False):
    target,url=checked_file(row,root)
    if verify(target,row):return {'path':str(target),'status':'VERIFIED_EXISTING'}
    if verify_only:return {'path':str(target),'status':'MISSING_OR_MISMATCH'}
    if target.exists():raise ValueError('Existing file differs; refusing overwrite: '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True)
    handle,name=tempfile.mkstemp(prefix=target.name+'.',suffix='.partial',dir=target.parent);temp=Path(name)
    try:
        headers={'User-Agent':'kohaku-moss-verified-downloader/1'}
        h=hashlib.sha256();size=0
        with os.fdopen(handle,'wb') as out,urlopen(Request(url,headers=headers),timeout=60) as response:
            for block in iter(lambda:response.read(8*1024**2),b''):
                size+=len(block)
                if size>row['size_bytes']:raise ValueError('Download exceeds pinned size')
                h.update(block);out.write(block)
            out.flush();os.fsync(out.fileno())
        if size!=row['size_bytes'] or h.hexdigest()!=row['sha256'].lower():raise ValueError('Downloaded size/SHA256 mismatch')
        # Publish only fully verified bytes; recheck an independently created target.
        if target.exists():
            if verify(target,row):return {'path':str(target),'status':'VERIFIED_CONCURRENT'}
            raise ValueError('Destination appeared during download; refusing overwrite')
        # Same-directory hard-link publication is atomic and never overwrites a
        # concurrently created destination. Fail safely on unsupported filesystems.
        os.link(temp,target)
        return {'path':str(target),'status':'DOWNLOADED_VERIFIED','bytes':size}
    finally:
        temp.unlink(missing_ok=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);ap.add_argument('--models-dir',type=Path,required=True);ap.add_argument('--assets',nargs='+',help='Asset IDs, space or comma separated; defaults to Base INT8 + INT8 TE + original VAE');g=ap.add_mutually_exclusive_group();g.add_argument('--dry-run',action='store_true');g.add_argument('--verify-only',action='store_true');a=ap.parse_args()
    manifest=json.loads(a.manifest.read_text());lookup={x['id']:x for x in manifest['assets']}
    selected=[part for item in (a.assets or manifest['default_assets']) for part in item.split(',')]
    plan=[]
    # Validate ALL selected metadata before creating directories or fetching bytes.
    for id in dict.fromkeys(selected):
        asset=lookup.get(id)
        if not asset:ap.error('Unknown asset '+id)
        if asset['status']!='public' or not asset['files']:ap.error(id+' is local_only/publish_pending')
        for row in asset['files']:
            target,url=checked_file(row,a.models_dir);plan.append((id,row,target,url))
    output={'mode':'dry-run' if a.dry_run else 'verify-only' if a.verify_only else 'download','total_selected_bytes':sum(x[1]['size_bytes'] for x in plan),'files':[]}
    for id,row,target,url in plan:
        result={'path':str(target),'url':url,'size_bytes':row['size_bytes'],'sha256':row['sha256'],'status':'PLANNED'} if a.dry_run else install(row,a.models_dir,verify_only=a.verify_only)
        output['files'].append({'asset':id,**result})
    print(json.dumps(output,ensure_ascii=False,indent=2));return 1 if any(x['status']=='MISSING_OR_MISMATCH' for x in output['files']) else 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError) as error:raise SystemExit(str(error))
