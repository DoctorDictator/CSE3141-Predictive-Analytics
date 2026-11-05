from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib,json,time,zipfile,shutil,requests
ROOT=Path(__file__).resolve().parents[1]/'data/raw/automatum'
ROOT.mkdir(parents=True,exist_ok=True)
REPO='AutomatumData/automatum-data-full-highway'
NAME='automatum_data_full_highway_drone_dataset.zip'
api=requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/main?expand=true',timeout=30);api.raise_for_status()
metadata=next(v for v in api.json() if v['path']==NAME)
size=metadata['size']; expected=metadata['lfs']['oid']
url=f'https://huggingface.co/datasets/{REPO}/resolve/main/{NAME}'
archive=ROOT/NAME
parts=ROOT/'download_parts';parts.mkdir(exist_ok=True)
chunk=8*1024*1024
start_time=time.monotonic();completed=0
print(f'Downloading {size:,} bytes from official repository; 8 resumable streams.',flush=True)
def fetch(index):
    low=index*chunk;high=min(size,low+chunk)-1;length=high-low+1
    target=parts/f'{index:04}.part'
    if target.exists() and target.stat().st_size==length:return length
    for attempt in range(5):
        try:
            with requests.get(url,params={'download':'true','chunk':str(index),'attempt':str(attempt)},headers={'Range':f'bytes={low}-{high}'},stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {low}-{high}/{size}':raise RuntimeError('Server did not honor requested byte range')
                with target.open('wb') as f:
                    for data in r.iter_content(256*1024):f.write(data)
            if target.stat().st_size!=length:raise RuntimeError('Incomplete chunk')
            return length
        except Exception:
            if attempt==4:raise
            time.sleep(min(8,attempt+1))
if not archive.exists() or archive.stat().st_size!=size:
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,i) for i in range((size+chunk-1)//chunk)]
        for future in as_completed(futures):
            completed+=future.result()
            print(f'Progress {completed/size:.1%}; {completed/1e9:.2f}/{size/1e9:.2f} GB; elapsed {time.monotonic()-start_time:.0f}s',flush=True)
    temporary=ROOT/(NAME+'.partial')
    with temporary.open('wb') as out:
        for i in range((size+chunk-1)//chunk):
            with (parts/f'{i:04}.part').open('rb') as src:shutil.copyfileobj(src,out,1024*1024)
    temporary.replace(archive)
print('Checking publisher SHA-256...',flush=True)
h=hashlib.sha256()
with archive.open('rb') as f:
    while block:=f.read(8*1024*1024):h.update(block)
assert h.hexdigest()==expected,'Archive checksum does not match publisher'
print('SHA-256 matches publisher.',flush=True)
with zipfile.ZipFile(archive) as z:
    entries=z.infolist(); expanded=sum(e.file_size for e in entries)
    print(f'Archive contains {len(entries)} entries; expanded size {expanded/1e9:.2f} GB.',flush=True)
    bad=z.testzip();assert bad is None,f'Corrupt ZIP entry: {bad}'
    names=z.namelist(); trajectories=[n for n in names if n.endswith('/dynamicWorld.json')]
    maps=[n for n in names if n.endswith('/staticWorld.xodr')]
    print(f'ZIP CRC checks passed. Trajectory files: {len(trajectories)}; road maps: {len(maps)}.',flush=True)
manifest=dict(repository=REPO,url=url,filename=NAME,size_bytes=size,sha256=h.hexdigest(),publisher_sha256=expected,checksum_verified=True,zip_crc_verified=True,archive_entries=len(entries),uncompressed_bytes=expanded,trajectory_files=len(trajectories),road_map_files=len(maps),recording_folders=[str(Path(n).parent) for n in trajectories])
(ROOT/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
r=requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/main/README.md',timeout=30);r.raise_for_status();(ROOT/'DATASET_README.md').write_text(r.text,encoding='utf-8')
# Delete only the exact temporary chunks created by this downloader, after integrity checks.
for i in range((size+chunk-1)//chunk):(parts/f'{i:04}.part').unlink(missing_ok=True)
parts.rmdir()
print('DOWNLOAD COMPLETE:',archive,flush=True)
