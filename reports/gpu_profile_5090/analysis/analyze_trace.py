import sqlite3,json,csv,collections,statistics,bisect,pathlib
import sys
root=pathlib.Path(sys.argv[1])
c=sqlite3.connect(root/'capture/baseline.sqlite')
names=dict(c.execute('select id,value from StringIds'))
duration=c.execute('select duration from ANALYSIS_DETAILS limit 1').fetchone()[0]
def merge(rows):
    out=[]
    for a,b in sorted(rows):
        a=max(0,a);b=min(duration,b)
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out
def total(rows):return sum(b-a for a,b in rows)
def overlap(a,b):
    i=j=0;v=0
    while i<len(a) and j<len(b):
        v+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return v
kernels=list(c.execute('select start,end,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
gpu=merge([(r[0],r[1]) for r in kernels]+list(c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY'))+list(c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMSET')))
idle=[];last=0
for a,b in gpu:
    if a>last:idle.append([last,a])
    last=b
if last<duration:idle.append([last,duration])
apis=list(c.execute('select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME order by start'))
syn=[(a,b) for a,b,n,t in apis if 'Synchronize' in names[n]]
alloc=[(a,b) for a,b,n,t in apis if 'Malloc' in names[n] or 'Free' in names[n]]
launch=[(a,b) for a,b,n,t in apis if 'LaunchKernel' in names[n]]
device=[(a,b) for a,b,n,t in apis if names[n].startswith('cudaDeviceSynchronize')]
families=collections.defaultdict(lambda:[0,0])
def family(name):
    n=name.lower()
    if any(x in n for x in ['rasterize','raster','interpolate','texture_fwd']):return 'nvdiffrast/render (symbol heuristic)'
    if any(x in n for x in ['knn','farthest','ballquery']):return 'PyTorch3D geometry (symbol heuristic)'
    if any(x in n for x in ['gemm','gemv','xmma','flash_fwd','fmha']):return 'convolution/GEMM/attention compute'
    if 'cudnn' in n and any(x in n for x in ['nhwc','nchw','bn_','batchnorm']):return 'cuDNN layout/batchnorm'
    if any(x in n for x in ['catarray','copy_kernel','elementwise','vectorized_elementwise','unrolled_elementwise']):return 'tensor elementwise/copy/concat'
    return 'other kernels'
for a,b,n,*_ in kernels:
    v=families[family(names[n])];v[0]+=1;v[1]+=b-a
kt=sum(b-a for a,b,*_ in kernels)
out=dict(capture_duration_s=duration/1e9,kernel_count=len(kernels),summed_kernel_s=kt/1e9,gpu_activity_union_s=total(gpu)/1e9,gpu_idle_s=total(idle)/1e9,gpu_busy_fraction=total(gpu)/duration,
    kernel_duration_us=dict(median=statistics.median((r[1]-r[0])/1e3 for r in kernels),under_10us=sum(r[1]-r[0]<10000 for r in kernels),under_100us=sum(r[1]-r[0]<100000 for r in kernels)),
    api_union_s=total(merge([(a,b) for a,b,_,_ in apis]))/1e9,
    sync_calls=len(syn),sync_api_sum_s=total(syn)/1e9,sync_api_union_s=total(merge(syn))/1e9,
    sync_overlap_gpu_busy_s=overlap(merge(syn),gpu)/1e9,sync_overlap_gpu_idle_s=overlap(merge(syn),idle)/1e9,
    allocation_overlap_gpu_idle_s=overlap(merge(alloc),idle)/1e9,launch_overlap_gpu_idle_s=overlap(merge(launch),idle)/1e9,
    gpu_idle_outside_cuda_api_s=(total(idle)-overlap(merge([(a,b) for a,b,_,_ in apis]),idle))/1e9,
    device_sync=dict(count=len(device),sum_s=total(device)/1e9,median_us=statistics.median((b-a)/1e3 for a,b in device),max_us=max((b-a)/1e3 for a,b in device),overlap_gpu_idle_s=overlap(merge(device),idle)/1e9),
    idle_gaps={str(th):dict(count=sum(b-a>=th for a,b in idle),sum_s=sum(b-a for a,b in idle if b-a>=th)/1e9) for th in [10000,100000,1000000,10000000]},
    kernel_families={k:dict(count=v[0],total_s=v[1]/1e9,share=v[1]/kt) for k,v in sorted(families.items(),key=lambda x:-x[1][1])},
    streams=list(c.execute('select streamId,count(*),sum(end-start)/1e9 from CUPTI_ACTIVITY_KIND_KERNEL group by streamId')))
starts=[a for a,b in gpu];ends=[b for a,b in gpu]
ds=[]
for a,b in device:
    idx=bisect.bisect_left(starts,b)
    pi=bisect.bisect_right(ends,a)-1
    ds.append(dict(start_s=a/1e9,duration_us=(b-a)/1e3,prior_activity_gap_us=(a-ends[pi])/1e3 if pi>=0 else None,next_activity_gap_us=(starts[idx]-b)/1e3 if idx<len(starts) else None))
out['device_sync_first_12']=ds[:12]
out['device_sync_alternating']={str(i):dict(count=len(device[i::2]),sum_us=total(device[i::2])/1e3) for i in [0,1]}
out['diagnostics']=[dict(zip([d[0] for d in cur.description],r)) for cur in [c.execute('select * from DIAGNOSTIC_EVENT')] for r in cur.fetchall()]
out['nvtx_schema']=list(c.execute('pragma table_info(NVTX_EVENTS)'))
(root/'stats/timeline-analysis.json').write_text(json.dumps(out,indent=2))
with (root/'stats/device-sync-events.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(ds[0]));w.writeheader();w.writerows(ds)
transfers=[]
for label,count,ns,bytes_ in c.execute('select e.label,count(*),sum(m.end-m.start),sum(m.bytes) from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind group by m.copyKind'):
    transfers.append(dict(direction=label,count=count,total_ns=ns,bytes=bytes_))
with (root/'stats/transfer-summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(transfers[0]));w.writeheader();w.writerows(transfers)
print(json.dumps(out,indent=2))
