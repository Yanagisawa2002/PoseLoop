import sqlite3,pathlib,json,bisect,collections,statistics
import sys
root=pathlib.Path(sys.argv[1]);c=sqlite3.connect(root/'capture/baseline.sqlite')
names=dict(c.execute('select id,value from StringIds'))
ks=list(c.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start'));starts=[x[0] for x in ks]
ds=list(c.execute("select r.start,r.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cudaDeviceSynchronize%' order by r.start"))
rs=[a for a,b,n in ks if names[n]=='RasterizeCudaFwdShaderKernel(RasterizeCudaFwdShaderParams)']
phases=collections.defaultdict(lambda:dict(kernel_count=0,kernel_ns=0,compute_ns=0,dominant_conv_ns=0,dominant_conv_calls=0))
valid=0;bad=[];dominant='sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize128x32x32_stage4_warpsize4x1x1_g1_tensor16x8x16_execute_kernel__5x_cudnn'
for i in range(1,len(ds)-1,2):
    a,b=ds[i][1],ds[i+1][0]
    marks=rs[bisect.bisect_left(rs,a):bisect.bisect_left(rs,b)]
    if len(marks)!=6:bad.append([a,b,len(marks)]);continue
    valid+=1
    for label,left,right in [('refine_render_1_to_5',marks[0],marks[5]),('score_render_6',marks[5],b)]:
        v=phases[label]
        for st,en,n in ks[bisect.bisect_left(starts,left):bisect.bisect_left(starts,right)]:
            d=en-st;name=names[n];v['kernel_count']+=1;v['kernel_ns']+=d
            if any(x in name.lower() for x in ['gemm','gemv','xmma','flash_fwd','fmha']):v['compute_ns']+=d
            if name==dominant:v['dominant_conv_ns']+=d;v['dominant_conv_calls']+=1
def union(rows,lo,hi):
    t=0;last=lo
    for a,b in sorted(rows):
        a=max(lo,a,last);b=min(hi,b)
        if b>a:t+=b-a;last=b
    return t
activities=[(a,b) for a,b,n in ks]+list(c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMCPY'))+list(c.execute('select start,end from CUPTI_ACTIVITY_KIND_MEMSET'))
inner=[r for r in ks if r[0]>=1e9 and r[1]<=59e9]
top=sum(b-a for a,b,n in inner if names[n]==dominant);kt=sum(b-a for a,b,n in inner)
o=dict(complete_registration_ranges=valid,range_exceptions=bad,phase_mapping='Inference from six RasterizeCudaFwdShaderKernel calls per complete registration: first five refine, sixth score. No model NVTX ranges added; phase boundaries are shader launch starts, not exact Python function boundaries.',phase_kernel_attribution=phases,
    inner_window=dict(start_s=1,end_s=59,duration_s=58,gpu_activity_union_s=union(activities,int(1e9),int(59e9))/1e9,kernel_s=kt/1e9,dominant_conv_s=top/1e9,dominant_conv_share=top/kt,kernel_count=len(inner)),
    registration_boundary_mapping=dict(post_count=len(ds[::2]),post_total_us=sum(b-a for a,b in ds[::2])/1e3,pre_count=len(ds[1::2]),pre_total_us=sum(b-a for a,b in ds[1::2])/1e3,method='Alternating closely spaced post/pre pair followed by depth erosion kernels and approximately 0.7s registration. Source loop has only these two device sync calls; attribution is inferred, not stack-sampled.'))
kernel_ids={x[0] for x in c.execute('select correlationId from CUPTI_ACTIVITY_KIND_KERNEL')}
launches=[(a,b,i,names[n]) for a,b,i,n in c.execute('select start,end,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME') if 'LaunchKernel' in names[n]]
missing=[x for x in launches if x[2] not in kernel_ids]
o['launch_correlation_coverage']=dict(launch_api_count=len(launches),launches_without_kernel=len(missing),interior_1_to_59s_launches_without_kernel=sum(1e9<=x[0]<59e9 for x in missing),unmatched_by_second=dict(collections.Counter(int(x[0]/1e9) for x in missing)),interpretation='Correlation check detects missing launch/kernel pairs, not every possible missing event; no completeness guarantee.')
(root/'stats/stage-analysis.json').write_text(json.dumps(o,indent=2));print(json.dumps(o,indent=2))
