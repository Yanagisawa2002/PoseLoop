import pathlib,json,hashlib,shutil,subprocess
r=pathlib.Path('/root/autodl-tmp/poseloop-a9-perf');old=pathlib.Path('/root/autodl-tmp/poseloop-v1.1-replay/foundationpose-primary')
a=[json.loads(x) for x in (old/'predictions.jsonl').read_text().splitlines()]
b=[json.loads(x) for x in (r/'primary/predictions.jsonl').read_text().splitlines()]
a=[x for x in a if x.get('record_type')=='prediction'];b=[x for x in b if x.get('record_type')=='prediction']
excluded={'registration_seconds','cuda_peak_allocated_bytes','run_lock_sha256'}
byid={x['item_id']:x for x in a};diffs=[]
for row in b:
    before=byid[row['item_id']]
    changes=[k for k in row.keys()|before.keys() if k not in excluded and row.get(k)!=before.get(k)]
    if changes:diffs.append({'item_id':row['item_id'],'fields':changes})
receipt=json.loads((r/'primary/completion-receipt.json').read_text())
result=dict(clean_count=len(a),profile_count=len(b),unique_items=len({x['item_id'] for x in b}),same_item_order=[x['item_id'] for x in a]==[x['item_id'] for x in b],ignored_provenance_timing_fields=sorted(excluded),semantic_differences=diffs,semantic_exact=len(a)==len(b)==820 and not diffs,profile_completion=receipt,free_disk_bytes=shutil.disk_usage(r).free)
result['clean_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [old/'predictions.jsonl',old/'completion-receipt.json',old/'run-lock.json']}
result['git_head']=subprocess.check_output(['git','-C','/root/autodl-tmp/PoseLoop-network','rev-parse','HEAD'],text=True).strip()
result['git_status']=subprocess.check_output(['git','-C','/root/autodl-tmp/PoseLoop-network','status','--short'],text=True)
result['foundationpose_tracked_status']=subprocess.check_output(['git','-C','/root/autodl-tmp/poseloop-v1.1-replay/foundationpose-runtime/sources/FoundationPose','status','--short','--untracked-files=no'],text=True)
(r/'receipts/primary-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
