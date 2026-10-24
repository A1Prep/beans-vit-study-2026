"""Execute the fixed, explicitly exploratory SigLIP2/DINOv2 follow-up.

Data preparation is performed by prepare_data.py. Runs are new and separate
from the original DINOv2 study. Both backbones finish training before any new
test inference; selected artifacts are hash-locked and checked before scoring.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--data-root',required=True,type=Path)
    ap.add_argument('--run-root',required=True,type=Path,help='New directory for model checkpoints; allow several GB')
    ap.add_argument('--out',required=True,type=Path,help='Evidence directory; weights excluded')
    ap.add_argument('--study-plan',type=Path,default=SOURCE.parent/'results/model_comparison/study_plan.json')
    ap.add_argument('--dinov2-path',type=Path)
    ap.add_argument('--siglip2-path',type=Path)
    ap.add_argument('--dinov2-acquisition',type=Path,default=SOURCE.parent/'results/model_acquisition.json')
    ap.add_argument('--siglip2-acquisition',type=Path,default=SOURCE.parent/'results/model_comparison/siglip2_acquisition.json')
    ap.add_argument('--device',default='auto')
    ap.add_argument('--phase',choices=['pilot','train','evaluate','analyse','all'],default='all')
    ap.add_argument('--resume',action='store_true',help='Continue only matching completed runs; incomplete runs are never overwritten')
    args=ap.parse_args()
    for key in ['manifest','data_root','run_root','out','study_plan']:
        setattr(args,key,getattr(args,key).resolve())
    plan=read(args.study_plan);settings=plan['settings'];seeds=settings['seeds']
    if digest(args.manifest)!=plan['dataset']['manifest_sha256']:
        raise ValueError('Dataset manifest differs from the recorded study plan; record a reproduction plan for a relocated regenerated manifest')
    args.out.mkdir(parents=True,exist_ok=True)
    timing_policy=args.out/'timing_analysis_amendment.json'
    if not timing_policy.is_file():
        raise ValueError('Record the descriptive timing policy in the new output folder before execution; the reproduction notebook plan cell creates timing_analysis_amendment.json without changing the original study')
    timing=read(timing_policy)
    if timing.get('before_new_test_evaluation') is not True or timing.get('raw_times_preserved') is not True:
        raise ValueError('Timing policy must preserve raw times and disclose recording before new test evaluation')
    pretrained_checks={}
    for key,model in plan['models'].items():
        local=getattr(args,key+'_path')
        if local:
            acquisition_path=getattr(args,key+'_acquisition')
            acquired=read(acquisition_path)
            if acquired.get('model_id',acquired.get('selected_model'))!=model['model_id'] or acquired['revision']!=model['revision']:
                raise ValueError('Local pretrained acquisition identity differs from the plan')
            files={}
            for entry in acquired['files']:
                name=entry['file']
                if Path(name).name!=name:raise ValueError('Acquisition files must be simple filenames')
                target=local/name
                if target.stat().st_size!=entry['bytes'] or digest(target)!=entry['sha256']:
                    raise ValueError(f'Local pretrained file differs from verified acquisition: {name}')
                files[name]=entry['sha256']
            if 'config.json' not in files or 'model.safetensors' not in files or 'preprocessor_config.json' not in files:
                raise ValueError('Local acquisition must bind weights, model config and preprocessing')
            pretrained_checks[key]={'model_id':model['model_id'],'revision':model['revision'],'local_path':str(local.resolve()),
                                    'acquisition_sha256':digest(acquisition_path),'verified_files':files}
        else:
            pretrained_checks[key]={'model_id':model['model_id'],'revision':model['revision'],'loading':'Official Hugging Face pinned revision; no local override'}
    write(args.out/'pretrained_input_verification.json',{'verified_utc':now(),'models':pretrained_checks})
    code=SOURCE/'followup_experiment.py'
    identity={'training_code_sha256':digest(code),'manifest_sha256':digest(args.manifest),
              'study_plan_sha256':digest(args.study_plan),'run_root':str(args.run_root)}
    identity_path=args.out/'execution_identity.json'
    if identity_path.exists():
        previous=read(identity_path)
        if any(previous.get(k)!=v for k,v in identity.items()):
            raise ValueError('Source, plan, manifest or run root changed during execution')
        if not args.resume:raise FileExistsError('Execution exists; use --resume with unchanged inputs')
    else:write(identity_path,{**identity,'started_utc':now(),'prior_test_exposure':True})
    common=['--manifest',str(args.manifest),'--data-root',str(args.data_root),
            '--batch-size',str(settings['batch_size']),'--device',args.device,
            '--epochs',str(settings['epochs']),'--patience',str(settings['patience']),
            '--head-lr',str(settings['head_lr']),'--backbone-lr',str(settings['backbone_lr']),
            '--weight-decay',str(settings['weight_decay'])]
    def call(script,*arguments):
        print(json.dumps({'event':'command','script':script,'arguments':list(map(str,arguments)),'utc':now()}),flush=True)
        subprocess.run([sys.executable,str(SOURCE/script),*map(str,arguments)],check=True)
    def model_args(key):
        model=plan['models'][key]
        values=['--model-id',model['model_id'],'--revision',model['revision']]
        local=getattr(args,key+'_path')
        if local:values+=['--local-model-path',str(local.resolve())]
        return values
    def verify_metadata(c,key,arm,seed):
        expected={'training_code_sha256':identity['training_code_sha256'],
                  'manifest_sha256':identity['manifest_sha256'],'arm':arm,'seed':seed,
                  'model_id':plan['models'][key]['model_id'],
                  'requested_revision':plan['models'][key]['revision'],
                  'batch_size':settings['batch_size'],'max_epochs':settings['epochs'],
                  'patience':settings['patience'],'head_lr':settings['head_lr'],
                  'backbone_lr':settings['backbone_lr'],'weight_decay':settings['weight_decay']}
        for k,v in expected.items():
            if c.get(k)!=v:raise ValueError(f'Run setting mismatch: {key}/{arm}/{seed} {k}')
    def verify_pilot(key):
        checks=read(args.run_root/key/'pilot/pilot_report.json')
        if len(checks)!=2 or {x['arm'] for x in checks}!=set(settings['arms']):
            raise ValueError('Exactly one pilot per planned arm is required')
        heads=[]
        for check in checks:
            if check['status']!='passed' or not check['epoch_benchmarked'] or check['training']['n']!=plan['dataset']['split_sizes']['train']:
                raise ValueError('Complete full-epoch pilot evidence is required')
            verify_metadata(check['metadata'],key,check['arm'],17)
            heads.append(check['metadata']['initial_head_sha256'])
        if len(set(heads))!=1:raise ValueError('Pilot classifier initializations differ')
    def verify_run(folder,key,arm,seed):
        c=read(folder/'config.json');s=read(folder/'summary.json')
        verify_metadata(c,key,arm,seed)
        if not (folder/'best.pt').is_file() or s['epochs_completed']<1:
            raise ValueError('Incomplete selected checkpoint')
        return c,s
    if args.phase in ['pilot','all']:
        for key in plan['models']:
            dest=args.run_root/key
            report=dest/'pilot/pilot_report.json'
            if report.exists() and args.resume:
                verify_pilot(key)
            else:call('followup_experiment.py','pilot',*common,*model_args(key),'--output-dir',dest,'--benchmark-epoch')
            verify_pilot(key)
            (args.out/key).mkdir(parents=True,exist_ok=True)
            shutil.copy2(report,args.out/key/'pilot_report.json')
        write(args.out/'pilot_completion.json',{'completed_utc':now(),'models':list(plan['models']),'all_passed':True})
    if args.phase in ['train','all']:
        # Pair model order by seed to reduce gross hardware/time ordering effects.
        for seed in seeds:
            for key in plan['models']:
                verify_pilot(key)
                for arm in settings['arms']:
                    folder=args.run_root/key/'runs'/arm/f'seed_{seed}'
                    if (folder/'summary.json').exists() and args.resume:
                        verify_run(folder,key,arm,seed)
                    else:call('followup_experiment.py','train',*common,*model_args(key),'--output-dir',args.run_root/key/'runs','--seeds',str(seed),'--arm',arm)
        write(args.out/'training_completion.json',{'completed_utc':now(),'runs':len(plan['models'])*len(seeds)*len(settings['arms'])})
    if args.phase in ['evaluate','all']:
        # Require every run before locking/evaluating either backbone.
        for key in plan['models']:
            entries=[]
            for seed in seeds:
                heads=[]
                for arm in settings['arms']:
                    folder=args.run_root/key/'runs'/arm/f'seed_{seed}'
                    c,s=verify_run(folder,key,arm,seed);heads.append(c['initial_head_sha256'])
                    entries.append({'arm':arm,'seed':seed,'config_sha256':digest(folder/'config.json'),
                                    'summary_sha256':digest(folder/'summary.json'),
                                    'checkpoint_sha256':digest(folder/'best.pt'),'best_epoch':s['best_epoch']})
                if len(set(heads))!=1:raise ValueError('Within-backbone head initialization differs')
            lock=args.out/key/'protocol.lock.json'
            if not lock.exists():
                previous_test_files=[p for pattern in ['metrics_test*.json','predictions_test*.csv']
                                     for p in (args.run_root/key/'runs').rglob(pattern)]
                if previous_test_files:
                    raise ValueError('Cannot create a new pre-evaluation lock for runs with existing test outputs')
                write(lock,{'locked_at_utc':now(),'training_code_sha256':identity['training_code_sha256'],
                  'manifest_sha256':identity['manifest_sha256'],'study_plan_sha256':identity['study_plan_sha256'],
                  'model':plan['models'][key]['model_id'],'revision':plan['models'][key]['revision'],
                  'seeds':seeds,'reference_seed':plan['analysis']['reference_seed'],
                  'original_dinov2_test_results_known_before_followup':True,
                  'new_followup_test_predictions_evaluated_before_lock':False,
                  'interpretation':'Exploratory reuse of previously evaluated test set; not a fresh confirmatory experiment',
                  'settings':{k:settings[k] for k in ['batch_size','epochs','patience','head_lr','backbone_lr','weight_decay']},'runs':entries})
        from validate_protocol import assert_locked
        for key in plan['models']:
            lock=args.out/key/'protocol.lock.json'
            for seed in seeds:
                for arm in settings['arms']:
                    folder=args.run_root/key/'runs'/arm/f'seed_{seed}'
                    provenance=assert_locked(args.manifest,folder/'best.pt',lock,code)
                    for corruption in (['clean','blur','brightness'] if seed==plan['analysis']['reference_seed'] else ['clean']):
                        stem='test'+('' if corruption=='clean' else '_'+corruption)
                        metric=folder/f'metrics_{stem}.json'
                        if metric.exists() and args.resume:
                            old=read(metric)
                            if old.get('checkpoint_sha256')!=provenance['checkpoint_sha256'] or old.get('protocol_lock_sha256')!=digest(lock) or not (folder/f'predictions_{stem}.csv').is_file():
                                raise ValueError('Existing evaluation provenance differs')
                        else:call('followup_experiment.py','evaluate',*common,'--checkpoint',folder/'best.pt','--output-dir',folder,'--split','test','--confirm-test-locked','--protocol-lock',lock,'--corruption',corruption)
                    dest=args.out/key/'runs'/arm/f'seed_{seed}'
                    shutil.copytree(folder,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.pt','*.tmp'))
        write(args.out/'evaluation_completion.json',{'completed_utc':now(),'test_runs':12,'additional_synthetic_conditions':8,'all_training_completed_first':True})
    if args.phase in ['analyse','all']:
        for key in plan['models']:
            call('analyse_results.py','--runs',args.out/key/'runs','--manifest',args.manifest,'--data-root',args.data_root,'--out',args.out/key)
            call('audit_model_comparison.py','--runs',args.run_root/key/'runs','--manifest',args.manifest,'--data-root',args.data_root,'--protocol-lock',args.out/key/'protocol.lock.json','--out',args.out/key/'overfitting','--device',args.device)
        call('analyse_model_comparison.py','--study',args.out,'--manifest',args.manifest,'--data-root',args.data_root)
        write(args.out/'completion.json',{'completed_utc':now(),'status':'completed','run_count':12,'prior_test_exposure':True,'study_plan_sha256':identity['study_plan_sha256'],'training_source_sha256':identity['training_code_sha256']})
    print('REQUESTED_PHASE_COMPLETE',flush=True)

if __name__=='__main__':main()
