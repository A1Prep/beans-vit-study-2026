"""Recompute metrics, paired uncertainty and figures from saved predictions.

This program never trains or selects a model. Run only after the protocol is
locked and predictions for both approaches have been saved.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

NAMES = ['Angular leaf spot', 'Bean rust', 'Healthy']
COLORS = {'frozen':'#376680', 'partial':'#b4623d'}
LABELS = {'frozen':'Frozen backbone', 'partial':'Partial fine-tuning'}

def metrics(y, p):
    return {'accuracy':float(accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)),
            'classification_report':classification_report(y,p,labels=[0,1,2],target_names=NAMES,zero_division=0,output_dict=True),
            'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist(),
            'diseased_as_healthy':int(((y!=2)&(p==2)).sum()),
            'healthy_as_diseased':int(((y==2)&(p!=2)).sum())}

def paired_bootstrap(y, a, b, n=2000, seed=20261020):
    """Fixed class counts; same resampled evaluation images in both arms."""
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(y==c) for c in range(3)]
    if any(len(g)==0 for g in groups): raise ValueError('All three classes required')
    values=[]
    for _ in range(n):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        values.append(f1_score(y[ix],b[ix],labels=[0,1,2],average='macro',zero_division=0)-f1_score(y[ix],a[ix],labels=[0,1,2],average='macro',zero_division=0))
    return {'difference_partial_minus_frozen':float(f1_score(y,b,labels=[0,1,2],average='macro')-f1_score(y,a,labels=[0,1,2],average='macro')),
            'ci95_percentile':np.quantile(values,[.025,.975]).tolist(),
            'resamples':n,'seed':seed,'method':'paired class-stratified percentile bootstrap',
            'limitation':'Conditional on observed classes and independent audited images; unknown plant/farm dependence is not corrected.'}

def save(fig, folder, name):
    fig.savefig(folder/(name+'.png'),dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,default=Path('.'),help='Folder containing train/validation/test images; relocates saved manifests')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    manifest=pd.read_csv(args.manifest)
    active=manifest[manifest.include.astype(str).str.lower().isin(['true','1'])].copy()
    expected=active.loc[active.split=='test',['image_id','label']].sort_values('image_id').reset_index(drop=True)
    if expected.empty or expected.image_id.duplicated().any():raise ValueError('Invalid audited test manifest')
    args.out.mkdir(parents=True,exist_ok=True)
    figs=args.out/'figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    rows=[];detail={};preds={}
    for arm in ['frozen','partial']:
        for folder in sorted((args.runs/arm).glob('seed_*')):
            file=folder/'predictions_test.csv'
            if not file.exists():continue
            seed=int(folder.name.split('_')[-1])
            p=pd.read_csv(file).sort_values('image_id').reset_index(drop=True)
            if p.image_id.duplicated().any(): raise ValueError('Repeated test image id')
            if not p[['image_id','label']].equals(expected):raise ValueError(f'Predictions differ from audited test units: {file}')
            probabilities=p[['prob_0','prob_1','prob_2']].to_numpy()
            if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1),1,atol=1e-5):raise ValueError('Invalid softmax scores')
            if not np.array_equal(probabilities.argmax(axis=1),p.prediction.to_numpy()):raise ValueError('Prediction does not match probability argmax')
            y=p.label.to_numpy();yp=p.prediction.to_numpy()
            m=metrics(y,yp)
            summary=json.loads((folder/'summary.json').read_text())
            row={'arm':arm,'seed':seed,'n_test':len(p),'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
                 'best_epoch':summary['best_epoch'],'training_seconds':summary['training_seconds'],
                 'epochs_completed':summary['epochs_completed'],'wall_seconds':summary['wall_seconds'],
                 'mean_training_epoch_seconds':summary['training_seconds']/summary['epochs_completed'],
                 'peak_cuda_allocated_bytes':summary['peak_cuda_allocated_bytes'],
                 'trainable_params':summary['trainable_params'],'total_params':summary['total_params']}
            rows.append(row);detail[f'{arm}_{seed}']=m;preds[arm,seed]=p
    table=pd.DataFrame(rows)
    if table.empty:raise RuntimeError('No saved test predictions found')
    if set(table.loc[table.arm=='frozen','seed'])!=set(table.loc[table.arm=='partial','seed']):raise ValueError('Arm seed sets differ')
    table.to_csv(args.out/'run_metrics.csv',index=False)
    grouped=table.groupby('arm').agg(runs=('seed','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),training_seconds_mean=('training_seconds','mean'),trainable_params=('trainable_params','first'))
    grouped.to_csv(args.out/'comparison.csv')
    paired=table.pivot(index='seed',columns='arm',values='macro_f1')
    paired['partial_minus_frozen']=paired['partial']-paired['frozen']
    paired.to_csv(args.out/'paired_seed_differences.csv')
    ref=17
    if ('frozen',ref) not in preds or ('partial',ref) not in preds:raise ValueError('Prespecified reference seed17 missing')
    a=preds['frozen',ref];b=preds['partial',ref]
    if not a[['image_id','label']].equals(b[['image_id','label']]):raise ValueError('Unmatched test samples')
    ci=paired_bootstrap(a.label.to_numpy(),a.prediction.to_numpy(),b.prediction.to_numpy())
    def image_path(row):
        candidates=[Path(row.path),args.data_root/str(row.path),args.data_root/str(row.image_id)]
        for candidate in candidates:
            if candidate.is_file(): return str(candidate)
        raise FileNotFoundError(row.image_id)
    active['path']=[image_path(row) for row in active.itertuples()]
    majority=int(active.loc[active.split=='train','label'].value_counts().sort_index().idxmax())
    sanity=metrics(a.label.to_numpy(),np.full(len(a),majority))
    stress=[]
    for arm in ['frozen','partial']:
        for condition in ['clean','blur','brightness']:
            file=args.runs/arm/f'seed_{ref}'/('predictions_test'+('' if condition=='clean' else '_'+condition)+'.csv')
            if not file.exists():continue
            frame=pd.read_csv(file).sort_values('image_id').reset_index(drop=True)
            if not a[['image_id','label']].equals(frame[['image_id','label']]):raise ValueError('Stress-test samples differ')
            score=metrics(frame.label.to_numpy(),frame.prediction.to_numpy())
            stress.append({'arm':arm,'seed':ref,'condition':condition,'accuracy':score['accuracy'],
                           'macro_f1':score['macro_f1'],'diseased_as_healthy':score['diseased_as_healthy']})
    pd.DataFrame(stress).to_csv(args.out/'stress_metrics.csv',index=False)
    per_class=[]
    for arm in ['frozen','partial']:
        for name in NAMES:
            per_class.append({'arm':arm,'seed':ref,'class':name,**detail[f'{arm}_{ref}']['classification_report'][name]})
    pd.DataFrame(per_class).to_csv(args.out/'per_class_metrics.csv',index=False)
    result={'reference_seed':ref,'paired_macro_f1_difference':ci,'majority_class_baseline':{'label':majority,**sanity},'runs':detail,'summary':json.loads(grouped.reset_index().to_json(orient='records')),'stress_tests':stress}
    (args.out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

    # Dataset balance and deterministic training examples.
    counts=pd.crosstab(active.label,active.split).reindex(index=[0,1,2],columns=['train','validation','test'],fill_value=0)
    fig,axs=plt.subplots(1,2,figsize=(11,3.8),gridspec_kw={'width_ratios':[1,1.1]})
    counts.index=NAMES
    counts.plot(kind='bar',ax=axs[0],color=['#376680','#c79c58','#73957c'],rot=0)
    axs[0].set_ylabel('Images');axs[0].set_xlabel('');axs[0].tick_params(axis='x',labelsize=10)
    axs[0].set_title('Audited dataset splits',pad=34);axs[0].legend(fontsize=10,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False)
    axs[1].axis('off')
    examples=[]
    for label in range(3):
        row=active[(active.split=='train')&(active.label==label)].sort_values('image_id').iloc[0]
        examples.append(np.asarray(Image.open(row.path).convert('RGB').resize((240,240))))
    axs[1].imshow(np.concatenate(examples,axis=1));axs[1].set_title('Training examples in class order',pad=12)
    save(fig,figs,'dataset_overview')

    fig,axs=plt.subplots(1,2,figsize=(10.5,3.9))
    for arm in ['frozen','partial']:
        hist=pd.read_csv(args.runs/arm/f'seed_{ref}'/'history.csv')
        axs[0].plot(hist.epoch,hist.train_loss,label=LABELS[arm]+' train',color=COLORS[arm])
        axs[0].plot(hist.epoch,hist.val_loss,label=LABELS[arm]+' validation',color=COLORS[arm],linestyle='--')
        axs[1].plot(hist.epoch,hist.val_macro_f1,label=LABELS[arm],color=COLORS[arm],marker='o',markersize=3)
    axs[0].set_title('Loss by epoch');axs[0].set_ylabel('Cross entropy');axs[0].legend(fontsize=11)
    axs[1].set_title('Validation macro-F1');axs[1].set_ylabel('Macro-F1');axs[1].legend(fontsize=11)
    for ax in axs:ax.set_xlabel('Epoch');ax.grid(alpha=.15)
    fig.suptitle('Prespecified reference run seed 17',fontsize=12,y=1.02)
    save(fig,figs,'learning_curves')

    fig,axs=plt.subplots(1,2,figsize=(10.2,4.2))
    for arm,ax in zip(['frozen','partial'],axs):
        mat=np.array(detail[f'{arm}_{ref}']['confusion_matrix'])
        ax.imshow(mat,cmap='Blues',vmin=0,vmax=max(mat.max(),1))
        for (i,j),v in np.ndenumerate(mat):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>mat.max()/2 else '#142330',fontsize=13)
        ax.set_xticks(range(3),['Leaf spot','Rust','Healthy']);ax.set_yticks(range(3),['Leaf spot','Rust','Healthy'])
        ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(LABELS[arm])
    fig.suptitle('Test confusion matrices for seed 17',fontsize=12,y=1.02)
    save(fig,figs,'confusion_matrices')

    fig,axs=plt.subplots(1,2,figsize=(10.3,3.7))
    for i,arm in enumerate(['frozen','partial']):
        r=table[table.arm==arm]
        axs[0].bar(i,r.macro_f1.mean(),yerr=(r.macro_f1.std(ddof=1) if len(r)>1 else 0),color=COLORS[arm],width=.55,capsize=5)
        axs[0].scatter(np.full(len(r),i),r.macro_f1,color='black',s=18,zorder=3)
        axs[1].bar(i,r.training_seconds.mean()/60,color=COLORS[arm],width=.55)
    for ax in axs:ax.set_xticks([0,1],['Frozen','Partial']);ax.grid(axis='y',alpha=.15)
    axs[0].set_ylabel('Test macro-F1');axs[0].set_ylim(0,1.03);axs[0].set_title('Mean and training-seed SD')
    axs[1].set_ylabel('Minutes');axs[1].set_title('Mean measured training time')
    save(fig,figs,'performance_and_cost')

    if len(stress)==6:
        fig,ax=plt.subplots(figsize=(7.5,3.6))
        frame=pd.DataFrame(stress)
        for i,arm in enumerate(['frozen','partial']):
            values=frame[frame.arm==arm].set_index('condition').loc[['clean','blur','brightness'],'macro_f1']
            ax.bar(np.arange(3)+(i-.5)*.34,values,width=.34,label=LABELS[arm],color=COLORS[arm])
        ax.set_xticks(range(3),['Original','Gaussian blur (radius 1)','Brightness × 0.7'])
        ax.set_ylabel('Test macro-F1');ax.set_ylim(0,1.03);ax.legend(fontsize=10,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False);ax.grid(axis='y',alpha=.15)
        ax.set_title('Synthetic sensitivity check, seed 17',pad=35);save(fig,figs,'stress_sensitivity')

    # Fixed rule: highest-confidence partial-model errors, then lowest-confidence
    # correct examples; ties broken by image_id. This is descriptive post-test QA.
    b=b.copy();b['confidence']=b[['prob_0','prob_1','prob_2']].max(axis=1)
    wrong=b[b.label!=b.prediction].sort_values(['confidence','image_id'],ascending=[False,True]).head(4)
    right=b[b.label==b.prediction].sort_values(['confidence','image_id'],ascending=[True,True]).head(max(2,6-len(wrong)))
    chosen=pd.concat([wrong,right]).head(6)
    chosen.to_csv(args.out/'selected_examples.csv',index=False)
    paths=active.set_index('image_id').path.to_dict()
    fig,axs=plt.subplots(2,3,figsize=(10.3,6.6))
    for ax,(_,r) in zip(axs.flat,chosen.iterrows()):
        ax.imshow(Image.open(paths[r.image_id]).convert('RGB'));ax.axis('off')
        ax.set_title(f'True: {NAMES[int(r.label)]}\nPredicted: {NAMES[int(r.prediction)]} ({r.confidence:.3f})',fontsize=11.5,color='#a0442d' if r.label!=r.prediction else '#244f3b')
    for ax in list(axs.flat)[len(chosen):]:ax.axis('off')
    fig.suptitle('Selected partial-model test predictions\nScores are softmax confidence, not calibrated probabilities',fontsize=11,y=1.01)
    save(fig,figs,'sample_predictions')
    print(json.dumps({'comparison':result['summary'],'paired_difference':ci,'reference_errors':{arm:detail[f'{arm}_{ref}']['diseased_as_healthy'] for arm in ['frozen','partial']}},indent=2))

if __name__=='__main__': main()
