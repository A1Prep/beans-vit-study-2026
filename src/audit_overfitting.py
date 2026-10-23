"""Post-hoc overfitting diagnostic of already-selected, immutable checkpoints.

Score train and validation with identical deterministic preprocessing and model
mode. Reuse the existing test scores for context; do not train, tune, select new
checkpoints, or claim this diagnostic was specified before test inspection.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import experiment as exp
from validate_protocol import assert_locked


def history_diagnostic(history: pd.DataFrame, best_epoch: int) -> dict:
    selected = history.loc[history.epoch == best_epoch]
    if len(selected) != 1:
        raise ValueError('Selected epoch must appear exactly once')
    selected = selected.iloc[0]
    final = history.iloc[-1]
    later = history[history.epoch > best_epoch]
    return {'selected_epoch': int(best_epoch), 'final_epoch': int(final.epoch),
            'selected_validation_loss': float(selected.val_loss),
            'final_validation_loss': float(final.val_loss),
            'final_minus_selected_validation_loss': float(final.val_loss-selected.val_loss),
            'selected_minus_final_validation_f1_pp': float(100*(selected.val_macro_f1-final.val_macro_f1)),
            'post_selection_epochs': len(later),
            'post_selection_epochs_with_higher_validation_loss': int((later.val_loss>selected.val_loss).sum()),
            'final_minus_selected_augmented_training_loss': float(final.train_loss-selected.train_loss)}


def save_figure(fig, out, name):
    fig.savefig(out/(name+'.png'), dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(out/(name+'.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', required=True, type=Path, help='Original run tree containing best.pt and recorded metadata')
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--data-root', required=True, type=Path)
    ap.add_argument('--protocol-lock', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = exp.runtime_device(args.device)
    records = exp.read_manifest(args.manifest, args.data_root)
    lock = json.loads(args.protocol_lock.read_text(encoding='utf-8'))
    seeds = lock.get('seeds', lock.get('settings', {}).get('seeds'))
    if not seeds:
        raise ValueError('Protocol must enumerate the paired seeds')
    scores=[]; histories=[]; details={}
    for arm in ['frozen','partial']:
        for seed in seeds:
            folder=args.runs/arm/f'seed_{seed}'
            checkpoint=folder/'best.pt'
            provenance=assert_locked(args.manifest, checkpoint, args.protocol_lock,
                                     Path(exp.__file__))
            model,metadata=exp.load_checkpoint(checkpoint,device)
            run_out=args.out/arm/f'seed_{seed}'
            run_out.mkdir(parents=True,exist_ok=True)
            measurements={}
            for split in ['train','validation']:
                subset=exp.split_records(records,split)
                dataset=exp.BeansImages(subset,metadata['preprocessing'],training=False)
                loader=exp.make_loader(dataset,args.batch_size,0,device)
                metric,rows=exp.predict(model,loader,device,amp=metadata['amp'] and device.type=='cuda')
                y=np.array([row['label'] for row in rows]);p=np.array([row['predicted_label'] for row in rows])
                assert abs(f1_score(y,p,labels=[0,1,2],average='macro')-metric['macro_f1'])<1e-12
                assert abs(accuracy_score(y,p)-metric['accuracy'])<1e-12
                exp.write_predictions(run_out/f'predictions_{split}_clean.csv',rows)
                measurements[split]=metric
            # These are already published results, not a new test evaluation.
            test=json.loads((folder/'metrics_test.json').read_text(encoding='utf-8'))
            if test['checkpoint_sha256']!=provenance['checkpoint_sha256']:
                raise ValueError('Recorded test metric belongs to a different checkpoint')
            saved_val=pd.read_csv(folder/'validation_predictions.csv').sort_values('image_id')
            clean_val=pd.read_csv(run_out/'predictions_validation_clean.csv').sort_values('image_id')
            if not saved_val[['image_id','label','predicted_label']].reset_index(drop=True).equals(
                    clean_val[['image_id','label','predicted_label']].reset_index(drop=True)):
                raise ValueError('Selected-checkpoint validation predictions changed')
            record={'arm':arm,'seed':seed,'selected_epoch':metadata['epoch']}
            for split,metric in [*measurements.items(),('test',test)]:
                for key in ['n','accuracy','macro_f1','loss']:
                    record[f'{split}_{key}']=metric[key]
            record['train_minus_validation_f1_pp']=100*(measurements['train']['macro_f1']-measurements['validation']['macro_f1'])
            record['train_minus_test_f1_pp']=100*(measurements['train']['macro_f1']-test['macro_f1'])
            record['validation_minus_train_loss']=measurements['validation']['loss']-measurements['train']['loss']
            scores.append(record)
            history=pd.read_csv(folder/'history.csv')
            histories.append({'arm':arm,'seed':seed,**history_diagnostic(history,metadata['epoch'])})
            exp.write_json(run_out/'metrics.json',{'diagnostic':'post-hoc, no optimization',
                'checkpoint_sha256':provenance['checkpoint_sha256'],'measurements':measurements,
                'existing_test_metrics':test,'validation_predictions_match_original':True})
            details[f'{arm}_{seed}']={'history':history,'checkpoint_epoch':metadata['epoch']}
            print(json.dumps(record),flush=True)
            del model
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    table=pd.DataFrame(scores);table.to_csv(args.out/'checkpoint_gaps.csv',index=False)
    temporal=pd.DataFrame(histories);temporal.to_csv(args.out/'history_diagnostics.csv',index=False)
    summary=table.groupby('arm').agg(
        runs=('seed','count'),train_macro_f1_mean=('train_macro_f1','mean'),
        validation_macro_f1_mean=('validation_macro_f1','mean'),test_macro_f1_mean=('test_macro_f1','mean'),
        train_accuracy_mean=('train_accuracy','mean'),validation_accuracy_mean=('validation_accuracy','mean'),
        train_loss_mean=('train_loss','mean'),validation_loss_mean=('validation_loss','mean'),
        train_minus_validation_f1_pp_mean=('train_minus_validation_f1_pp','mean'),
        train_minus_test_f1_pp_mean=('train_minus_test_f1_pp','mean'))
    summary.to_csv(args.out/'summary.csv')
    exp.write_json(args.out/'audit.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Post-hoc diagnostic of fixed selected checkpoints, after original test evaluation',
        'optimization_performed':False,'new_test_evaluation_performed':False,
        'train_and_validation_preprocessing':'Identical clean resize/normalization, model.eval(), no augmentation',
        'validation_predictions_match_original':True,
        'diagnostic_source_sha256':exp.file_sha256(Path(__file__)),
        'protocol_lock_sha256':exp.file_sha256(args.protocol_lock),
        'summary':json.loads(summary.reset_index().to_json(orient='records')),
        'limitations':['Training and validation are reused, not independent external evaluation.',
                      'A small gap cannot prove absence of overfitting, leakage or distribution shift.',
                      'Online augmented training losses use evolving parameters, unlike fixed-checkpoint evaluation.']})
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,len(seeds),figsize=(12,6.8),squeeze=False)
    for i,arm in enumerate(['frozen','partial']):
        for j,seed in enumerate(seeds):
            d=details[f'{arm}_{seed}'];h=d['history'];ax=axs[i,j]
            ax.plot(h.epoch,h.train_loss,label='Train (online, augmented)',color='#376680')
            ax.plot(h.epoch,h.val_loss,label='Validation (clean)',color='#b4623d')
            ax.axvline(d['checkpoint_epoch'],linestyle=':',color='#4e5c58',label='Selected epoch')
            ax.set_title(f'{arm.capitalize()}, seed {seed}');ax.set_xlabel('Epoch');ax.set_ylabel('Cross entropy')
            ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.04),frameon=False)
    fig.tight_layout(h_pad=2,w_pad=1.8)
    save_figure(fig,args.out,'all_seed_learning_curves')
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,arm in zip(axs,['frozen','partial']):
        frame=table[table.arm==arm]
        for _,row in frame.iterrows():
            ax.plot(range(3),[row[f'{s}_macro_f1']*100 for s in ['train','validation','test']],
                    marker='o',label=f"Seed {int(row.seed)}")
        ax.set_xticks(range(3),['Train','Validation','Test'])
        ax.set_title(arm.capitalize());ax.set_ylim(80,101);ax.set_ylabel('Macro-F1 (%)');ax.grid(alpha=.15)
        ax.legend(fontsize=9)
    fig.suptitle('Selected checkpoints: identical clean preprocessing\nTest scores reused; post-hoc diagnostic, no retraining',fontsize=11,y=1.08)
    fig.tight_layout();save_figure(fig,args.out,'checkpoint_generalisation')
    print('OVERFITTING_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
