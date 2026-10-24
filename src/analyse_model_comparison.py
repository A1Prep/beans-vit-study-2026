"""Audit and visualise the fixed exploratory DINOv2 / SigLIP2 comparison.

Read saved predictions only: this program performs no training, inference,
checkpoint selection or hyperparameter search. Original source results are not
modified. The published test split was known before this follow-up experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from analyse_results import metrics, paired_bootstrap, save, NAMES

MODELS = ('dinov2', 'siglip2')
ARMS = ('frozen', 'partial')
SEEDS = (17, 42, 2026)
REFERENCE_SEED = 17
MODEL_NAMES = {'dinov2': 'DINOv2-S/14', 'siglip2': 'SigLIP2-B/16'}
MODEL_IDS = {'dinov2': 'facebook/dinov2-small',
             'siglip2': 'google/siglip2-base-patch16-224'}
COLORS = {'dinov2': '#326783', 'siglip2': '#b35d39'}
LIMITS = [
    'Exploratory follow-up: the original DINOv2 test results were known before this study; the reused test split is not fresh confirmatory evidence.',
    'Model sizes, pretraining data, pooling and native preprocessing differ; the comparison does not isolate model age or pretraining objective.',
    'Fixed learning rates and adaptation depth were not separately optimised for each model.',
    'Three training seeds describe limited optimisation variability, not independent datasets or external population uncertainty.',
    'Unknown plant/farm dependence and pretraining overlap remain; no external dataset establishes geographic or field generalisation.',
    'A recent competitive model family does not establish the newest or best model in October 2026, or state-of-the-art performance on Beans.',
    'Clean training/validation diagnostics reuse development data; a small gap does not prove absence of overfitting.',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_close(actual, expected, context, atol=1e-10):
    if not np.isfinite(float(actual)) or not np.isclose(float(actual), float(expected), rtol=0, atol=atol):
        raise ValueError(f'{context}: saved {actual!r}, recomputed {expected!r}')


def summarise_epoch_times(values):
    """Describe every observed training epoch, including anomalous long times.

    Quantiles use NumPy's explicit linear interpolation. No observations are
    trimmed, corrected or assigned an unverified cause.
    """
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('Epoch times must be a nonempty vector of finite, nonnegative observations')
    q1, median, q3 = np.quantile(times, [.25, .5, .75], method='linear')
    return {'count': int(len(times)), 'total_seconds': float(times.sum()),
            'mean_seconds': float(times.mean()), 'median_seconds': float(median),
            'q1_seconds': float(q1), 'q3_seconds': float(q3), 'iqr_seconds': float(q3-q1),
            'min_seconds': float(times.min()), 'max_seconds': float(times.max())}


def active_manifest(path):
    frame = pd.read_csv(path)
    needed = {'image_id', 'label', 'split', 'include'}
    if not needed.issubset(frame.columns):
        raise ValueError(f'Manifest missing columns: {needed-set(frame.columns)}')
    frame = frame[frame.include.astype(str).str.lower().isin(['true', '1'])].copy()
    if frame.empty or frame.image_id.duplicated().any() or frame.image_id.isna().any():
        raise ValueError('Active manifest requires unique, nonempty image identifiers')
    if set(frame.label) != {0, 1, 2} or set(frame.split) != {'train', 'validation', 'test'}:
        raise ValueError('Expected all three classes and published train/validation/test splits')
    for split in ['train', 'validation', 'test']:
        if set(frame.loc[frame.split == split, 'label']) != {0, 1, 2}:
            raise ValueError(f'All classes are required in {split}')
    return frame


def read_predictions(path, expected):
    """Match every audited unit and independently score its saved prediction."""
    frame = pd.read_csv(path).sort_values('image_id').reset_index(drop=True)
    cols = ['image_id', 'label', 'prob_0', 'prob_1', 'prob_2']
    if not set(cols).issubset(frame.columns):
        raise ValueError(f'Prediction columns missing: {path}')
    target = expected[['image_id', 'label']].sort_values('image_id').reset_index(drop=True)
    if frame.image_id.duplicated().any() or not frame[['image_id', 'label']].equals(target):
        raise ValueError(f'Predictions differ from audited evaluation units: {path}')
    field = 'prediction' if 'prediction' in frame else 'predicted_label'
    if field not in frame:
        raise ValueError(f'Predicted label missing: {path}')
    if 'predicted_label' in frame and not np.array_equal(frame[field], frame.predicted_label):
        raise ValueError(f'Conflicting predicted-label columns: {path}')
    frame['prediction'] = frame[field]
    probabilities = frame[['prob_0', 'prob_1', 'prob_2']].to_numpy(dtype=float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1+1e-6).any()
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError(f'Invalid softmax scores: {path}')
    if not np.array_equal(probabilities.argmax(axis=1), frame.prediction.to_numpy()):
        raise ValueError(f'Prediction differs from score argmax: {path}')
    frame['confidence'] = probabilities.max(axis=1)
    return frame, metrics(frame.label.to_numpy(), frame.prediction.to_numpy())


def check_saved_metrics(saved, recomputed, n, context):
    if int(saved['n']) != n or saved['confusion_matrix'] != recomputed['confusion_matrix']:
        raise ValueError(f'Count or confusion matrix mismatch: {context}')
    for name in ['accuracy', 'macro_f1']:
        assert_close(saved[name], recomputed[name], f'{context} {name}')
    if len(saved['per_class']) != 3 or {int(x['label']) for x in saved['per_class']} != {0, 1, 2}:
        raise ValueError(f'Incomplete per-class metrics: {context}')
    for row in saved['per_class']:
        cls = recomputed['classification_report'][NAMES[int(row['label'])]]
        for name, source in [('precision', 'precision'), ('recall', 'recall'), ('f1', 'f1-score'), ('support', 'support')]:
            assert_close(row[name], cls[source], f'{context} class {row["label"]} {name}')


def collect_results(study, manifest_path):
    """Reject missing/extra runs or inconsistent saved metrics before plotting."""
    study = Path(study)
    active = active_manifest(manifest_path)
    expected = {split: active[active.split == split] for split in ['train', 'validation', 'test']}
    wanted = {(m, a, s) for m in MODELS for a in ARMS for s in SEEDS}
    actual = set()
    for model in MODELS:
        for path in (study/model/'runs').glob('*/seed_*'):
            if path.is_dir():
                try:
                    actual.add((model, path.parent.name, int(path.name.removeprefix('seed_'))))
                except ValueError as exc:
                    raise ValueError(f'Unexpected run directory: {path}') from exc
    if actual != wanted:
        raise ValueError(f'Exactly twelve planned runs required; missing={sorted(wanted-actual)}, extra={sorted(actual-wanted)}')
    rows, detail, predictions, histories, clean, per_class, source_hashes = [], {}, {}, {}, [], [], {}
    amendment_path = study/'timing_analysis_amendment.json'
    amendment = read_json(amendment_path)
    if (amendment.get('before_new_test_evaluation') is not True
            or amendment.get('training_changes') is not False
            or amendment.get('raw_times_preserved') is not True):
        raise ValueError('Timing amendment must record its pre-evaluation, descriptive-only scope')
    source_hashes['timing_analysis_amendment.json'] = digest(amendment_path)
    for model in MODELS:
        gaps_path = study/model/'overfitting'/'checkpoint_gaps.csv'
        gaps = pd.read_csv(gaps_path)
        if len(gaps) != 6 or gaps[['arm', 'seed']].duplicated().any() or set(map(tuple, gaps[['arm', 'seed']].to_numpy())) != {(a, s) for a in ARMS for s in SEEDS}:
            raise ValueError(f'Exactly six selected-checkpoint diagnostics required: {gaps_path}')
        source_hashes[gaps_path.relative_to(study).as_posix()] = digest(gaps_path)
        for arm in ARMS:
            for seed in SEEDS:
                folder = study/model/'runs'/arm/f'seed_{seed}'
                config = read_json(folder/'config.json')
                summary = read_json(folder/'summary.json')
                saved = read_json(folder/'metrics_test.json')
                history = pd.read_csv(folder/'history.csv')
                for content, name in [(config, 'config'), (summary, 'summary'), (saved, 'test metrics')]:
                    if content['arm'] != arm or int(content['seed']) != seed:
                        raise ValueError(f'Run identity mismatch in {name}: {folder}')
                if config['model_id'] != MODEL_IDS[model]:
                    raise ValueError(f'Model identity mismatch: {folder}')
                if config.get('manifest_sha256') != digest(manifest_path):
                    raise ValueError(f'Manifest hash mismatch: {folder}')
                if saved.get('split') != 'test' or saved.get('corruption') != 'clean':
                    raise ValueError(f'Expected clean test evaluation: {folder}')
                if int(saved['checkpoint_epoch']) != int(summary['best_epoch']):
                    raise ValueError(f'Test checkpoint epoch differs from selected epoch: {folder}')
                if len(history) != int(summary['epochs_completed']) or list(history.epoch) != list(range(1, len(history)+1)):
                    raise ValueError(f'Incomplete or unordered epoch history: {folder}')
                if int(summary['best_epoch']) not in history.epoch.values or int(summary['epochs_completed']) < 1:
                    raise ValueError(f'Selected epoch not in history: {folder}')
                timing = summarise_epoch_times(history.train_seconds.to_numpy())
                assert_close(summary['training_seconds'], timing['total_seconds'], f'{folder} total training time', atol=1e-6)
                for field in ['training_seconds', 'wall_seconds', 'peak_cuda_allocated_bytes', 'trainable_params', 'total_params']:
                    if not np.isfinite(float(summary[field])) or float(summary[field]) < 0:
                        raise ValueError(f'Invalid {field}: {folder}')
                if int(summary['trainable_params']) > int(summary['total_params']):
                    raise ValueError(f'Trainable parameter count exceeds total: {folder}')
                frame, score = read_predictions(folder/'predictions_test.csv', expected['test'])
                check_saved_metrics(saved, score, len(frame), str(folder))
                row = {'model': model, 'arm': arm, 'seed': seed, 'n_test': len(frame),
                       'accuracy': score['accuracy'], 'macro_f1': score['macro_f1'],
                       **{k: summary[k] for k in ['best_epoch', 'epochs_completed', 'training_seconds', 'wall_seconds', 'peak_cuda_allocated_bytes', 'trainable_params', 'total_params']}}
                row['mean_training_epoch_seconds'] = row['training_seconds']/row['epochs_completed']
                row['median_observed_training_epoch_seconds'] = timing['median_seconds']
                gap = gaps[(gaps.arm == arm) & (gaps.seed == seed)].iloc[0]
                if int(gap.selected_epoch) != int(summary['best_epoch']):
                    raise ValueError(f'Clean diagnostic checkpoint differs: {folder}')
                for split in ['train', 'validation', 'test']:
                    if split == 'test':
                        split_score, n = score, len(frame)
                    else:
                        clean_path = study/model/'overfitting'/arm/f'seed_{seed}'/f'predictions_{split}_clean.csv'
                        clean_frame, split_score = read_predictions(clean_path, expected[split])
                        n = len(clean_frame)
                        source_hashes[clean_path.relative_to(study).as_posix()] = digest(clean_path)
                    if int(gap[f'{split}_n']) != n:
                        raise ValueError(f'Clean diagnostic count mismatch: {folder} {split}')
                    for metric in ['accuracy', 'macro_f1']:
                        assert_close(gap[f'{split}_{metric}'], split_score[metric], f'{folder} clean {split} {metric}')
                        row[f'{split}_{metric}'] = split_score[metric]
                    if not np.isfinite(float(gap[f'{split}_loss'])) or float(gap[f'{split}_loss']) < 0:
                        raise ValueError(f'Invalid clean diagnostic loss: {folder} {split}')
                    row[f'{split}_loss'] = float(gap[f'{split}_loss'])
                for field, value in [
                    ('train_minus_validation_f1_pp', 100*(row['train_macro_f1']-row['validation_macro_f1'])),
                    ('train_minus_test_f1_pp', 100*(row['train_macro_f1']-row['test_macro_f1'])),
                    ('validation_minus_train_loss', row['validation_loss']-row['train_loss'])]:
                    assert_close(gap[field], value, f'{folder} {field}')
                    row[field] = value
                for name in ['config.json', 'summary.json', 'metrics_test.json', 'predictions_test.csv', 'history.csv']:
                    source_hashes[(folder/name).relative_to(study).as_posix()] = digest(folder/name)
                for label, class_name in enumerate(NAMES):
                    cls = score['classification_report'][class_name]
                    per_class.append({'model': model, 'arm': arm, 'seed': seed, 'label': label, 'class': class_name,
                                      'precision': cls['precision'], 'recall': cls['recall'], 'f1': cls['f1-score'], 'support': int(cls['support'])})
                rows.append(row)
                detail[f'{model}_{arm}_{seed}'] = score
                predictions[model, arm, seed] = frame
                histories[model, arm, seed] = history
                clean.append({'model': model, **gap.to_dict()})
    table = pd.DataFrame(rows)
    summary_rows = []
    for (model, arm), group in table.groupby(['model', 'arm'], sort=False):
        for field in ['trainable_params', 'total_params']:
            if group[field].nunique() != 1:
                raise ValueError(f'Parameter counts differ across seeds for {model}/{arm}')
        result = {'model': model, 'arm': arm, 'runs': len(group)}
        for field in ['accuracy', 'macro_f1', 'training_seconds', 'mean_training_epoch_seconds', 'train_macro_f1', 'validation_macro_f1', 'train_minus_validation_f1_pp', 'train_minus_test_f1_pp']:
            result[field+'_mean'] = float(group[field].mean())
            result[field+'_sd'] = float(group[field].std(ddof=1))
        result.update({'pooled_training_seconds': float(group.training_seconds.sum()),
                       'pooled_epochs': int(group.epochs_completed.sum()),
                       'pooled_training_seconds_per_epoch': float(group.training_seconds.sum()/group.epochs_completed.sum()),
                       'trainable_params': int(group.trainable_params.iloc[0]), 'total_params': int(group.total_params.iloc[0]),
                       'peak_cuda_allocated_bytes_max': int(group.peak_cuda_allocated_bytes.max())})
        pooled_times = np.concatenate([histories[model, arm, seed].train_seconds.to_numpy() for seed in SEEDS])
        pooled = summarise_epoch_times(pooled_times)
        for name in ['median_seconds', 'q1_seconds', 'q3_seconds', 'iqr_seconds', 'min_seconds', 'max_seconds']:
            result['pooled_epoch_'+name] = pooled[name]
        summary_rows.append(result)
    paired_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            a = table[(table.model == 'dinov2') & (table.arm == arm) & (table.seed == seed)].iloc[0]
            b = table[(table.model == 'siglip2') & (table.arm == arm) & (table.seed == seed)].iloc[0]
            paired_rows.append({'arm': arm, 'seed': seed, 'dinov2_macro_f1': a.macro_f1, 'siglip2_macro_f1': b.macro_f1,
                                'siglip2_minus_dinov2_macro_f1': b.macro_f1-a.macro_f1,
                                'dinov2_accuracy': a.accuracy, 'siglip2_accuracy': b.accuracy,
                                'siglip2_minus_dinov2_accuracy': b.accuracy-a.accuracy})
    a = predictions['dinov2', 'partial', REFERENCE_SEED]
    b = predictions['siglip2', 'partial', REFERENCE_SEED]
    interval = paired_bootstrap(a.label.to_numpy(), a.prediction.to_numpy(), b.prediction.to_numpy(), n=2000, seed=20261023)
    interval['difference_siglip2_minus_dinov2'] = interval.pop('difference_partial_minus_frozen')
    interval.update({'arm': 'partial', 'reference_seed': REFERENCE_SEED,
                     'scope': 'Descriptive interval for the seed-17 model pair on the previously evaluated test split; not an interval for the three-seed mean or a fresh confirmatory test.'})
    majority = int(expected['train'].label.value_counts().sort_index().idxmax())
    baseline = metrics(a.label.to_numpy(), np.full(len(a), majority))
    payload = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'study_status': 'Exploratory comparison on a previously evaluated test split',
        'prior_test_exposure': True, 'run_count': 12, 'seeds': list(SEEDS), 'reference_seed': REFERENCE_SEED,
        'primary_metric': 'macro-F1', 'primary_comparison': interval,
        'majority_class_baseline': {'label': majority, 'selection': 'Most frequent training class; smallest label breaks ties', **baseline},
        'summary': summary_rows, 'paired_model_differences': paired_rows,
        'runs': detail, 'clean_generalisation': clean,
        'timing_analysis': {
            'amendment': amendment,
            'raw_times_preserved': True,
            'figure_metric': 'Median observed training epoch time, pooled across all completed epochs in the three seeds',
            'quantiles': 'NumPy linear interpolation; IQR = Q3 - Q1; every observed epoch retained',
            'interpretation': 'Descriptive measured durations, not corrected hardware benchmarks. Different completed epoch counts weight seeds differently in pooled epoch summaries; raw run totals, run means and pooled means remain reported. No cause is assigned to unusually long epochs.'},
        'interpretation_limits': LIMITS,
        'verification': {'all_test_units_match_manifest': True, 'saved_test_metrics_match_recomputed': True,
                         'all_clean_split_accuracy_and_f1_recomputed': True, 'raw_sources_unchanged': True,
                         'manifest_sha256': digest(manifest_path), 'analysis_source_sha256': digest(__file__),
                         'source_hashes': source_hashes},
    }
    return {'active': active, 'table': table, 'summary': pd.DataFrame(summary_rows), 'paired': pd.DataFrame(paired_rows),
            'per_class': pd.DataFrame(per_class), 'payload': payload, 'predictions': predictions, 'histories': histories}


def make_figures(data, folder, data_root):
    folder.mkdir(parents=True, exist_ok=True)
    table, summary = data['table'], data['summary']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    configurations = [(m, a) for m in MODELS for a in ARMS]
    labels = [f'{MODEL_NAMES[m]}\n{a.capitalize()}' for m, a in configurations]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for i, (model, arm) in enumerate(configurations):
        rows = table[(table.model == model) & (table.arm == arm)]
        group = summary[(summary.model == model) & (summary.arm == arm)].iloc[0]
        alpha = .62 if arm == 'frozen' else 1
        axs[0].bar(i, 100*group.macro_f1_mean, yerr=100*group.macro_f1_sd,
                   color=COLORS[model], alpha=alpha, width=.65, capsize=4)
        axs[0].scatter(i+np.linspace(-.09, .09, len(rows)), rows.macro_f1*100, color='#172329', s=19, zorder=3)
        axs[1].bar(i, group.pooled_epoch_median_seconds, color=COLORS[model], alpha=alpha, width=.65)
        axs[1].errorbar(i, group.pooled_epoch_median_seconds,
                       yerr=np.array([[group.pooled_epoch_median_seconds-group.pooled_epoch_q1_seconds],
                                      [group.pooled_epoch_q3_seconds-group.pooled_epoch_median_seconds]]),
                       color='#172329', capsize=4, fmt='none')
        axs[1].text(i, group.pooled_epoch_q3_seconds, f'{group.pooled_epoch_median_seconds:.1f}', ha='center', va='bottom', fontsize=10)
    for ax in axs:
        ax.set_xticks(range(4), labels, fontsize=9)
        ax.grid(axis='y', alpha=.15)
    axs[0].set_ylim(0, 104); axs[0].set_ylabel('Test macro-F1 (%)')
    axs[0].set_title('Mean, sample SD and individual seeds')
    axs[1].set_ylim(bottom=0); axs[1].margins(y=.15); axs[1].set_ylabel('Observed training epoch time (seconds)')
    axs[1].set_title('Pooled epoch median and interquartile range')
    save(fig, folder, 'performance_cost')

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4), constrained_layout=True)
    for model, ax in zip(MODELS, axs):
        matrix = np.asarray(data['payload']['runs'][f'{model}_partial_{REFERENCE_SEED}']['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(matrix.sum(axis=1)))
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center', color='white' if value > matrix.max()/2 else '#142330')
        ax.set_xticks(range(3), ['Leaf spot', 'Rust', 'Healthy'])
        ax.set_yticks(range(3), ['Leaf spot', 'Rust', 'Healthy'])
        ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title(MODEL_NAMES[model])
    fig.suptitle('Partial fine-tuning: test confusion matrices, reference seed 17')
    save(fig, folder, 'partial_confusion')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for model in MODELS:
        history = data['histories'][model, 'partial', REFERENCE_SEED]
        selected = int(table[(table.model == model) & (table.arm == 'partial') & (table.seed == REFERENCE_SEED)].best_epoch.iloc[0])
        axs[0].plot(history.epoch, history.train_loss, color=COLORS[model], label=MODEL_NAMES[model]+' train (augmented)')
        axs[0].plot(history.epoch, history.val_loss, color=COLORS[model], linestyle='--', label=MODEL_NAMES[model]+' validation')
        axs[1].plot(history.epoch, history.val_macro_f1*100, color=COLORS[model], marker='o', markersize=3, label=MODEL_NAMES[model])
        for ax in axs:
            ax.axvline(selected, linestyle=':', linewidth=1, color=COLORS[model], alpha=.7)
    for ax in axs:
        ax.set_xlabel('Epoch'); ax.grid(alpha=.15); ax.legend(fontsize=9)
    axs[0].set_ylabel('Cross entropy'); axs[0].set_ylim(bottom=0)
    axs[1].set_ylabel('Validation macro-F1 (%)'); axs[1].set_ylim(0, 102)
    fig.suptitle('Partial fine-tuning, seed 17; dotted lines mark selected checkpoints')
    save(fig, folder, 'reference_learning_curves')

    frame = data['predictions']['siglip2', 'partial', REFERENCE_SEED]
    # All errors take precedence, capped at six, before lowest-confidence correct examples.
    wrong = frame[frame.label != frame.prediction].sort_values(['confidence', 'image_id'], ascending=[False, True]).head(6)
    right = frame[frame.label == frame.prediction].sort_values(['confidence', 'image_id'], ascending=[True, True]).head(6-len(wrong))
    chosen = pd.concat([wrong, right]).head(6)
    chosen.to_csv(folder.parent/'siglip_selected_examples.csv', index=False)
    records = data['active'].set_index('image_id')
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 7), constrained_layout=True)
    for ax, (_, row) in zip(axs.flat, chosen.iterrows()):
        record = records.loc[row.image_id]
        candidates = [Path(data_root)/str(row.image_id)]
        if 'path' in record:
            candidates += [Path(str(record.path)), Path(data_root)/str(record.path)]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Image required for sample figure: {row.image_id}')
        with Image.open(path) as image:
            ax.imshow(image.convert('RGB'))
        ax.axis('off')
        ax.set_title(f'True: {NAMES[int(row.label)]}\nPredicted: {NAMES[int(row.prediction)]} ({row.confidence:.3f})',
                     fontsize=10.5, color='#a0442d' if row.label != row.prediction else '#244f3b')
    for ax in list(axs.flat)[len(chosen):]: ax.axis('off')
    fig.suptitle('SigLIP2 partial, seed 17: highest-confidence errors, then least-confident correct\nSoftmax confidence is not calibrated probability', fontsize=11)
    save(fig, folder, 'siglip_sample_predictions')

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.9), sharey=True, constrained_layout=True)
    minimum = table[['train_macro_f1', 'validation_macro_f1', 'test_macro_f1']].min().min()*100
    low = max(0, np.floor(minimum/5)*5-5)
    for ax, (model, arm) in zip(axs, configurations):
        rows = table[(table.model == model) & (table.arm == arm)]
        values = rows[['train_macro_f1', 'validation_macro_f1', 'test_macro_f1']].to_numpy()*100
        for seed, scores in zip(rows.seed, values):
            ax.plot(range(3), scores, marker='o', markersize=3, alpha=.38, color=COLORS[model], linewidth=1)
        ax.plot(range(3), values.mean(axis=0), color=COLORS[model], marker='s', linewidth=2.3, label='Mean')
        ax.set_xticks(range(3), ['Train', 'Val.', 'Test']); ax.set_ylim(low, 101)
        ax.set_title(f'{MODEL_NAMES[model]}\n{arm.capitalize()}', fontsize=11); ax.grid(alpha=.15)
    axs[0].set_ylabel('Clean macro-F1 (%)')
    fig.suptitle('Selected checkpoints: bold means and faint individual seeds\nIdentical clean preprocessing within each model; test scores reused', fontsize=11)
    save(fig, folder, 'model_generalisation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    data = collect_results(args.study, args.manifest)
    data['table'].to_csv(args.study/'combined_run_metrics.csv', index=False)
    data['summary'].to_csv(args.study/'combined_summary.csv', index=False)
    data['paired'].to_csv(args.study/'paired_model_differences.csv', index=False)
    data['per_class'].to_csv(args.study/'combined_per_class_metrics.csv', index=False)
    pd.DataFrame(data['payload']['clean_generalisation']).to_csv(args.study/'combined_checkpoint_gaps.csv', index=False)
    make_figures(data, args.study/'figures', args.data_root)
    # Verify raw input bytes remained unchanged throughout this read-only analysis.
    for relative, expected in data['payload']['verification']['source_hashes'].items():
        if digest(args.study/relative) != expected:
            raise ValueError(f'Input changed during analysis: {relative}')
    (args.study/'comparison_analysis.json').write_text(json.dumps(data['payload'], indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'run_count': 12, 'summary': data['payload']['summary'],
                      'primary_comparison': data['payload']['primary_comparison']}, indent=2))


if __name__ == '__main__':
    main()
