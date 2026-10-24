"""Independent, CPU-only verification of completed comparison artifacts.

Does not import experiment/analysis code, deserialize models, or perform inference.
Written before completed results were available. Run only after analysis completes.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def same(a, b, label):
    if a != b:
        raise AssertionError(f'{label}: {a!r} != {b!r}')


def near(a, b, label, atol=1e-10):
    if not np.allclose(a, b, rtol=0, atol=atol) or not np.all(np.isfinite(a)):
        raise AssertionError(f'{label}: {a!r} != {b!r}')


def score(y, p):
    cm = np.zeros((3, 3), dtype=np.int64)
    for actual, predicted in zip(y, p):
        cm[actual, predicted] += 1
    per = []
    for k in range(3):
        tp = int(cm[k, k]); support = int(cm[k].sum()); predicted = int(cm[:, k].sum())
        precision = tp/predicted if predicted else 0.
        recall = tp/support if support else 0.
        f1 = 2*tp/(support+predicted) if support+predicted else 0.
        per.append(dict(label=k, precision=precision, recall=recall, f1=f1, support=support))
    return dict(n=len(y), accuracy=float(np.mean(y == p)), macro_f1=float(np.mean([c['f1'] for c in per])),
                confusion_matrix=cm.tolist(), per_class=per)


def predictions(path, expected):
    data = sorted(rows(path), key=lambda r: r['image_id'])
    same([(r['image_id'], int(r['label'])) for r in data], expected, f'{path} manifest units')
    y = np.array([int(r['label']) for r in data])
    p = np.array([int(r['predicted_label']) for r in data])
    prob = np.array([[float(r[f'prob_{c}']) for c in range(3)] for r in data])
    assert np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all(), path
    near(prob.sum(axis=1), np.ones(len(data)), f'{path} probabilities', 1e-5)
    same(prob.argmax(axis=1).tolist(), p.tolist(), f'{path} argmax')
    for r in data:
        if 'prediction' in r:
            same(int(r['prediction']), int(r['predicted_label']), f'{path} prediction aliases')
    return score(y, p), y, p


def check_metric(actual, saved, label):
    for key in ['n', 'confusion_matrix']:
        same(actual[key], saved[key], f'{label} {key}')
    for key in ['accuracy', 'macro_f1']:
        near(actual[key], saved[key], f'{label} {key}')
    for a, b in zip(actual['per_class'], sorted(saved['per_class'], key=lambda r: r['label'])):
        for key in a:
            near(a[key], b[key], f'{label} class {a["label"]} {key}')


def check_lock(project, root, lock_path, source, manifest_hash):
    lock = read(lock_path)
    same(sha(source), lock['training_code_sha256'], 'locked training source')
    same(manifest_hash, lock['manifest_sha256'], 'locked manifest')
    for e in lock['runs']:
        folder = root/e['arm']/f'seed_{e["seed"]}'
        for name, key in [('config.json', 'config_sha256'), ('best.pt', 'checkpoint_sha256'), ('summary.json', 'summary_sha256')]:
            if key in e:
                same(sha(folder/name), e[key], f'{folder/name} lock')
        cfg = read(folder/'config.json')
        same(cfg['arm'], e['arm'], 'arm'); same(cfg['seed'], e['seed'], 'seed')
        same(cfg['training_code_sha256'], lock['training_code_sha256'], 'config source')
        same(cfg['manifest_sha256'], manifest_hash, 'config manifest')
        same(cfg['model_id'], lock['model'], 'config model')
        same(cfg['requested_revision'], lock['revision'], 'config revision')
    return lock


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/beans_vit_project')
    ap.add_argument('--run-root', type=Path, default=Path('D:/CodexWork/beans-vit-project/model_comparison'))
    ap.add_argument('--original-run-root', type=Path, default=Path('D:/CodexWork/beans-vit-project/runs'))
    args = ap.parse_args(); project = args.project; study = project/'results/model_comparison'
    manifest = project/'results/data_audit/manifest.csv'; mh = sha(manifest)
    active = [r for r in rows(manifest) if r['include'].lower() in ['1', 'true', 'yes']]
    expected = {s: sorted((r['image_id'], int(r['label'])) for r in active if r['split'] == s)
                for s in ['train', 'validation', 'test']}
    plan = read(study/'study_plan.json'); identity = read(study/'execution_identity.json')
    same(mh, plan['dataset']['manifest_sha256'], 'planned manifest')
    same(identity['study_plan_sha256'], sha(study/'study_plan.json'), 'execution plan')
    same(identity['training_code_sha256'], sha(project/'src/followup_experiment.py'), 'execution source')
    analysis = read(study/'comparison_analysis.json')
    same(analysis['run_count'], 12, 'analysis run count')
    assert analysis['prior_test_exposure'], 'Missing prior-test disclosure'
    acquisition = read(study/'pretrained_input_verification.json')
    acquisition_checks = {}
    for model in ['dinov2', 'siglip2']:
        acqpath = project/'results/model_acquisition.json' if model == 'dinov2' else study/'siglip2_acquisition.json'
        acq = read(acqpath); verified = acquisition['models'][model]
        same(sha(acqpath), verified['acquisition_sha256'], 'acquisition evidence')
        for e in acq['files']:
            same(sha(Path(verified['local_path'])/e['file']), e['sha256'], 'pretrained file')
            same(e['sha256'], verified['verified_files'][e['file']], 'pretrained reference')
            if e.get('published_lfs_sha256'):
                same(e['sha256'], e['published_lfs_sha256'], 'official LFS digest')
        same(verified['revision'], plan['models'][model]['revision'], 'pretrained planned revision')
        acquisition_checks[model] = dict(acquisition_sha256=sha(acqpath), file_count=len(acq['files']))
    amendment_path = study/'timing_analysis_amendment.json'; amendment = read(amendment_path)
    assert amendment['before_new_test_evaluation'] and amendment['raw_times_preserved']
    assert not any(amendment[k] for k in ['training_changes', 'checkpoint_selection_changes', 'test_evaluation_changes'])
    same(amendment['observed_new_test_artifact_count'], 0, 'recorded pre-evaluation artifact count')
    motivation = amendment['motivation']
    for kind in ['history', 'summary']:
        same(sha(study/motivation[kind+'_relative_path']), motivation[kind+'_sha256'], 'timing amendment historical source')
    amendment_time = datetime.fromisoformat(amendment['created_utc'].replace('Z', '+00:00'))
    same(analysis['verification']['source_hashes']['timing_analysis_amendment.json'], sha(amendment_path), 'analysis timing amendment hash')
    same(analysis['timing_analysis']['amendment'], amendment, 'analysis timing policy')
    all_scores = {}; trainval = {}; clean = {}; stress = {}; heads = {}; summaries = {}; epoch_times = {}
    combined_rows = rows(study/'combined_run_metrics.csv')
    for model in ['dinov2', 'siglip2']:
        lockpath = study/model/'protocol.lock.json'
        lock = check_lock(project, args.run_root/model/'runs', lockpath, project/'src/followup_experiment.py', mh)
        same(lock['study_plan_sha256'], sha(study/'study_plan.json'), 'lock plan')
        assert amendment_time < datetime.fromisoformat(lock['locked_at_utc'].replace('Z', '+00:00')), 'Timing amendment must precede recorded new lock time'
        same(len(lock['runs']), 6, 'lock run count')
        assert lock['original_dinov2_test_results_known_before_followup']
        assert not lock['new_followup_test_predictions_evaluated_before_lock']
        for arm in ['frozen', 'partial']:
            for seed in [17, 42, 2026]:
                key = f'{model}_{arm}_{seed}'; folder = study/model/'runs'/arm/f'seed_{seed}'
                sourcefolder = args.run_root/model/'runs'/arm/f'seed_{seed}'
                config = read(folder/'config.json'); summary = read(folder/'summary.json'); summaries[key] = summary
                heads[model, arm, seed] = config['initial_head_sha256']
                for f in folder.iterdir():
                    if f.is_file(): same(sha(f), sha(sourcefolder/f.name), f'copied artifact {f}')
                for condition in ['clean', 'blur', 'brightness'] if seed == 17 else ['clean']:
                    stem = 'test' + ('' if condition == 'clean' else '_'+condition)
                    value, y, p = predictions(folder/f'predictions_{stem}.csv', expected['test'])
                    saved = read(folder/f'metrics_{stem}.json'); check_metric(value, saved, key+'/'+condition)
                    same(saved['manifest_sha256'], mh, 'test manifest')
                    same(saved['protocol_lock_sha256'], sha(lockpath), 'test lock')
                    entry = next(e for e in lock['runs'] if e['arm'] == arm and e['seed'] == seed)
                    same(saved['checkpoint_sha256'], entry['checkpoint_sha256'], 'test checkpoint')
                    same(saved['checkpoint_epoch'], summary['best_epoch'], 'selected epoch')
                    all_scores[key+'/'+condition] = value
                    if condition == 'clean':
                        clean[key] = (value, y, p)
                        reported = analysis['runs'][key]
                        same(value['confusion_matrix'], reported['confusion_matrix'], 'analysis confusion matrix')
                        for metric in ['accuracy', 'macro_f1']:
                            near(value[metric], reported[metric], 'analysis run metric')
                        names = ['Angular leaf spot', 'Bean rust', 'Healthy']
                        for c in value['per_class']:
                            cls = reported['classification_report'][names[c['label']]]
                            for target, source in [('precision', 'precision'), ('recall', 'recall'), ('f1', 'f1-score'), ('support', 'support')]:
                                near(c[target], cls[source], 'analysis per-class metric')
                    else: stress[key+'/'+condition] = value
                diag = study/model/'overfitting'/arm/f'seed_{seed}'
                saved_diag = read(diag/'metrics.json')
                for split in ['train', 'validation']:
                    value, _, _ = predictions(diag/f'predictions_{split}_clean.csv', expected[split])
                    check_metric(value, saved_diag['measurements'][split], key+'/'+split)
                    trainval[key+'/'+split] = value
                gap = next(r for r in analysis['clean_generalisation'] if r['model'] == model and r['arm'] == arm and int(r['seed']) == seed)
                for split in ['train', 'validation', 'test']:
                    value = clean[key][0] if split == 'test' else trainval[key+'/'+split]
                    for metric in ['n', 'accuracy', 'macro_f1']:
                        near(value[metric], gap[f'{split}_{metric}'], 'analysis clean diagnostic')
                near(100*(trainval[key+'/train']['macro_f1']-trainval[key+'/validation']['macro_f1']), gap['train_minus_validation_f1_pp'], 'analysis validation gap')
                near(100*(trainval[key+'/train']['macro_f1']-clean[key][0]['macro_f1']), gap['train_minus_test_f1_pp'], 'analysis test gap')
                hist = rows(folder/'history.csv')
                epoch_times[key] = np.array([float(r['train_seconds']) for r in hist])
                near(float(epoch_times[key].sum()), summary['training_seconds'], 'all raw epoch durations retained', 1e-6)
                combined = next(r for r in combined_rows if r['model'] == model and r['arm'] == arm and int(r['seed']) == seed)
                near(float(np.median(epoch_times[key])), float(combined['median_observed_training_epoch_seconds']), 'per-run epoch median')
                winner = min(hist, key=lambda r: (-float(r['val_macro_f1']), float(r['val_loss']), int(r['epoch'])))
                same(int(winner['epoch']), summary['best_epoch'], 'independent checkpoint selection')
                near(float(winner['val_macro_f1']), trainval[key+'/validation']['macro_f1'], 'selected validation F1')
        for seed in [17, 42, 2026]:
            same(heads[model, 'frozen', seed], heads[model, 'partial', seed], 'paired head initialization')
    for summary in analysis['summary']:
        keys = [f'{summary["model"]}_{summary["arm"]}_{s}' for s in [17, 42, 2026]]
        for metric in ['accuracy', 'macro_f1']:
            values = [clean[k][0][metric] for k in keys]
            near(float(np.mean(values)), summary[metric+'_mean'], 'analysis mean')
            near(float(np.std(values, ddof=1)), summary[metric+'_sd'], 'analysis sample SD')
        near(sum(summaries[k]['training_seconds'] for k in keys)/sum(summaries[k]['epochs_completed'] for k in keys),
             summary['pooled_training_seconds_per_epoch'], 'pooled training cost')
        times = np.concatenate([epoch_times[k] for k in keys]); q1, median, q3 = np.quantile(times, [.25, .5, .75])
        for field, value in dict(median=median, q1=q1, q3=q3, iqr=q3-q1, min=times.min(), max=times.max()).items():
            near(float(value), summary[f'pooled_epoch_{field}_seconds'], 'pooled observed epoch statistic')
    paired = []
    for saved in analysis['paired_model_differences']:
        a = clean[f'dinov2_{saved["arm"]}_{saved["seed"]}'][0]
        b = clean[f'siglip2_{saved["arm"]}_{saved["seed"]}'][0]
        delta = {m: b[m]-a[m] for m in ['accuracy', 'macro_f1']}
        for m, v in delta.items(): near(v, saved['siglip2_minus_dinov2_'+m], 'paired difference')
        paired.append(dict(arm=saved['arm'], seed=saved['seed'], **delta))
    _, y, a = clean['dinov2_partial_17']; _, yb, b = clean['siglip2_partial_17']
    same(y.tolist(), yb.tolist(), 'bootstrap paired labels')
    rng = np.random.default_rng(20261023); groups = [np.flatnonzero(y == c) for c in range(3)]; boot = []
    for _ in range(2000):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        boot.append(score(y[indices], b[indices])['macro_f1']-score(y[indices], a[indices])['macro_f1'])
    interval = np.quantile(boot, [.025, .975]).tolist()
    difference = score(y, b)['macro_f1']-score(y, a)['macro_f1']
    near(interval, analysis['primary_comparison']['ci95_percentile'], 'independent bootstrap')
    near(difference, analysis['primary_comparison']['difference_siglip2_minus_dinov2'], 'bootstrap observed difference')
    original = read(Path(__file__).with_name('original_comparison_review_baseline.json'))
    for name, expected_hash in original['files'].items(): same(sha(project/name), expected_hash, 'unchanged original artifact')
    original_lock = check_lock(project, args.original_run_root, project/'results/protocol.lock.json', project/'src/experiment.py', mh)
    same(original_lock['training_code_sha256'], '9dceb0bf7817cded87091d0954346118b85b7755bffa6ef14e6ae195f8a6de3f', 'original known training hash')
    result = dict(status='passed', verified_utc=datetime.now(timezone.utc).isoformat(),
        method='Independent NumPy confusion-matrix arithmetic and resampling; no project scoring imports, model deserialization or GPU operations',
        clean_test_runs_verified=len(clean), stress_test_conditions_verified=len(stress), clean_train_validation_metrics_verified=len(trainval),
        all_per_class_and_confusion_metrics_verified=True, paired_initial_heads_verified=True,
        checkpoint_lock_file_hashes_verified=True, preprocessing_and_weight_acquisition_hashes_verified=acquisition_checks,
        original_artifacts_unchanged=len(original['files']), original_training_sha256=original_lock['training_code_sha256'],
        timing_amendment_source_hashes_and_recorded_chronology_verified=True,
        all_per_run_epoch_medians_and_pooled_epoch_medians_quartiles_ranges_verified=True,
        paired_model_differences=paired, reference_partial_bootstrap=dict(difference_siglip2_minus_dinov2=difference, ci95_percentile=interval, resamples=2000, seed=20261023),
        recomputed_test_metrics=all_scores, recomputed_clean_development_metrics=trainval,
        verifier_sha256=sha(__file__), comparison_analysis_sha256=sha(study/'comparison_analysis.json'),
        limitations=['Artifact consistency does not independently establish historical lock timing or absence of prior test access.',
                    'Observed laptop timing includes unexplained long duration; all epochs retained. No causal explanation or controlled hardware benchmark is established.',
                    'This is exploratory reused-test evidence, not fresh confirmatory or external validation.',
                    'No grade or state-of-the-art claim is made.'])
    (study/'independent_review.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'clean_test_runs_verified', 'stress_test_conditions_verified', 'clean_train_validation_metrics_verified', 'reference_partial_bootstrap']}))


if __name__ == '__main__':
    main()
