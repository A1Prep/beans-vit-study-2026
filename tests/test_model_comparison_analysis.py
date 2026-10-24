"""Synthetic contract and tampering tests; no model inference or real results."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import analyse_model_comparison as analysis


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


class ComparisonAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='beans-comparison-analysis-test-')
        self.root = Path(self.temp.name)
        self.study = self.root/'study'
        write_json(self.study/'timing_analysis_amendment.json', {
            'before_new_test_evaluation': True, 'training_changes': False,
            'raw_times_preserved': True, 'fixture': 'Synthetic contract test, not experimental evidence'})
        self.images = self.root/'images'
        records = []
        for split in ['train', 'validation', 'test']:
            for i, label in enumerate([0, 0, 1, 1, 2, 2]):
                image_id = f'{split}/{i}.png'
                path = self.images/image_id
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (20, 20), (40+label*50, 100, 40)).save(path)
                records.append({'image_id': image_id, 'label': label, 'split': split,
                                'include': True, 'path': str(path)})
        self.manifest = self.root/'manifest.csv'
        manifest = pd.DataFrame(records)
        manifest.to_csv(self.manifest, index=False)
        for model in analysis.MODELS:
            gaps = []
            for arm in analysis.ARMS:
                for seed in analysis.SEEDS:
                    folder = self.study/model/'runs'/arm/f'seed_{seed}'
                    folder.mkdir(parents=True)
                    config = {'arm': arm, 'seed': seed, 'model_id': analysis.MODEL_IDS[model],
                              'manifest_sha256': analysis.digest(self.manifest)}
                    summary = {'arm': arm, 'seed': seed, 'best_epoch': 1, 'epochs_completed': 1,
                               'training_seconds': 1.0, 'wall_seconds': 1.5,
                               'peak_cuda_allocated_bytes': 32, 'trainable_params': 1, 'total_params': 9}
                    write_json(folder/'config.json', config)
                    write_json(folder/'summary.json', summary)
                    pd.DataFrame([{'epoch': 1, 'train_loss': .1, 'val_loss': .1, 'val_macro_f1': 1.0,
                                   'train_seconds': 1.0}]).to_csv(folder/'history.csv', index=False)
                    gap = {'arm': arm, 'seed': seed, 'selected_epoch': 1}
                    for split in ['train', 'validation', 'test']:
                        subset = manifest[manifest.split == split]
                        labels = subset.label.to_numpy()
                        predicted = labels.copy()
                        if model == 'dinov2' and split == 'test':
                            predicted[0] = 1
                        probability = np.full((6, 3), .05)
                        probability[np.arange(6), predicted] = .9
                        frame = pd.DataFrame({'image_id': subset.image_id.to_list(), 'label': labels,
                                              'prediction': predicted, 'predicted_label': predicted,
                                              'prob_0': probability[:, 0], 'prob_1': probability[:, 1], 'prob_2': probability[:, 2]})
                        score = analysis.metrics(labels, predicted)
                        for key in ['accuracy', 'macro_f1']:
                            gap[f'{split}_{key}'] = score[key]
                        gap[f'{split}_n'] = 6
                        gap[f'{split}_loss'] = .1
                        if split == 'test':
                            frame.to_csv(folder/'predictions_test.csv', index=False)
                            per_class = []
                            for label, name in enumerate(analysis.NAMES):
                                cls = score['classification_report'][name]
                                per_class.append({'label': label, 'precision': cls['precision'], 'recall': cls['recall'],
                                                  'f1': cls['f1-score'], 'support': cls['support']})
                            write_json(folder/'metrics_test.json', {'arm': arm, 'seed': seed, 'split': 'test',
                                'corruption': 'clean', 'checkpoint_epoch': 1, 'n': 6, 'accuracy': score['accuracy'],
                                'macro_f1': score['macro_f1'], 'confusion_matrix': score['confusion_matrix'], 'per_class': per_class})
                        else:
                            path = self.study/model/'overfitting'/arm/f'seed_{seed}'/f'predictions_{split}_clean.csv'
                            path.parent.mkdir(parents=True, exist_ok=True)
                            frame.to_csv(path, index=False)
                    gap['train_minus_validation_f1_pp'] = 100*(gap['train_macro_f1']-gap['validation_macro_f1'])
                    gap['train_minus_test_f1_pp'] = 100*(gap['train_macro_f1']-gap['test_macro_f1'])
                    gap['validation_minus_train_loss'] = 0.0
                    gaps.append(gap)
            pd.DataFrame(gaps).to_csv(self.study/model/'overfitting'/'checkpoint_gaps.csv', index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_pipeline_pairs_correct_direction_and_renders(self):
        data = analysis.collect_results(self.study, self.manifest)
        self.assertEqual(len(data['table']), 12)
        self.assertEqual(len(data['summary']), 4)
        primary = data['payload']['primary_comparison']
        self.assertGreater(primary['difference_siglip2_minus_dinov2'], 0)
        self.assertEqual(primary['seed'], 20261023)
        self.assertEqual(primary['resamples'], 2000)
        self.assertNotIn('difference_partial_minus_frozen', primary)
        self.assertTrue(data['payload']['prior_test_exposure'])
        self.assertTrue(np.allclose(data['summary'].pooled_training_seconds_per_epoch, 1))
        analysis.make_figures(data, self.study/'figures', self.images)
        self.assertEqual(len(list((self.study/'figures').glob('*.png'))), 5)
        self.assertEqual(len(list((self.study/'figures').glob('*.pdf'))), 5)
        chosen = pd.read_csv(self.study/'siglip_selected_examples.csv')
        self.assertEqual(len(chosen), 6)
        for relative, expected in data['payload']['verification']['source_hashes'].items():
            self.assertEqual(analysis.digest(self.study/relative), expected)

    def test_missing_or_extra_runs_rejected(self):
        missing = self.study/'siglip2'/'runs'/'partial'/'seed_2026'
        missing.rename(missing.with_name('unplanned'))
        with self.assertRaisesRegex(ValueError, 'Exactly twelve'):
            analysis.collect_results(self.study, self.manifest)
        missing.with_name('unplanned').rename(missing)
        (missing.parent/'seed_999').mkdir()
        with self.assertRaisesRegex(ValueError, 'Exactly twelve'):
            analysis.collect_results(self.study, self.manifest)

    def test_tampered_saved_metric_rejected(self):
        path = self.study/'dinov2'/'runs'/'frozen'/'seed_17'/'metrics_test.json'
        data = analysis.read_json(path)
        data['macro_f1'] = 1
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, 'macro_f1'):
            analysis.collect_results(self.study, self.manifest)

    def test_audited_labels_and_duplicate_units_rejected(self):
        path = self.study/'dinov2'/'runs'/'frozen'/'seed_17'/'predictions_test.csv'
        frame = pd.read_csv(path)
        original = frame.copy()
        frame.loc[0, 'label'] = 2
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'audited evaluation units'):
            analysis.collect_results(self.study, self.manifest)
        original.loc[0, 'image_id'] = original.loc[1, 'image_id']
        original.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'audited evaluation units'):
            analysis.collect_results(self.study, self.manifest)

    def test_invalid_probability_and_conflicting_prediction_rejected(self):
        path = self.study/'dinov2'/'runs'/'frozen'/'seed_17'/'predictions_test.csv'
        frame = pd.read_csv(path)
        original = frame.copy()
        frame.loc[0, 'prob_0'] = -1
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'Invalid softmax'):
            analysis.collect_results(self.study, self.manifest)
        original.loc[0, 'predicted_label'] = 2
        original.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'Conflicting predicted-label'):
            analysis.collect_results(self.study, self.manifest)

    def test_invented_clean_generalisation_gap_rejected(self):
        path = self.study/'dinov2'/'overfitting'/'checkpoint_gaps.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'train_minus_test_f1_pp'] = 0
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'train_minus_test_f1_pp'):
            analysis.collect_results(self.study, self.manifest)

    def test_epoch_outlier_retained_in_totals_but_not_median(self):
        # One long epoch changes total/mean; median and IQR describe typical
        # observed epochs without deleting or correcting the long measurement.
        times = [13.0]*8+[809.0, 13.0]
        result = analysis.summarise_epoch_times(times)
        self.assertEqual(result['count'], 10)
        self.assertEqual(result['total_seconds'], 926.0)
        self.assertEqual(result['mean_seconds'], 92.6)
        self.assertEqual(result['median_seconds'], 13.0)
        self.assertEqual(result['iqr_seconds'], 0.0)
        self.assertEqual(result['min_seconds'], 13.0)
        self.assertEqual(result['max_seconds'], 809.0)
        for invalid in [[], [1.0, float('nan')], [1.0, -1.0]]:
            with self.assertRaises(ValueError):
                analysis.summarise_epoch_times(invalid)

    def test_pooled_epoch_statistics_and_timing_provenance(self):
        folder = self.study/'siglip2'/'runs'/'frozen'/'seed_17'
        times = [13.0]*8+[809.0, 13.0]
        pd.DataFrame([{'epoch': i+1, 'train_loss': .1, 'val_loss': .1, 'val_macro_f1': 1.0,
                       'train_seconds': value} for i, value in enumerate(times)]).to_csv(folder/'history.csv', index=False)
        summary = analysis.read_json(folder/'summary.json')
        summary.update({'training_seconds': sum(times), 'epochs_completed': 10})
        write_json(folder/'summary.json', summary)
        # Bootstrap is independently exercised in the full integration test;
        # this test isolates timing and avoids unnecessary repeated resampling.
        with patch.object(analysis, 'paired_bootstrap', return_value={'difference_partial_minus_frozen': 0.1}):
            data = analysis.collect_results(self.study, self.manifest)
        run = data['table'].query("model == 'siglip2' and arm == 'frozen' and seed == 17").iloc[0]
        self.assertEqual(run.median_observed_training_epoch_seconds, 13.0)
        self.assertEqual(run.training_seconds, 926.0)
        group = data['summary'].query("model == 'siglip2' and arm == 'frozen'").iloc[0]
        self.assertEqual(group.pooled_epochs, 12)
        self.assertEqual(group.pooled_training_seconds, 928.0)
        self.assertEqual(group.pooled_epoch_median_seconds, 13.0)
        self.assertEqual(group.pooled_epoch_max_seconds, 809.0)
        self.assertEqual(group.pooled_epoch_min_seconds, 1.0)
        self.assertEqual(group.pooled_epoch_q1_seconds, 13.0)
        self.assertEqual(group.pooled_epoch_q3_seconds, 13.0)
        hashes = data['payload']['verification']['source_hashes']
        self.assertEqual(hashes['timing_analysis_amendment.json'], analysis.digest(self.study/'timing_analysis_amendment.json'))
        self.assertEqual(hashes['siglip2/runs/frozen/seed_17/history.csv'], analysis.digest(folder/'history.csv'))


if __name__ == '__main__':
    unittest.main()
