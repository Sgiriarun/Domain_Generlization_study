"""Additional entry points. Saved-input analyses are deliberately explicit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NAMES = dict(enumerate([
    'check_required_software', 'download_the_recordings',
    'check_recordings_and_participants', 'compare_heartbeat_detection_methods',
    'investigate_heart_rate_disagreements', 'prepare_reference_heart_rates',
    'prepare_ppg_signals', 'compare_the_four_datasets',
    'compare_signals_at_similar_heart_rates', 'explore_activity_and_signal_quality',
    'choose_windows', 'estimate_heart_rate_using_signal_frequency',
    'test_timeppg_on_new_participants', 'test_timeppg_on_a_new_dataset',
    'find_where_prediction_errors_increase', 'compare_training_and_test_heart_rates',
    'repeat_the_ptt_experiment', 'compare_fingertip_and_finger_base_signals',
    'check_the_other_channel_pairs', 'check_the_pretrained_pulseppg_model',
    'test_pulseppg_with_a_simple_predictor',
    'test_pulseppg_with_a_neural_network_predictor',
    'train_the_last_part_of_pulseppg', 'compare_models_and_remaining_errors',
    'create_result_tables_and_figures']))

def extra_plan(step, base, out, saved):
    py = sys.executable
    scripts = ROOT/'scripts'
    reports = ROOT/'reports'
    worker = [py, str(ROOT/'start_here/saved_worker.py'), str(step), str(out)]
    if step == 1:
        return [['bash', str(scripts/'download_datasets.sh'), 'all', str(out)]], []
    if step in (2,4,8,9,15,16,17,18,19,20,22,24):
        # These branches reuse audited original inputs, not arbitrary earlier runs.
        if step != 24 and not saved:
            raise ValueError('This step currently requires --use-saved-inputs; it reads the audited original data/reports. See README.')
        required = {
            2: ['datasets/raw'],
            4: ['reports/phase4a_ppg_dalia/window_comparison.csv',
                'reports/phase4g_wesad_hr/window_hr_and_quality.csv',
                'reports/phase4i_bidmc_hr/window_hr_and_quality.csv'],
            8: ['reports/phase6_distribution_shift/hr_matched_features.csv'],
            9: ['reports/phase6_distribution_shift/hr_matched_features.csv',
                'reports/phase5_ppg/window_channel_quality.csv',
                'reports/phase5_ppg/processed_recordings.csv'],
            15: ['reports/phase7_frozen_dataset/main_window_manifest.csv',
                 'reports/phase9_timeppg/lodo_full'],
            16: ['reports/phase9_timeppg/lodo_full/target_ptt_ppg/seed_17/target_test_predictions.csv',
                 'reports/phase9_timeppg/lodo_full/target_ptt_ppg/seed_29/target_test_predictions.csv',
                 'reports/phase9_timeppg/lodo_full/target_ptt_ppg/seed_43/target_test_predictions.csv'],
            17: ['reports/phase11_ptt_site_transfer/subject_site_metrics.csv',
                 'reports/phase11_ptt_site_transfer/paired_window_predictions.csv',
                 'reports/phase11_ptt_site_transfer/reverse_proximal_to_distal/subject_site_metrics.csv',
                 'reports/phase11_ptt_site_transfer/reverse_proximal_to_distal/paired_window_predictions.csv'],
            18: ['reports/phase11_ptt_site_transfer/pair_sensitivity/frozen_all_pair_summary.csv'],
            19: ['artifacts/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl',
                 'reports/phase7_frozen_dataset/main_window_manifest.csv'],
            20: ['artifacts/pulseppg/phase12_embeddings/metadata.json',
                 'artifacts/pulseppg/phase12_embeddings/embeddings_8s_50hz.npy',
                 'artifacts/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl'],
            22: ['artifacts/pulseppg/phase12_inputs_8s_50hz/metadata.json',
                 'artifacts/pulseppg/phase12_inputs_8s_50hz/windows_8s_50hz.npy',
                 'reports/phase12_pulseppg/03_head_and_tuning_ablation/nonlinear_head'],
        }
        return [worker], [ROOT/path for path in required.get(step, [])]
    if step in (12,13):
        windows = reports/'phase7_frozen_dataset' if saved else base/'10_choose_windows'
        roles = out/'three_fold_subject_roles.csv'
        cmds = []
        if step == 12:
            cmds.append([py, str(ROOT/'start_here/saved_worker.py'), '12', str(out), str(windows)])
        for dataset in ('BIDMC','WESAD','PTT-PPG','PPG-DaLiA'):
            for fold in (range(3) if step == 12 else [0]):
                script = 'train_within_dataset.py' if step == 12 else 'train_lodo_pilot.py'
                args = (['--dataset',dataset,'--test-fold',str(fold),'--validation-fold','-1',
                         '--splits-file',str(roles),'--role-column',f'within3_test_fold_{fold}_role']
                        if step == 12 else ['--target',dataset,'--full-data'])
                cmds.append([py,str(scripts/'models/timeppg'/script),*args,
                    '--phase7-dir',str(windows),'--output-root',str(out/'models'),
                    '--epochs','200','--patience','20','--seed','17'])
        return cmds,[windows/'main_window_manifest.csv',windows/'subject_splits.csv']
    if step == 21:
        if not saved:
            raise ValueError('Use --use-saved-inputs: this training uses the audited saved Pulse-PPG embeddings.')
        return [worker], [
                     ROOT/'artifacts/pulseppg/phase12_embeddings/metadata.json',
                     ROOT/'artifacts/pulseppg/phase12_embeddings/embeddings_8s_50hz.npy']
    raise ValueError(f'Unknown step {step}')
