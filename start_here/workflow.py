"""Safe, numbered entry points; original scientific calculations are unchanged."""
from __future__ import annotations

import argparse
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    0: 'check_required_software', 3: 'compare_heartbeat_detection_methods',
    5: 'prepare_reference_heart_rates', 6: 'prepare_ppg_signals',
    7: 'compare_the_four_datasets', 10: 'choose_windows',
    11: 'estimate_heart_rate_using_signal_frequency',
    14: 'find_where_prediction_errors_increase',
    23: 'compare_models_and_remaining_errors',
}
from extra_steps import NAMES as EXTRA_NAMES, extra_plan
NAMES.update(EXTRA_NAMES)


def output(base: Path, step: int) -> Path:
    return base / f'{step:02d}_{NAMES[step]}'


def pulse_cache_problem(step: int) -> str | None:
    """Accept only an exact historical manifest reconstruction for saved arrays."""
    cache = ('phase12_inputs_8s_50hz' if step == 22 else 'phase12_embeddings')
    metadata_path = ROOT/'artifacts/pulseppg'/cache/'metadata.json'
    manifest = ROOT/'reports/phase7_frozen_dataset/main_window_manifest.csv'
    checkpoint = ROOT/'artifacts/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl'
    if not metadata_path.is_file() or not manifest.is_file() or not checkpoint.is_file():
        return None  # Ordinary missing-path check gives the detailed path.
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    for label, path in [('manifest', manifest), ('checkpoint', checkpoint)]:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = metadata.get(f'{label}_sha256')
        if label == 'manifest' and actual != expected:
            # Phase 7 was regenerated after DaLiA condition_name was changed
            # from numeric activity codes to descriptive words. Reconstruct the
            # exact old CSV in memory and accept it only if its *full byte hash*
            # equals the cache provenance. The worker writes that CSV separately.
            import pandas as pd
            frame = pd.read_csv(manifest, low_memory=False)
            dalia = frame.dataset.eq('PPG-DaLiA')
            frame.loc[dalia, 'condition_name'] = frame.loc[dalia, 'condition_id'].astype('Int64').astype(str)
            restored = hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()
            if restored == expected:
                continue
        if actual != expected:
            return (f'Saved Pulse-PPG {cache} cache does not match current {label}: '
                    f'expected {expected}, found {actual}. Regenerate the cache from '
                    'the current inputs or restore the original input snapshot; do not bypass this check.')
    return None


def plan(step: int, base: Path, saved: bool):
    """Return commands and required files. Never silently fall back to old inputs."""
    out = output(base, step)
    py = sys.executable
    previous = ROOT / 'reports' if saved else base
    labels = (previous / 'phase4k_canonical_hr' if saved else output(base, 5)) / 'canonical_window_hr.csv'
    ppg = previous / 'phase5_ppg' if saved else output(base, 6)
    windows = previous / 'phase7_frozen_dataset' if saved else output(base, 10)
    if step == 0:
        return [[py, '-c', "import rq1_hr,numpy,pandas,scipy,sklearn,torch; print('Required imports succeeded')"],
                [py, '-m', 'pytest', '-q']], []
    if step == 3:
        scripts = ['validate_dalia_hr_pipeline.py', 'benchmark_dalia_xqrs.py',
                   'benchmark_ptt_xqrs.py', 'build_wesad_ecg_hr.py', 'build_bidmc_ecg_hr.py']
        folders = ['phase4a_ppg_dalia', 'phase4e_dalia_detector_comparison',
                   'phase4f_ptt_detector_comparison', 'phase4g_wesad_hr', 'phase4i_bidmc_hr']
        return [[py, str(ROOT/'scripts'/s), '--output-dir', str(out/f)] for s,f in zip(scripts,folders)], [ROOT/'datasets/raw']
    if step == 5:
        inputs = ROOT/'reports' if saved else output(base, 3)
        needed = [inputs / f for f in ['phase4a_ppg_dalia/window_comparison.csv',
            'phase4f_ptt_detector_comparison/window_comparison.csv',
            'phase4g_wesad_hr/window_hr_and_quality.csv', 'phase4i_bidmc_hr/window_hr_and_quality.csv']]
        return [[py, str(ROOT/'scripts/build_canonical_hr_labels.py'), '--reports-root', str(inputs), '--output-dir', str(out)]], needed
    configs = {
        6: ('build_phase5_ppg.py', ['--labels', str(labels)], [labels, ROOT/'datasets/raw']),
        7: ('analyse_phase6_distribution_shift.py', ['--phase5-dir', str(ppg)], [ppg/'window_channel_quality.csv', ppg/'processed_recordings.csv']),
        10: ('build_phase7_frozen_dataset.py', ['--phase5-dir', str(ppg)], [ppg/'window_channel_quality.csv', ppg/'processed_recordings.csv']),
        11: ('run_phase8_spectral_baseline.py', ['--phase7-dir', str(windows)], [windows/'main_window_manifest.csv']),
        14: ('models/timeppg/analyse_lodo_failures.py', [], [ROOT/'reports/phase9_timeppg/lodo_full']),
    }
    if step in configs:
        script, args, needed = configs[step]
        return [[py, str(ROOT/'scripts'/script), *args, '--output-dir', str(out)]], needed
    if step == 23:
        return [[py, str(Path(__file__).resolve()), '--pulse-worker', str(out)]], [ROOT/'reports/phase12_pulseppg/03_head_and_tuning_ablation']
    return extra_plan(step, base, out, saved)


def main(step: int):
    parser = argparse.ArgumentParser(description=f'{step:02d}: {NAMES[step].replace("_", " ")}. Default: preview only.')
    parser.add_argument('--run', action='store_true', help='Execute the displayed commands.')
    parser.add_argument('--allow-training', action='store_true', help='Explicitly permit model fitting in steps 12, 13, 20, 21 and 22.')
    parser.add_argument('--use-saved-inputs', action='store_true', help='Explicitly reuse original reports instead of earlier numbered outputs.')
    parser.add_argument('--results-dir', type=Path, default=ROOT/'study_results')
    args = parser.parse_args()
    base = args.results_dir.resolve()
    protected = [ROOT/name for name in ('reports', 'datasets', 'artifacts', 'src', 'scripts', 'start_here', 'PGR9')]
    if base == ROOT or any(base == path or path in base.parents for path in protected):
        parser.error('Choose a separate result folder, outside the original data, code and reports folders.')
    try:
        commands, required = plan(step, base, args.use_saved_inputs)
    except ValueError as exc:
        parser.error(str(exc))
    out = output(base, step)
    print('Output:', out)
    if step in (14,23):
        print('Reads original saved experiment predictions under reports/. Does not run model inference or training.')
    if step == 10:
        print('Creates the original Phase 7 split arrangement, NOT final TimePPG three-fold training roles.')
    for command in commands:
        import shlex
        print(shlex.join(command))
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print('Missing required inputs:', '\n'.join(missing))
    if not args.run:
        print('Preview only. Add --run to execute. Existing outputs will never be overwritten.')
        return
    if missing:
        parser.error('Required inputs missing. Run earlier steps or explicitly choose --use-saved-inputs.')
    if step in (20,21,22):
        problem = pulse_cache_problem(step)
        if problem:
            parser.error(problem)
    if step in (12,13,20,21,22) and not args.allow_training:
        parser.error('This step fits models and may take hours. Add --allow-training to proceed.')
    if out.exists():
        parser.error(f'Output already exists: {out}. Choose a fresh --results-dir; no files were overwritten.')
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'), env.get('PYTHONPATH','')])
    env['MPLCONFIGDIR'] = str(out/'plot_cache')
    env['MPLBACKEND'] = 'Agg'
    record = {'step': step, 'commands': commands, 'required_paths': [str(p) for p in required],
              'saved_inputs_requested': args.use_saved_inputs, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'running', 'python': sys.version}
    record['git_commit'] = subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    record['git_status'] = subprocess.run(['git','status','--short'], cwd=ROOT, text=True, capture_output=True).stdout
    meta = out/'run_details.json'
    meta.write_text(json.dumps(record, indent=2))
    try:
        with (out/'console.log').open('w') as log:
            for command in commands:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line, end=''); log.write(line); log.flush()
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
        record['status'] = 'completed'
    except BaseException as exc:
        record['status'] = 'failed'; record['error'] = str(exc)
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        meta.write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    # Internal subprocess: redirect only the analysis writer, never model inputs.
    p = argparse.ArgumentParser()
    p.add_argument('--pulse-worker', type=Path, required=True)
    a = p.parse_args()
    path = ROOT/'scripts/models/pulseppg/analyse_partial_finetuning.py'
    spec = importlib.util.spec_from_file_location('pulse_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = a.pulse_worker.resolve()
    module.FIG = module.OUT/'figures'
    module.main()
