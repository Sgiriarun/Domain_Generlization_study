"""Run original analyses with their output paths redirected to one fresh folder."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def module(relative: str, name: str):
    path = ROOT / 'scripts' / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    # dataclasses and standard imports expect the executing module to be
    # registered under its import name.
    sys.modules[name] = loaded
    try:
        spec.loader.exec_module(loaded)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return loaded


def invoke(loaded, *arguments):
    previous = sys.argv
    try:
        sys.argv = [str(loaded.__file__), *map(str, arguments)]
        loaded.main()
    finally:
        sys.argv = previous


def copy_file(source: Path, dest: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def historical_pulse_manifest(out: Path) -> Path:
    """Write an isolated exact copy of the manifest used for saved Phase 12."""
    import pandas as pd
    source = ROOT/'reports/phase7_frozen_dataset/main_window_manifest.csv'
    metadata = json.loads((ROOT/'artifacts/pulseppg/phase12_embeddings/metadata.json').read_text())
    frame = pd.read_csv(source, low_memory=False)
    dalia = frame.dataset.eq('PPG-DaLiA')
    frame.loc[dalia, 'condition_name'] = frame.loc[dalia, 'condition_id'].astype('Int64').astype(str)
    content = frame.to_csv(index=False).encode('utf-8')
    actual = hashlib.sha256(content).hexdigest()
    expected = metadata['manifest_sha256']
    if actual != expected:
        raise RuntimeError(f'Historical manifest reconstruction failed: {actual} != {expected}')
    path = out/'phase7_manifest_used_for_saved_pulse_cache.csv'
    path.write_bytes(content)
    (out/'manifest_reconciliation.json').write_text(json.dumps({
        'source': str(source.relative_to(ROOT)),
        'reconstructed_sha256': actual,
        'saved_cache_sha256': expected,
        'changed_column': 'PPG-DaLiA condition_name only',
        'rows': len(frame),
        'current_manifest_unchanged': True,
    }, indent=2) + '\n')
    return path


def run(step: int, out: Path, windows: Path | None):
    reports = ROOT / 'reports'
    if step == 2:
        m = module('build_recording_manifest.py', 'recording_manifest')
        m.OUTPUT = out / 'recordings.csv'
        invoke(m)
        invoke(module('validate_phase3.py', 'validate_recordings'),
               '--raw-root', ROOT/'datasets/raw', '--output-dir', out/'checks')
    elif step == 4:
        invoke(module('analyse_dalia_ecg_errors.py', 'dalia_disagreement'),
               '--output-dir', out/'dalia')
        invoke(module('analyse_wesad_s2_disagreement.py', 'wesad_disagreement'),
               '--output-dir', out/'wesad')
        for script, folder in [('inspect_wesad_s2_disagreement.py','wesad_detail'),
                               ('inspect_bidmc_disagreement.py','bidmc_detail')]:
            m = module(script, folder)
            m.OUTPUT = out / folder
            invoke(m)
    elif step == 8:
        source = reports/'phase6_distribution_shift'
        selected = sorted(p for p in source.glob('*') if p.is_file() and
                          any(word in p.name for word in ('hr_matched','dataset_classifier','shape_distance','waveform')))
        if not selected:
            raise FileNotFoundError('No saved similar-HR comparison tables found')
        for p in selected:
            copy_file(p, out/p.name)
        (out/'SOURCE.txt').write_text('Copied existing Phase 6 results; step 07 performs the calculation.\n')
    elif step == 9:
        m = module('analyse_phase6_domain_pcas.py', 'domain_pcas')
        m.PHASE5 = reports/'phase5_ppg'
        m.OUTPUT = out
        m.FIGURES = out/'figures'
        copy_file(reports/'phase6_distribution_shift/hr_matched_features.csv',
                  out/'hr_matched_features.csv')
        invoke(m)
    elif step == 12:
        m = module('models/timeppg/build_three_fold_splits.py', 'threefold_roles')
        m.INPUT = windows/'subject_splits.csv'
        m.OUTPUT = out/'three_fold_subject_roles.csv'
        invoke(m)
    elif step == 15:
        m = module('models/timeppg/analyse_train_test_hr_support.py', 'hr_support')
        m.OUT = out
        invoke(m)
    elif step == 16:
        m = module('models/timeppg/aggregate_ptt_lodo_seeds.py', 'ptt_seeds')
        m.OUT = out
        invoke(m)
    elif step == 17:
        source = reports/'phase11_ptt_site_transfer'
        for relative in ('subject_site_metrics.csv', 'paired_window_predictions.csv',
                         'reverse_proximal_to_distal/subject_site_metrics.csv',
                         'reverse_proximal_to_distal/paired_window_predictions.csv'):
            copy_file(source/relative, out/relative)
        m = module('models/timeppg/summarize_ptt_bidirectional_site.py', 'ptt_pair_summary')
        m.ROOT = out
        (out/'figures').mkdir(exist_ok=True)
        invoke(m)
        f = module('models/timeppg/freeze_ptt_bidirectional_site_outcome.py', 'ptt_pair_freeze')
        f.ROOT, f.FORWARD, f.REVERSE = (out, out/'paired_window_predictions.csv',
                                      out/'reverse_proximal_to_distal/paired_window_predictions.csv')
        invoke(f)
    elif step == 18:
        source = reports/'phase11_ptt_site_transfer/pair_sensitivity/frozen_all_pair_summary.csv'
        copy_file(source, out/'frozen_all_pair_summary.csv')
        m = module('models/timeppg/freeze_ptt_all_site_pairs.py', 'ptt_all_pairs')
        m.OUT = out
        invoke(m, '--plot-only')
    elif step == 19:
        m = module('models/pulseppg/input_compatibility_smoke.py', 'pulse_compatibility')
        m.OUTPUT = out/'smoke_metrics.json'
        invoke(m)
    elif step == 20:
        m = module('models/pulseppg/run_full_linear_probe.py', 'pulse_linear')
        m.MANIFEST = historical_pulse_manifest(out)
        m.OUTPUT = out
        invoke(m, '--stage', 'evaluate')
    elif step == 21:
        # The nonlinear module imports cached_features from run_full_linear_probe;
        # that function retains the helper module's globals, so set both.
        helper = module('models/pulseppg/run_full_linear_probe.py', 'run_full_linear_probe')
        sys.modules['run_full_linear_probe'] = helper
        m = module('models/pulseppg/run_nonlinear_head_ablation.py', 'pulse_nonlinear')
        m.MANIFEST = helper.MANIFEST = historical_pulse_manifest(out)
        invoke(m, '--seeds', '17,29,43', '--batch-size', '4096', '--output', out)
    elif step == 22:
        helper = module('models/pulseppg/run_full_linear_probe.py', 'run_full_linear_probe')
        sys.modules['run_full_linear_probe'] = helper
        m = module('models/pulseppg/run_partial_finetuning.py', 'pulse_partial')
        m.MANIFEST = helper.MANIFEST = historical_pulse_manifest(out)
        m.OUT = out
        invoke(m, '--stage', 'train', '--output', out)
    elif step == 24:
        figures = sorted((ROOT/'reports').glob('**/figures/*.png'))
        if not figures:
            raise FileNotFoundError('No saved result figures exist')
        rows = [{'figure': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size}
                for p in figures]
        (out/'saved_figure_index.json').write_text(json.dumps(rows, indent=2))
        print(f'Indexed {len(rows)} existing result figures. This step does not recalculate them.')
    else:
        raise ValueError(step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', type=int)
    parser.add_argument('output', type=Path)
    parser.add_argument('windows', nargs='?', type=Path)
    args = parser.parse_args()
    run(args.step, args.output.resolve(), args.windows)
