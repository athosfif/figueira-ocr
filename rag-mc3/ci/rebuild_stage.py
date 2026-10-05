#!/usr/bin/env python3
"""Reproduce exact audited AMD app/vendor from a small hash-pinned JSON handoff.

Download only official binary PyPI wheels on the remote CI runner. The original
AMD native import proof is retained as provenance, not claimed as a CI import
test or final-image GPU execution.
"""
import argparse
import concurrent.futures
import importlib.util
import json
import pathlib
import platform
import re
import shutil


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rebuild(handoff_path, expected, root, stage):
    here = pathlib.Path(__file__).resolve().parent
    producer = load('ci_layer_plan', here / 'stream_hf_layer.py')
    identity = load('ci_handoff_identity', here / 'export_handoff.py').vendor_identity
    prepare = load('audited_amd_prepare', root / 'scripts' / 'prepare_oci_layer.py')
    if producer.sha_file(handoff_path) != expected or handoff_path.stat().st_size > 256 * 1024:
        raise ValueError('Small AMD handoff SHA256/size mismatch')
    handoff = json.loads(handoff_path.read_text())
    if handoff.get('status') != 'verified_amd_runtime_handoff' or handoff.get('handoff_version') != 1 or handoff.get('model_revision') != producer.REVISION:
        raise ValueError('Pinned complete AMD handoff required')
    receipt = handoff['app_layer_receipt']
    quality = handoff['gpu_quality_gate']
    smoke = receipt.get('native_vendor_import_smoke', {})
    if (receipt.get('status') != 'stream_plan_prepared' or receipt.get('weights_included') is not True
            or receipt.get('model_revision') != producer.REVISION or receipt.get('dependency_closure_verified') is not True
            or smoke.get('model_loaded') is not False or smoke.get('gpu_inference') is not False):
        raise ValueError('Complete audited pinned AMD native prepare receipt is required before downloads')
    count = quality.get('queries_total', 0)
    if quality.get('quality_verified') is not True or not isinstance(count, int) or count < 14 or quality.get('queries_passed') != count:
        raise ValueError('Completed official and independent GPU quality gate required before downloads')
    if set(quality.get('runtime_sha256', {})) != producer.RUNTIME:
        raise ValueError('All seven frozen runtime hashes are required')
    for name, expected_hash in quality['runtime_sha256'].items():
        path = root / name
        if not path.is_file() or path.is_symlink() or producer.sha_file(path) != expected_hash:
            raise ValueError(f'Approved CI source differs from successful GPU pilot: {name}')
    closure = receipt['dependency_closure']
    if producer.sha_file(root / 'requirements.txt') != closure['requirements_sha256']:
        raise ValueError('Requirements differ from the full dependency closure')
    wheels = receipt['wheels']
    planned = {row['name']: row['version'] for row in closure['wheels']}
    actual = {row['name']: row['version'] for row in wheels}
    if actual != planned or len(actual) != len(wheels):
        raise ValueError('The handoff must contain the entire dependency closure, not an install delta')
    if any(row['name'].lower().replace('_', '-') in prepare.DENY for row in wheels):
        raise ValueError('Base ROCm/Torch packages must not be replaced')
    for row in wheels:
        if (not re.fullmatch('[a-z0-9][a-z0-9-]*', row['name'])
                or pathlib.PurePosixPath(row['filename']).name != row['filename']
                or not row['filename'].endswith('.whl') or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Safe official wheel filename/package/hash required before downloads')
    if stage.is_symlink() or any(p.is_symlink() for p in stage.parents) or stage.exists() and any(stage.iterdir()):
        raise ValueError('Use a new empty non-symlink CI stage')
    if platform.system() != 'Linux':
        raise ValueError('Rebuild only on the authorized Linux CI runner')
    if shutil.disk_usage(stage.parent).free < 4 * 1024**3:
        raise ValueError('At least 4 GiB free staging space required; weights will never be saved')
    wheel_dir = stage / 'wheels'
    tree = stage / 'layer'
    vendor = tree / 'app/vendor'
    wheel_dir.mkdir(parents=True)
    vendor.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        downloaded = list(pool.map(lambda row: prepare.fetch(row, wheel_dir), wheels))
    for row in downloaded:
        prepare.extract_wheel(wheel_dir / row['filename'], vendor)
    for denied in prepare.DENY:
        if (vendor / denied.replace('-', '_')).exists():
            raise ValueError('Vendored code would shadow a base ROCm package')
    for name in sorted(producer.RUNTIME):
        shutil.copyfile(root / name, tree / 'app' / name)
        # Same mode as the audited prepare's shutil.copyfile default.
        (tree / 'app' / name).chmod(0o644)
    (tree / 'app/output').mkdir()
    (tree / 'app/output').chmod(0o777)
    (tree / 'models/qwen').mkdir(parents=True)
    files = [prepare.describe(path, path.relative_to(tree).as_posix()) for path in sorted(tree.rglob('*'))]
    if identity(files) != handoff['vendor_identity']:
        raise ValueError('Rebuilt vendor content/modes differ from the exact AMD native-tested layer')
    files += handoff['model_manifest']
    receipt['method'] = 'CI replay of exact AMD app/vendor hashes with official PyPI wheels and pinned HF model streams'
    receipt['native_amd_proof_reused'] = True
    receipt['runtime_imports_verified_in_final_image'] = False
    receipt['model_bytes_saved_in_ci'] = 0
    for name, data in [('APP-LAYER-RECEIPT.json', receipt), ('APP-LAYER-FILES.json', files),
                       ('IMAGE-CONFIG.json', handoff['image_config']), ('GPU-QUALITY-GATE.json', quality)]:
        (stage / name).write_text(json.dumps(data, indent=2) + '\n')
    producer.plan(stage)  # All source/vendor/model/quality gates, no weight requests.
    return {'status': 'amd_stage_reproduced', 'vendor_identity': handoff['vendor_identity'],
            'wheels': len(downloaded), 'model_bytes_saved': 0, 'registry_contacted': False,
            'native_amd_proof_reused': True, 'final_container_gpu_execution_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=pathlib.Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--stage', type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rebuild(args.handoff, args.sha256, args.root, args.stage), indent=2))
