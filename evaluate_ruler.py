# -*- coding: utf-8 -*-
"""Fixed-sample RULER evaluation, original-prediction rescoring, and preflight.

CPU-only preflight/rescoring uses the standard library. Model inference imports
the pinned FLA/Transformers dependencies only when requested.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def score_prediction(prediction, answers):
    """Match the paper's postprocess_pred -> string_match_all protocol.

    Scoring source: EleutherAI/lm-evaluation-harness, RULER common_utils.py.
    This is answer containment, including fractional multi-answer scores.
    """
    if not answers:
        raise ValueError('Empty reference answer list')
    processed = re.sub(r'[\x00-\x1f]', '\n', str(prediction).strip()).strip().lower()
    return sum(str(answer).lower() in processed for answer in answers) / len(answers)


def validate_dataset(data_root):
    manifest = read_json(data_root / 'manifest.json')
    samples = {}
    for item in manifest['settings']:
        path = data_root / item['path']
        if sha256(path) != item['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {path}')
        rows = read_jsonl(path)
        if len(rows) != 500 or len({int(row['doc_id']) for row in rows}) != 500:
            raise ValueError(f'Expected 500 unique samples: {path}')
        if any(row['task'] != item['task'] or int(row['length']) != item['length']
               or not row['prompt'] or not row['answer'] for row in rows):
            raise ValueError(f'Invalid sample schema: {path}')
        samples[(item['task'], item['length'])] = rows
    if len(samples) != 11:
        raise ValueError('Expected exactly 11 main-table settings')
    return manifest, samples


def model_directory(root, item):
    folder = root / item['directory']
    if (folder / 'hf_model').is_dir():
        folder = folder / 'hf_model'
    return folder


def checkpoint_header(path):
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f'Incomplete checkpoint: {path}')
        length = struct.unpack('<Q', raw)[0]
        if length > 256 << 20:
            raise ValueError(f'Invalid safetensors header length: {path}')
        header = json.loads(stream.read(length))
    end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
    if 8 + length + end != path.stat().st_size:
        raise ValueError(f'Incomplete or corrupted safetensors data: {path}')
    return header


def preflight(args, models, samples):
    result = []
    for item in models:
        folder = model_directory(args.models_root, item)
        for name in ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']:
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        header = checkpoint_header(folder / 'model.safetensors')
        cfg = read_json(folder / 'config.json')
        expected_type = 'gated_deltanet' if item['architecture'] == 'gdn' else item['architecture']
        if cfg['model_type'] != expected_type:
            raise ValueError(f'Wrong architecture in {folder}')
        expected = item.get('recorded_checkpoint_sha256')
        hash_matches = None
        if args.check_weight_hashes and expected:
            hash_matches = sha256(folder / 'model.safetensors') == expected
            if not hash_matches:
                raise ValueError(f'Checkpoint differs from original paper checkpoint: {folder}')
        row = {'architecture':item['architecture'], 'initialization':item['initialization'],
               'directory':str(folder), 'tensors':len(header)-int('__metadata__' in header),
               'tokenizer_sha256':sha256(folder / 'tokenizer.json'),
               'weight_hash_matches':hash_matches, 'passed':True}
        result.append(row)
        print(f"[OK] {item['directory']}: complete checkpoint/config/tokenizer", flush=True)
    if len({r['tokenizer_sha256'] for r in result}) != 1:
        raise ValueError('The selected models do not use the same tokenizer')
    return {'dataset_samples':sum(map(len, samples.values())), 'models':result,
            'weight_hashes_checked':args.check_weight_hashes}


def summarize(output, models, metrics):
    output.mkdir(parents=True, exist_ok=True)
    rows, differences = [], []
    for item in models:
        key = item['architecture'] + '/' + item['initialization']
        measured = metrics[key]
        row = {'architecture':item['architecture'], 'initialization':item['initialization']}
        for cell in item['cells']:
            name = f"{cell['task']}_{cell['length']}"
            if name not in measured:
                continue
            value = measured[name]
            row[name] = value
            differences.append({'architecture':item['architecture'], 'initialization':item['initialization'],
                'setting':name, 'actual_percent':value, 'paper_percent':cell['accuracy_percent'],
                'difference_pp':value-cell['accuracy_percent'],
                'matches_paper':abs(value-cell['accuracy_percent']) < 1e-8})
        values = [v for k,v in row.items() if k not in ('architecture','initialization')]
        row['Avg.'] = sum(values)/len(values) if values else None
        row['complete_11_settings'] = len(values) == 11
        rows.append(row)
    fields = ['architecture','initialization'] + [f"{c['task']}_{c['length']}" for c in models[0]['cells']] + ['Avg.','complete_11_settings']
    with (output/'ruler_results.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    report = {'models':rows, 'comparisons':differences,
              'matching_cells':sum(c['matches_paper'] for c in differences),
              'total_cells':len(differences)}
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"Saved {len(rows)} rows; {report['matching_cells']}/{len(differences)} cells match the original paper results.")
    return report


def rescore_reference(args, models, samples):
    metrics = {}
    for item in models:
        values = {}
        for cell in item['cells']:
            task, length = cell['task'], cell['length']
            canonical = {int(r['doc_id']):r for r in samples[(task,length)]}
            path = ROOT/'reference/ruler_predictions'/cell['prediction_path']
            if sha256(path) != cell['prediction_sha256']:
                raise ValueError(f'Reference prediction checksum mismatch: {path}')
            preds = read_jsonl(path)
            if len(preds) != 500 or {int(r['doc_id']) for r in preds} != set(canonical):
                raise ValueError(f'Prediction/sample IDs do not align: {path}')
            scores = [score_prediction(r['prediction'],canonical[int(r['doc_id'])]['answer']) for r in preds]
            if any(abs(s-float(r['score'])) > 1e-12 for s,r in zip(scores,preds)):
                raise ValueError(f'Archived score does not match re-scoring: {path}')
            value = sum(scores)/len(scores)*100
            if abs(value-cell['accuracy_percent']) > 1e-8:
                raise ValueError(f'Recomputed result differs from paper: {path}')
            values[f'{task}_{length}'] = value
        metrics[item['architecture']+'/'+item['initialization']] = values
    return summarize(args.output_dir,models,metrics)


def restore_special_floats(value):
    # Original downloaded Mamba-2 configs encode infinity as a tagged object.
    if isinstance(value,dict):
        if set(value) == {'__float__'}:
            return float(value['__float__'])
        return {k:restore_special_floats(v) for k,v in value.items()}
    if isinstance(value,list):
        return [restore_special_floats(v) for v in value]
    return value


def load_checkpoint(folder, architecture, dtype, device):
    from initialization import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, GenerationConfig
    config = restore_special_floats(read_json(folder/'config.json'))
    config['use_cache'] = True
    model = build_model(architecture,config)
    model.load_state_dict(load_file(str(folder/'model.safetensors')),strict=True)
    # Same dtype collapse as the original evaluator, including Mamba-2 norms.
    model.to(device=device,dtype=dtype).eval()
    if (folder/'generation_config.json').is_file():
        model.generation_config = GenerationConfig.from_pretrained(folder,local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(folder,local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate(args, models, samples):
    import torch
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        raise RuntimeError('Model inference requires Linux + CUDA and the FLA dependencies. Preflight/rescoring can run on Windows/CPU.')
    dtype = getattr(torch,args.dtype)
    metrics = {}
    for item in models:
        arch,scheme = item['architecture'],item['initialization']
        model,tokenizer = load_checkpoint(model_directory(args.models_root,item),arch,dtype,args.device)
        values = {}
        for cell in item['cells']:
            task,length = cell['task'],cell['length'];setting=f'{task}_{length}'
            if args.settings and setting not in args.settings:
                continue
            rows = samples[(task,length)][:args.samples]
            dest = args.output_dir/arch/scheme/task/str(length)
            dest.mkdir(parents=True,exist_ok=True)
            predfile = dest/'predictions.jsonl'
            bs = args.batch_size or int(cell['batch_size'])
            signature = {'samples':args.samples,'max_new_tokens':args.max_new_tokens,'dtype':args.dtype,
                         'batch_size':bs,'model_directory':str(model_directory(args.models_root,item).resolve())}
            if args.skip_done and (dest/'result.json').is_file():
                previous = read_json(dest/'result.json')
                if previous['protocol'] != signature:
                    raise ValueError(f'Existing run uses a different protocol: {dest}')
                predictions = read_jsonl(predfile)
            else:
                predictions = []
                for start in range(0,len(rows),bs):
                    batch = rows[start:start+bs]
                    encoded = tokenizer([r['prompt'] for r in batch],padding=True,return_tensors='pt').to(args.device)
                    width = encoded['input_ids'].shape[1]
                    with torch.inference_mode():
                        generated = model.generate(**encoded,max_new_tokens=args.max_new_tokens,
                            do_sample=False,temperature=None,top_p=None,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,use_cache=True)
                    for source,tokens in zip(batch,generated):
                        prediction = tokenizer.decode(tokens[width:],skip_special_tokens=True)
                        predictions.append({'doc_id':source['doc_id'],'task':task,'length':length,
                            'prediction':prediction,'score':score_prediction(prediction,source['answer'])})
                    print(f'[{arch}/{scheme}] {setting}: {min(start+bs,len(rows))}/{len(rows)}',flush=True)
                with predfile.open('w',encoding='utf-8') as stream:
                    for row in predictions:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            if len(predictions) != len(rows) or {int(p['doc_id']) for p in predictions} != {int(r['doc_id']) for r in rows}:
                raise ValueError(f'Prediction set is incomplete: {dest}')
            gold = {int(r['doc_id']):r['answer'] for r in rows}
            accuracy = sum(score_prediction(p['prediction'],gold[int(p['doc_id'])]) for p in predictions)/len(rows)*100
            values[setting] = accuracy
            (dest/'result.json').write_text(json.dumps({'architecture':arch,'initialization':scheme,
                'accuracy_percent':accuracy,'protocol':signature},indent=2)+'\n',encoding='utf-8')
        metrics[arch+'/'+scheme] = values
        summarize(args.output_dir,[m for m in models if m['architecture']+'/'+m['initialization'] in metrics],metrics)
        del model,tokenizer
        gc.collect();torch.cuda.empty_cache()
    return summarize(args.output_dir,models,metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root',type=Path,help='Directory containing main_gdn_random, main_gdn_PoST, etc.')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/ruler')
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/ruler')
    parser.add_argument('--architecture',choices=['gdn','kda','gdn2','mamba2','rwkv7'])
    parser.add_argument('--initialization',choices=['random','post','dami'])
    parser.add_argument('--all',action='store_true',help='Evaluate all 15 conditions (the default without filters)')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--preflight',action='store_true',help='Check complete data/checkpoints without importing torch or running models')
    actions.add_argument('--rescore-reference',action='store_true',help='Re-score all archived paper predictions on CPU')
    parser.add_argument('--check-weight-hashes',action='store_true',help='Optional full checkpoint SHA-256 verification; reads every weight byte')
    parser.add_argument('--skip-done',action='store_true')
    parser.add_argument('--samples',type=int,default=500)
    parser.add_argument('--settings',nargs='+',help='Optional setting names, e.g. niah_single_1_1024')
    parser.add_argument('--batch-size',type=int,help='Override original per-setting batch sizes')
    parser.add_argument('--max-new-tokens',type=int,default=128)
    parser.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    if args.all and (args.architecture or args.initialization):parser.error('--all cannot be combined with a model filter')
    if not 1 <= args.samples <= 500 or args.max_new_tokens < 1 or (args.batch_size is not None and args.batch_size < 1):parser.error('Invalid sample count, generation length or batch size')
    registry = read_json(ROOT/'reference/ruler_models.json')
    models = [m for m in registry['models'] if (not args.architecture or m['architecture']==args.architecture)
              and (not args.initialization or m['initialization']==args.initialization)]
    _,samples = validate_dataset(args.data_root)
    known = {f'{t}_{n}' for t,n in samples}
    if args.settings and not set(args.settings) <= known:parser.error('Unknown RULER setting')
    if args.rescore_reference:
        rescore_reference(args,models,samples);return
    if args.models_root is None:parser.error('--models-root is required for preflight or model inference')
    check = preflight(args,models,samples)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'preflight.json').write_text(json.dumps(check,indent=2)+'\n',encoding='utf-8')
    if not args.preflight:evaluate(args,models,samples)


if __name__ == '__main__':
    main()
