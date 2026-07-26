"""VLind-Bench adapter with official CK/VP/CB/LP pipeline scoring."""

from __future__ import annotations

import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import LMUDataRoot, dump, load, toliststr


def infer_true_or_false(response: str) -> str:
    """Official VLind True/False extractor (InstructBLIP eval)."""
    normalized = (
        str(response).lower()
        .replace('\n', ' ')
        .replace(',', '')
        .replace('.', '')
        .split(' ')
    )
    for word in normalized:
        if word == 'true':
            return 'True'
        if word == 'false':
            return 'False'
    return 'NA'


def score_vlind_pipeline(data: pd.DataFrame, vote_thres: int = 2) -> dict:
    """Reproduce eval/score_pipeline.py (style=all).

    Primary metric for this repo's experiment ledger: ``LP_acc`` =
    ``d_pass_ratio_macro`` (pipeline LP accuracy, percent).
    """
    concept2num_instance = defaultdict(int)
    concept2num_image = defaultdict(int)
    concept2scores = defaultdict(lambda: defaultdict(float))

    grouped = data.groupby('global_id', sort=False)
    for global_id, group in grouped:
        by_key = {}
        for _, row in group.iterrows():
            key = str(row['query_key'])
            image_id = '' if pd.isna(row.get('image_id', '')) else str(row['image_id'])
            pred = infer_true_or_false(row['prediction'])
            if key in ('d1', 'd2'):
                by_key.setdefault(key, {})[image_id] = pred
            else:
                by_key[key] = pred

        concept = str(group.iloc[0]['concept'])
        # Reconstruct good images from LP rows present in the TSV.
        good_images = sorted(by_key.get('d1', {}).keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        if not good_images:
            continue

        concept2num_instance[concept] += 1
        concept2num_instance['total'] += 1
        concept2num_image[concept] += len(good_images)
        concept2num_image['total'] += len(good_images)

        a_pass = int(by_key.get('a1') == 'True' and by_key.get('a2') == 'False')
        b_pass = int(by_key.get('b1') == 'True' and by_key.get('b2') == 'False')
        c_raw = int(by_key.get('c1') == 'True' and by_key.get('c2') == 'False')
        c_pass = int(c_raw and a_pass)
        bc_pass = int(b_pass and c_pass)

        d_flags = []
        d_pass_flags = []
        for image_id in good_images:
            d_ok = int(
                by_key.get('d1', {}).get(image_id) == 'True'
                and by_key.get('d2', {}).get(image_id) == 'False'
            )
            d_flags.append(d_ok)
            d_pass_flags.append(int(d_ok and bc_pass))
        d_macro = sum(d_flags) / len(good_images)
        d_pass_macro = sum(d_pass_flags) / len(good_images)

        for bucket in (concept, 'total'):
            concept2scores[bucket]['a'] += a_pass
            concept2scores[bucket]['b'] += b_pass
            concept2scores[bucket]['c'] += c_raw
            concept2scores[bucket]['d_macro'] += d_macro
            concept2scores[bucket]['d_micro'] += sum(d_flags)
            concept2scores[bucket]['a_pass'] += a_pass
            concept2scores[bucket]['b_pass'] += b_pass
            concept2scores[bucket]['c_pass'] += c_pass
            concept2scores[bucket]['bc_pass'] += bc_pass
            concept2scores[bucket]['d_pass_macro'] += d_pass_macro
            concept2scores[bucket]['d_pass_micro'] += sum(d_pass_flags)
            concept2scores[bucket]['bc_pass_micro'] += bc_pass * len(good_images)

    def _ratio(num, den):
        return float(num) / float(den) if den else 0.0

    scores = {}
    for concept, score_dict in concept2scores.items():
        n_inst = concept2num_instance[concept]
        scores[concept] = {
            'n_instance': n_inst,
            'n_image': concept2num_image[concept],
            'CK_acc': 100.0 * _ratio(score_dict['a'], n_inst),
            'VP_acc': 100.0 * _ratio(score_dict['b'], n_inst),
            'CB_acc': 100.0 * _ratio(score_dict['c'], n_inst),
            'LP_raw_macro': 100.0 * _ratio(score_dict['d_macro'], n_inst),
            'CK_pass': 100.0 * _ratio(score_dict['a_pass'], n_inst),
            'VP_pass': 100.0 * _ratio(score_dict['b_pass'], n_inst),
            'CB_pass': 100.0 * _ratio(score_dict['c_pass'], score_dict['a_pass']),
            'LP_acc': 100.0 * _ratio(score_dict['d_pass_macro'], score_dict['bc_pass']),
            'LP_acc_micro': 100.0 * _ratio(
                score_dict['d_pass_micro'], score_dict['bc_pass_micro']),
        }

    total = scores.get('total', {})
    summary = {
        'CK_acc': total.get('CK_acc', 0.0),
        'VP_acc': total.get('VP_acc', 0.0),
        'CB_acc': total.get('CB_acc', 0.0),
        'LP_raw_macro': total.get('LP_raw_macro', 0.0),
        'CK_pass': total.get('CK_pass', 0.0),
        'VP_pass': total.get('VP_pass', 0.0),
        'CB_pass': total.get('CB_pass', 0.0),
        'LP_acc': total.get('LP_acc', 0.0),
        'LP_acc_micro': total.get('LP_acc_micro', 0.0),
        'n_instance': total.get('n_instance', 0),
        'n_image': total.get('n_image', 0),
        'vote_thres': vote_thres,
    }
    return {'summary': summary, 'by_concept': scores}


class VLindBenchDataset(ImageBaseDataset):
    """Official VLind-Bench expanded query set.

    Reads ``$LMUData/VLind-Bench.tsv`` from ``scripts/build_vlind_tsv.py``.
    Primary reported metric: ``LP_acc`` (pipeline language-prior accuracy).
    """

    TYPE = 'VQA'
    DATASET_URL = {
        'VLind-Bench': '',
    }
    DATASET_MD5 = {}

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        self.data_path = data_path
        if not osp.isfile(data_path):
            raise FileNotFoundError(
                f'Missing {data_path}. Run scripts/build_vlind_tsv.py first.')
        data = load(data_path)
        required = {
            'index', 'global_id', 'concept', 'stage', 'query_key',
            'image_path', 'question', 'answer',
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f'{dataset}.tsv is missing columns: {sorted(missing)}')
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            target_path = toliststr(line['image_path'])
        else:
            target_path = self.dump_image(line)

        messages = []
        if isinstance(target_path, list):
            messages.extend(
                dict(type='image', value=path) for path in target_path)
        else:
            messages.append(dict(type='image', value=target_path))
        messages.append(dict(type='text', value=str(line['question'])))
        return messages

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        data = load(eval_file)
        required = {
            'global_id', 'concept', 'query_key', 'prediction', 'answer',
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f'Eval file missing columns for VLind scoring: {sorted(missing)}')

        vote_thres = 2
        if 'vote_thres' in data.columns and len(data):
            try:
                vote_thres = int(data.iloc[0]['vote_thres'])
            except Exception:
                vote_thres = 2

        data = data.copy()
        data['extracted'] = [infer_true_or_false(x) for x in data['prediction']]
        data['hit'] = (data['extracted'] == data['answer']).astype(int)
        detail_path = eval_file.replace('.xlsx', '_vlind_detail.xlsx')
        if detail_path == eval_file:
            detail_path = eval_file + '.vlind_detail.xlsx'
        dump(data, detail_path)

        scored = score_vlind_pipeline(data, vote_thres=vote_thres)
        summary = scored['summary']

        # Flat one-row score table for VLMEvalKit consumers.
        score_df = pd.DataFrame([summary])
        score_tgt = eval_file.replace('.xlsx', '_score.csv')
        if score_tgt == eval_file:
            score_tgt = eval_file + '.score.csv'
        dump(score_df, score_tgt)

        concept_rows = []
        for concept, metrics in scored['by_concept'].items():
            if concept == 'total':
                continue
            row = {'concept': concept}
            row.update(metrics)
            concept_rows.append(row)
        if concept_rows:
            concept_df = pd.DataFrame(concept_rows).sort_values('concept')
            concept_tgt = eval_file.replace('.xlsx', '_score_by_concept.csv')
            if concept_tgt == eval_file:
                concept_tgt = eval_file + '.score_by_concept.csv'
            dump(concept_df, concept_tgt)

        # Also keep a numpy-friendly overall accuracy on atomic TF hits.
        summary['query_acc'] = float(100.0 * data['hit'].mean()) if len(data) else 0.0
        dump(pd.DataFrame([summary]), score_tgt)
        return score_df
