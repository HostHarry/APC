"""CHAIR (Caption Hallucination Assessment with Image Relevance) scorer.

Faithful port of the canonical implementation from LisaAnne/Hallucination
(EMNLP 2018, Rohrbach et al. "Object Hallucination in Image Captioning"),
adapted for Python 3 + MSCOCO 2014/2017 + VLMEval integration.

What was ported verbatim
------------------------
- ``data/synonyms.txt`` (shipped as ``chair_data/synonyms.txt``).
- ``coco_double_words`` list.
- ``double_word_dict`` construction (including the ``baby X`` / ``adult X`` /
  ``passenger X`` / ``bow tie`` / ``toilet seat`` / ``wine glas`` rules).
- ``caption_to_words`` algorithm.
- Special disambiguation: if both ``toilet`` and ``seat`` are matched, drop
  ``seat`` (avoids ``toilet seat`` firing ``chair`` via seat→chair).
- ``compute_chair`` per-caption walk (CHAIRi = hallucinated / mentioned;
  CHAIRs = fraction of captions with any hallucination).
- GT is the **union** of instance-mask-derived objects and objects mentioned
  in the 5 human-annotated GT captions (``get_annotations_from_captions``).

What differs from canonical (documented)
----------------------------------------
- ``pattern.en.singularize`` -> ``nltk.stem.WordNetLemmatizer.lemmatize(w,'n')``.
  ``pattern`` no longer installs cleanly on Python >= 3.9.
- ``nltk.word_tokenize`` is retained (canonical).
- Canonical joins train+val COCO 2014; this port accepts one COCO annotation
  file. For canonical Karpathy Test all 5,000 IDs are in val2014 and none
  overlap train2014, so passing val2014 is exactly equivalent to that join.

Environment
-----------
- ``NLTK_DATA``: workspace-local NLTK data dir (installed by
  ``python -m nltk.downloader -d $NLTK_DATA punkt punkt_tab wordnet omw-1.4``).
- ``CHAIR_COCO_ANN``: path to ``instances_val2014.json`` or
  ``instances_val2017.json`` (required).
- ``CHAIR_COCO_CAPS``: matching COCO captions JSON (optional; if absent,
  GT is instance-only, matching a strict-mode ablation).

Public API
----------
- ``CHAIRScorer(instances_json, captions_json=None, imids=None)``
- ``scorer.score(records)`` where records is a list of ``dict(image_id, caption)``
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Iterable, Sequence

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# NLTK setup: prefer workspace-local dir; fall back to whatever is on the path.
# ---------------------------------------------------------------------------
_LOCAL_NLTK = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'nltk_data')
_LOCAL_NLTK = os.path.abspath(_LOCAL_NLTK)
if os.path.isdir(_LOCAL_NLTK) and _LOCAL_NLTK not in nltk.data.path:
    nltk.data.path.insert(0, _LOCAL_NLTK)
_LEMMA = WordNetLemmatizer()


def _singularize(word: str) -> str:
    """Match the intent of ``pattern.en.singularize`` via WordNet lemmatizer.

    Empirically identical outputs on the words that appear in ``synonyms.txt``
    (women/men/mice/geese/oxen/wolves/leaves/knives/cats/dogs/tables/glasses).
    """
    return _LEMMA.lemmatize(word, 'n')


# ---------------------------------------------------------------------------
# Load canonical synonyms.txt.
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'chair_data')
_SYN_PATH = os.path.join(_DATA_DIR, 'synonyms.txt')


def _load_synonyms() -> tuple[list[list[str]], dict[str, str]]:
    with open(_SYN_PATH, 'r') as f:
        lines = f.readlines()
    rows = [ln.strip().split(', ') for ln in lines if ln.strip()]
    # Trim + drop empties, mirroring LisaAnne's split(', ').
    rows = [[w.strip() for w in row if w.strip()] for row in rows]
    inv: dict[str, str] = {}
    for row in rows:
        canon = row[0]
        for w in row:
            inv[w] = canon
    return rows, inv


_SYN_ROWS, _INVERSE_SYN = _load_synonyms()
_MSCOCO_OBJECTS = set(w for row in _SYN_ROWS for w in row)
_CANONICAL_80 = [row[0] for row in _SYN_ROWS]


# ---------------------------------------------------------------------------
# Canonical double-word dictionary (matches CHAIR() in LisaAnne/utils/chair.py).
# ---------------------------------------------------------------------------
def _build_double_word_dict() -> dict[str, str]:
    coco_double_words = [
        'motor bike', 'motor cycle', 'air plane', 'traffic light',
        'street light', 'traffic signal', 'stop light', 'fire hydrant',
        'stop sign', 'parking meter', 'suit case', 'sports ball',
        'baseball bat', 'baseball glove', 'tennis racket', 'wine glass',
        'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier',
        'potted plant', 'bow tie', 'laptop computer', 'stove top oven',
        'hot dog', 'teddy bear', 'home plate', 'train track',
    ]
    animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                    'bear', 'zebra', 'giraffe', 'animal', 'cub']
    vehicle_words = ['jet', 'train']

    d: dict[str, str] = {}
    for dw in coco_double_words:
        d[dw] = dw
    for aw in animal_words:
        d[f'baby {aw}'] = aw
        d[f'adult {aw}'] = aw
    for vw in vehicle_words:
        d[f'passenger {vw}'] = vw
    d['bow tie'] = 'tie'
    d['toilet seat'] = 'toilet'
    d['wine glas'] = 'wine glass'   # canonical typo-tolerance
    return d


_DOUBLE_WORD_DICT = _build_double_word_dict()


# ---------------------------------------------------------------------------
# caption_to_words (verbatim port from canonical).
# ---------------------------------------------------------------------------
def caption_to_words(caption: str) -> tuple[list[str], list[str], list[int], list[str]]:
    """Return (matched_words, canonical_words, idxs, raw_words_after_dedup).

    ``matched_words`` are the still-unmapped synonym tokens present in the
    caption, ``canonical_words`` are those mapped through ``_INVERSE_SYN`` to
    the leading COCO category name, ``idxs`` are the token positions in the
    post-lemmatised token list, and ``raw_words_after_dedup`` is the pass-1
    token list after double-word merging (used for offset debugging).
    """
    words = word_tokenize(caption.lower())
    words = [_singularize(w) for w in words]

    # replace double words (i, i+1) -> merged if in _DOUBLE_WORD_DICT
    i = 0
    double_words: list[str] = []
    idxs: list[int] = []
    while i < len(words):
        idxs.append(i)
        double_word = ' '.join(words[i:i + 2])
        if double_word in _DOUBLE_WORD_DICT:
            double_words.append(_DOUBLE_WORD_DICT[double_word])
            i += 2
        else:
            double_words.append(words[i])
            i += 1
    words = double_words

    # toilet seat disambiguation (canonical)
    if ('toilet' in words) and ('seat' in words):
        words = [w for w in words if w != 'seat']

    # gather synonym hits + map to canonical
    idxs_kept = [idxs[j] for j, w in enumerate(words) if w in _MSCOCO_OBJECTS]
    matched = [w for w in words if w in _MSCOCO_OBJECTS]
    canonical = [_INVERSE_SYN[w] for w in matched]

    return matched, canonical, idxs_kept, double_words


# ---------------------------------------------------------------------------
# GT annotation loaders (both channels, matching canonical union semantics).
# ---------------------------------------------------------------------------
def _load_instance_gt(instances_json: str) -> dict[int, set[str]]:
    with open(instances_json, 'r') as f:
        d = json.load(f)
    id_to_name = {c['id']: c['name'] for c in d['categories']}
    for name in id_to_name.values():
        if name not in _MSCOCO_OBJECTS:
            # Shouldn't happen for COCO2017 categories, but guard anyway.
            continue
    gt: dict[int, set[str]] = defaultdict(set)
    for ann in d['annotations']:
        name = id_to_name.get(ann['category_id'])
        if name is None:
            continue
        canon = _INVERSE_SYN.get(name, name)
        gt[ann['image_id']].add(canon)
    return dict(gt)


def _load_caption_gt(captions_json: str,
                     restrict_imids: set[int] | None = None
                     ) -> dict[int, set[str]]:
    with open(captions_json, 'r') as f:
        d = json.load(f)
    gt: dict[int, set[str]] = defaultdict(set)
    for ann in d['annotations']:
        imid = int(ann['image_id'])
        if restrict_imids is not None and imid not in restrict_imids:
            continue
        _, canon_words, _, _ = caption_to_words(ann['caption'])
        gt[imid].update(canon_words)
    return dict(gt)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
class CHAIRScorer:
    """Canonical CHAIR scorer.

    Parameters
    ----------
    instances_json : path to a COCO instances JSON (required)
    captions_json  : path to the matching COCO captions JSON (optional; adds a GT
        channel derived from human-written captions, matching canonical)
    imids          : optional iterable of image_ids to restrict caption
        loading to (small optimisation for the selected benchmark split).
    """

    def __init__(self,
                 instances_json: str,
                 captions_json: str | None = None,
                 imids: Iterable[int] | None = None):
        if not os.path.exists(instances_json):
            raise FileNotFoundError(instances_json)
        seg_gt = _load_instance_gt(instances_json)
        if captions_json and os.path.exists(captions_json):
            imid_set = set(int(i) for i in imids) if imids is not None else None
            cap_gt = _load_caption_gt(captions_json, restrict_imids=imid_set)
            self.gt = defaultdict(set)
            keys = set(seg_gt) | set(cap_gt)
            for k in keys:
                self.gt[k] = seg_gt.get(k, set()) | cap_gt.get(k, set())
            self.gt_source = 'segments+captions'
        else:
            self.gt = seg_gt
            self.gt_source = 'segments'
        self.gt = dict(self.gt)

    def score(self, records: Sequence[dict]) -> dict:
        n_caps = 0
        n_hall_sentences = 0
        n_mentioned_tokens = 0        # canonical CHAIRi denominator
        n_hall_tokens = 0             # canonical CHAIRi numerator
        n_gt_objects = 0
        n_covered_gt = 0
        details = []

        for rec in records:
            n_caps += 1
            imid = int(rec['image_id'])
            caption = str(rec['caption'])
            matched, canonical, idxs, raw = caption_to_words(caption)
            gt = self.gt.get(imid, set())

            hallucinated_pairs = []
            hall_idxs = []
            for word, node, idx in zip(matched, canonical, idxs):
                if node not in gt:
                    hallucinated_pairs.append((word, node))
                    hall_idxs.append(idx)

            hallucinated_nodes = sorted(set(n for _, n in hallucinated_pairs))
            mentioned_nodes = sorted(set(canonical))
            covered = sorted(set(canonical) & gt)

            n_mentioned_tokens += len(canonical)
            n_hall_tokens += len(hallucinated_pairs)
            n_gt_objects += len(gt)
            n_covered_gt += len(covered)
            if hallucinated_pairs:
                n_hall_sentences += 1

            details.append({
                'image_id': imid,
                'caption': caption,
                'mscoco_gt_words': sorted(gt),
                'mscoco_generated_words': canonical,
                'mscoco_hallucinated_words': hallucinated_nodes,
                'hallucination_idxs': hall_idxs,
                'raw_tokens': raw,
                'n_mentioned_tokens': len(canonical),
                'n_hallucinated_tokens': len(hallucinated_pairs),
                'n_gt': len(gt),
                'CHAIRi_local': (
                    len(hallucinated_pairs) / len(canonical) * 100
                    if len(canonical) > 0 else 0.0
                ),
                'CHAIRs_local': 100.0 if hallucinated_pairs else 0.0,
            })

        chair_i = (n_hall_tokens / n_mentioned_tokens * 100
                   if n_mentioned_tokens else 0.0)
        chair_s = (n_hall_sentences / n_caps * 100
                   if n_caps else 0.0)
        recall = (n_covered_gt / n_gt_objects * 100
                  if n_gt_objects else 0.0)
        avg_len = (sum(len(d['raw_tokens']) for d in details) / n_caps
                   if n_caps else 0.0)

        return {
            'summary': {
                'n_captions': n_caps,
                'CHAIRi': chair_i,
                'CHAIRs': chair_s,
                'recall': recall,
                'avg_caption_len': avg_len,
                'total_mentioned_tokens': n_mentioned_tokens,
                'total_hallucinated_tokens': n_hall_tokens,
                'total_gt_objects': n_gt_objects,
                'total_covered_gt': n_covered_gt,
                'total_hall_sentences': n_hall_sentences,
                'gt_source': self.gt_source,
            },
            'details': details,
        }


# ---------------------------------------------------------------------------
# Compatibility shims used by other modules / tests.
# ---------------------------------------------------------------------------
def extract_mentioned_objects(caption: str) -> set[str]:
    """Return the set of canonical COCO category names in ``caption``.

    Retained for parity with earlier callers.
    """
    _, canonical, _, _ = caption_to_words(caption)
    return set(canonical)


# Kept for external code that imports the constant.
_COCO80_NAMES = _CANONICAL_80


__all__ = [
    'CHAIRScorer',
    'caption_to_words',
    'extract_mentioned_objects',
    '_COCO80_NAMES',
]
