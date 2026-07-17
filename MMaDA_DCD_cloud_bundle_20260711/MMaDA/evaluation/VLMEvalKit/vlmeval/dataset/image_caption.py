import json
import os
import os.path as osp

from .image_base import ImageBaseDataset
from ..smp import *


class COCO_Caption_Scorer():
    def __init__(self, ref, gt):
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print('%s: %0.3f' % (m, sc * 100))
                total_scores['Bleu'] = [x * 100 for x in score]
            else:
                print('%s: %0.3f' % (method, score * 100))
                total_scores[method] = score * 100

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores


class ImageCaptionDataset(ImageBaseDataset):

    TYPE = 'Caption'

    DATASET_URL = {
        'COCO_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL.tsv',
    }

    DATASET_MD5 = {
        'COCO_VAL': '72a5079dead060269ac222c5aa5128af',
    }

    def load_data(self, dataset):
        data = super().load_data(dataset)
        if 'question' not in data:
            data['question'] = [(
                'Please describe this image in general. Directly provide the description, '
                'do not include prefix like "This image depicts". '
            )] * len(data)
        return data

    # It returns a dictionary of scores
    @classmethod
    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        ref, gt = {}, {}
        for i, line in enumerate(lines):
            ref[str(i)] = [str(line['prediction'])]
            gt[str(i)] = eval(line['answer'])

        scorer = COCO_Caption_Scorer(ref, gt)
        coco_caption_score_dict = scorer.compute_scores()
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(coco_caption_score_dict, score_pth)
        return coco_caption_score_dict


class CHAIRDataset(ImageBaseDataset):
    """CHAIR hallucination benchmark on MSCOCO val2017.

    - The TSV is built offline by ``scripts/build_chair_tsv.py`` and shipped
      alongside the workspace's ``LMUData/CHAIR.tsv``. No MD5 check + no
      download is performed (the classical CHAIR data comes from COCO, not
      opencompass).
    - The scorer is bundled in ``vlmeval.dataset.utils.chair`` and reads the
      COCO instances_val2017.json indicated by ``CHAIR_COCO_ANN`` (env var);
      failing that, it tries ``$LMUData/coco_instances_val2017.json``, then a
      well-known LLaVA-eval mirror path.
    """

    TYPE = 'Caption'

    # No auto-download; we bring our own TSV.
    DATASET_URL = {'CHAIR': ''}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        data = super().load_data(dataset)
        if 'question' not in data.columns:
            data['question'] = (
                'Please describe this image in detail.' * len(data)
            )
        return data

    @classmethod
    def evaluate(cls, eval_file, **kwargs):
        """Score CHAIR{i,s} + recall on the current predictions file.

        Uses canonical (Rohrbach 2018) union GT = instance-masks ∪
        caption-derived objects. Set ``CHAIR_STRICT_GT=1`` to force
        instance-masks-only (older behaviour / ablation).
        """
        from .utils.chair import CHAIRScorer

        data = load(eval_file)
        if 'image_id' not in data.columns:
            raise ValueError(
                "CHAIR predictions file is missing the 'image_id' column. "
                "The TSV built by scripts/build_chair_tsv.py must include it."
            )

        def _first_existing(cands):
            return next((p for p in cands if p and osp.exists(p)), None)

        ann_path = _first_existing([
            os.environ.get('CHAIR_COCO_ANN'),
            osp.join(os.environ.get('LMUData', './LMUData'),
                     'coco_instances_val2017.json'),
            '/home/user/大模型/LLava/data/coco/annotations/instances_val2017.json',
        ])
        if ann_path is None:
            raise FileNotFoundError(
                'Could not locate COCO instances_val2017.json. '
                'Set CHAIR_COCO_ANN=/path/to/instances_val2017.json.'
            )

        cap_path = None
        if os.environ.get('CHAIR_STRICT_GT', '0') != '1':
            cap_path = _first_existing([
                os.environ.get('CHAIR_COCO_CAPS'),
                osp.join(os.environ.get('LMUData', './LMUData'),
                         'coco_captions_val2017.json'),
                '/home/user/大模型/LLava/data/coco/annotations/captions_val2017.json',
            ])

        imids = [int(r['image_id']) for _, r in data.iterrows()]
        scorer = CHAIRScorer(
            instances_json=ann_path,
            captions_json=cap_path,
            imids=imids,
        )
        records = [
            dict(image_id=int(r['image_id']),
                 caption=str(r['prediction']))
            for _, r in data.iterrows()
        ]
        result = scorer.score(records)

        details_path = eval_file.replace('.xlsx', '_chair_details.jsonl')
        with open(details_path, 'w') as f:
            for d in result['details']:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

        score_path = eval_file.replace('.xlsx', '_chair_score.csv')
        import pandas as pd
        pd.DataFrame([result['summary']]).to_csv(score_path, index=False)
        print(f'[CHAIR] wrote {score_path}')
        print(f'[CHAIR] summary ({result["summary"]["gt_source"]}): '
              f'{result["summary"]}')
        return result['summary']
