from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from meteor.meteor import Meteor
import numpy as np
import sys

from bleu.bleu import computeMaps, bleuFromMaps


def main(hyp, ref, len):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        # for k, v in enumerate(hypothesis):
        #     aa = v.strip().lower().split()[1:len]
        #     k = [" ".join(v.strip().lower().split()[:len])]
        #     print(k)
        res = {k: [" ".join(v.strip().lower().split()[1:len])] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        gts = {k: [" ".join(v.strip().lower().split()[1:])] for k, v in enumerate(references)}

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: "), score_Meteor

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    print("ROUGe: "), score_Rouge

    score_Cider, scores_Cider = Cider().compute_score(gts, res)
    print("Cider: "), score_Cider

    return scores_Meteor, scores_Rouge


if __name__ == '__main__':
    hyp_path = sys.argv[1]
    ref_path = sys.argv[2]

    scores_Meteor, scores_Rouge = main(hyp_path, ref_path, 64)

    pred_file = open(hyp_path, 'r')
    pred_lines = pred_file.readlines()
    predictions = []
    for idx, row in enumerate(pred_lines):
        predictions.append(row[:-1])

    ref_file = open(ref_path, 'r')
    ref_lines = ref_file.readlines()
    references = []
    for idx, row in enumerate(ref_lines):
        references.append(row[:-1])

    pred_file.close()
    ref_file.close()

    (goldMap, predictionMap) = computeMaps(predictions, references)
    bleu, bleus = bleuFromMaps(goldMap, predictionMap)
    print("Bleu 4: ",bleu[0])

