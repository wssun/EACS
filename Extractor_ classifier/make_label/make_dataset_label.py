import json
import jsonlines
import numpy as np
from tqdm import tqdm
import rouge_not_a_wrapper as my_rouge
from identifier_splitting import split_identifier_into_parts

from utils import *

ex_codes_num = []
ex_words_num = []


def split_code_to_blocks(code_token_list):
    first_lf_brace = True

    lf_brace_up = 0

    lf_bracket_up = 0
    idxs = []

    end_idx = 0

    for idx, i in enumerate(code_token_list):
        if i == ';' and not lf_brace_up and not lf_bracket_up:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '(':
            lf_bracket_up += 1
        elif i == ')':
            lf_bracket_up -= 1
        elif i == '{' and not first_lf_brace:
            lf_brace_up += 1
        elif i == '}' and (lf_brace_up == 1):
            idxs.append((end_idx, idx))
            end_idx = idx + 1
            lf_brace_up -= 1
        elif i == '}' and (lf_brace_up != 1):
            lf_brace_up -= 1
        elif i == '{' and first_lf_brace:
            idxs.append((0, idx))
            end_idx = idx + 1
            first_lf_brace = False

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if idx == 0:
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))
    return code_seqs


def split_java_to_seqs(code_token_list):
    lf_bracket_up = 0
    idxs = []

    end_idx = 0

    for idx, i in enumerate(code_token_list):
        if i == ';' and not lf_bracket_up:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '(':
            lf_bracket_up += 1
        elif i == ')':
            lf_bracket_up -= 1
        elif i == '{':
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '}':
            end_idx = idx + 1

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if code_snap[-1] == '{':
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))
    return code_seqs


def get_code_ex(code_seqs, nl_seq):
    if len(code_seqs) == 0 or len(nl_seq) == 0:
        return [], [], [], [], [], None

    global ex_codes_num
    global ex_words_num
    fs = []
    ps = []
    rs = []
    for i, code_seq in enumerate(code_seqs):
        rouge_l_f, rouge_l_p, rouge_l_r = my_rouge.rouge_l_summary_level([code_seq], nl_seq)
        fs.append(rouge_l_f)
        ps.append(rouge_l_p)
        rs.append(rouge_l_r)

    scores = np.array(rs)
    id_sort_by_scores = np.argsort(scores)[::-1]
    max_Rouge_l_r = 0.0
    ex_ids = []
    ex_codes = []

    for i in range(len(code_seqs)):
        new_ex_ids = sorted(ex_ids + [id_sort_by_scores[i]])
        new_ex_codes = [code_seqs[idx] for idx in new_ex_ids]
        _, _, Rouge_l_r = my_rouge.rouge_l_summary_level(new_ex_codes, nl_seq)

        if Rouge_l_r > max_Rouge_l_r:
            ex_ids = new_ex_ids
            ex_codes = new_ex_codes
            max_Rouge_l_r = Rouge_l_r

    ex_codes_num.append(len(ex_codes))
    ex_words = ' '.join(ex_codes).split(' ')
    ex_words_num.append(len(ex_words))
    return ex_codes, ex_ids, fs, ps, rs, max_Rouge_l_r


def make_py_dataset(input_path, output_path, language):
    total_num = 0
    no_ex_blocks_num = 0
    no_ex_seqs_num = 0
    with open(input_path, encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        for idx, line in tqdm(enumerate(in_f)):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            code_tokens = js['code_tokens']
            codes = ' '.join(code_tokens).replace('\n', '')
            code_tokens = codes.split()

            code_seqs = split_java_to_seqs(code_tokens)
            code_seqs = ' <spt> '.join(code_seqs)

            code_seqs = split_identifier_into_parts(code_seqs)
            code_seqs = ' '.join(code_seqs)
            code_seqs_lower = code_seqs.lower()
            code_seqs = code_seqs_lower.split(' <spt> ')

            code = ' '.join(code_tokens).replace('\n', '')
            code = split_identifier_into_parts(code)
            code_lower = [s.lower() for s in code]

            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = split_identifier_into_parts(nl)
            nl_seq = [' '.join(nl).lower()]

            js['cleaned_nl'] = nl_seq
            js['cleaned_codes'] = code_lower

            # ex_blocks, ex_ids, fs, ps, rs, max_Rouge_l_r = get_code_ex(code_blocks, nl_seq)
            # if len(ex_blocks) == 0:
            #     ex_blocks = code_lower
            #     no_ex_blocks_num += 1
            js['cleaned_blocks'] = []
            js['cleaned_blocks_ex'] = []

            ex_seqs, ex_ids, fs, ps, rs, max_Rouge_l_r = get_code_ex(code_seqs, nl_seq)
            if len(ex_seqs) == 0:
                ex_seqs = code_lower
                no_ex_seqs_num += 1
            js['cleaned_seqs'] = code_seqs
            js['cleaned_seqs_ex'] = ex_seqs
            js['ex_labels'] = [1 if i in ex_ids else 0 for i in range(len(code_seqs))]
            # js['ex_labels'] = []

            out_f.write(js)

            total_num += 1

    print('total num:', total_num)
    print('no ex blocks num:', no_ex_blocks_num)
    print('no ex seqs num:', no_ex_seqs_num)

    print('no ex blocks %:', np.round(no_ex_blocks_num / total_num, 4))
    print('no ex seqs %:', np.round(no_ex_seqs_num / total_num, 4))


if __name__ == '__main__':
    language = 'java'

    input_root = f'../../dataset/codesearchnet/{language}/'
    output_root = f'../../dataset/codesearchnet/{language}-cls/'

    dataset_file = ['train.jsonl', 'valid.jsonl', 'test.jsonl']

    for i in dataset_file:
        input_path = input_root + i
        output_path = output_root + i
        make_py_dataset(input_path, output_path, language)
