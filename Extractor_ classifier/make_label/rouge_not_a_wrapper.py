from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import itertools
import numpy as np
import re
from collections import defaultdict
import pdb


def _get_ngrams(n, text):
  ngram_dict = defaultdict(int)
  text_length = len(text)
  max_index_ngram_start = text_length - n + 1
  for i in range(max_index_ngram_start):
    if n > 1:
      ngram_dict[tuple(text[i:i + n])] += 1
    else:
      ngram_dict[text[i]] += 1
  return ngram_dict, max_index_ngram_start

def _preprocess(sentence):
  s = sentence.lower()
  try:
    # s = re.sub('-', ' - ', s.decode('utf-8'))
    s = re.sub('-', ' - ', s)
  except Exception as e:
    print(e)
  #s = re.sub('-', ' - ', s)
  s = re.sub('[^A-Za-z0-9\-]', ' ', s) # replace not A~Z, a~z, 0~9 to a space
  s = s.strip()
  return s

def _split_into_words(sentences):

  return list(itertools.chain(*[_preprocess(s).split() for s in sentences]))


def _get_word_ngrams(n, sentences):
  assert len(sentences) > 0
  assert n > 0

  words = _split_into_words(sentences)
  return _get_ngrams(n, words)


def _len_lcs(x, y):
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _recon_lcs(x, y):
  i, j = len(x), len(y)
  table = _lcs(x, y)
  if table[i, j] == 0:
    return []
  
  lcs = []
  while 1:
    if i == 0 or j == 0:
      break
    elif x[i - 1] == y[j - 1]:
      lcs = [(x[i - 1], i - 1)] + lcs
      i = i - 1
      j = j - 1
    elif table[i - 1, j] > table[i, j - 1]:
      i = i - 1
    else:
      j = j - 1

  '''
  def _recon(i, j):
    """private recon calculation"""
    if i == 0 or j == 0:
      return []
    elif x[i - 1] == y[j - 1]:
      return _recon(i - 1, j - 1) + [(x[i - 1], i - 1)]
    elif table[i - 1, j] > table[i, j - 1]:
      return _recon(i - 1, j)
    else:
      return _recon(i, j - 1)

  LCS = _recon(len(x), len(y))
  pdb.set_trace()
  '''
  return lcs


def rouge_n(evaluated_sentences, reference_sentences, n=2):
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    #raise ValueError("Collections must contain at least 1 sentence.")
    return 0.0, 0.0, 0.0

  evaluated_ngrams, evaluated_count = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams, reference_count = _get_word_ngrams(n, reference_sentences)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_count = 0
  for ngram in reference_ngrams:
    if ngram in evaluated_ngrams:
      count1 = reference_ngrams[ngram]
      count2 = evaluated_ngrams[ngram]
      hit = count1 if count1 < count2 else count2
      overlapping_count += hit

  return _f_p_r_1(overlapping_count, reference_count, evaluated_count)


def _f_p_r_1(l, m, n):
  r = l / m if m > 0 else 0.0
  p = l / n if n > 0 else 0.0

  if r + p == 0:
    f = 0.0
  else:
    f = 2.0 * ((r * p) / (r + p))
  return f, p, r


def _f_p_r_2(l, m, n):
  r = l / m if m > 0 else 0.0
  p = l / n if n > 0 else 0.0
  
  beta = p / (r + 1e-12)
  num = (1 + (beta**2)) * r * p
  denom = r + ((beta**2) * p)
  f =  num / (denom + 1e-12)
  return f, p, r


def _union_lcs(evaluated_sentences, reference_sentence):
  if len(evaluated_sentences) <= 0:
    return set()
    #raise ValueError("Collections must contain at least 1 sentence.")

  lcs_union = set()
  reference_words = _split_into_words([reference_sentence])
  combined_lcs_length = 0
  for eval_s in evaluated_sentences:
    evaluated_words = _split_into_words([eval_s])
    lcs = set(_recon_lcs(reference_words, evaluated_words))
    lcs_union = lcs_union.union(lcs) # a list of tuple (hit_unigram, index in reference)

  return lcs_union


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    return 0.0, 0.0, 0.0
    #raise ValueError("Collections must contain at least 1 sentence.")

  # unigram dictionary for reference and evaluated sentences

  # m, n为词总数
  ref_1gram_dict, m = _get_word_ngrams(1, reference_sentences)
  eval_1gram_dict, n = _get_word_ngrams(1, evaluated_sentences)

  total_hits = 0
  for ref_s in reference_sentences:
    ref_hits = list(_union_lcs(evaluated_sentences, ref_s))
    for w in ref_hits:
      if ref_1gram_dict[w[0]] > 0 and eval_1gram_dict[w[0]] > 0:
        total_hits += 1
        ref_1gram_dict[w[0]] -= 1
        eval_1gram_dict[w[0]] -= 1
  return _f_p_r_1(total_hits, m, n)

