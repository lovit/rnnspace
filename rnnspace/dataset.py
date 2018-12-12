import os
from .utils import installpath

def load_lalaland(score=False):
    path = installpath + '/data/lalaland_text_score.txt'
    with open(path, encoding='utf-8') as f:
        texts, scores = zip(*[doc[:-1].split('\t') for doc in f])
    if not score:
        return texts
    scores = [int(s) for s in scores]
    return texts, scores