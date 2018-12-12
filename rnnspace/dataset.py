import os
import pickle
from .utils import installpath

def load_lalaland(score=False):
    path = installpath + '/data/lalaland_text_score.txt'
    with open(path, encoding='utf-8') as f:
        texts, scores = zip(*[doc[:-1].split('\t') for doc in f])
    if not score:
        return texts
    scores = [int(s) for s in scores]
    return texts, scores

def load_lalaland_train_data():
    path = installpath + '/data/lalaland_train_data.pkl'
    with open(path, 'rb') as f:
        params = pickle.load(f)
    images = params['images']
    labels = params['labels']
    idx_to_vocab = params['idx_to_vocab']
    idx_to_tag = params['idx_to_tag']
    return images, labels, idx_to_vocab, idx_to_tag