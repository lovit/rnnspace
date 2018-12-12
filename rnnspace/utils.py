from collections import Counter
import os
import torch

installpath = os.path.sep.join(
    os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])

def space_tag(sent, nonspace=0, space=1):
    """
    :param sent: str
        Input sentence
    :param nonspace: Object
        Non-space tag. Default is 0, int type
    :param space: Object
        Space tag. Default is 1, int type

    It returns
    ----------
    chars : list of character
    tags : list of tag

    (example)
        sent  = '이건 예시문장입니다'
        chars = list('이건예시문장입니다')
        tags  = [0,1,0,0,0,0,0,0,1]
    """

    sent = sent.strip()
    chars = list(sent.replace(' ',''))
    tags = [nonspace]*(len(chars) - 1) + [space]
    idx = 0

    for c in sent:
        if c == ' ':
            tags[idx-1] = space
        else:
            idx += 1

    return chars, tags

def scan_vocabulary(texts, min_count=1):
    """
    :param texts: list of str
        Input sentences
    :param min_count: int
        Minimum frequency of vocabulary occurrence. Default is 1

    It returns
    ----------
    idx_to_vocab: list of str
    vocab_to_idx: dict, str to int
    """

    counter = Counter(vocab for text in texts for vocab in text)
    counter = {vocab:count for vocab, count in counter.items() if count >=min_count}
    idx_to_vocab = [vocab for vocab in sorted(counter, key=lambda x:-counter[x])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def to_idx(item, mapper, unknown=None):
    """
    :param item: Object
        Object to be encoded
    :param mapper: dict
        Dictionary from item to idx
    :param unknown: int
        Index of unknown item. If None, use len(mapper)

    It returns
    ----------
    idx : int
        Index of item
    """

    if unknown is None:
        unknown = len(mapper)
    return mapper.get(item, unknown)

def to_item(idx, idx_to_vocab, unknown='Unk'):
    """
    :param idx: int
        Index of item
    :param idx_to_vocab: list of Object
        Mapper from index to item object
    :param unknown: Object
        Return value when the idx is outbound of idx_to_vocab
        Default is 'Unk', str type

    It returns
    ----------
    object : object
        Item that corresponding idx
    """

    if 0 <= idx < len(idx_to_vocab):
        return idx_to_vocab[idx]
    return unknown

def sent_to_xy(sent, vocab_to_idx):
    """
    :param sent: str
        Input sentence
    :param vocab_to_idx: dict
        Dictionary from character to index

    It returns
    ----------
    idxs : torch.LongTensor
        Encoded character sequence
    tags : torch.LongTensor
        Space tag sequence
    """

    chars, tags = space_tag(sent)
    idxs = torch.LongTensor(
        [to_idx(c, vocab_to_idx) for c in chars])
    tags = torch.LongTensor(tags)
    return idxs, tags