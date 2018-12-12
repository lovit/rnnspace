import os

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