import argparse
import os
#import lzma
from backports import lzma
import gzip
import random

def read_sentences(path):
    with lzma.open(path, 'rt') as inf:
        sent = []
        for line in inf:
            if line.startswith('#'):
                continue
            if line == '\n':
                if sent:
                    yield sent
                    sent = []
                continue
            idx, line = line.split('\t', maxsplit=1)
            if '-' in idx or "." in idx:
                continue
            fields = line.rstrip().split('\t')
            token = fields[0]
            sent.append(token)
        if sent:
            yield sent


def select_rand_sentences(dir):
    dirs = os.listdir(dir)
    return random.choice(dirs)

def get_sents(path, nr_sents):
    i = 0
    dict_words = dict()
    while i < nr_sents:
        path_f = select_rand_sentences(path)
        path_f = os.path.join(path, path_f)
        s = read_sentences(path_f)
            #se = next(s)
        for se in s:
            print("Sent:{}".format(se))
            i += 1
            for t in se:
                if t not in dict_words:
                    dict_words[t] = 0
                dict_words[t] +=1
            if i > nr_sents:
                break

    return dict_words

def get_TTR(dict_words):
    t_count = 0
    for v in dict_words.values():
        t_count +=v
    typ_count = len(dict_words)
    print("typ_count: {}, t_count: {}".format(str(typ_count), str(t_count)))
    return typ_count/t_count

def main(args):
    dict_words = get_sents(args.path_dirs, int(args.nr_sent))
    ttr = get_TTR(dict_words)
    print(str(ttr))



if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_dirs")
    parser.add_argument("nr_sent")
    args = parser.parse_args()
    main(args)


