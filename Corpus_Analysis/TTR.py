"""
get TTR from random samples
"""
import argparse
import os
#import lzma
from backports import lzma
import gzip
import random
import numpy as np
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
    nr_toks = 0
    dict_words = dict()
    while i < nr_sents:
        path_f = select_rand_sentences(path)
        path_f = os.path.join(path, path_f)
        s = read_sentences(path_f)
            #se = next(s)
        for se in s:
            i += 1
            for t in se:
                nr_toks+=1
                if t not in dict_words:
                    dict_words[t] = 0
                dict_words[t] +=1
            if i > nr_sents:
                break

    return dict_words, nr_toks

def get_TTR(dict_words):
    t_count = 0
    for v in dict_words.values():
        t_count +=v
    typ_count = len(dict_words)
    print("typ_count: {}, t_count: {}".format(str(typ_count), str(t_count)))
    return typ_count/t_count

def main(args):
    ttrs = []
    all_nr_toks = []
    for i in range(1,int(args.nr_samples)):
        dict_words, nr_toks = get_sents(args.path_dirs, int(args.nr_sent))
        all_nr_toks.append(nr_toks)
        ttr = get_TTR(dict_words)
        print(str(ttr))
        ttrs.append(ttr)
    mean_ttr = np.mean(ttrs)
    sd_ttr = np.std(ttrs)
    mean_nrtoks = np.mean(all_nr_toks)
    sd_nrtoks = np.std(all_nr_toks)
    print("mean ttr out of {} samples: {}".format(str(args.nr_samples), str(mean_ttr)))
    print("sd: {}".format(str(sd_ttr)))
    print("nr of toks, mean: {}, std: {}".format(str(mean_nrtoks), str(sd_nrtoks)))

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_dirs")
    parser.add_argument("nr_sent")
    parser.add_argument("nr_samples")
    args = parser.parse_args()
    main(args)


