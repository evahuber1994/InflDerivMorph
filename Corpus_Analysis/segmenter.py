"""
segment words with smor
"""
from collections import Counter
import gzip
import re
import os
import subprocess


# import morfessor

class Segmenter:
    def __init__(self, path_fst, path_segment):
        self._path_sfst = path_fst
        self._path_rules = path_segment

    def segment_word(self, word):
        # os.system("echo {} | {} {} >> {}".format(word, self.path_sfst, self.path_rules))
        print(self.path_sfst)
        result = subprocess.run("echo {} | {} {}".format(word, self.path_sfst, self.path_rules), shell=True,
                                stdout=subprocess.PIPE).stdout.decode('utf-8')
        return (self.convert_to_string_list(result))

    def segment_list_of_words(self, word_array):
        res = []
        for w in word_array:
            res.append(self.segment_word(w))
        return res

    def convert_to_string_list(self, strii):
        strii = strii.strip()
        l = strii.split("\n")
        l[0] = l[0].strip("> ")
        return l
    @property
    def path_sfst(self):
        return self._path_sfst
    @property
    def path_rules(self):
        return self._path_rules


def read_rules(rule_path):
    prefixes = []
    suffixes = []
    infixes = []
    with open(rule_path) as file:
        next(file)
        for l in file:
            l = l.strip()
            if not l: continue
            if l[1] == "pfx":
                prefixes.append((l[0], l[2]))
            elif l[2] == "sfx":
                suffixes.append((l[0], l[2]))
            else:
                infixes.append((l[0], l[2]))
    return prefixes, suffixes, infixes


prefixes = dict()  # dict with key (relation name): (prefix, pos)
suffixes = dict()  # dict with key (relation name): (suffix, pos)


def read_conll(file_path, segmenter):
    dict_all = dict()
    with open(file_path) as conll_file:
        for l in conll_file:
            l = l.strip()
            if not l: continue
            line = l.split('\t')
            wf = line[0]
            pos = line[1]
            upos = line[2]
            morph = line[3]
            segmenter.segment_word(wf)

    return dict_all

def get_morphemes(list_segmented, pos):
    print(list_segmented)
    word = ""
    print(pos)
    for w in list_segmented[1:]:
        if pos in w:
            word =w
            break
    if "#" in word or  "~" in word:
        word = re.split('<#>|<~>', word)
        idx = word[-1].index("<")
        info = word[-1][idx:]
        word[-1] = word[-1][:idx]
        return word, info
    else:
        info = word.split("<")
        return [], info

def count(all_words):
    return Counter(all_words)


def main():
    path_corpus = ''

    path_fst = '/home/evahu/Documents/Master/Master_Dissertation/morphological_analyser/SFST-1.4.7e/SFST/src/fst-infl2'
    path_seg = '/home/evahu/Documents/Master/Master_Dissertation/morphological_analyser/SFST-1.4.7e/SFST/src/zmorge-20150315-smor_newlemma.ca'
    seg = Segmenter(path_fst, path_seg)


    segmented_word = seg.segment_word("europäisch")
    print(segmented_word)
    pos_dict = {"VERB": "+V", "NOUN": "+NN", "PROPN": "+NPROP", "ADJ": "+ADJ"}
    pos = pos_dict[]
    if len(segmented_word) > 1:
        if "no result" not in segmented_word[1]:
            get_morphemes(segmented_word, pos)


if __name__ == "__main__":
    main()
