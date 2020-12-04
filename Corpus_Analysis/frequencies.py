"""
get frequencies from corpus
"""
import argparse
import os
#import lzma
from backports import lzma
import gzip
def get_save_frequencies(in_path, out_path):
    dict_all = dict()
    dirs = os.listdir(in_path)
    for i,f in enumerate(dirs):
        print("{}/{}".format(str(i), str(len(dirs))))
        path = os.path.join(in_path, f)
        with lzma.open(path, 'rt') as rf:
            for l in rf:
                l = l.strip()
                if not l: continue
                if l.startswith("#"): continue
                line = l.split("\t")
                if "-" in line[0]: continue
                word = "{}_{}_{}".format(line[1], line[3], line[5])
                if word in dict_all:
                    dict_all[word] +=1
                else:
                    dict_all[word] = 1
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\n".format("word form", "pos", "morph", "count"))
        for k,v in dict_all.items():
            word_split = k.split("_")
            wf.write("{}\t{}\t{}\t{}\n".format(word_split[0], word_split[1], word_split[2], str(v)))



def main(paths):
    get_save_frequencies(paths.directory_path, paths.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory_path")
    parser.add_argument("out_path")
    args = parser.parse_args()
    main(args)