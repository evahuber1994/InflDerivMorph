import argparse

def add_frequencies(dict_count, in_path, out_path, word="w1"):
    print("gets frequencies for results file")
    assert word == "w1" or word == "w2", "invalid word "
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("relation", word, "rank", "counts", "pos", "morph"))
        with open(in_path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split("\t")
                if word == 'w1':
                    tw = line[1]
                else:
                    tw = line[2]
                try:
                    counts = dict_count[tw]
                    for c in counts:
                        wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(line[0], tw, line[2], c[3], c[1], c[2]))
                except:
                    print("{} not in frequency dict".format(tw))
                    wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(line[0],tw, line[2], "NA", "NA", "NA"))


def add_frequencies_to_words(dict_count, in_path, out_path, word):
    print("gets frequencies for word file")
    print("word type", word)
    assert word == "w1" or word == "w2", "invalid word "
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\t{}\n".format("relation", word, "counts", "pos", "morph"))
        with open(in_path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split("\t")
                if word == "w1":
                    tw = line[1].lower()
                elif word == "w2":
                    tw = line[2].lower()
                else:
                    print("invalid word")
                if tw in dict_count:
                    counts = dict_count[tw]
                    for c in counts:
                        if len(c) > 2:
                            wf.write("{}\t{}\t{}\t{}\t{}\n".format(line[0], tw, c[-1], c[1], c[2]))
                else:
                    print("{} not in frequency dict".format(tw))
                    wf.write("{}\t{}\t{}\t{}\t{}\n".format(line[0], tw, "NA", "NA", "NA"))
def read_count_dict(in_path):
    #("word form", "pos", "morph", "count"
    dict_count = dict()
    with open(in_path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split("\t")
            if line[0].lower() not in dict_count:
                dict_count[line[0].lower()] = []
            dict_count[line[0].lower()].append((line))
    return dict_count
def main(args):
    count_dict = read_count_dict(args.frequencies)
    print("finished reading count dict")
    if args.type_file =="results":
        add_frequencies(count_dict, args.file_path, args.out_path, word =args.word_type)
    else:
        add_frequencies_to_words(count_dict, args.file_path, args.out_path, word=args.word_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frequencies")
    parser.add_argument("file_path")
    parser.add_argument("out_path")
    parser.add_argument("word_type")
    parser.add_argument("type_file", help="results or words")
    args = parser.parse_args()
    main(args)