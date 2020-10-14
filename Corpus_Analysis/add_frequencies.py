import argparse

def add_frequencies(dict_count, in_path, out_path):
    with open(out_path, 'w') as wf:
        wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("relation", "target_word", "rank", "counts", "pos", "morph"))
        with open(in_path, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split("\t")
                tw = line[1]
                try:
                    counts = dict_count[tw]
                    for c in counts:
                        wf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(line[0], line[1], line[2], c[3], c[1], c[2]))
                except:
                    print("{} not in frequency dict".format(tw))
                    continue
def read_count_dict(in_path):
    #("word form", "pos", "morph", "count"
    dict_count = dict()
    with open(in_path, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split("\t")
            if line[0] not in dict_count:
                dict_count[line[0].lower()] = []
            dict_count[line[0].lower()].append((line))
    return dict_count
def main(args):
    count_dict = read_count_dict(args.frequencies)
    add_frequencies(count_dict, args.file_path, args.out_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frequencies")
    parser.add_argument("file_path")
    parser.add_argument("out_path")
    args = parser.parse_args()
    main(args)