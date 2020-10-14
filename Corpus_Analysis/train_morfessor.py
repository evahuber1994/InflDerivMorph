import morfessor
import os
import gzip

#dir = '/zdv/sfb833-a3/treebanks/taz/r3'
d = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/Corpus_Analysis/corpus_sample'
out = 'words.txt'
test_words = ['aufgehen', 'untergehen', 'gegangen', 'Schöne', 'schön', 'Kleiner', 'wiedersehen', 'sagtest', 'reinhören', 'Vaters', 'Kastanienbaums', 'lieber']
words = []
count = 0
#for d in os.listdir(dir):
#    d = os.path.join(dir, d)
for f in os.listdir(d):
    count += 1
    path = os.path.join(d, f)
    with open(out, 'w') as wf:
        o = open
        if path.endswith('.gz'):
            o = gzip.open
        with o(path, 'rt') as fr:
            for l in fr:
                l = l.strip()
                if not l: continue
                line = l.split('\t')
                if len(line) > 1:
                    wf.write(line[1] + "\n")

    io = morfessor.MorfessorIO()
    train_data = list(io.read_corpus_file(out))
    print("finished reading data")
    if count == 1:
        seg_model = morfessor.BaselineModel()
    else:
        seg_model = io.read_binary_model_file('model_out.bin')
    seg_model.load_data(train_data)
    print("start training")
    seg_model.train_batch()
    for w in test_words:
        print(count, seg_model.viterbi_segment(w))
    io.write_binary_file('model_out.bin', seg_model)


