"""
get syncretic slots
"""
import pandas as pd
path = 'ALL_IN_ONE.csv'
out_path = 'DE/inf_analysed_per_relation.csv'
out_path2 = 'DE/inf_readme.txt'
out_path3 = 'DE/inf_relation_pairs_words.csv'
df_out_path = "DE/inf_df.csv"
df_all = pd.read_csv(path, delimiter="\t")
df = df_all[df_all['relation'].str.startswith("INF")]
rels = set(df['relation'])
dict_rels = dict()
for r in rels:
    sub_df = df[df['relation'] ==r]
    #dict_subdf = sub_df.set_index('relation').to_dict()
    dict_rels[r] = dict(zip(sub_df.w1, sub_df.w2))




new_dict = dict()
df_rels = pd.DataFrame(index=rels, columns=rels)
nrs = {k:[] for k in rels}
for r1 in rels:
    for r2 in rels:
        d1 = dict_rels[r1]
        d2 = dict_rels[r2]
        new_dict[r1+"+" + r2] = []
        c = 0
        c_not = 0
        for k,v in d1.items():
            w2_d1 = v
            try:
                w2_d2 = d2[k]
            except:
                continue
            if w2_d1 == w2_d2:
                new_dict[r1+"+" + r2].append((k, w2_d1, w2_d2))
                c +=1
            else:
                c_not +=1
        c_tot = c + c_not
        df_rels[r1][r2] = (c, c_tot)
        if c>0:
            nrs[r1].append((r2, c, c_tot))

df_rels.to_csv(df_out_path, sep="\t")


with open(out_path, 'w') as wf:
    wf.write("{}\t{}\t{}\t{}\t{}\n".format("relation1", "relation2", "count_same", "total_count", "percentage_overlap"))
    for k,v in nrs.items():
        for w in v:
            c1 = w[1]
            c2 = w[2]
            avg_c = (c1/c2)*100

            wf.write("{}\t{}\t{}\t{}\t{}\n".format(k, w[0], str(c1), str(c2), str(avg_c)))





with open(out_path2, 'w') as wf:
    wf.write("relation\tnumber\n")
    for k,v in dict_rels.items():
        wf.write("{}\t{}\n".format(k, str(len(v))))

with open(out_path3, 'w') as wf:
    wf.write("{}\t{}\t{}\t{}\t{}\n".format("r1", "r2", "base", "w2.1", "w2.2"))
    for k,v in new_dict.items():
        k = k.split("+")
        for w in v:
            wf.write("{}\t{}\t{}\t{}\t{}\n".format(k[0],k[1], w[0], w[1], w[2]))

