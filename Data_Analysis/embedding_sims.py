from data_reader import read_deriv, FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
file = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/rus_data_conll/DERIVATION_TH50/Derivbase_finalfilter_thresh80.csv'
emb_path = '/embeddings/rus_65/model.fifu'
out_path = '/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/rus_data_conll/DERIVATION_TH50/embedding_sims'
rels, w1s, w2s = read_deriv(file)

feature_extractor = FeatureExtractor(emb_path, 100)
with open(out_path, 'w') as wf:
    for w1, w2, r in zip(w1s, w2s, rels):
        w1_emb = feature_extractor.get_embedding(w1)
        w2_emb = feature_extractor.get_embedding(w2)
        sim = cosine_similarity(w1_emb.reshape(1,-1), w2_emb.reshape(1,-1))
        wf.write("{}\t{}\t{}\t{}\n".format(r, w1, w2, str(round(sim[0][0], 2))))

