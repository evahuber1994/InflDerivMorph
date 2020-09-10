import numpy as np
import math
import csv
from sklearn.metrics.pairwise import cosine_similarity
import random
class Ranker:
    def __init__(self, path_predictions, target_words, vocabulary_matrix, lab2idx, idx2lab, path_results):
        self._vocabulary_matrix = vocabulary_matrix
        self._lab2idx = lab2idx
        self._idx2lab = idx2lab
        # self._extractor = embedding_extractor
        # self._vocabulary = self.extractor.vocab.words #list with whole vocabulary
        # print("voc", len(self._vocabulary))
        # #self._vocabulary = target_words
        # #random.shuffle(self._vocabulary)
        # self._lab2idx = {word:i+1 for i, word in enumerate(self.vocabulary)}
        # self._lab2idx['UNK'] = 0
        #     #dict(zip(self.vocabulary, range(len(self.vocabulary)))) #dict with vocabulary and indices
        # self._idx2lab = {i:word for word,i in self.lab2idx.items()}
        # self._vocabulary_matrix = np.zeros((len(self.vocabulary)+1, embedding_dim)) # matrix with whole vocabulary
        # self._vocabulary_matrix[0] = np.random.rand(1,embedding_dim)
        #
        # for i in range(len(self.vocabulary)):
        #     self._vocabulary_matrix[i+1] = embedding_extractor.get_embedding(self.vocabulary[i])
        # print("built vocabualry matrix in ranker")
        self._predicted_embeddings = np.load(path_predictions, allow_pickle=True)
        #here open embeddings an save in prediction_matrix
        self._prediction_matrix = None

        self._target_words = target_words
        #self._target_embeddings = self._extractor.get_array_embeddings(self.target_words)
        self._path_results = path_results
        self._ranks, gold_sims, self._preds_sims = self.get_rank()
        self._quartiles = self.calculate_quartiles(self.ranks)
        self._reciprank = self.reciprocal_rank(self.ranks)

        #if save_detailed:

        self.save_metrics(self.target_words,self.ranks, self.quartiles, self.reciprank,self.preds_sims)


    def get_rank(self):
        ranks = []
        gold_similarities = []
        prediction_similarities = []


        #target_ids = [self.lab2idx[lab] for lab in self.target_words]
        target_ids = []
        for lab in self.target_words:
            if lab in self.lab2idx.keys():
                target_ids.append(self.lab2idx[lab])
            else:
                target_ids.append(self.lab2idx['UNK'])


        #matrix of target representations ordered
        target_repr = np.take(self.vocabulary_matrix, target_ids, axis=0)
        print("shape of matrix of target representations", target_repr.shape)
        #similarities between all words in vocabulary and target representations (derived or inflected forms) 
        target_similarities = cosine_similarity(self.vocabulary_matrix, target_repr)#not sure
        print("shape of matrix that should contain similarities of voc to target repr", target_similarities.shape)
        for i in range(self.predicted_embeddings.shape[0]):

            target_prediction_similarity = cosine_similarity(self.predicted_embeddings[i].reshape(1,-1), target_repr[i].reshape(1,-1))
            print("target to prediction similarity", target_prediction_similarity)
            prediction_similarities.append(float(target_prediction_similarity[0]))
            gold_similarities.append(target_similarities[:, i])
            # delete similarity ebtween target label and itself
            target_sims = np.delete(target_similarities[:, i], target_ids[i])
            print("target sims", len(target_sims), target_sims)
            #higher_ranks1 = np.nonzero(target_sims > target_prediction_similarity[0])    #nonzero returns tuple, therefore pick first element
            rank = np.count_nonzero(target_sims > target_prediction_similarity) +1
            #print("nonzero instead of counts", higher_ranks1)
            print("items at higher ranks than target", rank)
            #self.save_ranks(higher_ranks.tolist(), target_ids[i])
            #ranks.append(len(higher_ranks) + 1)
            ranks.append(rank)
        return ranks, gold_similarities, prediction_similarities
    def metric(self):
        pass

    @staticmethod
    def reciprocal_rank(ranks):
        """
        get reciprocal rank (1/r)
        :param ranks: list of ranks
        :return: list of RR
        """
        return [float("{:.2f}".format(1/r)) for r in ranks]


    @staticmethod
    def calculate_quartiles(ranks):
        """
        get the quartiles for the data
        :param ranks: a list of ranks
        :return: the three quartiles we are interested in, string representation of percentage of data that are rank 1
        and percentage of data that are
        """
        sorted_data = sorted(ranks)
        leq5 = sum([1 for rank in sorted_data if rank <= 5])
        leq1 = sum([1 for rank in sorted_data if rank == 1])
        if len(ranks) < 3:
            return ranks, "%.2f%% of ranks = 1; %.2f%% of ranks <=5" % (
                (100 * leq1 / float(len(sorted_data))), (100 * leq5 / float(len(sorted_data))))
        mid_index = math.floor((len(sorted_data) - 1) / 2)
        if len(sorted_data) % 2 != 0:
            quartiles = list(map(np.median, [sorted_data[0:mid_index], sorted_data, sorted_data[mid_index + 1:]]))
        else:
            quartiles = list(map(np.median, [sorted_data[0:mid_index + 1], sorted_data, sorted_data[mid_index + 1:]]))
        return quartiles, "%.2f%% of ranks = 1; %.2f%% of ranks <=5" % (
            (100 * leq1 / float(len(sorted_data))), (100 * leq5 / float(len(sorted_data))))

    def save_metrics(self, target_words, ranks, q, rr, preds_sims):
        with open(self.path_results, 'w') as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(("target_word", "rank", "recip_rank", "similarity"))
            for inst in zip(target_words, ranks, rr, preds_sims):
                writer.writerow(inst)


        path_coarse = self.path_results.strip(".csv") + "_average.csv"
        with open(path_coarse, 'w') as file:
            file.write("average rank:\t {} \n".format(sum(ranks)/len(ranks)))
            file.write("average recip rank:\t {} \n".format(sum(rr)/len(rr)))
            file.write("average similarity:\t {} \n".format(sum(preds_sims)/len(preds_sims)))
            file.write("quartiles:\t {} \n".format(q))


            #file.write("ranks:\t {} \n ".format(str(ranks)))
            ##file.write("quartiles:\t {} \n".format(str(q)))
            #file.write("reciprocal rank:\t {} \n".format(str(rr)))
            #file.write("preds_sim: \t {} \n".format(str(preds_sims)))

    def save_fine_metrics(self):
        pass

    def save_ranks(self, ranks, target):
        #print("r", ranks)
        #print("t", target)
        ranks_words = [self.idx2lab[idx] for idx in ranks]
        return ranks_words.append(self.idx2lab[target])


    # @property
    # def vocabulary(self):
    #     return self._vocabulary
    # @property
    # def extractor(self):
    #     return self._extractor

    @property
    def vocabulary_matrix(self):
        return self._vocabulary_matrix

    @property
    def target_words(self):
        return self._target_words

    @property
    def lab2idx(self):
        return self._lab2idx

    @property
    def idx2lab(self):
        return self._idx2lab

    # @property
    # def target_embeddings(self):
    #     return self._target_embeddings

    @property
    def reciprank(self):
        return self._reciprank
    @property
    def quartiles(self):
        return self._quartiles
    @property
    def vocabulary_matrix(self):
        return self._vocabulary_matrix

    @property
    def predicted_embeddings(self):
         return self._predicted_embeddings

    @property
    def path_results(self):
        return self._path_results

    @property
    def preds_sims(self):
        return self._preds_sims

    @property
    def ranks(self):
        return self._ranks