import numpy as np
import math
import csv
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.metrics import f1_score, accuracy_score
class Ranker:
    def __init__(self, path_predictions, target_words, relations, vocabulary_matrix, lab2idx, idx2lab):
        self._vocabulary_matrix = vocabulary_matrix
        self._lab2idx = lab2idx
        self._idx2lab = idx2lab

        self._predicted_embeddings = np.load(path_predictions, allow_pickle=True)

        self._target_words = target_words

        if relations is not None:
            self._relations = relations
            self._ranks, gold_sims, self._preds_sims, self._predicted_words, self._relations_predictions = self.get_rank()
            dict_relations = dict()
            for rank, rel in zip(self.relations_predictions):
                if rel not in dict_relations:
                    dict_relations[rel] = [rank]
                else:
                    dict_relations[rel].append(rank)
            self._dict_results_per_relation = dict()
            for k,v in dict_relations.items():
                prec5 = self.precision_at_rank(5, v)
                prec1 = self.precision_at_rank(1, v)
                self._dict_results_per_relation[k] = (prec5, prec1)

        else:
            self._ranks, gold_sims, self._preds_sims, self._predicted_words = self.get_rank()
        self._quartiles = self.calculate_quartiles(self.ranks)
        self._reciprank = self.reciprocal_rank(self.ranks)


        #for all ranks it calculates precision at rank 5 and 1
        self._precision_at_rank_5 = self.precision_at_rank(5, self.ranks)
        self._precision_at_rank_1 = self.precision_at_rank(1, self.ranks)

        if relations is not None:
            self._relations = relations
        #self.save_metrics(self.target_words,self.ranks, self.reciprank,self.preds_sims)


    def get_rank(self):
        ranks = []
        gold_similarities = []
        prediction_similarities = []
        predicted_words = []
        all_relations = []
        #predicted_attribute = self.index2label[np.argmax(composed_attributes_similarity)]
        #predicted_labels.append(predicted_attribute)
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

            predicted_words.append(self.idx2lab[np.argmax(target_prediction_similarity)])
            # delete similarity ebtween target label and itself
            target_sims = np.delete(target_similarities[:, i], target_ids[i])
            print("target sims", len(target_sims), target_sims)
            #higher_ranks1 = np.nonzero(target_sims > target_prediction_similarity[0])    #nonzero returns tuple, therefore pick first element
            rank = np.count_nonzero(target_sims > target_prediction_similarity) +1
            #print("nonzero instead of counts", higher_ranks1)
            print("items at higher ranks than target", rank)
            #self.save_ranks(higher_ranks.tolist(), target_ids[i])
            #ranks.append(len(higher_ranks) + 1)
            all_relations.append(self.relations[i])
            if rank > 100:
                rank = 100
            ranks.append(rank)
        if self.relations is not None:
            return ranks, gold_similarities, prediction_similarities, predicted_words, all_relations
        else:
            return ranks, gold_similarities, prediction_similarities, predicted_words
    @staticmethod
    def precision_at_rank(k, ranks):
        """
            Computes the number of times a rank is equal or lower to a given rank.
            :param k: the rank for which the precision is computed
            :param ranks: a list of ranks
            :return: the precision at a certain rank (float)
        """
        assert k >= 1
        correct = len([rank for rank in ranks if rank <= k])
        return correct / len(ranks)

    def performance_metrics(self):
        """
        calculates weighted f1 and accuracy
        :return: accuracy and f1
        """
        f1 = f1_score(y_true=self.target_words, y_pred=self.predicted_words, average="weighted")
        acc = accuracy_score(y_true=self.target_words, y_pred=self.predicted_words)
        return acc, f1

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

    def save_metrics_per_relation(self, path):
        with open(path, 'w') as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(("target_word", "rank", "recip_rank", "similarity"))
            for inst in zip(self.target_words, self.ranks, self.reciprank, self.prediction_similarities):
                writer.writerow(inst)


    def save_fine_metrics(self):
        pass

    def save_ranks(self, ranks, target):
        #print("r", ranks)
        #print("t", target)
        ranks_words = [self.idx2lab[idx] for idx in ranks]
        return ranks_words.append(self.idx2lab[target])

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


    @property
    def relations(self):
        return self._relations
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
    def preds_sims(self):
        return self._preds_sims

    @property
    def ranks(self):
        return self._ranks
    @property
    def precision_at_rank_1(self):
        return self._precision_at_rank_1
    @property
    def precision_at_rank_5(self):
        return self._precision_at_rank_5

    @property
    def prediction_similarities(self):
        return self.prediction_similarities

    @property
    def predicted_words(self):
        return self._predicted_words

    @property
    def relations_predictions(self):
         return self._relations_predictions

    @property
    def dict_results_per_relation(self):
         return self._dict_results_per_relation