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
            self._ranks, self._gold_sims, self._preds_sims, self._relations_predictions = self.get_rank(rel=True)
            print("ranks calculated")
            self._dict_relations = dict()
            self._dict_sims = dict()
            self._list_target_words = []
            for rank, rel, tw, sim1, sim2 in zip(self.ranks, self.relations, self.target_words, self.gold_sims, self.preds_sims):
                if rel not in self._dict_relations:
                    self._dict_relations[rel] = [rank]
                else:
                    self._dict_relations[rel].append(rank)
                if rel not in self._dict_sims:
                    self._dict_sims[rel] = [sim2]
                else:
                    self._dict_sims[rel].append(sim2)
                self._list_target_words.append((rel, tw, rank))

            self._dict_results_per_relation = dict()

            for k, v in self._dict_relations.items():
                prec5 = self.precision_at_rank(5, v)
                prec1 = self.precision_at_rank(1, v)
                prec50 = self.precision_at_rank(50, v)
                prec80 = self.precision_at_rank(80, v)
                sims = self._dict_sims[k]

                #sim1 = sum(sims[0])/len(sims[0])
                sim2 = np.mean(sims)
                self._dict_results_per_relation[k] = (prec1, prec5, prec50, prec80, sim2)

        else:
            self._ranks, gold_sims, self._preds_sims = self.get_rank()
        self._quartiles = self.calculate_quartiles(self.ranks)
        self._reciprank = self.reciprocal_rank(self.ranks)

        # for all ranks it calculates precision at rank 5 and 1
        self._precision_at_rank_5 = self.precision_at_rank(5, self.ranks)
        self._precision_at_rank_1 = self.precision_at_rank(1, self.ranks)
        self._precision_at_rank_50 = self.precision_at_rank(50, self.ranks)
        self._precision_at_rank_80 = self.precision_at_rank(80, self.ranks)

        if relations is not None:
            self._relations = relations
        # self.save_metrics(self.target_words,self.ranks, self.reciprank,self.preds_sims)

    def get_rank(self, rel=False):
        ranks = []
        gold_similarities = []
        prediction_similarities = []
        all_relations = []

        target_ids = []
        for lab in self.target_words:
            if lab in self.lab2idx.keys():
                target_ids.append(self.lab2idx[lab])
            else:
                target_ids.append(self.lab2idx['UNK'])

        # matrix of target representations ordered
        target_repr = np.take(self.vocabulary_matrix, target_ids, axis=0)

        target_similarities = cosine_similarity(self.vocabulary_matrix, target_repr)

        for i in range(self.predicted_embeddings.shape[0]):

            target_prediction_similarity = cosine_similarity(self.predicted_embeddings[i].reshape(1, -1),
                                                             target_repr[i].reshape(1, -1))
            prediction_similarities.append(float(target_prediction_similarity[0]))
            #print("pred sim:", float(target_prediction_similarity[0]))
            gold_similarities.append(target_similarities[:, i])
            # delete similarity ebtween target label and itself
            target_sims = np.delete(target_similarities[:, i], target_ids[i])
            rank = np.count_nonzero(target_sims > target_prediction_similarity) + 1
            if rel:
                all_relations.append(self.relations[i])
            if rank > 100:
                rank = 100
            ranks.append(rank)
        if rel:
            return ranks, gold_similarities, prediction_similarities, all_relations
        else:
            return ranks, gold_similarities, prediction_similarities

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



    @staticmethod
    def reciprocal_rank(ranks):
        """
        get reciprocal rank (1/r)
        :param ranks: list of ranks
        :return: list of RR
        """
        return [float("{:.2f}".format(1 / r)) for r in ranks]

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

    def save_ranks(self, path):
        with open(path, 'w') as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(("relation", "target_word", "rank"))
            for word in self.list_target_words:
                writer.writerow(word)

    def save_fine_metrics(self):
        pass

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
    def ranks(self):
        return self._ranks

    @property
    def precision_at_rank_1(self):
        return self._precision_at_rank_1

    @property
    def precision_at_rank_5(self):
        return self._precision_at_rank_5

    @property
    def precision_at_rank_80(self):
        return self._precision_at_rank_80

    @property
    def precision_at_rank_50(self):
        return self._precision_at_rank_50

    @property
    def prediction_similarities(self):
        return self._preds_sims,

    @property
    def relations_predictions(self):
        return self._relations_predictions

    @property
    def dict_results_per_relation(self):
        return self._dict_results_per_relation

    @property
    def dict_relations(self):
        return self._dict_relations

    @property
    def list_target_words(self):
        return self._list_target_words

    @property
    def preds_sims(self):
        return self._preds_sims
    @property
    def gold_sims(self):
        return self._gold_sims