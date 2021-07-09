import time
import datetime
import numpy as np
import scipy
import tensorflow as tf
from sklearn import preprocessing as prep
import pandas as pd
from tqdm import tqdm
import copy
from scipy import stats


class timer(object):
    def __init__(self, name='default'):
        """
        timer object to record running time of functions, not for micro-benchmarking
        usage is:
            $ timer = utils.timer('name').tic()
            $ timer.toc('process A').tic()

        :param name: label for the timer
        """
        self._start_time = None
        self._name = name
        self.tic()

    def tic(self):
        self._start_time = time.time()
        return self

    def toc(self, message):
        elapsed = time.time() - self._start_time
        message = '' if message is None else message
        print('[{0:s}] {1:s} elapsed [{2:s}]'.format(self._name, message, timer._format(elapsed)))
        return self

    def reset(self):
        self._start_time = None
        return self

    @staticmethod
    def _format(s):
        delta = datetime.timedelta(seconds=s)
        d = datetime.datetime(1, 1, 1) + delta
        s = ''
        if (d.day - 1) > 0:
            s = s + '{:d} days'.format(d.day - 1)
        if d.hour > 0:
            s = s + '{:d} hr'.format(d.hour)
        if d.minute > 0:
            s = s + '{:d} min'.format(d.minute)
        s = s + '{:d} s'.format(d.second)
        return s


def batch_eval_recall(_sess, tf_eval, eval_feed_dict, recall_k, eval_data, R):
    """
    given EvalData and DropoutNet compute graph in TensorFlow, runs batch evaluation

    :param _sess: tf session
    :param tf_eval: the evaluate output symbol in tf
    :param eval_feed_dict: method to parse tf, pick from EvalData method
    :param recall_k: list of thresholds to compute recall at (information retrieval recall)
    :param eval_data: EvalData instance
    :return: recall array at thresholds matching recall_k
    """

    idcg_array = np.arange(recall_k[-1]) + 1
    idcg_array = 1 / np.log2(idcg_array + 1)
    idcg_table = np.zeros(recall_k[-1])
    for i in range(recall_k[-1]):
        idcg_table[i] = np.sum(idcg_array[:(i + 1)])

    tf_eval_preds_batch = []
    for (batch, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        # input users and items, return the top item ids for each user
        tf_eval_preds = _sess.run(tf_eval,
                                  feed_dict=eval_feed_dict(
                                      batch, eval_start, eval_stop, R, eval_data))
        tf_eval_preds_batch.append(tf_eval_preds)
    # tf_eval_preds = _sess.run(tf_eval, feed_dict=eval_feed_dict(R, eval_data))
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)  # ranked item ids for all test users
    tf.local_variables_initializer().run()

    preds_all = tf_eval_preds
    recall = []
    precision = []
    ndcg = []
    for at_k in recall_k:
        preds_k = preds_all[:, :at_k]
        at_k = preds_k.shape[1]
        y = eval_data.R_test_inf

        x = scipy.sparse.coo_matrix((np.ones_like(preds_k).reshape(-1), 
                                     (np.repeat(np.arange(y.shape[0]), at_k), preds_k.reshape(-1))), shape=y.shape)

        z = y.multiply(x)

        recall_users = np.divide((np.sum(z, 1)), np.sum(y, 1))
        recall.append(np.mean(recall_users))

        precision_users = np.sum(z, 1) / at_k
        precision.append(np.mean(precision_users))

        rows = x.row
        cols = x.col
        y_csr = y.tocsr()
        dcg_array = y_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
        dcg = np.sum(dcg_array * idcg_array[:at_k].reshape((1, -1)), axis=1)
        idcg = np.sum(y, axis=1) - 1
        idcg[np.where(idcg >= at_k)] = at_k-1
        idcg = idcg_table[idcg.astype(int)]

        ndcg_users = dcg.reshape([-1]) / idcg.reshape([-1])
        ndcg.append(np.mean(ndcg_users))

    return recall, precision, ndcg, preds_all


def negative_sampling(pos_user_array, pos_item_array, neg, item_warm):
    user_pos = pos_user_array.reshape((-1))
    user_neg = np.tile(pos_user_array, neg).reshape((-1))
    pos = pos_item_array.reshape((-1))
    neg = np.random.choice(item_warm, size=(neg * pos_user_array.shape[0]), replace=True).reshape((-1))
    target_pos = np.ones_like(pos)
    target_neg = np.zeros_like(neg)
    return np.concatenate((user_pos, user_neg)), np.concatenate((pos, neg)), \
           np.concatenate((target_pos, target_neg))


class BiasEvaluator(object):
    def __init__(self, data_name, test_eval, old_cold_idx):
        user_cold_test_like = list(np.load('../Data/' + data_name + '/user_cold_test_like.npy', allow_pickle=True))
        item_AS_list_all = np.load('../Data/' + data_name + '/item_audience_size_list.npy')
        item_old2new_id_dict = test_eval.test_item_ids_map
        user_old2new_id_dict = test_eval.test_user_ids_map

        self.num_user = len(user_old2new_id_dict)
        self.num_item = len(item_old2new_id_dict)

        # filter the item_AS_list_all with items involed in test phase
        item_AS_list = np.zeros(len(item_old2new_id_dict)).astype(np.float32)
        for i in range(len(item_AS_list_all)):
            if i in item_old2new_id_dict:
                item_AS_list[item_old2new_id_dict[i]] = item_AS_list_all[i]

        # get cold_idx with new id
        itemIdsNew = copy.copy(old_cold_idx)
        for i in range(len(old_cold_idx)):
            itemIdsNew[i] = item_old2new_id_dict[old_cold_idx[i]]
        self.cold_idx = itemIdsNew

        # get user_like_list for cold test
        self.cold_test_like = [[] for _ in range(len(user_old2new_id_dict))]
        for old_uid in range(len(user_cold_test_like)):
            if old_uid in user_old2new_id_dict:
                old_test_like = user_cold_test_like[old_uid]
                test_like = []
                for old_iid in old_test_like:
                    if old_iid in item_old2new_id_dict:
                        test_like.append(item_old2new_id_dict[old_iid])
                self.cold_test_like[user_old2new_id_dict[old_uid]] = np.array(test_like).astype(int)

        self.item_cold_pop = item_AS_list[self.cold_idx]

    def bias_analysis(self, rank_matrix, k=100):
        item_attention_count = np.zeros(self.num_item)
        item_count = np.zeros(self.num_item)
        for u in (range(self.num_user)):
            u_rank_list = rank_matrix[u]
            u_cold_like_set = set(self.cold_test_like[u])

            match_item_set = set([])
            for rank, iid in enumerate(u_rank_list):
                if rank == k:
                    break
                if iid in u_cold_like_set:
                    item_attention_count[iid] += (1. / np.log2(rank + 2))
                    item_count[iid] += 1.
                    match_item_set.add(iid)

            unmatch_item_set = u_cold_like_set - match_item_set
            for iid in unmatch_item_set:
                # item_attention_count[iid] += (1. / np.log2(k + 2))
                item_attention_count[iid] += 0.
                item_count[iid] += 1.

        item_avg_attention = item_attention_count / (item_count + 1e-7)
        item_cold_avg_attention = item_avg_attention[self.cold_idx]

        minority_10_n = int(len(item_cold_avg_attention) * 0.1)
        minority_20_n = int(len(item_cold_avg_attention) * 0.2)
        minority_30_n = int(len(item_cold_avg_attention) * 0.3)
        minority_50_n = int(len(item_cold_avg_attention) * 0.5)
        minority_10_idx = np.argpartition(item_cold_avg_attention, minority_10_n)[:minority_10_n]
        minority_20_idx = np.argpartition(item_cold_avg_attention, minority_20_n)[:minority_20_n]
        minority_30_idx = np.argpartition(item_cold_avg_attention, minority_30_n)[:minority_30_n]
        minority_50_idx = np.argpartition(item_cold_avg_attention, minority_50_n)[:minority_50_n]

        majority_10_idx = np.argpartition(item_cold_avg_attention, -minority_10_n)[-minority_10_n:]

        print('$' * 60)
        print('$$ PCC att for cold = ' + str(stats.pearsonr(item_cold_avg_attention + 1e-7, self.item_cold_pop + 1e-7)))
        print('$$ avg att for cold = ' + str(np.mean(item_cold_avg_attention)))
        print('$$ Minority 10% attention = ' + str(np.mean(item_cold_avg_attention[minority_10_idx])))
        print('$$ Minority 20% attention = ' + str(np.mean(item_cold_avg_attention[minority_20_idx])))
        print('$$ Minority 30% attention = ' + str(np.mean(item_cold_avg_attention[minority_30_idx])))
        print('$$ Minority 50% attention = ' + str(np.mean(item_cold_avg_attention[minority_50_idx])))
        print('$' * 60)
        print('$$ Majority 10% attention = ' + str(np.mean(item_cold_avg_attention[majority_10_idx])))
        print('$' * 60)
