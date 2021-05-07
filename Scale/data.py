import numpy as np
import tensorflow as tf
import scipy.sparse
import utils
import pandas as pd

"""
This module contains class and methods related to data used in DropoutNet  
"""


def load_eval_data(test_data, name, train_data, warm_test):
    timer = utils.timer()
    if warm_test is not None:
        test_data = np.concatenate([test_data, warm_test], axis=0)
    timer.toc('read %s triplets %s' % (name, test_data.shape)).tic()
    eval_data = EvalData(
        test_data,
        train=train_data
    )
    timer.toc('loaded %s' % name).tic()
    print(eval_data.get_stats_string())
    return eval_data


class EvalData:

    def __init__(self, test_triplets, train):
        # build map both-ways between compact and original indices
        # compact indices only contains:
        #  1) items in test set
        #  2) users who interacted with such test items

        # item ids in test set
        self.test_item_ids = np.unique(test_triplets['iid'])

        # item global id to test local id mapping, and local id to global id mapping
        self.test_item_ids_map = {iid: i for i, iid in enumerate(self.test_item_ids)}
        self.test_item_new2old_list = np.zeros(len(self.test_item_ids_map)).astype(int)
        for old in self.test_item_ids_map:
            self.test_item_new2old_list[self.test_item_ids_map[old]] = old

        # user ids in test set
        self.test_user_ids = np.unique(test_triplets['uid'])

        # user global id to test local id mapping
        self.test_user_ids_map = {user_id: i for i, user_id in enumerate(self.test_user_ids)}
        self.test_user_new2old_list = np.zeros(len(self.test_user_ids_map)).astype(int)
        for old in self.test_user_ids_map:
            self.test_user_new2old_list[self.test_user_ids_map[old]] = old

        # generate a sparse user-item matrix by local ids for the test set
        _test_i_for_inf = [self.test_user_ids_map[_t[0]] for _t in test_triplets]
        _test_j_for_inf = [self.test_item_ids_map[_t[1]] for _t in test_triplets]
        self.R_test_inf = scipy.sparse.coo_matrix(
            (np.ones(len(_test_i_for_inf)),
             (_test_i_for_inf, _test_j_for_inf)),
            shape=[len(self.test_user_ids), len(self.test_item_ids)]
        ).tolil(copy=False)

        if train is not None:
            train_ij_for_inf = [(self.test_user_ids_map[_t[0]], self.test_item_ids_map[_t[1]]) for _t
                                in train
                                if _t[1] in self.test_item_ids_map and _t[0] in self.test_user_ids_map]

            self.R_train_inf = scipy.sparse.coo_matrix((
                np.ones(len(train_ij_for_inf)),
                zip(*train_ij_for_inf)), shape=self.R_test_inf.shape).tolil(copy=False)
        else:
            self.R_train_inf = None

        # allocate fields
        self.tf_eval_train = None  #
        self.eval_batch = None  # batch windows for evaluation, which is for batches of users

    def init_tf(self, eval_batch_size):

        eval_l = self.R_test_inf.shape[0]  # number of users in the test set

        # generate batch windows for evaluation, which is for batches of users
        self.eval_batch = [(x, min(x + eval_batch_size, eval_l)) for x in range(0, eval_l, eval_batch_size)]

        if self.R_train_inf is not None:
            self.tf_eval_train = []
            for (eval_start, eval_finish) in self.eval_batch:
                _ui = self.R_train_inf[eval_start:eval_finish, :].tocoo()
                _ui = np.concatenate([_ui.row.reshape((-1, 1)), _ui.col.reshape((-1, 1))], axis=1)
                self.tf_eval_train.append(
                    tf.SparseTensorValue(
                        indices=_ui,
                        values=np.full(_ui.shape[0], -100000, dtype=np.float32),
                        dense_shape=[eval_finish - eval_start, self.R_train_inf.shape[1]]
                    )
                )
        else:
            self.tf_eval_train = None

    def get_stats_string(self):
        return ('\tn_test_users:[%d]\n\tn_test_items:[%d]' % (len(self.test_user_ids), len(self.test_item_ids))
                + '\n\tR_train_inf: %s' % (
                    'no R_train_inf for cold' if self.R_train_inf is None else 'shape=%s nnz=[%d]' % (
                        str(self.R_train_inf.shape), len(self.R_train_inf.nonzero()[0])
                    )
                )
                + '\n\tR_test_inf: shape=%s nnz=[%d]' % (
                    str(self.R_test_inf.shape), len(self.R_test_inf.nonzero()[0])
                ))
