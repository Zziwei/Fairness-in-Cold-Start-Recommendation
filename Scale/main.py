import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn import datasets
import data
import model
import scipy.sparse

import argparse
from tqdm import tqdm
import os
import pickle
import copy


np.random.seed(0)
tf.set_random_seed(0)


def main():

    data_path = args.data
    alg = args.alg

    model_select = args.model_select

    recall_at = [15, 30, 200]
    eval_batch_size = args.eval_batch_size  # the batch size for a user batch during evaluation
    eval_every = args.eval_every
    num_epoch = args.epoch
    batch_size = args.bs

    _lr = args.lr
    _decay_lr_every = 2
    _lr_decay = 0.9

    dat = load_data(data_path, alg)

    num_user = dat['num_user']
    num_item = dat['num_item']

    item_warm = dat['item_warm']

    bias_evaluator = dat['BiasEvaluator']

    cold_test_eval = dat['cold_test_eval']
    cold_vali_eval = dat['cold_vali_eval']

    R = dat['R'].T
    mask = dat['R_train'].toarray()

    R = R - np.min(R)
    R_output = R

    ''' do the score rescale'''
    pos_sum = np.sum(mask, axis=1, keepdims=True)
    pos_mean = np.zeros_like(pos_sum)
    pos_std = np.zeros_like(pos_sum)
    pos_mean[item_warm, :] = np.sum(R_output[item_warm, :] * mask[item_warm, :], axis=1, keepdims=True) / pos_sum[item_warm, :]
    pos_std[item_warm, :] = np.sum((R_output[item_warm, :] - pos_mean[item_warm, :]) ** 2 * mask[item_warm, :], axis=1, keepdims=True) / pos_sum[item_warm, :]
    pos_mean = pos_mean ** args.alpha
    weights = np.zeros_like(pos_sum)
    weights[item_warm, :] = 0 + np.max(pos_mean[item_warm, :]) / pos_mean[item_warm, :]
    R_output = R_output * weights * mask + (1 - mask) * R_output

    timer = utils.timer(name='main').tic()

    # prep eval
    timer.tic()
    cold_test_eval.init_tf(eval_batch_size)
    cold_vali_eval.init_tf(eval_batch_size)
    timer.toc('initialized eval for tf').tic()

    dropout_net = model.Model(model_select=model_select, num_user=num_user, num_item=num_item, reg=args.reg)

    config = tf.ConfigProto(allow_soft_placement=True)

    dropout_net.build_model()
    dropout_net.build_predictor(recall_at)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        timer.toc('initialized tf')

        row_index = item_warm

        for epoch in range(num_epoch):

            '''train recommendation'''
            np.random.shuffle(row_index)
            n_targets = len(row_index)
            data_batch = [(n, min(n + batch_size, n_targets)) for n in range(0, n_targets, batch_size)]
            loss_all_epoch = 0.
            loss_r_epoch = 0.
            loss_reg_epoch = 0.
            for (start, stop) in tqdm(data_batch):

                item_batch_idx = row_index[start:stop]
                _, loss_out, loss_r_out, loss_reg_out = sess.run(
                    [dropout_net.all_optimizer, dropout_net.loss_all, dropout_net.loss_r, dropout_net.reg_loss],
                    feed_dict={
                        dropout_net.R_input: R[item_batch_idx, :],
                        dropout_net.R_output: R_output[item_batch_idx, :],
                        dropout_net.lr_placeholder: _lr,
                        dropout_net.is_training: True
                })
                loss_all_epoch += loss_out
                loss_r_epoch += loss_r_out
                loss_reg_epoch += loss_reg_out
                if np.isnan(loss_out):
                    print('epoch=%d all_loss=%.4f r_loss=%.4f reg_loss=%.4f' % (
                        epoch, loss_all_epoch, loss_r_epoch, loss_reg_epoch))
                    raise Exception('f is nan')

            if epoch % _decay_lr_every == 0:
                _lr = _lr_decay * _lr
                print('decayed lr:' + str(_lr))

            if (epoch + 1) % eval_every == 0:
                cold_test_recall, cold_test_precision, \
                cold_test_ndcg, rank_matrix = utils.batch_eval_recall(sess, dropout_net.eval_preds_cold,
                                                                      eval_feed_dict=dropout_net.get_eval_dict,
                                                                      recall_k=recall_at, eval_data=cold_test_eval, R=R)
                bias_evaluator.bias_analysis(rank_matrix)

                item_new2old_list = np.array(cold_test_eval.test_item_new2old_list)
                user_new2old_list = np.array(cold_test_eval.test_user_new2old_list)
                np.save('../Data/' + data_path + '/item_new2old_list_Debias_scale_' + str(args.alg) + '.npy', item_new2old_list)
                np.save('../Data/' + data_path + '/user_new2old_list_Debias_scale_' + str(args.alg) + '.npy', user_new2old_list)
                np.save('../Data/' + data_path + '/rank_matrix_Debias_scale_' + str(args.alg) + '.npy', rank_matrix)

                timer.toc('epoch=%d all_loss=%.4f r_loss=%.4f reg_loss=%.4f' % (
                    epoch, loss_all_epoch, loss_r_epoch, loss_reg_epoch)).tic()
                print('\t\t\t' + '\t '.join([('@' + str(i)).ljust(6) for i in recall_at]))

                print('Curr test recall   \t%s' % (
                    ' '.join(['%.6f' % i for i in cold_test_recall]),
                ))
                print('Curr test precision\t%s' % (
                    ' '.join(['%.6f' % i for i in cold_test_precision]),
                ))
                print('Curr test ndcg     \t%s' % (
                    ' '.join(['%.6f' % i for i in cold_test_ndcg]),
                ))

                print('!' * 150)


def load_data(data_name, alg):
    timer = utils.timer(name='main').tic()
    data_path = '../Data/' + data_name

    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        n_users = info['num_user']
        n_items = info['num_item']

    train_file = data_path + '/train_df.csv'
    cold_test_file = data_path + '/cold_test_df.csv'
    cold_vali_file = data_path + '/cold_vali_df.csv'

    dat = {}
    dat['num_item'] = n_items
    dat['num_user'] = n_users

    # load preference data
    timer.tic()

    if alg == 'KNN':
        dat['R'] = np.load(data_path + '/KNN_R.npy')
    else:
        u_pref = np.load(data_path + '/U_emb_' + alg + '.npy')
        v_pref = np.load(data_path + '/I_emb_' + alg + '.npy')
        R = np.matmul(u_pref, v_pref.T)
        dat['R'] = R

    # load train, vali, test
    timer.tic()
    train_df = pd.read_csv(train_file, dtype=np.int32)
    dat['item_warm'] = train_df['iid'].unique()

    item_pop = np.array(train_df['iid'].value_counts())
    item_pop_id = np.array(train_df['iid'].value_counts().index)
    item_pop_list = np.zeros(n_items)
    for i in range(len(item_pop_id)):
        item_pop_list[item_pop_id[i]] = item_pop[i]

    dat['item_warm_pop'] = item_pop
    dat['item_pop_list'] = item_pop_list

    train = train_df.values.ravel().view(dtype=[('uid', np.int32), ('iid', np.int32)])

    timer.toc('read train triplets %s' % train.shape).tic()

    cold_test = pd.read_csv(cold_test_file, dtype=np.int32).values.ravel().view(dtype=[('uid', np.int32),
                                                                                       ('iid', np.int32)])
    cold_vali = pd.read_csv(cold_vali_file, dtype=np.int32).values.ravel().view(dtype=[('uid', np.int32),
                                                                                       ('iid', np.int32)])

    dat['cold_vali_eval'] = data.load_eval_data(cold_vali, name='cold_vali_eval', train_data=None, warm_test=None)
    dat['cold_test_eval'] = data.load_eval_data(cold_test, name='cold_test_eval', train_data=None, warm_test=None)

    dat['BiasEvaluator'] = utils.BiasEvaluator(data_name, dat['cold_test_eval'], np.unique(cold_test['iid']))

    dat['R_train'] = scipy.sparse.coo_matrix((np.ones(len(train_df)),
                                              (train_df['iid'].values,
                                              train_df['uid'].values)),
                                             shape=(n_items, n_users)).tolil(copy=False)

    return dat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale")
    parser.add_argument('--data', type=str, default='ml1m', help='path to eval in the downloaded folder')
    parser.add_argument('--alg', type=str, default='Heater', help='algorithm')

    parser.add_argument('--model-select', nargs='+', type=int,
                        default=[100],
                        help='specify the fully-connected architecture, starting from input,'
                             ' numbers indicate numbers of hidden units',
                        )
    parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
    parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--bs', type=int, default=50, help='data batch size')
    parser.add_argument('--eval_batch_size', type=int, default=40000, help='eval_batch_size')

    parser.add_argument('--reg', type=float, default=0.00001, help='reg')
    parser.add_argument('--alpha', type=float, default=4, help='alpha')

    args = parser.parse_args()
    for key in vars(args):
        print(key + ":" + str(vars(args)[key]))
    main()
