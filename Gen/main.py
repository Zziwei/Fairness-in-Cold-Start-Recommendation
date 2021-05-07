import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import data
import model
import scipy.sparse

import argparse
from tqdm import tqdm
import os
import pickle
import copy
import time


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
    # R = R - np.min(R)

    R_output = copy.copy(R)
    if args.alg == 'NLinMap' or args.alg == 'KNN':
        R_output = (R - np.mean(R, axis=0, keepdims=True)) / (np.std(R, axis=0, keepdims=True) + 1e-9)

    item_like_users = dat['item_like_users']

    timer = utils.timer(name='main').tic()

    # prep eval
    timer.tic()
    cold_test_eval.init_tf(eval_batch_size)
    cold_vali_eval.init_tf(eval_batch_size)
    timer.toc('initialized eval for tf').tic()

    debias_model = model.Model(model_select=model_select, num_user=num_user, num_item=num_item,
                                adv_layers=args.adv_layers, alpha=args.alpha, reg=args.reg,
                                sample_num=args.max_pos_num, max_pos_num=args.max_pos_num)

    config = tf.ConfigProto(allow_soft_placement=True)

    debias_model.build_model()
    debias_model.build_predictor(recall_at)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        timer.toc('initialized tf')

        row_index = item_warm

        for epoch in range(num_epoch):

            rec_epoch = 20
            bias_epoch = 5

            if epoch == rec_epoch:
                _lr = args.lr / 2

            '''train Gen'''
            if epoch >= rec_epoch and (epoch - rec_epoch) % bias_epoch == 0:
                if epoch == rec_epoch:
                    tmp = 10
                else:
                    tmp = bias_epoch
                for i in range(tmp):
                    ''' generate new R_fetch for this epoch '''
                    start_time = time.time()
                    item_fetch_users = np.zeros((num_item, args.max_pos_num))
                    for i in item_like_users:
                        random_pos = np.random.choice(item_like_users[i], args.max_pos_num, replace=True)
                        item_fetch_users[i, :] = random_pos
                    print('-' * 20 + ' generate fetch idx for this epoch, used ' + str(
                        time.time() * 1000 - start_time * 1000) + 'ms ' + '-' * 20)

                    a_row_index = copy.copy(row_index)
                    np.random.shuffle(a_row_index)
                    a_n_targets = len(a_row_index)
                    a_data_batch = [(n, min(n + batch_size, a_n_targets)) for n in range(0, a_n_targets, batch_size)]
                    a_loss_a_epoch = 0.
                    for (a_start, a_stop) in tqdm(a_data_batch):
                        a_item_batch_idx = a_row_index[a_start:a_stop]
                        a_fetch_users_batch = item_fetch_users[a_item_batch_idx, :]
                        a_batch_num = a_stop - a_start
                        a_fetch_idx_batch = np.concatenate([np.repeat(np.arange(a_batch_num), args.max_pos_num).reshape((-1, 1)), a_fetch_users_batch.reshape((-1, 1))], axis=1)
                        a_fetch_idx_batch = a_fetch_idx_batch.astype(int)

                        _, loss_out = sess.run(
                            [debias_model.a_optimizer, debias_model.loss_a],
                            feed_dict={
                                debias_model.R_input: R[a_item_batch_idx, :],
                                debias_model.R_output: R_output[a_item_batch_idx, :],
                                debias_model.seed_input: np.random.standard_normal(args.max_pos_num).reshape((-1, 1)),
                                debias_model.fetch_idx_input: a_fetch_idx_batch,
                                debias_model.lr_placeholder: _lr,
                                debias_model.is_training: True
                            })

                        a_loss_a_epoch += loss_out
                        if np.isnan(loss_out):
                            raise Exception('f is nan')
                    timer.toc('a_loss=%.4f' % a_loss_a_epoch).tic()

            '''train recommendation'''
            ''' generate new R_fetch for this epoch '''
            start_time = time.time()
            item_fetch_users = np.zeros((num_item, args.max_pos_num))
            for i in item_like_users:
                random_pos = np.random.choice(item_like_users[i], args.max_pos_num, replace=True)
                item_fetch_users[i, :] = random_pos
            print('-' * 20 + ' generate fetch idx for this epoch, used ' + str(time.time() * 1000 - start_time * 1000) + 'ms ' + '-' * 20)

            np.random.shuffle(row_index)
            n_targets = len(row_index)
            data_batch = [(n, min(n + batch_size, n_targets)) for n in range(0, n_targets, batch_size)]
            loss_all_epoch = 0.
            loss_r_epoch = 0.
            loss_a_epoch = 0.
            loss_reg_epoch = 0.
            for (start, stop) in tqdm(data_batch):
                item_batch_idx = row_index[start:stop]
                fetch_users_batch = item_fetch_users[item_batch_idx, :]
                batch_num = stop - start
                fetch_idx_batch = np.concatenate([np.repeat(np.arange(batch_num), args.max_pos_num).reshape((-1, 1)), fetch_users_batch.reshape((-1, 1))], axis=1)
                fetch_idx_batch = fetch_idx_batch.astype(int)

                if epoch >= rec_epoch:
                    _, loss_out, loss_r_out, loss_a_out, loss_reg_out = sess.run(
                        [debias_model.all_optimizer, debias_model.loss_all, debias_model.loss_r, debias_model.loss_ra, debias_model.reg_loss],
                        feed_dict={
                            debias_model.R_input: R[item_batch_idx, :],
                            debias_model.R_output: R_output[item_batch_idx, :],
                            debias_model.seed_input: np.random.standard_normal(args.max_pos_num).reshape((-1, 1)),
                            debias_model.fetch_idx_input: fetch_idx_batch,
                            debias_model.lr_placeholder: _lr,
                            debias_model.is_training: True})
                else:
                    _, loss_out, loss_r_out, loss_a_out, loss_reg_out = sess.run(
                        [debias_model.r_optimizer, debias_model.loss_all, debias_model.loss_r, debias_model.loss_ra, debias_model.reg_loss],
                        feed_dict={
                            debias_model.R_input: R[item_batch_idx, :],
                            debias_model.R_output: R_output[item_batch_idx, :],
                            debias_model.seed_input: np.random.standard_normal(args.max_pos_num).reshape((-1, 1)),
                            debias_model.fetch_idx_input: fetch_idx_batch,
                            debias_model.lr_placeholder: _lr,
                            debias_model.is_training: True})

                loss_all_epoch += loss_out
                loss_r_epoch += loss_r_out
                loss_a_epoch += loss_a_out
                loss_reg_epoch += loss_reg_out
                if np.isnan(loss_out):
                    raise Exception('recommendation loss is nan')

            if epoch % _decay_lr_every == 0 and _lr > 0.0005:
                _lr = _lr_decay * _lr
                print('decayed lr:' + str(_lr))

            if (epoch < rec_epoch and (epoch + 1) % 2 == 0) or (epoch >= rec_epoch and (epoch + 1) % eval_every == 0):
                cold_test_recall, cold_test_precision, \
                cold_test_ndcg, rank_matrix = utils.batch_eval_recall(sess, debias_model.eval_preds_cold,
                                                                      eval_feed_dict=debias_model.get_eval_dict,
                                                                      recall_k=recall_at, eval_data=cold_test_eval, R=R)
                bias_evaluator.bias_analysis(rank_matrix)

                item_new2old_list = np.array(cold_test_eval.test_item_new2old_list)
                user_new2old_list = np.array(cold_test_eval.test_user_new2old_list)
                np.save('../Data/' + data_path + '/item_new2old_list_Debias_gen_' + str(args.alg) + '.npy', item_new2old_list)
                np.save('../Data/' + data_path + '/user_new2old_list_Debias_gen_' + str(args.alg) + '.npy', user_new2old_list)
                np.save('../Data/' + data_path + '/rank_matrix_Debias_gen_' + str(args.alg) + '.npy', rank_matrix)

                timer.toc('epoch=%d all_loss=%.4f r_loss=%.4f reg_loss=%.4f a_loss=%.4f' % (
                    epoch, loss_all_epoch, loss_r_epoch, loss_reg_epoch, loss_a_epoch)).tic()
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

    dat = {'num_item': n_items, 'num_user': n_users}

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
    item_warm = train_df['iid'].unique()
    dat['item_warm'] = item_warm

    timer.toc('read train triplets %s' % len(train_df)).tic()

    cold_test = pd.read_csv(cold_test_file, dtype=np.int32).values.ravel().view(dtype=[('uid', np.int32),
                                                                                       ('iid', np.int32)])
    cold_vali = pd.read_csv(cold_vali_file, dtype=np.int32).values.ravel().view(dtype=[('uid', np.int32),
                                                                                       ('iid', np.int32)])

    dat['cold_vali_eval'] = data.load_eval_data(cold_vali, name='cold_vali_eval', train_data=None, warm_test=None)
    dat['cold_test_eval'] = data.load_eval_data(cold_test, name='cold_test_eval', train_data=None, warm_test=None)

    dat['BiasEvaluator'] = utils.BiasEvaluator(data_name, dat['cold_test_eval'], np.unique(cold_test['iid']))


    item_like_users = dict()

    for i in item_warm:
        cur_like_user = dat['R_train'][i, :].tocoo().col
        item_like_users[i] = cur_like_user
    dat['item_like_users'] = item_like_users
    return dat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gen")
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

    parser.add_argument('--adv_layers', nargs='+', type=int, default=[50, 50], help='adversarial layers')
    parser.add_argument('--alpha', type=float, default=200, help='alpha')

    parser.add_argument('--reg', type=float, default=0.00001, help='reg')

    parser.add_argument('--max_pos_num', type=int, default=1000, help='max_pos_num')

    args = parser.parse_args()
    for key in vars(args):
        print(key + ":" + str(vars(args)[key]))
    main()
