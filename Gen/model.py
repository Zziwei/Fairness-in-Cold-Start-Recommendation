import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.set_random_seed(0)


def l2_norm(para):
    return tf.reduce_sum(tf.square(para))


def dense_batch_fc(x, units, is_training, scope, do_norm=False):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init)
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        if do_norm:
            h2 = tf.contrib.layers.batch_norm(
                h1,
                decay=0.9,
                center=True,
                scale=True,
                is_training=is_training,
                scope=scope + '_bn')
            return h2, l2_norm(h1_w) + l2_norm(h1_b)
        else:
            return h1, l2_norm(h1_w) + l2_norm(h1_b)


def dense_batch_fc_tanh(x, units, is_training, scope, do_norm=False):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init)
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        if do_norm:
            h2 = tf.contrib.layers.batch_norm(
                h1,
                decay=0.9,
                center=True,
                scale=True,
                is_training=is_training,
                scope=scope + '_bn')
            return tf.nn.relu(h2, scope + '_tanh'), l2_norm(h1_w) + l2_norm(h1_b)
        else:
            return tf.nn.relu(h1, scope + '_tanh'), l2_norm(h1_w) + l2_norm(h1_b)


class Model:

    def __init__(self, model_select, num_user,  num_item, adv_layers, alpha, reg, sample_num, max_pos_num):
        self.reg = reg

        self.num_user = num_user
        self.num_item = num_item

        self.model_select = model_select

        self.adv_layers = adv_layers
        self.alpha = alpha
        self.sample_num = sample_num
        self.max_pos_num = max_pos_num

        # inputs
        self.is_training = None
        self.R_input = None
        self.eval_trainR = None


        # outputs in the model
        self.preds = None  # predicted scores for given user-item pairs

        self.lr_placeholder = None

        # predictor
        self.eval_preds_cold = None

    def build_model(self):
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[], name='learn_rate')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.R_input = tf.placeholder(tf.float32, shape=[None, self.num_user], name='R_input')
        self.R_output = tf.placeholder(tf.float32, shape=[None, self.num_user], name='R_output')

        self.seed_input = tf.placeholder(tf.float32, shape=[self.sample_num, 1], name='seed_input')

        self.fetch_idx_input = tf.placeholder(tf.int32, shape=[None, 2], name='fetch_idx_input')

        """ 
        the main component of the debiasing NN, which is an AutoEncoder 
        """
        self.reg_loss = 0
        with tf.variable_scope("AutoEncoder", reuse=tf.AUTO_REUSE):
            last = self.R_input
            for ihid, hid in enumerate(self.model_select):
                last, reg = dense_batch_fc(last, hid, self.is_training, 'layer_%d' % (ihid + 1), do_norm=True)
                self.reg_loss += reg

            W = tf.Variable(tf.truncated_normal([last.get_shape().as_list()[1], self.num_user], stddev=0.01), name='W')
            b = tf.Variable(tf.zeros([1, self.num_user]), name='b')
            self.preds = tf.matmul(last, W) + b
            self.reg_loss += l2_norm(W) + l2_norm(b)
        self.reg_loss *= self.reg
        para_r = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AutoEncoder")

        """
        the generator component to generate an ideal distribution for positive user-item pairs
        """
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):
            feature = self.seed_input

            for ihid, hid in enumerate(self.adv_layers):
                feature, _ = dense_batch_fc_tanh(feature, hid, self.is_training,
                                                 'adv_layer_%d' % (ihid + 1), do_norm=True)
            adv_emb_w = tf.Variable(
                tf.truncated_normal([feature.get_shape().as_list()[1], 1], stddev=0.01),
                name='adv_emb_w')
            adv_emb_b = tf.Variable(tf.zeros([1, 1]), name='adv_emb_b')
            pos_gen = tf.matmul(feature, adv_emb_w) + adv_emb_b  # max_pos_num x 1
        para_a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")

        with tf.variable_scope("loss"):
            pos_scr = tf.reshape(tf.gather_nd(self.preds, self.fetch_idx_input), [-1, self.max_pos_num])  # batch_size x max_pos_num
            pos_scr = tf.reshape(pos_scr, [-1, self.max_pos_num, 1])  # batch_size x max_pos_num x 1

            scr_gen_diff = tf.reduce_sum(tf.reshape(pos_gen, [1, -1, 1]) - pos_scr, axis=1, keepdims=True)  # batch_size x 1 x 1

            self.MMD_mask = tf.reshape(tf.to_float(tf.greater_equal(scr_gen_diff, tf.reduce_mean(scr_gen_diff))), [-1, 1])  # batch_size x 1

            L = 1

            # MMD_scr
            self.MMD_scr = tf.tile(pos_scr, [1, 1, self.max_pos_num]) - tf.transpose(pos_scr, [0, 2, 1])  # batch_size x max_pos_num x max_pos_num
            self.MMD_scr = tf.exp(-1 * self.MMD_scr ** 2 / L)  # batch_size x max_pos_num x max_pos_num
            self.MMD_scr = tf.reshape(tf.reduce_sum(self.MMD_scr, axis=[1, 2]), [-1, 1]) / (self.max_pos_num ** 2)  # batch_size x 1

            # MMD_gen
            self.MMD_gen = tf.tile(pos_gen, [1, self.sample_num]) - tf.transpose(pos_gen)  # sample_num x sample_num
            self.MMD_gen = tf.exp(-1 * self.MMD_gen ** 2 / L)  # sample_num x sample_num
            self.MMD_gen = tf.reduce_sum(self.MMD_gen) / (self.sample_num ** 2)  # 1 x 1

            # MMD_scr_gen
            pos_gen_tile = tf.tile(tf.expand_dims(pos_gen, 0), [tf.shape(self.R_input)[0], 1, 1])  # batch_size x sample_num x 1
            self.MMD_scr_gen = tf.tile(pos_scr, [1, 1, self.sample_num]) - tf.transpose(pos_gen_tile, [0, 2, 1])  # batch_size x max_pos_num x sample_num
            self.MMD_scr_gen = tf.exp(-1 * self.MMD_scr_gen ** 2 / L)  # batch_size x max_pos_num x sample_num
            self.MMD_scr_gen = tf.reshape(tf.reduce_sum(self.MMD_scr_gen, axis=[1, 2]), [-1, 1]) / (self.max_pos_num * self.sample_num)  # batch_size x 1

            self.MMD = self.MMD_scr - 2 * self.MMD_scr_gen + self.MMD_gen  # batch_size x 1

            self.loss_r = tf.reduce_mean(tf.reduce_sum((self.preds - self.R_output) ** 2, axis=1, keep_dims=True) ** 0.5)

            self.loss_a = tf.reduce_mean(self.MMD) * self.alpha
            self.loss_ra = tf.reduce_sum(self.MMD * self.MMD_mask) * self.alpha / (tf.reduce_sum(self.MMD_mask) + 1e-7)

            self.loss_rec = self.reg_loss + self.loss_r

            self.loss_all = self.loss_r + self.loss_ra + self.reg_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.r_optimizer = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss_rec,
                                                                                             var_list=para_r)
            self.a_optimizer = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss_a,
                                                                                             var_list=para_a)
            self.all_optimizer = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss_all,
                                                                                               var_list=para_r)

    def build_predictor(self, recall_at):
        self.user_input = tf.placeholder(tf.int32, shape=[None], name="user_input")

        with tf.variable_scope("eval"):
            self.eval_preds_cold = tf.nn.embedding_lookup(tf.transpose(self.preds), self.user_input)
            _, self.eval_preds_cold = tf.nn.top_k(self.eval_preds_cold, k=recall_at[-1], sorted=True,
                                                  name='topK_net_cold')

    def get_eval_dict(self, _i, _eval_start, _eval_finish, R, eval_data):
        _eval_dict = {
            self.R_input: R[eval_data.test_item_ids, :],
            self.user_input: eval_data.test_user_ids[_eval_start: _eval_finish],
            self.is_training: False
        }
        return _eval_dict
