import tensorflow as tf


def l2_norm(para):
    return tf.reduce_sum(tf.square(para))


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
            return h2, l2_norm(h1_w) + l2_norm(h1_b)
        else:
            return h1, l2_norm(h1_w) + l2_norm(h1_b)


class Model:

    def __init__(self, model_select, num_user,  num_item, reg):
        self.reg = reg

        self.num_user = num_user
        self.num_item = num_item

        self.model_select = model_select

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

        """ 
        the main component of the debiasing NN, which is an AutoEncoder 
        """
        self.reg_loss = 0
        with tf.variable_scope("AutoEncoder", reuse=tf.AUTO_REUSE):
            last = self.R_input
            for ihid, hid in enumerate(self.model_select):
                last, reg = dense_batch_fc_tanh(last, hid, self.is_training, 'layer_%d' % (ihid + 1), do_norm=True)
                self.reg_loss += reg

            W = tf.Variable(tf.truncated_normal([last.get_shape().as_list()[1], self.num_user], stddev=0.01), name='W')
            b = tf.Variable(tf.zeros([1, self.num_user]), name='b')
            self.preds = tf.matmul(last, W) + b
            self.reg_loss += l2_norm(W) + l2_norm(b)
        self.reg_loss *= self.reg
        para_r = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AutoEncoder")

        with tf.variable_scope("loss"):
            self.loss_r = tf.reduce_mean(tf.reduce_sum((self.preds - self.R_output) ** 2, axis=1, keep_dims=True) ** 0.5)

            self.loss_all = self.loss_r + self.reg_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.r_optimizer = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss_r,
                                                                                             var_list=para_r)
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
