""" This code builds a convolutional GAN.

Version
    my_can: convolutional network is used in generator and discriminator; random index is used
    to produce batch data.
    my_wgan: wasserstein loss is used.
    my_can2: large data that cannot fit into GPU are pinned to CPU instead. As a result,
    ReadFile class is used.
    my_can2_1: the epoch for ReadFile class is not set (i.e. infinite); sample evaluation function is added.
    my_can2_2: the discriminator is defined on image patches. Various patch strategies are tried.
    my_can2_3: built on my_can2_1, the model structure is altered to include ResNet and depthwise
    separable convolutions.
    my_can2_4: the generator is constructed using multiple groups of variables, each refining the results of
    previous group. Failed.
    my_can2_5: built on my_can2_3, the training process is changed to test multi-GPU performance.
    my_can2_6: built on my_can2_3, with multiple modifications.
            1. grey_scale image is used;
            2. a new gradient penalty is used
            3. eval_sampling now allows input of z_batch;
            4. evaluate_metric is redesigned to calculate the score for some z_batch (which can also be given as input).
            5. two discriminator nets are trained in parallel.
    Among them, 3 and 4 are also applied to my_can2_3.
    my_aegan: buildt on my_wgan; VAE/GAN model is tested.
    2017.10.13: the image format in current code has changed to [batch_size, height, weight, channels]
    my_aegan2: built on can2_3; VAE/GAN model is tested.
    my_age: built on aegan2; adversarial generator-decoder is tested
    my_age2: deterministic encoder is used
    my_daegan: built on age2; gan with denoising feature matching is tested. However, different from the paper,
    the discriminator is built upon the features of encoder.
    my_daegan2: built on my_daegan; the decoder needs to do both discrimination and reconstruction
    my_age3: an encoder and a decoder are trained with both discrimination and reconstruction error
    my_lgan: a ladder structure is used, where each layer in GAN is trying to learn downsampled inputs.
    my_lgan2: spectral normalization is used for discriminator
    my_sngan2: several new loss function is tested.

"""
# check the summary:
# tensorboard --logdir="/home/richard/Desktop/Data/celebA_0_log/test"
# check GPU workload
# nvidia-smi.exe

# default modules
import numpy as np
import tensorflow as tf

# helper functions
from GeneralTools.misc_fun import FLAGS
from GeneralTools.input_func import ReadTFRecords
from GeneralTools.graph_func import prepare_folder, rollback, write_sprite_wrapper, \
    global_step_config, multi_opt_config, embedding_image_wrapper, GenerativeModelMetric
from GeneralTools.layer_func import Net, Routine
from GeneralTools.math_func import MeshCode, rand_mmd_g_xy, get_squared_dist, mixture_mmd_g, \
    mixture_mmd_t, rand_mmd_g_xn, witness_mix_g, witness_mix_t, mat_slice, moving_average_copy

########################################################################
# from GeneralTools.misc_fun import FLAGS

"""
Class definition
"""


class SNGan(object):
    def __init__(
            self, architecture, misc=None, optimizer='adam', do_summary=True,
            do_summary_image=True, num_summary_image=8, image_transpose=False):
        """ This function initializes a ladder adversarial network.

        :param architecture: a dictionary, e.g.
            {'input': [(3, 32, 32)],  # channel, height, width, activation function
            # (code len, activation)
             'code': [(100, 'norm'), (1024, 4, 4, 'lrelu')],
            # num_feature_maps, kernel_size, stride_size, padding
             'generator': [(48, 3, 3, 1, 1, 'same', 'lrelu'),
                          (96, 3, 3, 1, 1, 'same', 'lrelu'),
                          (192, 3, 3, 1, 1, 'same', 'relu', 5),
                          (96, 3, 3, 1, 1, 'same', 'lrelu'),
                          (96, 3, 3, 1, 1, 'same', 'lrelu'),
                          (3, 3, 3, 1, 1, 'same', 'lrelu')],
             # num_feature_maps, kernel_size, stride_size, padding
             'discriminator': [(96, 3, 3, 1, 1, 'same', 'relu'),
                          (48, 3, 3, 1, 1, 'same', 'relu')]}
        :param misc: extra parameters to the model
        :param optimizer:
        :param do_summary:
        :param do_summary_image:
        :param num_summary_image:
        :param image_transpose:
        """
        # default parameters
        self.optimizer_type = ['sgd', 'momentum', 'adam', 'rmsprop']
        self.data_format = FLAGS.IMAGE_FORMAT
        # structure parameters
        self.architecture = architecture
        self.misc = misc
        self.optimizer = optimizer
        # input parameters
        self.channels = self.architecture['input'][0][0]
        self.height = self.architecture['input'][0][1]
        self.width = self.architecture['input'][0][2]
        self.D = np.prod(self.architecture['input'][0], dtype=np.int32)
        self.d = self.architecture['code'][0][0]
        self.s = self.architecture['discriminator'][-1]['out']
        # control parameters
        self.do_summary = do_summary
        self.do_summary_image = do_summary_image
        self.num_summary_image = num_summary_image
        self.loss_names = None
        self.global_step = None
        self.step_per_epoch = None

        if image_transpose:  # for dataset like MNIST, image needs to be transposed
            if FLAGS.IMAGE_FORMAT == 'channels_first':
                self.perm = [0, 1, 3, 2]
            elif FLAGS.IMAGE_FORMAT == 'channels_last':
                self.perm = [0, 2, 1, 3]
        else:
            self.perm = None

        # initialize network
        self.graph = None
        self.Gen = None
        self.Dis = None

    def init_net(self):
        """ This function initializes the network

        :return:
        """
        # initialize the generator network
        g_net = Net(
            self.architecture['generator'], net_name='gen',
            data_format=FLAGS.IMAGE_FORMAT)
        # define layer connections in generator
        self.Gen = Routine(g_net)
        self.Gen.add_input_layers([64, self.d], [0])
        self.Gen.seq_links(list(range(g_net.num_layers)))
        self.Gen.add_output_layers([g_net.num_layers - 1])

        # initialize the generator network
        d_net = Net(
            self.architecture['discriminator'], net_name='dis',
            data_format=FLAGS.IMAGE_FORMAT)
        # define layer connections in generator
        self.Dis = Routine(d_net)
        self.Dis.add_input_layers([64] + list(self.architecture['input'][0]), [0])
        self.Dis.seq_links(list(range(d_net.num_layers)))
        self.Dis.add_output_layers([d_net.num_layers - 1])

    ###################################################################
    def sample_codes(self, batch_size, name='codes'):
        """ This function samples from normal distribution, or performs reparameterization trick

        :param batch_size:
        :param name:
        :return:
        """
        # convert code mean and std to code
        return tf.random_normal([batch_size, self.d], mean=0.0, stddev=1.0, name=name)

    ###################################################################
    def gradient_penalty(self, x, x_gen, batch_size):
        """ This function calculates the gradient penalty used in wassersetin gan

        :param x:
        :param x_gen:
        :param batch_size:
        :return:
        """
        uni = tf.random_uniform(
            shape=[batch_size, 1, 1, 1],  # [batch_size, channels, height, width]
            minval=0.0, maxval=1.0,
            name='uniform')
        x_hat = tf.identity(
            tf.add(tf.multiply(x, uni), tf.multiply(x_gen, tf.subtract(1.0, uni))),
            name='x_hat')
        s_x_hat = self.Dis(x_hat, is_training=False)
        g_x_hat = tf.reshape(
            tf.gradients(s_x_hat, x_hat, name='gradient_x_hat')[0],
            [batch_size, -1])
        loss_grad_norm = tf.reduce_mean(
            tf.square(tf.norm(g_x_hat, ord=2, axis=1) - 1))
        return loss_grad_norm

    def mmd_gradient_penalty(self, x, x_gen, s_x, s_gen, batch_size, mode='fixed_g'):
        """ This function calculates the gradient penalty used in mmd-gan

        :param x: real images
        :param x_gen: generated images
        :param s_x: scores of real images
        :param s_gen: scores of generated images
        :param batch_size:
        :param mode:
        :return:
        """
        uni = tf.random_uniform(
            shape=[batch_size, 1, 1, 1],  # [batch_size, channels, height, width]
            minval=0.0, maxval=1.0,
            name='uniform')
        x_hat = tf.identity(
            tf.add(tf.multiply(x, uni), tf.multiply(x_gen, tf.subtract(1.0, uni))),
            name='x_hat')
        s_x_hat = self.Dis(x_hat, is_training=False)
        # get witness function w.r.t. x, x_gen
        dist_zx = get_squared_dist(s_x_hat, s_x, mode='xy', name='dist_zx', do_summary=self.do_summary)
        dist_zy = get_squared_dist(s_x_hat, s_gen, mode='xy', name='dist_zy', do_summary=self.do_summary)
        if mode == 'fixed_g_gp':
            witness = witness_mix_g(
                dist_zx, dist_zy, sigma=[1.0, 2.0, 4.0, 8.0, 16.0],
                name='witness', do_summary=self.do_summary)
        elif mode == 'fixed_t_gp':
            witness = witness_mix_t(
                dist_zx, dist_zy, alpha=[0.25, 0.5, 0.9, 2.0, 25.0], beta=2.0,
                name='witness', do_summary=self.do_summary)
        else:
            raise NotImplementedError('gradient penalty: {} not implemented'.format(mode))
        g_x_hat = tf.reshape(
            tf.gradients(witness, x_hat, name='gradient_x_hat')[0],
            [batch_size, -1])
        loss_grad_norm = tf.reduce_mean(
            tf.square(tf.norm(g_x_hat, ord=2, axis=1) - 1))
        return loss_grad_norm

    ###################################################################
    def __gpu_task__(
            self, batch_size=64, is_training=False, x_batch=None,
            opt_op=None, z_batch=None):
        """ This function defines the task on a gpu

        :param batch_size:
        :param is_training:
        :param x_batch: 4-D tensor, either in channels_first or channels_last format
        :param opt_op:
        :param z_batch:
        :return:
        """
        if is_training:
            # sample new data, [batch_size*2, height, weight, channels]
            z_rand = self.sample_codes(batch_size, name='z_rand')
            x_gen = self.Gen(z_rand, is_training=True)
            # score x_batch and x_gen
            # s_x = self.Dis(x_batch, is_training=True)
            s_x, s_gen = tf.split(
                self.Dis(tf.concat([x_batch, x_gen], axis=0), is_training=True),
                num_or_size_splits=2, axis=0)

            # loss function
            if self.misc[0] == 'logistic':  # logistic loss
                loss_dis = tf.reduce_mean(tf.nn.softplus(s_gen) + tf.nn.softplus(-s_x))
                loss_gen = tf.reduce_mean(tf.nn.softplus(-s_gen))
            elif self.misc[0] == 'hinge':
                loss_dis = tf.reduce_mean(tf.nn.relu(1.0 + s_gen)) + tf.reduce_mean(tf.nn.relu(1.0 - s_x))
                loss_gen = tf.reduce_mean(-s_gen)
            elif self.misc[0] == 'wasserstein':  # wasserstein loss
                loss_gen = tf.reduce_mean(s_x) - tf.reduce_mean(s_gen)
                loss_penalty = self.gradient_penalty(x_batch, x_gen, batch_size)
                loss_dis = - loss_gen + self.misc[1] * loss_penalty
            elif self.misc[0] == 'fixed_g':  # mmd gaussian with fixed sigma
                dist_xx, dist_xy, dist_yy = get_squared_dist(s_gen, s_x, z_score=False, do_summary=self.do_summary)
                # loss_gen = mixture_mmd_g(
                #     dist_xx, dist_xy, dist_yy, batch_size, sigma=[0.125, 0.4, 1.0, 2.0, 4.0],
                #     name='mmd', do_summary=self.do_summary)
                loss_gen = mixture_mmd_g(
                    dist_xx, dist_xy, dist_yy, batch_size, sigma=[1.0, 2.0, 4.0, 8.0, 16.0],
                    name='mmd', do_summary=self.do_summary)
                loss_dis = -loss_gen
            elif self.misc[0] == 'fixed_g_gp':  # mmd gaussian with fixed sigma and gradient penalty
                dist_xx, dist_xy, dist_yy = get_squared_dist(s_gen, s_x, z_score=False, do_summary=self.do_summary)
                loss_gen = mixture_mmd_g(  # if you change sigma here, do forget to change it in mmd_gradient_penalty
                    dist_xx, dist_xy, dist_yy, batch_size, sigma=[1.0, 2.0, 4.0, 8.0, 16.0],
                    name='mmd', do_summary=self.do_summary)
                loss_penalty = self.mmd_gradient_penalty(x_batch, x_gen, s_x, s_gen, batch_size, mode=self.misc[0])
                loss_dis = -loss_gen + self.misc[1] * loss_penalty
            elif self.misc[0] == 'fixed_g_mix':
                # mmd gaussian with fixed sigma and gradient penalty
                # mix sx and sg when discriminator is too strong
                sigma = [1.0, 2.0, 4.0, 8.0, 16.0]
                # sigma = [0.125, 0.4, 1.0, 2.0, 4.0]
                pair_dist = get_squared_dist(tf.concat((s_gen, s_x), axis=0))
                dxx = pair_dist[0:batch_size, 0:batch_size]
                dyy = pair_dist[batch_size:, batch_size:]
                dxy = pair_dist[0:batch_size, batch_size:]
                loss_gen = mixture_mmd_g(
                    dxx, dxy, dyy, batch_size, sigma=sigma,
                    name='mmd', do_summary=self.do_summary)
                # get mmd_average
                mmd_average = moving_average_copy(loss_gen, 'mmd_average', rho=0.01)
                # mix real and generated data
                mmd_target = 2.0
                rho = 1e-2
                mix_prob = tf.get_variable(
                    'prob', dtype=tf.float32, initializer=tf.constant(0.0), trainable=False)
                tf.add_to_collection(
                    tf.GraphKeys.UPDATE_OPS,
                    tf.assign(
                        mix_prob,
                        tf.clip_by_value(
                            mix_prob + rho * (mmd_average - mmd_target), clip_value_min=0.0, clip_value_max=0.5)))
                uni = tf.random_uniform([batch_size], 0.0, 1.0, dtype=tf.float32, name='uni')
                coin = tf.greater(uni, mix_prob, name='coin')  # coin for using original data
                # get mixed distance
                coin_x_yc = tf.concat((coin, tf.logical_not(coin)), axis=0)
                coin_xc_y = tf.concat((tf.logical_not(coin), coin), axis=0)
                dxx_mix = mat_slice(pair_dist, coin_x_yc)
                dyy_mix = mat_slice(pair_dist, coin_xc_y)
                dxy_mix = mat_slice(pair_dist, coin_x_yc, coin_xc_y)
                loss_mix = mixture_mmd_g(
                    dxx_mix, dxy_mix, dyy_mix, batch_size, sigma=sigma,
                    name='mmd_mix', do_summary=self.do_summary)
                loss_dis = -loss_mix

                if self.do_summary:
                    tf.summary.histogram('squared_dist/dxx', dxx)
                    tf.summary.histogram('squared_dist/dyy', dyy)
                    tf.summary.histogram('squared_dist/dxy', dxy)
                    tf.summary.scalar('loss/mean_mmd', mmd_average)
                    tf.summary.scalar('loss/mix_prob', mix_prob)

            elif self.misc[0] == 'fixed_t':  # mmd t with fixed alpha
                dist_xx, dist_xy, dist_yy = get_squared_dist(s_gen, s_x, z_score=False, do_summary=self.do_summary)
                # loss_gen = mixture_mmd_t(
                #     dist_xx, dist_xy, dist_yy, batch_size, alpha=[0.25, 0.4, 0.65, 1.3, 25.0], beta=0.25,
                #     name='mmd', do_summary=self.do_summary)
                loss_gen = mixture_mmd_t(  # this is the kernel used in original paper
                    dist_xx, dist_xy, dist_yy, batch_size, alpha=[0.2, 0.5, 1, 2, 5.0], beta=2.0,
                    name='mmd', do_summary=self.do_summary)
                loss_dis = -loss_gen
            elif self.misc[0] == 'fixed_t_gp':  # mmd gaussian with fixed sigma and gradient penalty
                dist_xx, dist_xy, dist_yy = get_squared_dist(s_gen, s_x, z_score=False, do_summary=self.do_summary)
                loss_gen = mixture_mmd_t(  # if you change alpha here, do forget to change it in mmd_gradient_penalty
                    dist_xx, dist_xy, dist_yy, batch_size, alpha=[0.25, 0.5, 0.9, 2.0, 25.0], beta=2.0,
                    name='mmd', do_summary=self.do_summary)
                loss_penalty = self.mmd_gradient_penalty(x_batch, x_gen, s_x, s_gen, batch_size, mode=self.misc[0])
                loss_dis = -loss_gen + self.misc[1] * loss_penalty
            elif self.misc[0] == 'rand_g':
                # mmd g with random sigma
                # omega = tf.random_uniform([], 0.05, 0.9, dtype=tf.float32, name='omega')
                # d_x_x, d_x_y, d_y_y = get_squared_dist(s_gen, s_x, z_score=False, do_summary=self.do_summary)
                # dist_yy_raw = get_squared_dist(tf.reshape(x_batch, shape=(batch_size, -1)), z_score=False)
                # m = tf.constant(batch_size, tf.float32)
                # k = matrix_mean_wo_diagonal(dist_yy_raw, m) / matrix_mean_wo_diagonal(dist_yy, m)
                # loss_mmd = rand_mmd_g_xy(
                #     d_x_x, d_x_y, d_y_y, batch_size, omega=omega, max_iter=3,
                #     name='mmd', do_summary=self.do_summary)
                # tf.summary.scalar('k', k)
                # loss_gen = loss_mmd
                # loss_dis = -loss_gen

                # a fixed sigma and a random sigma
                # dist_xx, dist_xy, dist_yy = get_squared_dist(s_gen, s_x, z_score=False, do_summary=self.do_summary)
                #
                # def f1():
                #     omega = tf.random_uniform([], 0.05, 0.9, dtype=tf.float32, name='omega')
                #     loss = rand_mmd_g_xy(
                #         dist_xx, dist_xy, dist_yy, batch_size, omega=omega, max_iter=3, name='mmd_v')
                #     return loss
                #
                # def f2():
                #     loss = mixture_mmd_g(
                #         dist_xx, dist_xy, dist_yy, batch_size, sigma=[0.125, 0.4, 1.0, 2.0, 4.0], name='mmd_f')
                #     loss = loss / 5.0
                #     return loss
                #
                # coin = tf.random_uniform([], 0.0, 1.0, dtype=tf.float32, name='coin')
                # f1_prob = tf.cast(
                #     0.00 + self.global_step / self.step_per_epoch / 64.0, dtype=tf.float32, name='vg_prob')
                # loss_gen = tf.cond(coin < f1_prob, lambda: f1(), lambda: f2())
                # loss_dis = -loss_gen

                # random sigma with three samples, sg sr and normal
                # sn = tf.random_normal(
                #     (batch_size, self.s), mean=0.0, stddev=np.sqrt(self.misc[1] / 2.0 / self.s), dtype=tf.float32)
                # omega = tf.random_uniform([], 0.05, 0.9, dtype=tf.float32)
                # d_x_x, d_y_y, d_z_z, d_x_y, d_x_z, d_y_z = squared_dist_triplet(
                #     s_gen, s_x, sn, name='squared_dist', do_summary=self.do_summary)
                # loss_gen = rand_mmd_g_xy(
                #     d_x_x, d_x_y, d_y_y, batch_size, omega=omega,
                #     max_iter=3, name='mmd_gr', do_summary=self.do_summary)
                # loss_rn = rand_mmd_g_xy(
                #     d_z_z, d_y_z, d_y_y, batch_size, omega=omega,
                #     max_iter=3, name='mmd_rn', do_summary=self.do_summary)
                # loss_dis = loss_rn - loss_gen
                #
                # if self.do_summary:
                #     tf.summary.scalar('loss/normal', loss_rn)

                # random sigma with three samples, sg sr and normal. Any expectation term under normal distribution is
                # calculated analytically.
                omega = tf.random_uniform([], self.misc[2], self.misc[3], dtype=tf.float32)
                d_x_x, d_x_y, d_y_y = get_squared_dist(s_gen, s_x, z_score=False, do_summary=self.do_summary)
                # from DeepLearning.my_test import hinge_mmd_rand_g_xy
                # loss_gr, hinge_loss_gr = hinge_mmd_rand_g_xy(
                #     d_x_x, d_x_y, d_y_y, batch_size, omega=omega,
                #     max_iter=3, name='mmd_gr', do_summary=self.do_summary)
                loss_gr = rand_mmd_g_xy(
                    d_x_x, d_x_y, d_y_y, batch_size, omega=omega,
                    max_iter=3, name='mmd_gr', do_summary=self.do_summary)
                # from DeepLearning.my_test import var_bound_rand_mmd_g
                # loss_gr = var_bound_rand_mmd_g(
                #     d_x_x, d_x_y, d_y_y, batch_size, max_iter=3, name='mmd_gr', do_summary=self.do_summary)
                loss_gn = rand_mmd_g_xn(
                    s_gen, self.misc[1], batch_size, self.s, dist_xx=d_x_x, omega=omega,
                    max_iter=3, name='mmd_gn', do_summary=self.do_summary)
                loss_rn = rand_mmd_g_xn(
                    s_x, self.misc[1], batch_size, self.s, dist_xx=d_y_y, omega=omega,
                    max_iter=3, name='mmd_rn', do_summary=self.do_summary)
                # decide the weight
                # mmd_target = 0.2
                # get mmd_average
                mmd_average = moving_average_copy(loss_gr, 'mmd_average', rho=0.01)
                # tf.add_to_collection(
                #     tf.GraphKeys.UPDATE_OPS,
                #     tf.assign(weight, tf.clip_by_value(weight - 0.01 * (mmd_average - mmd_target), 0.01, 1.0)))
                if self.do_summary:
                    tf.summary.scalar('loss/mean_mmd', mmd_average)
                #     tf.summary.scalar('loss/weight', weight)
                # apply weighted loss
                loss_gen = loss_gr
                loss_dis = loss_rn - loss_gr
                # loss_gen = loss_gr
                # loss_dis = loss_rn - 0.1 / tf.maximum(mmd_average, 0.1) * loss_gr

                if self.do_summary:
                    tf.summary.scalar('loss/gr', loss_gr)
                    tf.summary.scalar('loss/gn', loss_gn)
                    tf.summary.scalar('loss/rn', loss_rn)

            elif self.misc[0] == 'rand_g_mix':  # mix sx and sg when discriminator is too strong
                # calculate pairwise distance
                pair_dist = get_squared_dist(tf.concat((s_gen, s_x), axis=0))
                dxx = pair_dist[0:batch_size, 0:batch_size]
                dyy = pair_dist[batch_size:, batch_size:]
                dxy = pair_dist[0:batch_size, batch_size:]
                # mmd loss
                omega = tf.random_uniform([], self.misc[2], self.misc[3], dtype=tf.float32)
                loss_gr = rand_mmd_g_xy(
                    dxx, dxy, dyy, batch_size, omega=omega,
                    max_iter=3, name='mmd_gr', do_summary=self.do_summary)
                loss_gn = rand_mmd_g_xn(
                    s_gen, self.misc[1], batch_size, self.s, dist_xx=dxx, omega=omega,
                    max_iter=3, name='mmd_gn', do_summary=self.do_summary)
                loss_rn = rand_mmd_g_xn(
                    s_x, self.misc[1], batch_size, self.s, dist_xx=dyy, omega=omega,
                    max_iter=3, name='mmd_rn', do_summary=self.do_summary)
                # get mmd_average
                mmd_average = moving_average_copy(loss_gr, 'mmd_average', rho=0.01)
                # mix real and generated data
                mmd_target = 0.2
                rho = 1e-2
                mix_prob = tf.get_variable(
                    'prob', dtype=tf.float32, initializer=tf.constant(0.0), trainable=False)
                tf.add_to_collection(
                    tf.GraphKeys.UPDATE_OPS,
                    tf.assign(
                        mix_prob,
                        tf.clip_by_value(
                            mix_prob + rho * (mmd_average - mmd_target), clip_value_min=0.0, clip_value_max=0.5)))
                uni = tf.random_uniform([batch_size], 0.0, 1.0, dtype=tf.float32, name='uni')
                coin = tf.greater(uni, mix_prob, name='coin')  # coin for using original data
                # get mixed distance
                coin_x_yc = tf.concat((coin, tf.logical_not(coin)), axis=0)
                coin_xc_y = tf.concat((tf.logical_not(coin), coin), axis=0)
                dxx_mix = mat_slice(pair_dist, coin_x_yc)
                dyy_mix = mat_slice(pair_dist, coin_xc_y)
                dxy_mix = mat_slice(pair_dist, coin_x_yc, coin_xc_y)
                loss_gr_mix = rand_mmd_g_xy(
                    dxx_mix, dxy_mix, dyy_mix, batch_size, omega=omega,
                    max_iter=3, name='mmd_gr_mix', do_summary=self.do_summary)
                # apply weighted loss
                loss_gen = loss_gr
                loss_dis = loss_rn - loss_gr_mix

                if self.do_summary:
                    tf.summary.histogram('squared_dist/dxx', dxx)
                    tf.summary.histogram('squared_dist/dyy', dyy)
                    tf.summary.histogram('squared_dist/dxy', dxy)
                    tf.summary.scalar('loss/mean_mmd', mmd_average)
                    tf.summary.scalar('loss/gr', loss_gr)
                    tf.summary.scalar('loss/gn', loss_gn)
                    tf.summary.scalar('loss/rn', loss_rn)
                    tf.summary.scalar('loss/gr_mix', loss_gr_mix)
                    tf.summary.scalar('loss/mix_prob', mix_prob)
            else:
                raise NotImplementedError('Not implemented.')

            # form loss list
            # sigma = [layer.sigma for layer in self.Dis.net.layers]
            loss_list = [loss_gen, loss_dis]
            self.loss_names = '<loss_gen>, <loss_dis>'
            if self.misc[0] in {'rand_g', 'rand_g_mix'}:
                loss_list.append(mmd_average)

            # compute gradient
            # grads is a list of (gradient, variable) tuples
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            vars_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dis")
            grads_dis = opt_op[0].compute_gradients(loss_dis, var_list=vars_dis)
            vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gen")
            grads_gen = opt_op[1].compute_gradients(loss_gen, var_list=vars_gen)
            grads_list = [grads_dis, grads_gen]

            # summary op is always pinned to CPU
            # add summary to loss and intermediate variables
            if self.do_summary:
                # self.get_sigma()s
                tf.summary.scalar('loss/gen', loss_gen)
                tf.summary.scalar('loss/dis', loss_dis)
                tf.summary.histogram('x/x', x_batch)
                tf.summary.histogram('x/x_gen', x_gen)
                tf.summary.histogram('x/sx', s_x)
                tf.summary.histogram('x/sg', s_gen)
                g_x = tf.reshape(tf.gradients(s_x, x_batch)[0], [batch_size, -1])
                g_x_norm = tf.norm(g_x, ord=2, axis=1)
                tf.summary.histogram('x/g_x_norm', g_x_norm)
                g_gen = tf.reshape(tf.gradients(s_gen, x_gen)[0], [batch_size, -1])
                g_gen_norm = tf.norm(g_gen, ord=2, axis=1)
                tf.summary.histogram('x/g_gen_norm', g_gen_norm)
                for layer in self.Gen.net.layers:
                    for key in layer.ops:
                        kernel_norm = getattr(layer.ops[key], 'kernel_norm', None)
                        if kernel_norm is not None:
                            tf.summary.scalar('norm/' + layer.ops[key].name_in_err, kernel_norm)
                for layer in self.Dis.net.layers:
                    for key in layer.ops:
                        kernel_norm = getattr(layer.ops[key], 'kernel_norm', None)
                        if kernel_norm is not None:
                            tf.summary.scalar('norm/' + layer.ops[key].name_in_err, kernel_norm)

            return grads_list, loss_list
        else:
            if z_batch is None:
                z_batch = self.sample_codes(batch_size, name='z_te')
            # generate new images
            x_gen = self.Gen(z_batch, is_training=False)
            return x_gen

    def get_x_batch(self, filename, batch_size, file_repeat=1, num_threads=7, shuffle_file=False):
        """ This function reads image data

        :param filename:
        :param batch_size:
        :param file_repeat:
        :param num_threads:
        :param shuffle_file: bool, whether to shuffle the filename list
        :return:
        """
        # read data
        training_data = ReadTFRecords(
            filename, self.D, num_labels=0, dtype=tf.string, batch_size=batch_size,
            file_repeat=file_repeat, num_threads=num_threads, shuffle_file=shuffle_file)
        # training_data = PreloadGPU(filename, num_instance, self.D, num_threads=num_threads)
        # convert matrix data to image tensor and scale them to [-1, 1]
        training_data.shape2image(self.channels, self.height, self.width)
        x_batch = training_data.next_batch()
        # convert x_combo to grey scale images
        # x_batch = tf.image.rgb_to_grayscale(x_batch)  # [batch_size, height, width, 1]
        # for dataset like MNIST, image needs to be transposed
        if self.perm is not None:
            x_batch = tf.transpose(x_batch, perm=self.perm)

        return x_batch

    ###################################################################
    def training(self, filename, agent, num_instance, lr_list, end_lr=1e-7,
                 max_step=None, batch_size=64, num_threads=7, gpu='/gpu:0'):
        """ This function defines the training process

        :param filename:
        :param agent:
        :param num_instance:
        :param lr_list:
        :param end_lr:
        :param max_step:
        :type max_step: int
        :param batch_size:
        :param num_threads:
        :param gpu: which gpu to use
        :return:
        """
        from math import gcd
        self.step_per_epoch = np.floor(num_instance / batch_size).astype(np.int32)
        if max_step >= self.step_per_epoch:
            file_repeat = int(batch_size / gcd(num_instance, batch_size))
            shuffle_file = False
        else:
            if isinstance(filename, str) or (isinstance(filename, (list, tuple)) and len(filename) == 1):
                raise AttributeError(
                    'max_step should be larger than step_per_epoch when there is a single file.')
            else:
                # for large dataset, the data are stored in multiple files. If all files cannot be visited
                # within max_step, consider shuffle the filename list every max_step
                file_repeat = 1
                shuffle_file = True

        print('N: {}; Batch: {}; file_repeat: {}'.format(num_instance, batch_size, file_repeat))

        # build the graph
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device(gpu):
            self.init_net()
            # get next batch
            x_batch = self.get_x_batch(filename, batch_size, file_repeat, num_threads, shuffle_file)
            print('Shape of input batch: {}'.format(x_batch.get_shape().as_list()))

            # setup training process
            # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.global_step = global_step_config()
            _, opt_ops = multi_opt_config(
                lr_list, end_lr=end_lr,
                optimizer=self.optimizer, global_step=self.global_step)
            # assign tasks
            with tf.variable_scope(tf.get_variable_scope()):
                # calculate loss and gradients
                grads_list, loss_list = self.__gpu_task__(
                    batch_size=batch_size, is_training=True, x_batch=x_batch,
                    opt_op=opt_ops)
            # apply the gradient
            if agent.imbalanced_update is None:
                dis_op = opt_ops[0].apply_gradients(grads_list[0], global_step=self.global_step)
                gen_op = opt_ops[1].apply_gradients(grads_list[1])
                op_list = [dis_op, gen_op]
            elif isinstance(agent.imbalanced_update, (list, tuple)):
                if agent.imbalanced_update[0] == 1:
                    dis_op = opt_ops[0].apply_gradients(grads_list[0], global_step=self.global_step)
                    gen_op = opt_ops[1].apply_gradients(grads_list[1])
                    op_list = [dis_op, gen_op]
                elif agent.imbalanced_update[1] == 1:
                    dis_op = opt_ops[0].apply_gradients(grads_list[0])
                    gen_op = opt_ops[1].apply_gradients(grads_list[1], global_step=self.global_step)
                    op_list = [dis_op, gen_op]
                else:
                    raise AttributeError('One of the imbalanced_update must be 1.')
            elif isinstance(agent.imbalanced_update, str):
                dis_op = opt_ops[0].apply_gradients(grads_list[0])
                gen_op = opt_ops[1].apply_gradients(grads_list[1], global_step=self.global_step)
                op_list = [dis_op, gen_op]
            else:
                raise AttributeError('Imbalanced_update not identified.')

            # summary op is always pinned to CPU
            # add summary for all trainable variables
            if self.do_summary:
                for grads in grads_list:
                    for var_grad, var in grads:
                        tf.summary.histogram('grad_' + var.name.replace(':', '_'), var_grad)
                        tf.summary.histogram(var.name.replace(':', '_'), var)
                summary_op = tf.summary.merge_all()
            else:
                summary_op = None
            # add summary for final image reconstruction
            if self.do_summary_image:
                tf.get_variable_scope().reuse_variables()
                summary_image_op = self.summary_image_sampling(x_batch)
            else:
                summary_image_op = None

            # run the session
            print('loss_list name: {}.'.format(self.loss_names))
            agent.train(
                op_list, loss_list,
                self.global_step, max_step, self.step_per_epoch,
                summary_op, summary_image_op)

    ###################################################################
    def summary_image_sampling(self, x):
        """ This function randomly samples instances to compare with real samples.
        It returns a summary op.

        :param x:
        :return:
        """
        # down sample x
        x_real = x[0:self.num_summary_image, :]
        # generate new images
        x_gen = self.__gpu_task__(batch_size=self.num_summary_image, is_training=False)
        # do clipping
        x_gen = tf.clip_by_value(x_gen, clip_value_min=-1, clip_value_max=1)
        # tf.summary.image only accepts [batch_size, height, width, channels]
        if FLAGS.IMAGE_FORMAT == 'channels_first':
            x_real = tf.transpose(x_real, perm=(0, 2, 3, 1))
            x_gen = tf.transpose(x_gen, perm=(0, 2, 3, 1))
        # add summaries
        summaries_image = tf.summary.image('Ir', x_real, max_outputs=self.num_summary_image)
        summaries_gen = tf.summary.image('Ig', x_gen, max_outputs=self.num_summary_image)
        summary_image_op = tf.summary.merge([summaries_image, summaries_gen])

        return summary_image_op

    ###################################################################
    def eval_sampling(self, filename, sub_folder, mesh_num=None, mesh_mode=0,
                      if_invert=False, z_batch=None, num_threads=7, real_sample=False,
                      get_dis_score=True, do_sprite=True, do_embedding=False, ckpt_file=None):
        """ This function randomly generates samples and writes them to sprite.

        :param filename:
        :param sub_folder:
        :param mesh_num:
        :param if_invert:
        :param mesh_mode:
        :param z_batch: if provided, z_batch will be used to generate images.
        :param num_threads:
        :param real_sample: True if real sample should also be obtained
        :param get_dis_score: bool, whether to calculate the scores from the discriminator
        :param do_sprite:
        :param do_embedding:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :return:
        """
        # prepare folder
        ckpt_folder, summary_folder, _ = prepare_folder(filename, sub_folder=sub_folder)
        # check inputs
        if mesh_num is None:
            mesh_num = (10, 10)
        elif z_batch is not None:
            assert z_batch.shape[0] == mesh_num[0] * mesh_num[1]
        batch_size = mesh_num[0] * mesh_num[1]
        if do_embedding is True:
            get_dis_score = True
            real_sample = True

        # build the network graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.init_net()
            if real_sample:
                x_batch = self.get_x_batch(filename, batch_size, num_threads)
            else:
                x_batch = tf.constant(0)

            # sample validation instances
            if z_batch is None:
                code = MeshCode(self.d, mesh_num=mesh_num)
                z_batch = code.get_batch(mesh_mode)
            else:
                z_batch = tf.constant(z_batch, name='Z')
            # generate new images
            x_gen = self.__gpu_task__(z_batch=z_batch, is_training=False)
            # do clipping
            x_gen = tf.clip_by_value(x_gen, clip_value_min=-1, clip_value_max=1)

            # get discriminator scores
            if get_dis_score and real_sample:
                s_x, s_gen = tf.split(
                    self.Dis(tf.concat([x_batch, x_gen], axis=0), is_training=False),
                    num_or_size_splits=2, axis=0)
            else:
                s_x = tf.constant(0)
                s_gen = tf.constant(0)

            print('Graph configuration finished...')
            # calculate the value of x_gen
            var_list = [z_batch, x_gen, x_batch, s_x, s_gen]
            _temp, global_step_value = rollback(var_list, ckpt_folder, ckpt_file=ckpt_file)
            z_batch_value, x_gen_value, x_real_value, s_x_value, s_gen_value = _temp

        # write to files
        if do_sprite:
            if real_sample:
                write_sprite_wrapper(
                    x_real_value, mesh_num, filename, file_folder=summary_folder,
                    file_index='_r_' + sub_folder + '_' + str(global_step_value) + '_' + str(mesh_mode),
                    if_invert=if_invert, image_format=FLAGS.IMAGE_FORMAT)
            write_sprite_wrapper(
                x_gen_value, mesh_num, filename, file_folder=summary_folder,
                file_index='_g_' + sub_folder + '_' + str(global_step_value) + '_' + str(mesh_mode),
                if_invert=if_invert, image_format=FLAGS.IMAGE_FORMAT)

        # do visualization for code_value
        if do_embedding:
            # transpose image data if necessary
            if real_sample:
                x_as_image = np.transpose(x_real_value, axes=self.perm) if self.perm is not None else x_real_value
                x_gen_as_image = np.transpose(x_gen_value, axes=self.perm) if self.perm is not None else x_gen_value
                # concatenate real and generated images, codes and labels
                s_x_value = np.concatenate((s_x_value, s_gen_value), axis=0)
                x_as_image = np.concatenate((x_as_image, x_gen_as_image), axis=0)
                labels = np.concatenate(  # 1 for real, 0 for gen
                    (np.ones(batch_size, dtype=np.int), np.zeros(batch_size, dtype=np.int)), axis=0)
                # embedding
                mesh_num = (mesh_num[0] * 2, mesh_num[1])
                embedding_image_wrapper(
                    s_x_value, filename, var_name='x_vs_xg', file_folder=summary_folder,
                    file_index='_x_vs_xg_' + sub_folder + '_' + str(global_step_value) + '_' + str(mesh_mode),
                    labels=labels, images=x_as_image, mesh_num=mesh_num,
                    if_invert=if_invert, image_format=FLAGS.IMAGE_FORMAT)

        return z_batch_value

    def mdl_score(self, filename, sub_folder, batch_size, num_batch=10, model='v1', ckpt_file=None, num_threads=7):
        """ This function calculates the scores for the real and generated samples

        :param filename:
        :param sub_folder:
        :param batch_size:
        :param num_batch:
        :param model: whether to use inception_v1 or inception_v3 (v3 is not working for now)
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param num_threads:
        :return:
        """
        # prepare folder
        ckpt_folder, summary_folder, _ = prepare_folder(filename, sub_folder=sub_folder)

        # build the network graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.init_net()
            x_batch = self.get_x_batch(filename, batch_size, num_threads)

            # generate new images
            z_batch = self.sample_codes(batch_size)
            x_gen = self.__gpu_task__(z_batch=z_batch, is_training=False)
            # do clipping
            x_gen = tf.clip_by_value(x_gen, clip_value_min=-1, clip_value_max=1)

            metric = GenerativeModelMetric(model=model)
            if model == 'v1':
                scores = metric.inception_score_and_fid_v1(
                    x_batch, x_gen, num_batch=num_batch, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file)
            # elif model == 'v3':  # inception v3 model is not working
            #     scores = metric.inception_score_and_fid_v3(
            #         x_batch, x_gen, num_batch=num_batch, inception_batch=batch_size, ckpt_folder=ckpt_folder)
            else:
                raise NotImplementedError('Model {} not implemented.'.format(model))

            return scores
