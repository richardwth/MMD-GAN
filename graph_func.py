import json
import math
import os.path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import plotly as py
import time

from tensorflow.contrib import gan as tfgan
from tensorflow.python.client import timeline
from GeneralTools.input_func import ReadFile
from GeneralTools.misc_fun import FLAGS


def _create_variable_(name, initializer, fan_size, trainable=True, weight_scale=1.0):
    """ This function pins variables to cpu

    tf.get_variable is used instead of tf.Variable, so that variable with the same name will not be re-initialized
    """
    # define initialization method
    if initializer == 'zeros':
        initializer_fun = tf.zeros(fan_size)
    elif initializer == 'ones':
        initializer_fun = tf.multiply(tf.ones(fan_size), weight_scale)
    elif initializer == 'xavier':
        initializer_fun = xavier_init(fan_size[0], fan_size[1], weight_scale=weight_scale)
    elif initializer == 'normal_in':
        initializer_fun = tf.random_normal(fan_size, mean=0.0, stddev=tf.divide(1.0, tf.sqrt(fan_size[0])),
                                           dtype=tf.float32)
    else:
        raise AttributeError('Initialization method not supported.')

    return tf.get_variable(name=name, initializer=initializer_fun, trainable=trainable)


def create_variable(name, initializer, fan_size, trainable=True, weight_scale=1.0, pin_to_cpu=True):
    """ This function pins variables to cpu

    tf.get_variable is used instead of tf.Variable, so that variable with the same name will not be re-initialized
    """
    if pin_to_cpu:
        with tf.device('/cpu:0'):
            var = _create_variable_(name, initializer, fan_size, trainable, weight_scale)
    else:
        var = _create_variable_(name, initializer, fan_size, trainable, weight_scale)

    return var


def xavier_init(fan_in, fan_out, weight_scale=1.0):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    high = tf.multiply(weight_scale, tf.sqrt(6.0 / tf.add(fan_in, fan_out)))

    return tf.random_uniform((fan_in, fan_out), minval=tf.negative(high), maxval=high, dtype=tf.float32)


class SynTower(object):
    def __init__(self):
        """ This class contains several methods to synchronize variables across towers

        """
        pass

    @staticmethod
    def average_grads(tower_grads):
        """ This function averages the tower_grads

        :param tower_grads: the list of gradients calculated from opt_op.compute_gradient
        :return:
        """
        average_grads_list = []
        for grad_var_tuple in zip(*tower_grads):
            # a = zip(x,y) aggregates elements from each list into a = [(x0,y0),(x1,y1),...]
            # b = zip(*a) unzip tuples in a. Let a2=list(a), b2=list(b), then b2[i][j]=a2[j][i]
            # zip objects can only be unpack once using list(), tuple(), etc
            # grad_var_tuple takes the form: ((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1))
            grads = []
            for g, _ in grad_var_tuple:
                # first add one more dimension to g in the first dimension,
                # so that later we can add grads along the first dimension, aka, the tower dimension
                grads.append(tf.expand_dims(g, 0))  # grads now is a list of vectors
            # concatenate grads along the first dimension so that it will become a matrix
            grad_matrix = tf.concat(values=grads, axis=0)
            # average grad
            grad = tf.reduce_mean(grad_matrix, 0)
            # get the name for this grad
            var_name = grad_var_tuple[0][1]
            # append the results
            average_grads_list.append((grad, var_name))
        return average_grads_list

    @staticmethod
    def average_var(tower_var):
        """ This function averages the tower_var

        :param tower_var: a zip of list of variables, zip([[a0, b0, c0, ...], [a1, b1, c1, ...]])
        :return:
        """
        average_var_list = []
        for var_tuple in zip(*tower_var):
            # extract var from var_tuple
            _vars = []
            for v in var_tuple:
                _vars.append(tf.expand_dims(v, 0))
            var_matrix = tf.concat(values=_vars, axis=0)
            # average var
            var_mean = tf.reduce_mean(var_matrix, 0)
            # append the results
            average_var_list.append(var_mean)
        return average_var_list

    @staticmethod
    def stack_var(tower_var, axis=0):
        """ This function stacks the tower_var

        :param tower_var:
        :param axis:
        :return:
        """
        stack_var_list = []
        for var_tuple in zip(*tower_var):
            # extract var from var_tuple
            var_matrix = tf.concat(values=list(var_tuple), axis=axis)
            # append the results
            stack_var_list.append(var_matrix)
        return stack_var_list


def average_tower_grads(tower_grads):
    """ This function averages the tower_grads

    Inputs:
    tower_grads - a list of lists of (gradient, variable) tuples
    """
    average_grads = []
    for grad_var_tuple in zip(*tower_grads):
        # a = zip(x,y) aggregates elements from each list into a = [(x0,y0),(x1,y1),...]
        # b = zip(*a) unzip tuples in a. Let a2=list(a), b2=list(b), then b2[i][j]=a2[j][i]
        # zip objects can only be unpack once using list(), tuple(), etc
        # grad_var_tuple takes the form: ((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1))
        grads = []
        for g, _ in grad_var_tuple:
            # first add one more dimension to g in the first dimension,
            # so that later we can add grads along the first dimension, aka, the tower dimension
            grads.append(tf.expand_dims(g, 0))  # grads now is a list of vectors
        # concatenate grads along the first dimension so that it will become a matrix
        grad = tf.concat(values=grads, axis=0)
        # average grad
        grad = tf.reduce_mean(grad, 0)
        # get the name for this grad
        var_name = grad_var_tuple[0][1]
        # append the results
        average_grads.append((grad, var_name))
    return average_grads


def prepare_folder(filename, sub_folder='', set_folder=True):
    """ This function prepares the folders

    :param filename:
    :param sub_folder:
    :param set_folder:
    :return:
    """
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]

    ckpt_folder = os.path.join(FLAGS.DEFAULT_OUT, filename + '_ckpt', sub_folder)
    if not os.path.exists(ckpt_folder) and set_folder:
        os.makedirs(ckpt_folder)
    summary_folder = os.path.join(FLAGS.DEFAULT_OUT, filename + '_log', sub_folder)
    if not os.path.exists(summary_folder) and set_folder:
        os.makedirs(summary_folder)
    save_path = os.path.join(ckpt_folder, filename + '.ckpt')

    return ckpt_folder, summary_folder, save_path


def prepare_embedding_folder(summary_folder, filename, file_index=''):
    """ This function prepares the files for embedding

    :param summary_folder:
    :param filename:
    :param file_index:
    :return:
    """
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]

    embedding_path = os.path.join(summary_folder, filename + file_index + '_embedding.ckpt')
    label_path = os.path.join(summary_folder, filename + file_index + '_label.tsv')
    sprite_path = os.path.join(summary_folder, filename + file_index + '.png')

    return embedding_path, label_path, sprite_path


def write_metadata(label_path, labels, names=None):
    """ This function writes raw_labels to file for embedding

    :param label_path: file name, e.g. '...\\metadata.tsv'
    :param labels: raw_labels
    :param names: interpretation for raw_labels, e.g. ['plane','auto','bird','cat']
    :return:
    """
    metadata_file = open(label_path, 'w')
    metadata_file.write('Name\tClass\n')
    if names is None:
        i = 0
        for label in labels:
            metadata_file.write('%06d\t%s\n' % (i, str(label)))
            i = i + 1
    else:
        for label in labels:
            metadata_file.write(names[label])
    metadata_file.close()


def write_sprite(sprite_path, images, mesh_num=None, if_invert=False):
    """ This function writes images to sprite image for embedding

    This function was taken from:
    https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/debugging/embedding.ipynb

    The input image must be channels_last format.

    :param sprite_path: file name, e.g. '...\\a_sprite.png'
    :param images: ndarray, [batch_size, height, width(, channels)], values in range [0,1]
    :param if_invert: bool, if true, invert images: images = 1 - images
    :param mesh_num: nums of images in the row and column, must be a tuple
    :return:
    """
    if len(images.shape) == 3:  # if dimension of image is 3, extend it to 4
        images = np.tile(images[..., np.newaxis], (1, 1, 1, 3))
    if images.shape[3] == 1:  # if last dimension is 1, extend it to 3
        images = np.tile(images, (1, 1, 1, 3))
    # scale image to range [0,1]
    images = images.astype(np.float32)
    image_min = np.min(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose((1, 2, 3, 0)) - image_min).transpose((3, 0, 1, 2))
    image_max = np.max(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose((1, 2, 3, 0)) / image_max).transpose((3, 0, 1, 2))
    if if_invert:
        images = 1 - images
    # check mesh_num
    if mesh_num is None:
        print('Mesh_num will be calculated as sqrt of batch_size')
        batch_size = images.shape[0]
        sprite_size = int(np.ceil(np.sqrt(batch_size)))
        mesh_num = (sprite_size, sprite_size)
        # add paddings if batch_size is not the square of sprite_size
        padding = ((0, sprite_size ** 2 - batch_size), (0, 0), (0, 0)) + ((0, 0),) * (images.ndim - 3)
        images = np.pad(images, padding, mode='constant', constant_values=0)
    elif isinstance(mesh_num, list):
        mesh_num = tuple(mesh_num)
    # Tile the individual thumbnails into an image
    new_shape = mesh_num + images.shape[1:]
    images = images.reshape(new_shape).transpose((0, 2, 1, 3) + tuple(range(4, images.ndim + 1)))
    images = images.reshape((mesh_num[0] * images.shape[1], mesh_num[1] * images.shape[3]) + images.shape[4:])
    images = (images * 255).astype(np.uint8)
    # save images to file
    from scipy.misc import imsave
    imsave(sprite_path, images)


def write_sprite_wrapper(
        images, mesh_num, filename, file_folder=None, file_index='',
        if_invert=False, image_format='channels_last'):
    """ This is a wrapper function for write_sprite.

    :param images: ndarray, [batch_size, height, width(, channels)], values in range [0,1]
    :param mesh_num: mus tbe tuple (row_mesh, column_mesh)
    :param filename:
    :param file_folder:
    :param file_index:
    :param if_invert: bool, if true, invert images: images = 1 - images
    :param image_format: the default is channels_last; if channels_first is provided, transpose will be done.
    :return:
    """
    # check inputs
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]
    if isinstance(mesh_num, list):
        mesh_num = tuple(mesh_num)
    if file_folder is None:
        file_folder = FLAGS.DEFAULT_OUT
    if image_format in {'channels_first', 'NCHW'}:  # convert to [batch_size, height, width, channels]
        images = np.transpose(images, axes=(0, 2, 3, 1))
    # set up file location
    sprite_path = os.path.join(file_folder, filename + file_index + '.png')
    # write to files
    if os.path.isfile(sprite_path):
        print('This file already exists: ' + sprite_path)
    else:
        write_sprite(sprite_path, images, mesh_num=mesh_num, if_invert=if_invert)


def embedding_latent_code(
        latent_code, file_folder, embedding_path, var_name='codes',
        label_path=None, sprite_path=None, image_size=None):
    """ This function visualize latent_code using tSNE or PCA. The results can be viewed
    on tensorboard.

    :param latent_code: 2-D data
    :param file_folder:
    :param embedding_path:
    :param var_name:
    :param label_path:
    :param sprite_path:
    :param image_size:
    :return:
    """
    # register a session
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    # prepare a embedding variable
    # note this must be a variable, not a tensor
    embedding_var = tf.Variable(latent_code, name=var_name)
    sess.run(embedding_var.initializer)

    # configure the embedding
    from tensorflow.contrib.tensorboard.plugins import projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # add metadata (label) to embedding; comment out if no metadata
    if label_path is not None:
        embedding.metadata_path = label_path
    # add sprite image to embedding; comment out if no sprites
    if sprite_path is not None:
        embedding.sprite.image_path = sprite_path
        embedding.sprite.single_image_dim.extend(image_size)
    # finalize embedding setting
    embedding_writer = tf.summary.FileWriter(file_folder)
    projector.visualize_embeddings(embedding_writer, config)
    embedding_saver = tf.train.Saver([embedding_var], max_to_keep=1)
    embedding_saver.save(sess, embedding_path)
    # close all
    sess.close()


def embedding_image_wrapper(
        latent_code, filename, var_name='codes', file_folder=None, file_index='',
        labels=None, images=None, mesh_num=None, if_invert=False, image_format='channels_last'):
    """ This function is a wrapper function for embedding_image

    :param latent_code:
    :param filename:
    :param var_name:
    :param file_folder:
    :param file_index:
    :param labels:
    :param images: ndarray, [batch_size, height, width(, channels)], values in range [0,1]
    :param mesh_num:
    :param if_invert:
    :param image_format: the default is channels_last; if channels_first is provided, transpose will be done.
    :return:
    """
    # check inputs
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]
    if file_folder is None:
        file_folder = FLAGS.DEFAULT_OUT
    # prepare folder
    embedding_path, label_path, sprite_path = prepare_embedding_folder(file_folder, filename, file_index)
    # write label to file if labels are given
    if labels is not None:
        if os.path.isfile(label_path):
            print('Label file already exist.')
        else:
            write_metadata(label_path, labels)
    else:
        label_path = None
    # write images to file if images are given
    if images is not None:
        # if image is in channels_first format, convert to channels_last
        if image_format == 'channels_first':
            images = np.transpose(images, axes=(0, 2, 3, 1))
        image_size = images.shape[1:3]  # [height, width]
        if os.path.isfile(sprite_path):
            print('Sprite file already exist.')
        else:
            write_sprite(sprite_path, images, mesh_num=mesh_num, if_invert=if_invert)
    else:
        image_size = None
        sprite_path = None
    if os.path.isfile(embedding_path):
        print('Embedding file already exist.')
    else:
        embedding_latent_code(
            latent_code, file_folder, embedding_path, var_name=var_name,
            label_path=label_path, sprite_path=sprite_path, image_size=image_size)


def get_ckpt(ckpt_folder, ckpt_file=None):
    """ This function gets the ckpt states. In case an older ckpt file is needed, provide the name in ckpt_file

    :param ckpt_folder:
    :param ckpt_file:
    :return:
    """
    ckpt = tf.train.get_checkpoint_state(ckpt_folder)
    if ckpt_file is None:
        return ckpt
    else:
        index_file = os.path.join(ckpt_folder, ckpt_file+'.index')
        if os.path.isfile(index_file):
            ckpt.model_checkpoint_path = os.path.join(ckpt_folder, ckpt_file)
        else:
            raise FileExistsError('{} does not exist.'.format(index_file))

        return ckpt


def print_tensor_in_ckpt(ckpt_folder, all_tensor_values=False, all_tensor_names=False):
    """ This function print the list of tensors in checkpoint file.

    Example:
    from GeneralTools.graph_func import print_tensor_in_ckpt
    ckpt_folder = '/home/richard/PycharmProjects/myNN/Results/cifar_ckpt/sngan_hinge_2e-4_nl'
    print_tensor_in_ckpt(ckpt_folder)

    :param ckpt_folder:
    :param all_tensor_values: Boolean indicating whether to print the values of all tensors.
    :param all_tensor_names: Boolean indicating whether to print all tensor names.
    :return:
    """
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

    if not isinstance(ckpt_folder, str):  # if list, use the name of the first file
        ckpt_folder = ckpt_folder[0]

    output_folder = os.path.join(FLAGS.DEFAULT_OUT, ckpt_folder)
    print(output_folder)
    ckpt = tf.train.get_checkpoint_state(output_folder)
    print(ckpt)
    print_tensors_in_checkpoint_file(
        file_name=ckpt.model_checkpoint_path, tensor_name='',
        all_tensors=all_tensor_values, all_tensor_names=all_tensor_names)


def graph_configure(
        initial_lr, global_step_name='global_step', lr_decay_steps=None,
        end_lr=1e-7, optimizer='adam'):
    """ This function configures global_step and optimizer

    :param initial_lr:
    :param global_step_name:
    :param lr_decay_steps:
    :param end_lr:
    :param optimizer:
    :return:
    """
    global_step = global_step_config(name=global_step_name)
    learning_rate, opt_op = opt_config(initial_lr, lr_decay_steps, end_lr, optimizer, global_step)

    return global_step, learning_rate, opt_op


def global_step_config(name='global_step'):
    """ This function is a wrapper for global step

    """
    global_step = tf.get_variable(
        name=name,
        shape=[],
        dtype=tf.int32,
        initializer=tf.constant_initializer(0),
        trainable=False)

    return global_step


def opt_config(
        initial_lr, lr_decay_steps=None, end_lr=1e-7,
        optimizer='adam', name_suffix='', global_step=None, target_step=1e5):
    """ This function configures optimizer.

    :param initial_lr:
    :param lr_decay_steps:
    :param end_lr:
    :param optimizer:
    :param name_suffix:
    :param global_step:
    :param target_step:
    :return:
    """
    if optimizer in ['SGD', 'sgd']:
        # sgd
        if lr_decay_steps is None:
            lr_decay_steps = np.round(target_step * np.log(0.96) / np.log(end_lr / initial_lr)).astype(np.int32)
        learning_rate = tf.train.exponential_decay(  # adaptive learning rate
            initial_lr,
            global_step=global_step,
            decay_steps=lr_decay_steps,
            decay_rate=0.96,
            staircase=False)
        opt_op = tf.train.GradientDescentOptimizer(
            learning_rate, name='GradientDescent'+name_suffix)
        print('GradientDescent Optimizer is used.')
    elif optimizer in ['Momentum', 'momentum']:
        # momentum
        if lr_decay_steps is None:
            lr_decay_steps = np.round(target_step * np.log(0.96) / np.log(end_lr / initial_lr)).astype(np.int32)
        learning_rate = tf.train.exponential_decay(  # adaptive learning rate
            initial_lr,
            global_step=global_step,
            decay_steps=lr_decay_steps,
            decay_rate=0.96,
            staircase=False)
        opt_op = tf.train.MomentumOptimizer(
            learning_rate, momentum=0.9, name='Momentum'+name_suffix)
        print('Momentum Optimizer is used.')
    elif optimizer in ['Adam', 'adam']:  # adam
        # Occasionally, adam optimizer may cause the objective to become nan in the first few steps
        # This is because at initialization, the gradients may be very big. Setting beta1 and beta2
        # close to 1 may prevent this.
        learning_rate = tf.constant(initial_lr)
        # opt_op = tf.train.AdamOptimizer(
        #     learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-8, name='Adam'+name_suffix)
        opt_op = tf.train.AdamOptimizer(
            learning_rate, beta1=0.5, beta2=0.999, epsilon=1e-8, name='Adam' + name_suffix)
        print('Adam Optimizer is used.')
    elif optimizer in ['RMSProp', 'rmsprop']:
        # RMSProp
        learning_rate = tf.constant(initial_lr)
        opt_op = tf.train.RMSPropOptimizer(
            learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, name='RMSProp'+name_suffix)
        print('RMSProp Optimizer is used.')
    else:
        raise AttributeError('Optimizer {} not supported.'.format(optimizer))

    return learning_rate, opt_op


def multi_opt_config(
        lr_list, lr_decay_steps=None, end_lr=1e-7,
        optimizer='adam', global_step=None, target_step=1e5):
    """ This function configures multiple optimizer

    :param lr_list: a list, e.g. [1e-4, 1e-3]
    :param lr_decay_steps:
    :param end_lr:
    :param optimizer: a string, or a list same len as lr_multiplier
    :param global_step:
    :param target_step:
    :return:
    """
    num_opt = len(lr_list)
    if isinstance(optimizer, str):
        optimizer = [optimizer]
    # if one lr_multiplier is provided, configure one op
    # in this case, multi_opt_config is the same as opt_config
    if num_opt == 1:
        learning_rate, opt_op = opt_config(
            lr_list[0], lr_decay_steps, end_lr,
            optimizer[0], '', global_step, target_step)
    else:
        if len(optimizer) == 1:  # match the length of lr_multiplier
            optimizer = optimizer*num_opt
        # get a list of (lr, opt_op) tuple
        lr_opt_combo = [
            opt_config(
                lr_list[i], lr_decay_steps, end_lr,
                optimizer[i], '_'+str(i), global_step, target_step)
            for i in range(num_opt)]
        # separate lr and opt_op
        learning_rate = [lr_opt[0] for lr_opt in lr_opt_combo]
        opt_op = [lr_opt[1] for lr_opt in lr_opt_combo]

    return learning_rate, opt_op


class TimeLiner:
    def __init__(self):
        """ This class creates a timeline object that can be used to trace the timeline of
            multiple steps when called at each step.

        """
        self._timeline_dict = None

    ###################################################################
    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    ###################################################################
    def save(self, trace_file):
        with open(trace_file, 'w') as f:
            json.dump(self._timeline_dict, f)


def rollback(var_list, ckpt_folder, ckpt_file=None):
    """ This function provides a shortcut for reloading a model and calculating a list of variables

    :param var_list:
    :param ckpt_folder:
    :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
    :return:
    """
    global_step = global_step_config()
    # register a session
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    # initialization
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # load the training graph
    saver = tf.train.Saver(max_to_keep=2)
    ckpt = get_ckpt(ckpt_folder, ckpt_file=ckpt_file)
    if ckpt is None:
        raise FileNotFoundError('No ckpt Model found at {}.'.format(ckpt_folder))
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Model reloaded.')
    # run the session
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    var_value, global_step_value = sess.run([var_list, global_step])
    coord.request_stop()
    coord.join(threads)
    sess.close()
    print('Variable calculated.')

    return var_value, global_step_value


class MySession(object):
    def __init__(
            self, do_save=False, do_trace=False, save_path=None,
            load_ckpt=False, log_device=False, ckpt_var_list=None):
        """ This class provides shortcuts for running sessions.
        It needs to be run under context managers. Example:
        with MySession() as sess:
            var1_value, var2_value = sess.run_once([var1, var2])

        :param do_save:
        :param do_trace:
        :param save_path:
        :param load_ckpt:
        :param log_device:
        :param ckpt_var_list: list of variables to save / restore
        """
        # somehow it gives error: "global_step does not exist or is not created from tf.get_variable".
        # self.global_step = global_step_config()
        self.log_device = log_device
        # register a session
        # self.sess = tf.Session(config=tf.ConfigProto(
        #     allow_soft_placement=True,
        #     log_device_placement=log_device,
        #     gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device))
        # initialization
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.coord = None
        self.threads = None
        print('Graph initialization finished...')
        # configuration
        self.ckpt_var_list = ckpt_var_list
        if do_save:
            self.saver = self._get_saver_()
            self.save_path = save_path
        else:
            self.saver = None
            self.save_path = None
        self.summary_writer = None
        self.do_trace = do_trace
        self.load_ckpt = load_ckpt

    def __enter__(self):
        """ The enter method is called when "with" statement is used.

        :return:
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ The exit method is called when leaving the scope of "with" statement

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        print('Session finished.')
        if self.summary_writer is not None:
            self.summary_writer.close()
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()

    def _get_saver_(self):
        # create a saver to save all variables
        # Saver op should always be assigned to cpu, and it should be
        # created after all variables have been defined; otherwise, it
        # only save those variables already created.
        with tf.device('/cpu:0'):
            if self.ckpt_var_list is None:
                return tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
            else:
                return tf.train.Saver(var_list=self.ckpt_var_list, max_to_keep=2)

    def _load_ckpt_(self, ckpt_folder=None, ckpt_file=None):
        """ This function loads a checkpoint model

        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :return:
        """
        if self.load_ckpt and (ckpt_folder is not None):
            ckpt = get_ckpt(ckpt_folder, ckpt_file=ckpt_file)
            if ckpt is None:
                print('No ckpt Model found at {}. Model training from scratch.'.format(ckpt_folder))
            else:
                if self.saver is None:
                    self.saver = self._get_saver_()
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print('Model reloaded.')
        else:
            print('No ckpt model is loaded for current calculation.')

    def _check_thread_(self):
        """ This function initializes the coordinator and threads

        :return:
        """
        if self.threads is None:
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def run_once(self, var_list, ckpt_folder=None, ckpt_file=None, ckpt_var_list=None, feed_dict=None):
        """ This functions calculates var_list.

        :param var_list:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param ckpt_var_list: the variable to load in order to calculate var_list
        :param feed_dict:
        :return:
        """
        if ckpt_var_list is not None:
            self.ckpt_var_list = ckpt_var_list
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file)
        self._check_thread_()
        # var_value, global_step_value = self.sess.run([var_list, self.global_step])
        #
        # return var_value, global_step_value

        var_value = self.sess.run(var_list, feed_dict=feed_dict)

        return var_value

    def run(self, *args, **kwargs):
        return self.run_once(*args, **kwargs)

    def run_m_times(
            self, var_list, ckpt_folder=None, ckpt_file=None,
            max_iter=10000, trace=False, ckpt_var_list=None, feed_dict=None):
        """ This functions calculates var_list for multiple iterations, as often done in
        Monte Carlo analysis.

        :param var_list:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param max_iter:
        :param trace: if True, keep all outputs of m iterations
        :param ckpt_var_list: the variable to load in order to calculate var_list
        :param feed_dict:
        :return:
        """
        if ckpt_var_list is not None:
            self.ckpt_var_list = ckpt_var_list
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file)
        self._check_thread_()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        start_time = time.time()
        if trace:
            var_value_list = []
            for i in range(max_iter):
                var_value, _ = self.sess.run([var_list, extra_update_ops], feed_dict=feed_dict)
                var_value_list.append(var_value)
        else:
            for i in range(max_iter - 1):
                _, _ = self.sess.run([var_list, extra_update_ops], feed_dict=feed_dict)
            var_value_list, _ = self.sess.run([var_list, extra_update_ops], feed_dict=feed_dict)
        # global_step_value = self.sess.run([self.global_step])
        print('Calculation took {:.3f} sec.'.format(time.time() - start_time))
        return var_value_list

    @staticmethod
    def print_loss(loss_value, step=0, epoch=0):
        print('Epoch {}, global steps {}, loss_list {}'.format(
            epoch, step,
            ['{}'.format(['<{:.2f}>'.format(l_val) for l_val in l_value])
             if isinstance(l_value, (np.ndarray, list))
             else '<{:.3f}>'.format(l_value)
             for l_value in loss_value]))

    def full_run(self, op_list, loss_list, max_step, step_per_epoch, global_step, summary_op=None,
                 summary_image_op=None, summary_folder=None, ckpt_folder=None, ckpt_file=None, print_loss=True,
                 query_step=500, imbalanced_update=None):
        """ This function run the session with all monitor functions.

        :param op_list: the first op in op_list runs every extra_steps when the rest run once.
        :param loss_list: the first loss is used to monitor the convergence
        :param max_step:
        :param step_per_epoch:
        :param global_step:
        :param summary_op:
        :param summary_image_op:
        :param summary_folder:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param print_loss:
        :param query_step:
        :param imbalanced_update: a list indicating the period to update each ops in op_list;
            the first op must have period = 1 as it updates the global step
        :return:
        """
        # prepare writer
        if (summary_op is not None) or (summary_image_op is not None):
            self.summary_writer = tf.summary.FileWriter(summary_folder, self.sess.graph)
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file)
        # run the session
        self._check_thread_()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        start_time = time.time()
        if imbalanced_update is None:
            for step in range(max_step):
                # update the model
                loss_value, _, _, global_step_value = self.sess.run(
                    [loss_list, op_list, extra_update_ops, global_step])
                # check if model produces nan outcome
                assert not np.isnan(loss_value[0]), \
                    'Model diverged with loss = {} at step {}'.format(loss_value, step)

                # add summary and print loss every query step
                if global_step_value % query_step == (query_step-1):
                    if summary_op is not None:
                        summary_str = self.sess.run(summary_op)
                        self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                    if print_loss:
                        epoch = step // step_per_epoch
                        self.print_loss(loss_value, global_step_value, epoch)

                # save model at last step
                if step == max_step - 1:
                    if self.saver is not None:
                        self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
                    if summary_image_op is not None:
                        summary_image_str = self.sess.run(summary_image_op)
                        self.summary_writer.add_summary(summary_image_str, global_step=global_step_value)

        elif isinstance(imbalanced_update, (list, tuple)):
            num_ops = len(op_list)
            assert len(imbalanced_update) == num_ops, 'Imbalanced_update length does not match ' \
                                                      'that of op_list. Expected {} got {}.'.format(
                num_ops, len(imbalanced_update))

            for step in range(max_step):
                # get update ops
                global_step_value = self.sess.run(global_step)
                update_ops = [op_list[i] for i in range(num_ops) if global_step_value % imbalanced_update[i] == 0]

                # update the model
                loss_value, _, _, global_step_value = self.sess.run([loss_list, update_ops, extra_update_ops])
                # check if model produces nan outcome
                assert not np.isnan(loss_value[0]), \
                    'Model diverged with loss = {} at step {}'.format(loss_value, step)

                # add summary and print loss every query step
                if global_step_value % query_step == (query_step - 1):
                    if summary_op is not None:
                        summary_str = self.sess.run(summary_op)
                        self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                    if print_loss:
                        epoch = step // step_per_epoch
                        self.print_loss(loss_value, global_step_value, epoch)

                # save model at last step
                if step == max_step - 1:
                    if self.saver is not None:
                        self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
                    if summary_image_op is not None:
                        summary_image_str = self.sess.run(summary_image_op)
                        self.summary_writer.add_summary(summary_image_str, global_step=global_step_value)

        elif imbalanced_update == 'dynamic':
            # This case is used for sngan_mmd_rand_g only
            mmd_average = 0.0
            for step in range(max_step):
                # get update ops
                global_step_value = self.sess.run(global_step)
                update_ops = op_list if \
                    global_step_value < 1000 or \
                    np.random.uniform(low=0.0, high=1.0) < 0.1 / np.maximum(mmd_average, 0.1) else \
                    op_list[1:]

                # update the model
                loss_value, _, _, global_step_value = self.sess.run([loss_list, update_ops, extra_update_ops])
                # check if model produces nan outcome
                assert not np.isnan(loss_value[0]), \
                    'Model diverged with loss = {} at step {}'.format(loss_value, step)

                # add summary and print loss every query step
                if global_step_value % query_step == (query_step - 1):
                    if summary_op is not None:
                        summary_str = self.sess.run(summary_op)
                        self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                    if print_loss:
                        epoch = step // step_per_epoch
                        self.print_loss(loss_value, global_step_value, epoch)

                # save model at last step
                if step == max_step - 1:
                    if self.saver is not None:
                        self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
                    if summary_image_op is not None:
                        summary_image_str = self.sess.run(summary_image_op)
                        self.summary_writer.add_summary(summary_image_str, global_step=global_step_value)

        # calculate sess duration
        duration = time.time() - start_time
        print('Training for {} steps took {:.3f} sec.'.format(max_step, duration))

    def debug_mode(self, op_list, loss_list, global_step, summary_op=None, summary_folder=None, ckpt_folder=None,
                   ckpt_file=None, max_step=200, print_loss=True, query_step=100, imbalanced_update=None):
        """ This function do tracing to debug the code. It will burn-in for 25 steps, then record
        the usage every 5 steps for 5 times.

        :param op_list:
        :param loss_list:
        :param global_step:
        :param summary_op:
        :param summary_folder:
        :param max_step:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param print_loss:
        :param query_step:
        :param imbalanced_update: a list indicating the period to update each ops in op_list;
            the first op must have period = 1 as it updates the global step
        :return:
        """
        if self.do_trace or (summary_op is not None):
            self.summary_writer = tf.summary.FileWriter(summary_folder, self.sess.graph)
        if self.do_trace:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            multi_runs_timeline = TimeLiner()
        else:
            run_options = None
            run_metadata = None
            multi_runs_timeline = None

        # run the session
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file)
        self._check_thread_()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(extra_update_ops)
        start_time = time.time()
        if imbalanced_update is None:
            for step in range(max_step):
                if self.do_trace and (step >= max_step - 5):
                    # update the model in trace mode
                    loss_value, _, global_step_value, _ = self.sess.run(
                        [loss_list, op_list, global_step, extra_update_ops],
                        options=run_options, run_metadata=run_metadata)
                    # add time line
                    self.summary_writer.add_run_metadata(
                        run_metadata, tag='step%d' % global_step_value, global_step=global_step_value)
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    chrome_trace = trace.generate_chrome_trace_format()
                    multi_runs_timeline.update_timeline(chrome_trace)
                else:
                    # update the model
                    loss_value, _, global_step_value, _ = self.sess.run(
                        [loss_list, op_list, global_step, extra_update_ops])
                # check if model produces nan outcome
                assert not np.isnan(loss_value[0]), \
                    'Model diverged with loss = {} at step {}'.format(loss_value, step)

                # print(loss_value) and add summary
                if global_step_value % query_step == 0:
                    if print_loss:
                        self.print_loss(loss_value, global_step_value)
                    if summary_op is not None:
                        summary_str = self.sess.run(summary_op)
                        self.summary_writer.add_summary(summary_str, global_step=global_step_value)

                # conditional save, used in abnormal case
                if any([lv > 30000 for lv in loss_value]) and (step > 400):
                    # save the model
                    self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
                    # add summary
                    summary_str = self.sess.run(summary_op)
                    self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                    print('Training Stopped early as loss diverged.')
                    break

                # save the mdl if for loop completes normally
                if step == max_step - 1 and self.saver is not None:
                    self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)

        elif isinstance(imbalanced_update, (list, tuple)):
            num_ops = len(op_list)
            assert len(imbalanced_update) == num_ops, 'Imbalanced_update length does not match ' \
                                                      'that of op_list. Expected {} got {}.'.format(
                num_ops, len(imbalanced_update))
            for step in range(max_step):
                # get update ops
                global_step_value = self.sess.run(global_step)
                update_ops = [op_list[i] for i in range(num_ops) if global_step_value % imbalanced_update[i] == 0]

                if self.do_trace and (step >= max_step - 5):
                    # update the model in trace mode
                    loss_value, _, _ = self.sess.run(
                        [loss_list, update_ops, extra_update_ops],
                        options=run_options, run_metadata=run_metadata)
                    # add time line
                    self.summary_writer.add_run_metadata(
                        run_metadata, tag='step%d' % global_step_value, global_step=global_step_value)
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    chrome_trace = trace.generate_chrome_trace_format()
                    multi_runs_timeline.update_timeline(chrome_trace)
                else:
                    # update the model
                    loss_value, _, _ = self.sess.run([loss_list, update_ops, extra_update_ops])
                # check if model produces nan outcome
                assert not np.isnan(loss_value[0]), \
                    'Model diverged with loss = {} at step {}'.format(loss_value, step)

                # print(loss_value)
                if print_loss and (step % query_step == 0):
                    self.print_loss(loss_value, global_step_value)

                # add summary
                if (summary_op is not None) and (global_step_value % 100 == 99):
                    summary_str = self.sess.run(summary_op)
                    # add summary and print out loss
                    self.summary_writer.add_summary(summary_str, global_step=global_step_value)

                # conditional save, used in abnormal case
                if any([lv > 30000 for lv in loss_value]) and (step > 400):
                    self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
                    # add summary
                    summary_str = self.sess.run(summary_op)
                    self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                    print('Training Stopped early as loss diverged.')
                    break

                if step == max_step - 1 and self.saver is not None:
                    self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)

        elif isinstance(imbalanced_update, str) and imbalanced_update == 'dynamic':
            # This case is used for sngan_mmd_rand_g only
            mmd_average = 0.0
            for step in range(max_step):
                # get update ops
                global_step_value = self.sess.run(global_step)
                update_ops = op_list if \
                    global_step_value < 1000 or \
                    np.random.uniform(low=0.0, high=1.0) < 0.1 / np.maximum(mmd_average, 0.1) else \
                    op_list[1:]

                if self.do_trace and (step >= max_step - 5):
                    # update the model in trace mode
                    loss_value, _, _ = self.sess.run(
                        [loss_list, update_ops, extra_update_ops],
                        options=run_options, run_metadata=run_metadata)
                    # add time line
                    self.summary_writer.add_run_metadata(
                        run_metadata, tag='step%d' % global_step_value, global_step=global_step_value)
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    chrome_trace = trace.generate_chrome_trace_format()
                    multi_runs_timeline.update_timeline(chrome_trace)
                else:
                    # update the model
                    loss_value, _, _ = self.sess.run([loss_list, update_ops, extra_update_ops])
                # check if model produces nan outcome
                assert not np.isnan(loss_value[0]), \
                    'Model diverged with loss = {} at step {}'.format(loss_value, step)

                # update mmd_average
                mmd_average = loss_value[2]

                # print(loss_value)
                if print_loss and (step % query_step == 0):
                    self.print_loss(loss_value, global_step_value)

                # add summary
                if (summary_op is not None) and (global_step_value % 100 == 99):
                    summary_str = self.sess.run(summary_op)
                    # add summary and print out loss
                    self.summary_writer.add_summary(summary_str, global_step=global_step_value)

                if step == max_step - 1 and self.saver is not None:
                    self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)

        # calculate sess duration
        duration = time.time() - start_time
        print('Training for {} steps took {:.3f} sec.'.format(max_step, duration))
        # save tracing file
        if self.do_trace:
            trace_file = os.path.join(summary_folder, 'timeline.json')
            multi_runs_timeline.save(trace_file)


class Agent(object):
    def __init__(
            self, filename, sub_folder, load_ckpt=False, do_trace=False,
            do_save=True, debug_mode=False, debug_step=800, query_step=500,
            log_device=False, imbalanced_update=None, print_loss=True):
        """ Agent is a wrapper for the MySession class, used for training and evaluating complex model

        :param filename:
        :param sub_folder:
        :param load_ckpt:
        :param do_trace:
        :param do_save:
        :param debug_mode:
        :param log_device:
        :param query_step:
        :param imbalanced_update:
        """
        self.ckpt_folder, self.summary_folder, self.save_path = prepare_folder(filename, sub_folder=sub_folder)
        self.load_ckpt = load_ckpt
        self.do_trace = do_trace
        self.do_save = do_save
        self.debug = debug_mode
        self.debug_step = debug_step
        self.log_device = log_device
        self.query_step = query_step
        self.imbalanced_update = imbalanced_update
        self.print_loss = print_loss

    def train(
            self, op_list, loss_list, global_step, max_step=None, step_per_epoch=None,
            summary_op=None, summary_image_op=None, imbalanced_update=None):
        """ This method do the optimization process to minimizes loss_list

        :param op_list: [net0_op, net1_op, net2_op]
        :param loss_list: [loss0, loss1, loss2]
        :param global_step:
        :param max_step:
        :param step_per_epoch:
        :param summary_op:
        :param summary_image_op:
        :param imbalanced_update:
        :return:
        """
        # Check inputs
        if imbalanced_update is not None:
            self.imbalanced_update = imbalanced_update
        if self.imbalanced_update is not None:
            assert isinstance(self.imbalanced_update, (list, tuple, str)), \
                'Imbalanced_update must be a list, tuple or str.'

        if self.debug is None:
            # sess = tf.Session(config=tf.ConfigProto(
            #     allow_soft_placement=True,
            #     log_device_placement=False))
            writer = tf.summary.FileWriter(logdir=self.summary_folder, graph=tf.get_default_graph())
            writer.flush()
            # graph_protobuf = str(tf.get_default_graph().as_default())
            # with open(os.path.join(self.summary_folder, 'graph'), 'w') as f:
            #     f.write(graph_protobuf)
            print('Graph printed.')
        elif self.debug is True:
            print('Debug mode is on.')
            print('Remember to load ckpt to check variable values.')
            with MySession(self.do_save, self.do_trace, self.save_path, self.load_ckpt, self.log_device) as sess:
                sess.debug_mode(op_list, loss_list, global_step, summary_op, self.summary_folder, self.ckpt_folder,
                                max_step=self.debug_step, print_loss=self.print_loss,
                                imbalanced_update=self.imbalanced_update)
        elif self.debug is False:
            with MySession(self.do_save, self.do_trace, self.save_path, self.load_ckpt) as sess:
                sess.full_run(op_list, loss_list, max_step, step_per_epoch, global_step, summary_op, summary_image_op,
                              self.summary_folder, self.ckpt_folder, print_loss=self.print_loss,
                              query_step=self.query_step, imbalanced_update=self.imbalanced_update)
        else:
            raise AttributeError('Current debug mode is not supported.')


def data2sprite(
        filename, image_size, mesh_num=None, if_invert=False,
        num_threads=6, file_suffix='', image_transpose=False,
        grey_scale=False, separate_channel=False, image_format='channels_last'):
    """ This function reads data and writes them to sprite. Extra outputs are
    greyscaled image and images in each RGB channel

    :param filename: ['celebA_0']
    :param image_size: [height, width, channels]
    :param mesh_num: ()
    :param if_invert:
    :param num_threads:
    :param file_suffix:
    :param image_transpose: for dataset like MNIST, image needs to be transposed
    :param grey_scale: if true, also plot the grey-scaled image
    :param separate_channel: if true, also plot the red
    :param image_format: the default is channels_last; if channels_first is provided, transpose will be done.
    :return:
    """
    # check inputs
    height, width, channels = image_size
    data_dimension = np.prod(image_size, dtype=np.int32)
    if mesh_num is None:
        mesh_num = (10, 10)
    batch_size = np.prod(mesh_num, dtype=np.int32)
    if image_transpose:  # for dataset like MNIST, image needs to be transposed
        perm = [0, 2, 1, 3]
    else:
        perm = None
    if channels == 1:
        grey_scale = False
        separate_channel = False
    # prepare folder
    _, summary_folder, _ = prepare_folder(filename, sub_folder=file_suffix)

    # read data
    training_data = ReadFile(filename, data_dimension, 0, num_threads=num_threads)
    # training_data = PreloadGPU(filename, num_instance, self.D, num_threads=num_threads)
    # convert matrix data to image tensor channels_first or channels_last format and scale them to [-1, 1]
    training_data.shape2image(channels, height, width)

    # build the network graph
    with tf.Graph().as_default():
        # get next batch
        training_data.scheduler(batch_size=batch_size)
        x_batch, _ = training_data.next_batch()
        print('Graph configuration finished...')
        # convert x_batch to channels_last format
        if image_format == 'channels_first':
            x_batch = np.transpose(x_batch, axes=(0, 2, 3, 1))

        # calculate the value of x_batch and grey_scaled image
        if grey_scale:
            x_batch_gs = tf.image.rgb_to_grayscale(x_batch)
            with MySession() as sess:  # loss is a list of tuples
                x_batch_value, x_batch_gs_value = sess.run_once([x_batch, x_batch_gs])
        else:
            with MySession() as sess:  # loss is a list of tuples
                x_batch_value = sess.run_once(x_batch)
            x_batch_gs_value = None

    # for dataset like MNIST, image needs to be transposed
    if image_transpose:
        x_batch_value = np.transpose(x_batch_value, axes=perm)
    # write to files
    write_sprite_wrapper(
        x_batch_value, mesh_num, filename, file_folder=summary_folder,
        file_index='_real', if_invert=if_invert, image_format='channels_last')
    if grey_scale:
        # for dataset like MNIST, image needs to be transposed
        if image_transpose:
            x_batch_gs_value = np.transpose(x_batch_gs_value, axes=perm)
        # write to files
        write_sprite_wrapper(
            x_batch_gs_value, mesh_num, filename, file_folder=summary_folder,
            file_index='_real_greyscale', if_invert=if_invert, image_format='channels_last')
    if separate_channel:
        channel_name = ['_R', '_G', '_B']
        for i in range(channels):
            write_sprite_wrapper(
                x_batch_value[:, :, :, i], mesh_num, filename, file_folder=summary_folder,
                file_index='_real' + channel_name[i], if_invert=if_invert, image_format='channels_last')


class Fig(object):
    """ This class uses following two packages for figure plotting:
        import matplotlib.pyplot as plt
        import plotly as py
    """
    def __init__(self, fig_def=None, sub_mode=False):
        # change default figure setup
        self.dict = {'grid': False, 'title': 'Figure', 'x_label': 'x', 'y_label': 'y'}
        self._reset_fig_def_(fig_def)
        self.sub_mode = sub_mode

        # register plotly just in case
        # py.tools.set_credentials_file(username=FLAGS.PLT_ACC, api_key=FLAGS.PLT_KEY)

    def new_figure(self, *args, **kwargs):
        if not self.sub_mode:
            return plt.figure(*args, **kwargs)

    def new_sub_figure(self, *args, **kwargs):
        if self.sub_mode:
            return plt.subplot(*args, **kwargs)

    def show_figure(self, sub_mode=None):
        if sub_mode is not None:
            self.sub_mode = sub_mode
        if not self.sub_mode:
            plt.show()

    def _reset_fig_def_(self, fig_def):
        if fig_def is not None:
            for key in fig_def:
                self.dict[key] = fig_def[key]

    def _add_figure_labels_(self):
        plt.grid(self.dict['grid'])
        plt.title(self.dict['title'])
        plt.xlabel(self.dict['x_label'])
        plt.ylabel(self.dict['y_label'])

    def hist(self, data_list, bins='auto', fig_def=None):
        """ Histogram plot

        :param data_list:
        :param bins:
        :param fig_def:
        :return:
        """
        # check inputs
        self._reset_fig_def_(fig_def)

        # plot figure
        self.new_figure()
        plt.hist(data_list, bins)
        self._add_figure_labels_()
        # plt.colorbar()
        self.show_figure()

    def hist2d(self, x=None, x0=None, x1=None, bins=10, data_range=None, log_norm=False, fig_def=None):
        """

        :param x: either x or x0, x1 is given
        :param x0:
        :param x1:
        :param bins:
        :param data_range:
        :param log_norm: if log normalization is used
        :param fig_def:
        :return:
        """
        from matplotlib.colors import LogNorm
        # check inputs
        self._reset_fig_def_(fig_def)
        if x is not None:
            x0 = x[:, 0]
            x1 = x[:, 1]
        if data_range is None:
            data_range = [[-1.0, 1.0], [-1.0, 1.0]]
        num_instances = x0.shape[0]
        if num_instances > 200:
            count_min = np.ceil(num_instances/bins/bins*0.05)  # bins under this value will not be displayed
            print('hist2d; counts under {} will be ignored.'.format(count_min))
        else:
            count_min = None

        # plot figure
        self.new_figure()
        if log_norm:
            plt.hist2d(x0, x1, bins, range=data_range, norm=LogNorm(), cmin=count_min)
        else:
            plt.hist2d(x0, x1, bins, range=data_range, cmin=count_min)
        self._add_figure_labels_()
        plt.colorbar()
        self.show_figure()

    def plot(self, y, x=None, fig_def=None):
        """ line plot

        :param y:
        :param x:
        :param fig_def:
        :return:
        """
        # check inputs
        self._reset_fig_def_(fig_def)

        # plot figure
        self.new_figure()
        if x is None:
            plt.plot(y)
        else:
            plt.plot(x, y)
        self._add_figure_labels_()
        self.show_figure()

    def scatter(self, x=None, x0=None, x1=None, fig_def=None):
        """ scatter plot

        :param x: The data is given either as x, a [N, 2] matrix or x0, x1, each a [N] vector
        :param x0:
        :param x1:
        :param fig_def:
        :return:
        """
        # check inputs
        self._reset_fig_def_(fig_def)
        if x is not None:
            x0 = x[:, 0]
            x1 = x[:, 1]

        # plot figure
        self.new_figure()
        plt.scatter(x0, x1)
        self._add_figure_labels_()
        self.show_figure()

    def group_scatter(self, data, labels, fig_def=None):
        """ scatter plot with labels

        :param data: either a tuple (data1, data2, ...) or list, or a matrix
        :param labels: a tuple (label1, label2, ...) or list; its length either matches the length of data tuple or
            the number of rows in data matrix
        :param fig_def:
        :return:
        """
        if isinstance(data, tuple):
            assert isinstance(labels, tuple), 'if data is tuple, label must be tuple'
            assert len(labels) == len(data), \
                'Length not match: len(labels)={} while len(data)={}'.format(len(labels), len(data))
        else:  # data is a numpy array
            unique_labels = tuple(np.unique(labels))
            data_tuple = ()
            for label in unique_labels:
                index = labels == label
                data_tuple = data_tuple + (data[index, :],)
            data = data_tuple
            labels = unique_labels

        # plot
        self._reset_fig_def_(fig_def)
        fig = self.new_figure()
        ax = fig.add_subplot(1, 1, 1)
        for sub_data, label in zip(data, labels):
            x = sub_data[:, 0]
            y = sub_data[:, 1]
            ax.scatter(x, y, label=label)

        self._add_figure_labels_()
        plt.legend(loc=0)
        self.show_figure()

    def text_scatter(self, data, texts, color_labels=None, fig_def=None):
        """ scatter plot with texts

        :param data: either a tuple (data1, data2, ...) or list, or a matrix.
        :param texts: either a tuple (txt1, txt2, ...) or list; its length either matches the length of data tuple or
            the number of rows in data matrix
        :param color_labels: either a tuple (C1, C2, ...) or list; its length either matches the length of data tuple
            or the number of rows in data matrix. If provided, label us used to decide the color of texts.
        :param fig_def:
        :return:
        """
        if isinstance(data, tuple):
            assert isinstance(texts, tuple), 'if data is tuple, label must be tuple'
            assert len(texts) == len(data), \
                'Length not match: len(texts)={} while len(data)={}'.format(len(texts), len(data))
            if color_labels is not None:
                assert isinstance(color_labels, tuple), 'if data is tuple, colors must be tuple'
                assert len(color_labels) == len(data), \
                    'Length not match: len(colors)={} while len(data)={}'.format(len(color_labels), len(data))
        else:  # data is a numpy array. in this case, texts and color_labels must all be numpy array
            if color_labels is None:  # one class
                data = (data,)
                texts = (texts,)
                color_labels = ('k',)
            else:
                unique_colors = tuple(np.unique(color_labels))
                data_tuple = ()
                text_tuple = ()
                for color in unique_colors:
                    index = color_labels == color
                    data_tuple = data_tuple + (data[index, :],)
                    text_tuple = text_tuple + (texts[index])
                data = data_tuple
                texts = text_tuple
                color_labels = unique_colors

        if not isinstance(color_labels[0], str):
            color_temp = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
            color_labels = tuple([color_temp[color % 10] for color in color_labels])

        # plot
        self._reset_fig_def_(fig_def)
        # fig = plt.figure(figsize=(9.6, 7.2))
        fig = self.new_figure(figsize=(9.6, 7.2))
        ax = fig.add_subplot(1, 1, 1)
        for sub_data, sub_texts, color in zip(data, texts, color_labels):
            ax.scatter(sub_data[:, 0], sub_data[:, 1], color='w')
            for datum, text in zip(sub_data, sub_texts):
                # ax.text(datum[0], datum[1], s=text, color=color)
                ax.annotate(text, xy=(datum[0], datum[1]), color=color, ha='center', va='center', size='x-small')

        self._add_figure_labels_()
        self.show_figure()

    def contour(self, z, x=None, y=None, custom_level=False, fig_def=None):
        """ contour plot

        :param z: the contour level, a [d, d] matrix
        :param x:
        :param y:
        :param custom_level:
        :param fig_def:
        :return:
        """
        # check inputs
        self._reset_fig_def_(fig_def)
        # obtain levels
        if custom_level:
            z_max = np.percentile(z, q=99)
            z_min = np.percentile(z, q=1)
            levels = np.linspace(z_min, z_max, 10)
        else:
            levels = None

        # plot figure
        self.new_figure()
        if levels is None:
            if x is None or y is None:
                c_s = plt.contour(z)
            else:
                c_s = plt.contour(x, y, z)
        else:
            if x is None or y is None:
                c_s = plt.contour(z, levels=levels)
            else:
                c_s = plt.contour(x, y, z, levels=levels)
        plt.clabel(c_s, inline=1, fontsize=10)
        self._add_figure_labels_()
        self.show_figure()

    @staticmethod
    def add_line(p1, p2, color='C0'):
        """ This function adds a line to current plot without changing the bound of x or y axis

        The default colors range from 'C0' to 'C9'.

        :param p1:
        :param p2:
        :param color
        :return:
        """
        import matplotlib.lines as ml

        ax = plt.gca()
        xl, xu = ax.get_xbound()

        if p2[0] == p1[0]:
            xl = xu = p1[0]
            yl, yu = ax.get_ybound()
        else:
            yu = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xu - p1[0])
            yl = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xl - p1[0])

        line = ml.Line2D([xl, xu], [yl, yu], color=color)
        ax.add_line(line)

        return line


def print_pb_to_event(model_path, event_folder):
    """ This function print a pre-trained model to event_folder so that it can be viewed by tensorboard

    :param model_path: for example, FLAGS.INCEPTION_V3
    :param event_folder: for example, '/home/richard/PycharmProjects/myNN/Code/inception_v3/'
    :return:
    """
    from Code.import_pb_to_tensorboard import import_to_tensorboard

    import_to_tensorboard(model_path, event_folder)


class GenerativeModelMetric(object):
    def __init__(self, image_format=None, model='v1', model_path=None):
        """ This class defines several metrics using pre-trained classifier inception v1.

        :param image_format:
        """
        if model_path is None:
            self.model = model
            if model == 'v1':
                self.inception_graph_def = tfgan.eval.get_graph_def_from_disk(FLAGS.INCEPTION_V1)
            elif model == 'v3':
                self.inception_graph_def = tfgan.eval.get_graph_def_from_disk(FLAGS.INCEPTION_V3)
            else:
                raise NotImplementedError('Model {} not implemented.'.format(model))
        else:
            self.model = 'custom'
            self.inception_graph_def = tfgan.eval.get_graph_def_from_disk(model_path)
        if image_format is None:
            self.image_format = FLAGS.IMAGE_FORMAT
        else:
            self.image_format = image_format

        # preserved for inception v3
        self._pool3_v3_ = None
        self._logits_v3_ = None

    def _inception_v1_(self, image):
        """ This function runs the inception v1 model on images and give logits output.

        Note: if other layers of inception model is needed, change the output_tensor option in tfgan.eval.run_inception

        :param image:
        :return:
        """
        image_size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
        if self.image_format in {'channels_first', 'NCHW'}:
            image = tf.transpose(image, perm=(0, 2, 3, 1))
        if image.get_shape().as_list()[1] != image_size:
            image = tf.image.resize_bilinear(image, [image_size, image_size])

        # inception score uses the logits:0 while FID uses pool_3:0.
        logits, pool3 = tfgan.eval.run_inception(
            image, graph_def=self.inception_graph_def, input_tensor='Mul:0', output_tensor=['logits:0', 'pool_3:0'])

        return logits, pool3

    def inception_v1(self, images):
        """ This function runs the inception v1 model on images and give logits output.

        Note: if other layers of inception model is needed, change the output_tensor option in tfgan.eval.run_inception.
        Note: for large inputs, e.g. [10000, 64, 64, 3], it is better to run iterations containing this function.

        :param images:
        :return:
        """
        num_images = images.get_shape().as_list()[0]
        if num_images > 2500:
            raise MemoryError('The input is too big to possibly fit into memory. Consider using multiple runs.')
        if images.get_shape().as_list()[0] >= 400:
            # Note: need to validate the code below

            # somehow tfgan.eval.classifier_score does not work properly when splitting the datasets.
            # The following code is copied from:
            # https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
            num_batch = np.ceil(images.get_shape().as_list()[0] / 100).astype(np.int)
            generated_images_list = tf.split(images, num_or_size_splits=num_batch, axis=0)
            logits, pool3 = tf.map_fn(
                fn=self._inception_v1_,
                elems=tf.stack(generated_images_list),
                dtype=(tf.float32, tf.float32),
                parallel_iterations=1,
                back_prop=False,
                swap_memory=True,
                name='RunClassifier')
            logits = tf.concat(tf.unstack(logits), 0)
            pool3 = tf.concat(tf.unstack(pool3), 0)
        else:
            logits, pool3 = self._inception_v1_(images)

        return logits, pool3

    @staticmethod
    def inception_score_from_logits(logits):
        """ This function estimates the inception score from logits output by inception_v1

        :param logits:
        :return:
        """
        if type(logits) == np.ndarray:
            logits = tf.constant(logits, dtype=tf.float32)
        return tfgan.eval.classifier_score_from_logits(logits)

    @staticmethod
    def fid_from_pool3(x_pool3, y_pool3):
        """ This function estimates Fréchet inception distance from pool3 of inception model

        :param x_pool3:
        :param y_pool3:
        :return:
        """
        if type(x_pool3) == np.ndarray:
            x_pool3 = tf.constant(x_pool3, dtype=tf.float32)
        if type(y_pool3) == np.ndarray:
            y_pool3 = tf.constant(y_pool3, dtype=tf.float32)
        return tfgan.eval.frechet_classifier_distance_from_activations(x_pool3, y_pool3)

    def inception_score_and_fid_v1(self, x_batch, y_batch, num_batch=10, ckpt_folder=None, ckpt_file=None):
        """ This function calculates inception scores and FID based on inception v1.
        Note: batch_size * num_batch needs to be larger than 2048, otherwise the convariance matrix will be
        ill-conditioned.

        According to TensorFlow v1.7 (below), this is actually inception v3 model.
        Somehow the downloaded file says it's v1.
        code link: https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib \
        /gan/python/eval/python/classifier_metrics_impl.py

        Steps:
        1, the pool3 and logits are calculated for x_batch and y_batch with sess
        2, the pool3 and logits are passed to corresponding metrics

        :param ckpt_file:
        :param x_batch: tensor, one batch of x in range [-1, 1]
        :param y_batch: tensor, one batch of y in range [-1, 1]
        :param num_batch:
        :param ckpt_folder: check point folder
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :return:
        """
        assert self.model == 'v1', 'GenerativeModelMetric is not initialized with model="v1".'
        if ckpt_folder is None:
            raise AttributeError('ckpt_folder must be provided.')

        x_logits, x_pool3 = self.inception_v1(x_batch)
        y_logits, y_pool3 = self.inception_v1(y_batch)

        with MySession(load_ckpt=True) as sess:
            inception_outputs = sess.run_m_times(
                [x_logits, y_logits, x_pool3, y_pool3],
                ckpt_folder=ckpt_folder, ckpt_file=ckpt_file,
                max_iter=num_batch, trace=True)

        # get logits and pool3
        x_logits_np = np.concatenate([inc[0] for inc in inception_outputs], axis=0)
        y_logits_np = np.concatenate([inc[1] for inc in inception_outputs], axis=0)
        x_pool3_np = np.concatenate([inc[2] for inc in inception_outputs], axis=0)
        y_pool3_np = np.concatenate([inc[3] for inc in inception_outputs], axis=0)
        print('logits calculated. Shape = {}.'.format(x_logits_np.shape))
        print('pool3 calculated. Shape = {}.'.format(x_pool3_np.shape))
        # calculate scores
        inc_x = self.inception_score_from_logits(x_logits_np)
        inc_y = self.inception_score_from_logits(y_logits_np)
        xp3_1, xp3_2 = np.split(x_pool3_np, indices_or_sections=2, axis=0)
        fid_xx = self.fid_from_pool3(xp3_1, xp3_2)
        fid_xy = self.fid_from_pool3(x_pool3_np, y_pool3_np)

        with MySession() as sess:
            scores = sess.run_once([inc_x, inc_y, fid_xx, fid_xy])

        return scores

    def _initialize_inception_v3_(self):
        """ This function adds inception v3 model to the graph and changes the tensor shape from [1, h, w, c]
        to [None, h, w, c] so that the inception 3 model can handle arbitrary input batch size.

        Note: This function was obtained online. It did not work as expected.

        :return:
        """
        # add inception graph to current graph
        with tf.gfile.FastGFile(FLAGS.INCEPTION_V3, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        # change the shape[0] of each tensor along the graph up to pool3_output
        with tf.Session() as sess:
            pool3_output = sess.graph.get_tensor_by_name('pool_3:0')
            ops = pool3_output.graph.get_operations()
            for op_idx, op in enumerate(ops):
                for o in op.outputs:
                    shape = o.get_shape().as_list()
                    if len(shape) > 0:
                        shape[0] = None
                    o.set_shape(shape)  # online resource uses o._shape = tf.TensorShape(shape), which did not work.

            # define pool3 and logits
            # self._pool3_v3_ = tf.squeeze(pool3_output)  # squeeze remove dimensions of 1
            # print(sess.graph.get_tensor_by_name('Mul:0').get_shape().as_list())
            # print(self._pool3_v3_.get_shape().as_list())
            self._pool3_v3_ = tf.reshape(pool3_output, shape=[pool3_output.get_shape()[0], 2048])
            print(self._pool3_v3_.get_shape().as_list())
            weight = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
            self._logits_v3_ = tf.matmul(self._pool3_v3_, weight)

    def inception_v3(self, images, batch_size=100):
        """ This function runs the inception v3 model on images and give logits output.

        Note: if other layers of inception model is needed, change the output_tensor option in
        self._initialize_inception_v3_

        :param images:
        :type images: ndarray
        :param batch_size:
        :return:
        """
        # prepare
        if self.image_format in {'channels_first', 'NCHW'}:
            images = np.transpose(images, axes=(0, 2, 3, 1))
        images = images * 127.5 + 127.5  # rescale batch_image to [0, 255]
        num_images = images.shape[0]
        num_batches = int(math.ceil(num_images / batch_size))

        # run iterations
        pool3 = []
        logits = []
        with tf.Session() as sess:
            for i in range(num_batches):
                batch_image = images[(i * batch_size):min((i + 1) * batch_size, num_images)]
                batch_pool3, batch_logits = sess.run(
                    [self._pool3_v3_, self._logits_v3_], feed_dict={'ExpandDims:0': batch_image})
                pool3.append(batch_pool3)
                logits.append(batch_logits)
            pool3 = np.concatenate(pool3, axis=0)
            logits = np.concatenate(logits, axis=0)

        return logits, pool3

    def inception_score_and_fid_v3(
            self, x_batch, y_batch, num_batch=10, inception_batch=100, ckpt_folder=None, ckpt_file=None):
        """ This function calculates inception scores and FID based on inception v1.
        Note: batch_size * num_batch needs to be larger than 2048, otherwise the convariance matrix will be
        ill-conditioned.

        Steps:
        1. a large number of images are generated
        1, the pool3 and logits are calculated from numpy arrays x_images and y_images
        2, the pool3 and logits are passed to corresponding metrics

        :param x_batch:
        :param y_batch:
        :param num_batch:
        :param inception_batch:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :return:
        """
        assert self.model == 'v3', 'GenerativeModelMetric is not initialized with model="v3".'
        # initialize inception v3
        self._initialize_inception_v3_()

        # generate x_batch, get logits and pool3
        with MySession(load_ckpt=True) as sess:
            x_image_list = sess.run_m_times(
                x_batch, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file, max_iter=num_batch, trace=True)
            x_images = np.concatenate(x_image_list, axis=0)
        print('x_image obtained, shape: {}'.format(x_images.shape))
        x_logits_np, x_pool3_np = self.inception_v3(x_images, batch_size=inception_batch)
        print('logits calculated. Shape = {}.'.format(x_logits_np.shape))
        print('pool3 calculated. Shape = {}.'.format(x_pool3_np.shape))

        # generate y_batch, get logits and pool3
        with MySession(load_ckpt=True) as sess:
            y_image_list = sess.run_m_times(
                y_batch, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file, max_iter=num_batch, trace=True)
            y_images = np.concatenate(y_image_list, axis=0)
        print('y_image obtained, shape: {}'.format(x_images.shape))
        y_logits_np, y_pool3_np = self.inception_v3(y_images, batch_size=inception_batch)

        # calculate scores
        inc_x = self.inception_score_from_logits(x_logits_np)
        inc_y = self.inception_score_from_logits(y_logits_np)
        xp3_1, xp3_2 = np.split(x_pool3_np, indices_or_sections=2, axis=0)
        fid_xx = self.fid_from_pool3(xp3_1, xp3_2)
        fid_xy = self.fid_from_pool3(x_pool3_np, y_pool3_np)

        with MySession() as sess:
            scores = sess.run_once([inc_x, inc_y, fid_xx, fid_xy])

        return scores
