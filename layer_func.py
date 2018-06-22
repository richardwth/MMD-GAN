""" This code contains general functions that may help build other models

"""

# default modules
import numpy as np
import tensorflow as tf
from GeneralTools.misc_fun import FLAGS
from GeneralTools.math_func import power_iter, power_iter_conv, power_iter_atrous_conv, power_iter_transpose_conv


########################################################################
def weight_initializer(act_fun, init_w_scale=1.0):
    """ This function includes initializer for several common activation functions.
    The initializer will be passed to tf.layers.

    :param act_fun:
    :param init_w_scale:
    :return:
    """

    if FLAGS.WEIGHT_INITIALIZER == 'default':
        if init_w_scale == 0.0:
            initializer = tf.zeros_initializer()
        else:
            # if custom_info[0] in ['s', 'spectral'] or custom_info[1] in ['s', 'spectral']:
            #     initializer = lambda shape, dtype, partition_info: \
            #         spectral_norm_variable_initializer(shape, dtype, partition_info)
            # else:
            if (act_fun == 'relu') or (act_fun == 'lrelu'):
                initializer = tf.variance_scaling_initializer(
                    scale=2.0 * init_w_scale, mode='fan_in', distribution='normal')
            # elif act_fun == 'tanh':
            #     initializer = tf.variance_scaling_initializer(
            #         scale=1.0 * init_w_scale, mode='fan_avg', distribution='uniform')
            #     initializer = tf.contrib.layers.variance_scaling_initializer(
            #         factor=3.0 * init_w_scale, mode='FAN_AVG', uniform=True)
            elif act_fun == 'sigmoid':
                initializer = tf.variance_scaling_initializer(
                    scale=16.0 * init_w_scale, mode='fan_avg', distribution='uniform')
                # initializer = tf.contrib.layers.variance_scaling_initializer(
                #     factor=48.0 * init_w_scale, mode='FAN_AVG', uniform=True)
            else:  # xavier initializer
                initializer = tf.variance_scaling_initializer(
                    scale=1.0 * init_w_scale, mode='fan_avg', distribution='uniform')
                # initializer = tf.contrib.layers.variance_scaling_initializer(
                #     factor=3.0 * init_w_scale, mode='FAN_AVG', uniform=True)
    elif FLAGS.WEIGHT_INITIALIZER == 'sn_paper':
        # paper on spectral normalization used truncated_normal_initializer
        print('You are using custom initializer.')
        initializer = tf.truncated_normal_initializer(stddev=0.02)
    else:
        raise NotImplementedError('The initializer {} is not implemented.'.format(FLAGS.WEIGHT_INITIALIZER))

    return initializer


#######################################################################
def spectral_norm_variable_initializer(shape, dtype=tf.float32, partition_info = None):
    """ This function provides customized initializer for tf.get_variable()

    :param shape:
    :param dtype:
    :param partition_info: this is required by tf.layers, but ignored in many tf.initializer. Here we ignore it.
    :return:
    """
    variable = tf.random_normal(shape=shape, stddev=1.0, dtype=dtype)

    if len(shape) > 2:
        var_reshaped = tf.reshape(variable, shape=[-1, shape[-1]])
        sigma = tf.svd(var_reshaped, full_matrices=False, compute_uv=False)[0]
    else:
        sigma = tf.svd(variable, full_matrices=False, compute_uv=False)[0]

    return variable / (sigma + FLAGS.EPSI)


########################################################################
def leaky_relu(features, name=None):
    """ This function defines leaky rectifier linear unit

    :param features:
    :param name:
    :return:
    """
    # return tf.maximum(tf.multiply(0.1, features), features, name=name)
    return tf.nn.leaky_relu(features, alpha=0.1, name=name)


########################################################################
def get_std_act_fun(act_fun_name):
    """ This function gets the standard activation function from tensorflow

    :param act_fun_name:
    :return:
    """
    if act_fun_name == 'linear':
        act_fun = tf.identity
    elif act_fun_name == 'relu':
        act_fun = tf.nn.relu
    elif act_fun_name == 'crelu':
        act_fun = tf.nn.crelu
    elif act_fun_name == 'elu':
        act_fun = tf.nn.elu
    elif act_fun_name == 'lrelu':
        act_fun = leaky_relu
    elif act_fun_name == 'selu':
        act_fun = tf.nn.selu
    elif act_fun_name == 'softplus':
        act_fun = tf.nn.softplus
    elif act_fun_name == 'softsign':
        act_fun = tf.nn.softsign
    elif act_fun_name == 'sigmoid':
        act_fun = tf.nn.sigmoid
    elif act_fun_name == 'tanh':
        act_fun = tf.nn.tanh
    elif act_fun_name == 'crelu':
        # concatenated ReLU doubles the depth of the activations
        # CReLU only supports NHWC
        act_fun = tf.nn.crelu
    elif act_fun_name == 'elu':
        act_fun = tf.nn.elu
    else:
        raise NotImplementedError('Function {} is not implemented.'.format(act_fun_name))

    return act_fun


########################################################################
def apply_activation(layer_input, act_fun, name=None):
    """ This function applies element-wise activation function

    Inputs:
    layer_batch - inputs to activation function
    layer_activation - name of activation function

    """
    if isinstance(act_fun, str):
        act_fun = get_std_act_fun(act_fun)
    layer_output = act_fun(layer_input, name=name)

    return layer_output


########################################################################
# @ops.RegisterGradient("DepthwiseConv2dNativeBackpropInput")
# def _DepthwiseConv2DNativeBackpropInputGrad(op, grad):
#     """ The derivatives for depth-wise deconvolution.
#
#     :param op: the depth-wise deconvolution op.
#     :param grad: the tensor representing the gradient w.r.t. the output
#     :returns: the gradients w.r.t. the input and the filter
#     """
#     # print(op.inputs[1])
#     # print(op.get_attr("data_format"))
#     return [None,
#             nn_ops.depthwise_conv2d_native_backprop_filter(
#                 grad,
#                 array_ops.shape(op.inputs[1]),
#                 op.inputs[2],
#                 op.get_attr("strides"),
#                 op.get_attr("padding"),
#                 data_format=op.get_attr("data_format")),
#             nn_ops.depthwise_conv2d_native(
#                 grad,
#                 op.inputs[1],
#                 op.get_attr("strides"),
#                 op.get_attr("padding"),
#                 data_format=op.get_attr("data_format"))]

########################################################################
def periodic_shuffling(layer_input, scale_factor, scale_up=True, data_format='channels_first'):
    """ This function defines a periodic shuffling operation, proposed in following paper:

    Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A., Bishop, R., … Wang, Z. (2016).
    Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.
    In IEEE Conference on Computer Vision and Pattern Recognition (pp. 1874–1883).

    2017.11.18. The shape of gradient for tf.depth_to_space with data_format NCHW is wrong.
    As a result, transpose is used to convert input to 'NHWC'

    2018.01.25. Starting from TensorFlow 1.5, support for 'NCHW' is added.

    :param layer_input: 4-D tensor in format [batch_size, height, width, channels] or
        [batch_size, channels, height, width]
    :param scale_factor: integer, scaling factor
    :param scale_up: if True, do upsampling, otherwise, do downsampling
    :param data_format: integer, number of output channels
    :return x: 4-D tensors, feature maps,
        for upsampling, [batch_size, channels / scale_factor ** 2, height * scale_factor, width * scale_factor]
        for downsampling, [batch_size, channels * scale_factor ** 2, height / scale_factor, width / scale_factor]
    """
    if data_format in ['channels_first', 'NCHW']:
        image_format = 'NCHW'
    elif data_format in ['channels_last', 'NHWC']:
        image_format = 'NHWC'
    else:
        raise AttributeError('Data format {} not supported.'.format(data_format))

    def shuffling(image, _format):
        if scale_up:
            return tf.depth_to_space(image, scale_factor, data_format=_format)
        else:
            return tf.space_to_depth(image, scale_factor, data_format=_format)

    if FLAGS.TENSORFLOW_VERSION in {'1.1.0', '1.2.0', '1.3.0', '1.4.0'}:
        if data_format in ['channels_first', 'NCHW']:
            layer_output = tf.transpose(layer_input, perm=(0, 2, 3, 1))
            layer_output = shuffling(layer_output, 'NHWC')
            layer_output = tf.transpose(layer_output, perm=(0, 3, 1, 2))
            # print('Shape after depth_to_space: {}'.format(layer_output.get_shape().as_list()))
        elif data_format in ['channels_last', 'NHWC']:
            layer_output = shuffling(layer_input, 'NHWC')
        else:
            raise AttributeError('Image format not supported.')
    else:
        layer_output = shuffling(layer_input, image_format)

    return layer_output


########################################################################
def bilinear_additive_upsampling(layer_input, scale_factor, channel_out=None, data_format='channels_last'):
    """ This function defines a bilinear additive upsampling operation, proposed in following paper:

    Wojna, Z., Ferrari, V., Guadarrama, S., Silberman, N., Chen, L.-C., Fathi, A., & Uijlings, J. (2017).
    The Devil is in the Decoder.

    :param layer_input: 4-D tensor, in format [batch_size, height, width, channels] or
        [batch_size, channels, height, width]
    :param scale_factor: integer, scaling factor; theoretically, channels = channel_out * r^2
    :param channel_out: integer, number of output channels
    :param data_format: 'channels_first' or 'channels_last'
    :return:
    """
    # check inputs
    if channel_out is None:
        channel_out = 1
    if data_format == 'channel_first':  # convert to [batch_size, height, width, channels]
        layer_input = tf.transpose(layer_input, perm=(0, 2, 3, 1))
    # channels must equal to channel_out * r^2
    channel_to_add = scale_factor ^ 2
    required_channel = channel_to_add * channel_out
    batch_size, height, width, num_channel = layer_input.get_shape().as_list()
    assert num_channel == required_channel, \
        'Num of channel mis-match, required: %d, actual %d.' % (required_channel, num_channel)
    # do upsampling
    layer_sampled = tf.image.resize_bilinear(
        layer_input, [height * scale_factor, width * scale_factor], align_corners=True)
    # add every channel_to_add channels
    layer_output = tf.reduce_sum(
        tf.reshape(layer_sampled, shape=[batch_size, height, width, channel_out, channel_to_add]),
        axis=-1)
    if data_format == 'channel_first':  # convert to [batch_size, channels, height, width]
        layer_output = tf.transpose(layer_output, perm=(0, 3, 1, 2))

    return layer_output


########################################################################
class Crop(object):
    def __init__(self, num_crop):
        """ This class divides image input into num_crop[0]-by-num_crop[1] crops.
        Its methods below defines the format of output.

        Attention when using this class.
        tensorflow has provided functions like extract_image_patches, space_to_batch, etc to crop images.
        For example, do_crop and stack2batch can be realized by space_to_batch
        Check this link for more functions:
        https://www.tensorflow.org/api_guides/python/image#resize_images

        This function does not support input of type [batch, channel, height, width] currently.

        :param num_crop: for randomly
        """
        self.num_crop = num_crop
        self.layer_output = None
        self.batch = None
        self.channel = None
        self.height = None
        self.width = None

    def do_crop(self, layer_input):
        """ This function divides image input into num_crop[0]-by-num_crop[1] crops.
        After do-crop, stack2batch and stack2new_dimension can be used to decide how
        the crops will be stacked.

        :param layer_input: 4-D tensor,in format [batch_size, height, width, channels]
        :return:
        """
        # get shape info
        [self.batch, height, width, self.channel] = layer_input.get_shape().as_list()
        self.height = int(height / self.num_crop[0])  # new_height
        self.width = int(width / self.num_crop[1])  # new_weight

        # [1, batch_size, height, width, channel]
        layer_input = tf.expand_dims(layer_input, axis=0)
        # [crop[1], batch_size, height, new_width, channel]
        layer_input = tf.concat(tf.split(layer_input, num_or_size_splits=self.num_crop[1], axis=3), axis=0)
        # [1, crop[1], batch_size, height, new_width, channel]
        layer_input = tf.expand_dims(layer_input, axis=0)
        # [crop[0], crop[1], batch_size, new_height, new_width, channel]
        self.layer_output = tf.concat(tf.split(layer_input, num_or_size_splits=self.num_crop[0], axis=3), axis=0)

    def do_overlap_crop(self, layer_input):
        pass

    def stack2batch(self):
        """ This function stacks the crops into the batch_sie axis. Examples:
        [[x1], [x2]] ===> [[x1c11], [x1c12], [x1c21], [x1c22], [x2c11], [x2c12], [x2c21], [x2c22]].

        :return:
        """
        # [batch_size, crop[0], crop[1], new_height, new_width, channel]
        layer_output = tf.transpose(self.layer_output, perm=[2, 0, 1, 3, 4, 5])
        # [batch_size*crop[0]*crop[1], new_height, new_width, channel]
        layer_output = tf.reshape(layer_output, [-1, self.channel, self.height, self.width])

        return layer_output

    def stack2new_dimension(self):
        """ This function stacks the crops into the first dimension. Examples:
        [[x1], [x2]] ===> [[c11x1], [c11x2], [c12x1], [c12x2], [c21x1], [c21x2], [c22x1], [c22x2]].

        :return:
        """
        # [crop[0]*crop[1], batch_size, new_height, new_width, channel]
        return tf.reshape(self.layer_output, [-1, self.batch, self.height, self.width, self.channel])

    def do_random_crop(self, layer_input, size=None, stack2batch=False):
        """ This function randomly samples num_crop crops from the image.
        The crops will be stacked to new dimension if stack2batch=False

        :param layer_input: 4-D tensor,in format [batch_size, height, width, channels]
        :param size: list with two elements, [crop_height, crop_width]
        :param stack2batch:

        :return layer_output: if num_crop=1 or stack2batch=True, 4-D tensor; otherwise, 5-D tensor.
        """
        # check inputs
        if size is None:
            size = [16, 16]

        # get shape info
        [self.batch, _, _, self.channel] = layer_input.get_shape().as_list()
        self.height = size[0]  # new_height
        self.width = size[1]  # new_weight
        # do crop
        if self.num_crop == 1:
            # [batch_size, new_height, new_width, channel]
            layer_output = tf.random_crop(
                layer_input, size=[self.batch, self.height, self.width, self.channel])
        else:
            # [1, batch_size, height, width, channel]
            layer_input = tf.expand_dims(layer_input, axis=0)
            crops = []
            for i in range(self.num_crop):
                crops.append(
                    tf.random_crop(
                        layer_input, size=[1, self.batch, self.height, self.width, self.channel]))
            # [num_crops, batch_size, new_height, new_width, channel]
            layer_output = tf.concat(crops, axis=0)

            if stack2batch is True:
                # [batch_size, num_crops, new_height, new_width, channel]
                layer_output = tf.transpose(layer_output, perm=[1, 0, 2, 3, 4])
                # [batch_size*num_crops, new_height, new_width, channel]
                layer_output = tf.reshape(layer_output, shape=[-1, self.height, self.width, self.channel])

        # if num_crop=1 or stack2batch=True, 4-D tensor; otherwise, 5-D tensor.
        return layer_output

    def do_group_crop(self, layer_input, kernel_size):
        """ This function first divides the image into num_crop[0]-by-num_crop[1] crops; then like
        convolution, it pads the peripheral crops, and construct a big crop from kernel_size[0]-by-
        kernel_size[1] neighbouring crops. The big crops overlap, and cover the whole input images.

        :param layer_input: 4-D tensor,in format [batch_size, height, width, channels]
        :param kernel_size: a list/tuple of two elements
        :return: [batch_size, new_height x2, new_width x2, channel]
        """
        # check inputs
        if kernel_size[0] > self.num_crop[0] or kernel_size[0] > self.num_crop[0]:
            raise AttributeError('Kernel size should be smaller than crop size to avoid identical crops')

        # get shape info
        [self.batch, height, width, self.channel] = layer_input.get_shape().as_list()
        num_pixel_h = int(height / self.num_crop[0])  # num_pixel in height direction of each group
        num_pixel_w = int(width / self.num_crop[1])  # num_pixel in width direction of each group
        kernel_h = kernel_size[0] * num_pixel_h
        kernel_w = kernel_size[1] * num_pixel_w
        # print('Kernel size {}x{}'.format(kernel_h, kernel_w))
        # [1, batch_size, height, width, channel]
        layer_input = tf.expand_dims(layer_input, axis=0)

        # do group conv-crop
        crops = []
        for i in range(self.num_crop[0] + kernel_size[0] - 1):
            for j in range(self.num_crop[1] + kernel_size[1] - 1):
                pixel_i = tf.multiply(i + 1, num_pixel_h)
                pixel_j = tf.multiply(j + 1, num_pixel_w)
                # get the crop out of image
                range_h = [tf.maximum(0, pixel_i - kernel_h), tf.minimum(pixel_i, height)]
                range_w = [tf.maximum(0, pixel_j - kernel_w), tf.minimum(pixel_j, width)]
                crop_ij = layer_input[:, :, range_h[0]:range_h[1], range_w[0]:range_w[1], :]
                # pad the crop with zeros
                pad_h = [tf.maximum(0, kernel_h - pixel_i), tf.maximum(0, pixel_i - height)]
                pad_w = [tf.maximum(0, kernel_w - pixel_j), tf.maximum(0, pixel_j - width)]
                crops.append(tf.pad(crop_ij, paddings=[[0, 0], [0, 0], pad_h, pad_w, [0, 0]]))
        # [num_crops, batch_size, new_height, new_width, channel]
        layer_output = tf.concat(crops, axis=0)
        layer_output.set_shape(
            [(self.num_crop[0] + kernel_size[0] - 1) * (self.num_crop[1] + kernel_size[1] - 1),
             self.batch, kernel_h, kernel_w, self.channel])

        return layer_output


########################################################################
def spectral_norm(kernel, layer_def, scope_prefix='', v=None, num_iter=1):
    """ This function calculates the spectral normalization of the weight matrix using power iterations.

    The application of spectral normal to NN is proposed in following papers:
    Yoshida, Y., & Miyato, T. (2017).
    Spectral Norm Regularization for Improving the Generalizability of Deep Learning.
    Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2017).
    Spectral Normalization for Generative Adversarial Networks,

    :param kernel: a weight tensor with the following size:
        dense: [fan_in, fan_out]
        conv: [height, weight, channels_in, channels_out]
        separate conv: not supported yet
    :param layer_def: a dictionary with keys ['op', 'dilation', 'strides', 'padding', 'data_format', 'input_shape'];
        for the case layer_def['op'] == 'tc', layer_def also has key 'output_shape'; the value for 'input_shape' and
        'output_shape' must be list
    :param scope_prefix:
    :param v;
    :param num_iter: number of power iterations
    :return:
    """
    with tf.variable_scope(scope_prefix + 'SN', reuse=tf.AUTO_REUSE):

        if FLAGS.SPECTRAL_NORM_MODE == 'sn_paper':  # methods from the original paper
            w_shape = kernel.get_shape().as_list()
            w_reshaped = tf.reshape(kernel, [-1, w_shape[-1]])
            # initialize right singular vector
            if v is None:
                v = tf.get_variable(
                    'in_rand', [1, w_shape[-1]], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(), trainable=False)
            # update left and right singular vector
            sigma, v_update = power_iter(w_reshaped, v=v)
            # update u after each iteration
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(v, v_update))

        elif FLAGS.SPECTRAL_NORM_MODE in {'PICO', 'default'}:  # proposed PICO
            if layer_def['op'] == 'd':
                if layer_def['input_shape'][1] == 1 or kernel.get_shape().as_list()[1] == 1:
                    # in this case, spectral norm is simply vector norm
                    sigma = tf.norm(kernel, ord='euclidean')
                else:
                    # v: 1-by-q vector
                    if v is None:
                        v = tf.get_variable(
                            'in_rand', shape=[1, layer_def['input_shape'][1]], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(), trainable=False)
                    # do one power iteration
                    sigma, _update, _ = tf.while_loop(
                        cond=lambda _1, _2, i: i < num_iter,
                        body=lambda _1, u, i: power_iter(kernel, u=u, step=i),
                        loop_vars=(tf.constant(0.0, dtype=tf.float32), v, tf.constant(0, dtype=tf.int32)))
                    # update v
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(v, _update))
            elif layer_def['op'] == 'c':
                if layer_def['dilation'] == 1:
                    # v: 1-H-W-in or 1-in-H-W tensor
                    if v is None:
                        v = tf.get_variable(
                            'in_rand', shape=[1] + layer_def['input_shape'][1:], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(), trainable=False)
                    conv_def = {key: layer_def[key] for key in ['strides', 'padding', 'data_format']}
                    # do one power iteration
                    sigma, _update, _ = tf.while_loop(
                        cond=lambda _1, _2, i: i < num_iter,
                        body=lambda _1, u, i: power_iter_conv(kernel, u, conv_def, step=i),
                        loop_vars=(tf.constant(0.0, dtype=tf.float32), v, tf.constant(0, dtype=tf.int32)))
                elif layer_def['dilation'] > 1:
                    # v: 1-H-W-in tensor. The tf.nn.atrous_conv2d does not support NCHW format
                    if layer_def['data_format'] in ['NCHW', 'channels_first']:
                        shape = [layer_def['input_shape'][2], layer_def['input_shape'][3], layer_def['input_shape'][1]]
                    else:
                        shape = layer_def['input_shape'][1:]
                    if v is None:
                        v = tf.get_variable(
                            'in_rand', shape=[1] + shape, dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(), trainable=False)
                    conv_def = {key: layer_def[key] for key in ['dilation', 'padding']}
                    # do one power iteration
                    sigma, _update, _ = tf.while_loop(
                        cond=lambda _1, _2, i: i < num_iter,
                        body=lambda _1, u, i: power_iter_atrous_conv(kernel, u, conv_def, step=i),
                        loop_vars=(tf.constant(0.0, dtype=tf.float32), v, tf.constant(0, dtype=tf.int32)))
                else:
                    raise AttributeError('Layer dilation is incorrectly set')

                # update v
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(v, _update))
            elif layer_def['op'] == 'tc':
                # for a kernel, its spectral norm does not depend on whether it is used in conv or transpose conv
                # but here we setup the process for transpose conv anyway
                if layer_def['dilation'] == 1:
                    # v: 1-H-W-in or 1-in-H-W tensor
                    if v is None:
                        v = tf.get_variable(
                            'in_rand', shape=[1] + layer_def['input_shape'][1:], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(), trainable=False)
                    conv_def = {key: layer_def[key] for key in ['strides', 'padding', 'data_format', 'output_shape']}
                    if conv_def['output_shape'][0] != 1:
                        conv_def['output_shape'] = [1] + list(conv_def['output_shape'][1:])
                    # do one power iteration
                    sigma, _update, _ = tf.while_loop(
                        cond=lambda _1, _2, i: i < num_iter,
                        body=lambda _1, u, i: power_iter_transpose_conv(kernel, u, conv_def, step=i),
                        loop_vars=(tf.constant(0.0, dtype=tf.float32), v, tf.constant(0, dtype=tf.int32)))
                else:
                    raise AttributeError('TC layer with dilation > 1 is not supported yet.')

                # update v
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(v, _update))
            elif layer_def['op'] == 'sc':
                raise NotImplementedError('SC layer is not supported yet.')
            else:
                raise AttributeError('Layer_type {} not supported.'.format(layer_def['op']))
        else:
            raise NotImplementedError('The mode {} is not implemented.'.format(FLAGS.SPECTRAL_NORM_MODE))

        return sigma


########################################################################
def l2_norm(w, scope_prefix=''):
    """ This function calculates l2 normal (Frobenius norm) of the weight matrix.

    :param w: a list of kernels. The size of each kernel should be:
        dense: [fan_in, fan_out]
        conv: [height, weight, channels_in, channels_out]
        separate conv: not supported yet
    :param scope_prefix:
    :return:
    """
    # tf.norm is slightly faster than tf.sqrt(tf.reduce_sum(tf.square()))
    # it is important that axis=None; in this case, norm(w) = norm(vec(w))
    with tf.name_scope(scope_prefix + 'l2'):
        return tf.norm(w[0], ord='euclidean', axis=None) + FLAGS.EPSI


########################################################################
def local_response_normalization(x, scope_prefix='', data_format='channels_first'):
    """ This function applies a variant of local response normalization to the activations.
    It normalizes x by the sqrt of squared sum along channel dimension.

    The local response normalization is used in following paper:
    Karras T., Aila T., Laine S., Lehtinen J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability, And Variation,

    :param x:
    :param scope_prefix:
    :param data_format:
    :return:
    """
    with tf.name_scope(scope_prefix + 'LRN'):
        axis = 1 if data_format in ['channels_first', 'NCHW'] else -1
        return tf.divide(x, tf.sqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + FLAGS.EPSI))


class ParametricOperation(object):
    """ This class defines the parametric operation applied to the input

    """
    def __init__(self, design, input_shape, name_scope=None, scope_prefix='', data_format=None):
        """ This function initializes the linear operation

        :param design: dict with keys:
            'op': 'i' - identity,
                'k' - multiplication with trainable scalar kernel
                'c' - conv
                'd' - dense
                'tc' - transpose conv
                'sc' - separable conv
                'max' - max pool
                'avg' - average pool
                'b' - add bias
                'bn' - batch normalization
                'lrn' - local response normalization,
                'sum' - sum pool (used in sn_paper)
            'out': number of features/channels in the output
            'act': the activation function to use, e.g. 'linear',
            'act_k': whether to multiple a constant to compensate the norm loss at activation function, e.g. True
            'w_nm': None - no normalization, 's' - spectral normalization, 'l2' - He normalization
            'kernel': conv kernel size, e.g. 3,
            'strides': conv stride size, e.g. 1,
            'dilation': conv dilation size, e.g. 1,
            'padding': conv padding method, e.g. 'SAME' or 'VALID'
            'init_w_scale': initialization scale for kernel in conv or dense; if 0.0, then kernel is initialzied as 0.0
            'bn_center': batch normalization, whether to add offset beta, e.g. True
            'bn_scale': batch normalization, whether to multiply by gamma, e.g. True
            'bn_b_init': batch normalization, initializer for beta; if not provided, tf.zeros_initializer()
            'bn_w_init': batch normalization, initializer for gamma; if not provided, tf.ones_initializer()
            'bn_w_const': batch normalization, constraint for gamma
        :param input_shape:
        :param name_scope:
        :param scope_prefix: used to indicate layer/net info when print error
        :param data_format: 'channels_first' or 'channels_last'
        """
        self.design = design
        self.name_scope = 'kernel' if name_scope is None else name_scope
        self.name_in_err = scope_prefix + self.name_scope
        # IO
        self.input_shape = input_shape
        self.data_format = data_format
        if self.data_format == 'channels_first':
            self.data_format_alias = 'NCHW'
        elif self.data_format == 'channels_last':
            self.data_format_alias = 'NHWC'
        else:
            self.data_format_alias = self.data_format
        # format stride and dilation
        if 'strides' in self.design:
            self.strides = [1, 1, self.design['strides'], self.design['strides']] \
                if self.data_format == 'channels_first' \
                else [1, self.design['strides'], self.design['strides'], 1]
        if 'dilation' in self.design:
            if self.design['strides'] > 1 and self.design['dilation'] > 1:
                self.dilation = [1, 1, 1, 1]
                print('{}: when stride > 1, dilation > 1 is ignored.'.format(self.name_in_err))
            else:
                if self.design['op'] == 'c':
                    self.dilation = [1, 1, self.design['dilation'], self.design['dilation']] \
                        if self.data_format == 'channels_first' \
                        else [1, self.design['dilation'], self.design['dilation'], 1]
                elif self.design['op'] == 'sc':
                    self.dilation = [self.design['dilation'], self.design['dilation']]

        # calculate kernel shape and output_shape
        self._get_shape_()
        # initialize other parameters
        self.kernel = None
        self.multiplier = None
        self.kernel_norm = None
        self.is_kernel_norm_set = False

    def _get_shape_(self):
        """ This function calculates the kernel shape and the output shape

        :return:
        """
        if self.design['op'] == 'i':
            self.output_shape = self.input_shape
        if self.design['op'] == 'k':
            self.output_shape = self.input_shape
            self.kernel_shape = []
        elif self.design['op'] == 'd':  # dense layer
            self.kernel_shape = [self.input_shape[1], self.design['out']]
            self.output_shape = [self.input_shape[0], self.design['out']]
        elif self.design['op'] == 'c':  # conv layer
            if self.data_format == 'channels_first':
                fan_in, h, w = self.input_shape[1:]
            else:
                h, w, fan_in = self.input_shape[1:]
            self.kernel_shape = [self.design['kernel'], self.design['kernel'], fan_in, self.design['out']]
            h = spatial_shape_after_conv(
                h, self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            w = spatial_shape_after_conv(
                w, self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            self.output_shape = [self.input_shape[0], self.design['out'], h, w] \
                if self.data_format == 'channels_first' else [self.input_shape[0], h, w, self.design['out']]
        elif self.design['op'] == 'tc':  # transpose conv layer
            if self.data_format == 'channels_first':
                fan_in, h, w = self.input_shape[1:]
            else:
                h, w, fan_in = self.input_shape[1:]
            self.kernel_shape = [self.design['kernel'], self.design['kernel'], self.design['out'], fan_in]
            h = spatial_shape_after_transpose_conv(
                h, self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            w = spatial_shape_after_transpose_conv(
                w, self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            self.output_shape = [self.input_shape[0], self.design['out'], h, w] \
                if self.data_format == 'channels_first' else [self.input_shape[0], h, w, self.design['out']]
        elif self.design['op'] == 'sc':  # separate conv layer
            if self.data_format == 'channels_first':
                fan_in, h, w = self.input_shape[1:]
            else:
                h, w, fan_in = self.input_shape[1:]
            depthwise_shape = [self.design['kernel'], self.design['kernel'], fan_in, 1]
            pointwise_shape = [1, 1, fan_in, self.design['out']]
            self.kernel_shape = [depthwise_shape, pointwise_shape]
            h = spatial_shape_after_conv(
                h, self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            w = spatial_shape_after_conv(
                w, self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            self.output_shape = [self.input_shape[0], self.design['out'], h, w] \
                if self.data_format == 'channels_first' else [self.input_shape[0], h, w, self.design['out']]
        elif self.design['op'] in {'max', 'avg', 'sum'}:
            # max pool and average pool
            # sum pool is similar to avg, but instead of average, it does sum up
            if self.data_format == 'channels_first':
                fan_in, h, w = self.input_shape[1:]
                self.kernel_shape = [1, 1, self.design['kernel'], self.design['kernel']]
            else:
                h, w, fan_in = self.input_shape[1:]
                self.kernel_shape = [1, self.design['kernel'], self.design['kernel'], 1]
            h = spatial_shape_after_conv(  # pooling is simplified conv
                h, self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            w = spatial_shape_after_conv(
                w, self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            self.output_shape = [self.input_shape[0], self.design['out'], h, w] \
                if self.data_format == 'channels_first' else [self.input_shape[0], h, w, self.design['out']]
        elif self.design['op'] in {'b', 'bias'}:
            self.kernel_shape = self.input_shape[1] if self.data_format == 'channels_first' else self.input_shape[-1]
            self.output_shape = self.input_shape
        elif self.design['op'] in {'bn', 'batch_norm', 'lrn'}:
            self.output_shape = self.input_shape
        else:
            raise AttributeError('{}: type {} not supported'.format(self.name_in_err, self.design['op']))

    def _input_check_(self, op_input):
        """ Check the shape of the input

        :param op_input:
        :return:
        """
        input_shape = op_input.get_shape().as_list()
        assert self.input_shape[1:] == input_shape[1:], \
            '{}: the input shape {} does not match existed shape {}.'.format(
                self.name_in_err, input_shape[1:], self.input_shape[1:])

    def _output_check_(self, op_output):
        """ Check the shape of the output

        :param op_output:
        :return:
        """
        output_shape = op_output.get_shape().as_list()
        assert self.output_shape[1:] == output_shape[1:], \
            '{}: the output shape {} does not match existed shape {}.'.format(
                self.name_in_err, output_shape[1:], self.output_shape[1:])

    def init_kernel(self):
        """ This function initialize the kernels

        For simplification of name scopes in tensorboard, all variables are initialized here.

        :return:
        """
        if self.design['op'] in {'d', 'c', 'tc', 'sc'}:  # some ops may require different initial scale for kernel
            kernel_init = weight_initializer(self.design['act'], self.design['init_w_scale']) \
                if 'init_w_scale' in self.design else weight_initializer(self.design['act'])
        elif self.design['op'] == 'k':
            kernel_init = tf.zeros_initializer() \
                if 'init_w_scale' in self.design and self.design['init_w_scale'] == 0.0 else tf.ones_initializer()
        else:
            kernel_init = None

        if self.design['op'] in {'d', 'c', 'tc', 'k'}:  # dense, conv, transpose conv layer or scalar weight layer
            self.kernel = tf.get_variable(
                'kernel', self.kernel_shape, dtype=tf.float32, initializer=kernel_init, trainable=True)
        elif self.design['op'] == 'sc':  # separate conv layer
            depthwise_shape = self.kernel_shape[0]
            pointwise_shape = self.kernel_shape[1]
            self.kernel = [
                tf.get_variable(
                    'depthwise_kernel', depthwise_shape,
                    dtype=tf.float32, initializer=kernel_init, trainable=True),
                tf.get_variable(
                    'pointwise_kernel', pointwise_shape,
                    dtype=tf.float32, initializer=kernel_init, trainable=True)]
        elif self.design['op'] in {'b', 'bias'}:
            # self.kernel = tf.get_variable(
            #     'bias', shape=self.kernel_shape, dtype=tf.float32, trainable=True,
            #     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.002))
            self.kernel = tf.get_variable(
                'bias', shape=self.kernel_shape, dtype=tf.float32, trainable=True,
                initializer=tf.zeros_initializer())

        # initialize kernel norm
        self._get_weight_norm_()

    def _get_weight_norm_(self):
        """ This function get the weight of the norm

        :return:
        """
        if not self.is_kernel_norm_set:
            if 'w_nm' in self.design and self.design['w_nm'] in {'s', 'l2'}:
                if self.design['op'] in {'d', 'c', 'tc'}:
                    if self.design['w_nm'] in {'s'}:
                        layer_def = {'op': self.design['op'], 'input_shape': self.input_shape}
                        # apply spectral normalization to get w_norm
                        if self.design['op'] in ['d']:
                            self.kernel_norm = spectral_norm(self.kernel, layer_def, num_iter=1)
                        elif self.design['op'] in ['c', 'tc']:
                            layer_def.update({key: self.design[key] for key in ['strides', 'dilation', 'padding']})
                            layer_def['data_format'] = self.data_format_alias
                            if self.design['op'] == 'tc':
                                layer_def['output_shape'] = self.output_shape
                            self.kernel_norm = spectral_norm(self.kernel, layer_def, num_iter=1)
                        self._get_multiplier_()
                    else:
                        # apply he normalization to get w_norm
                        self.kernel_norm = l2_norm(self.kernel)
                else:
                    raise NotImplementedError(
                        '{}: kernel norm for {} has not been implemented.'.format(
                            self.name_in_err + '/' + self.name_scope, self.design['op']))

            self.is_kernel_norm_set = True

    def _get_multiplier_(self):
        """ This function reimburse the norm loss caused by the activation function

        :return:
        """
        if 'w_nm' in self.design and self.design['w_nm'] in ['spectral', 's']:
            if self.design['act_k'] is True:
                if self.design['act'] == 'lrelu':
                    self.multiplier = 1.0 / 0.55
                elif self.design['act'] in {'crelu', 'linear'}:
                    self.multiplier = 1.0
                elif self.design['act'] == 'relu':
                    self.multiplier = 2.0
                else:
                    self.multiplier = 1.0
            else:
                self.multiplier = 1.0

    def __call__(self, op_input, control_ops=None, is_training=True):
        """ This function applies the linear operation to the input

        :param op_input:
        :param control_ops:
        :param is_training:
        :return:
        """
        with tf.variable_scope(self.name_scope, reuse=tf.AUTO_REUSE):  # in case a layer has many kernels
            self._input_check_(op_input)
            with tf.control_dependencies(control_ops):
                # initialize the kernel
                self.init_kernel()
                # check multiplier
                if 'w_nm' in self.design and self.design['w_nm'] in ['spectral', 's', 'l2']:
                    multiplier = 1.0 / self.kernel_norm \
                        if self.multiplier is None else self.multiplier / self.kernel_norm
                else:
                    multiplier = None

                # apply ops
                if self.design['op'] == 'i':  # identity layer
                    op_output = op_input
                elif self.design['op'] == 'k':  # multiplication with trainable scalar kernel
                    op_output = op_input * self.kernel
                    if 'bound' in self.design:  # set bound to prevent gradient explosion
                        lb, hb = self.design['bound']
                        tf.add_to_collection(
                            tf.GraphKeys.UPDATE_OPS,
                            tf.assign(self.kernel, tf.clip_by_value(self.kernel, lb, hb)))
                elif self.design['op'] == 'd':  # dense layer
                    kernel = self.kernel if multiplier is None or multiplier == 1.0 else self.kernel * multiplier
                    op_output = tf.matmul(op_input, kernel)
                elif self.design['op'] == 'c':  # conv layer
                    kernel = self.kernel if multiplier is None or multiplier == 1.0 else self.kernel * multiplier
                    op_output = tf.nn.conv2d(
                        op_input, kernel, self.strides, self.design['padding'],
                        data_format=self.data_format_alias, dilations=self.dilation)
                elif self.design['op'] == 'tc':  # transpose conv layer
                    if self.design['dilation'] > 1:  # atrous_conv2d_transpose does not support NCHW
                        raise NotImplementedError(
                            '{}: atrous_conv2d_transpose has not been implemented.'.format(self.name_in_err))
                    else:
                        # conv2d_transpose needs specified output_shape including the batch size, which
                        # may change during training and test
                        output_shape = [op_input.get_shape().as_list()[0]] + self.output_shape[1:]
                        op_output = tf.nn.conv2d_transpose(
                            op_input, self.kernel, output_shape,
                            self.strides, self.design['padding'], self.data_format_alias)
                elif self.design['op'] == 'sc':  # separate conv layer
                    op_output = tf.nn.separable_conv2d(
                        op_input, self.kernel[0], self.kernel[1], self.strides, self.design['padding'],
                        data_format=self.data_format_alias, rate=self.dilation)
                elif self.design['op'] == 'max':  # max pool
                    op_output = tf.nn.max_pool(
                        op_input, self.kernel_shape, self.strides, self.design['padding'],
                        data_format=self.data_format_alias)
                elif self.design['op'] == 'avg':  # ave pool
                    op_output = tf.nn.avg_pool(
                        op_input, self.kernel_shape, self.strides, self.design['padding'],
                        data_format=self.data_format_alias)
                elif self.design['op'] == 'sum':
                    op_output = tf.nn.avg_pool(
                        op_input, self.kernel_shape, self.strides, self.design['padding'],
                        data_format=self.data_format_alias)
                    op_output = op_output * self.design['kernel'] ** 2
                elif self.design['op'] in {'b', 'bias'}:
                    if len(self.input_shape) == 2:
                        op_output = tf.add(op_input, self.kernel)
                    elif len(self.input_shape) == 4:
                        op_output = tf.nn.bias_add(op_input, self.kernel, self.data_format_alias)
                    else:
                        raise AttributeError('{}: does not support bias_add operation.'.format(self.name_in_err))
                elif self.design['op'] in {'bn', 'batch_norm'}:
                    axis = 1 if self.data_format == 'channels_first' else -1
                    center = self.design['bn_center'] if 'bn_center' in self.design else True
                    scale = self.design['bn_scale'] if 'bn_scale' in self.design else True
                    beta_init = self.design['bn_b_init'] if 'bn_b_init' in self.design else tf.zeros_initializer()
                    gamma_init = self.design['bn_w_init'] if 'bn_w_init' in self.design else tf.ones_initializer()
                    gamma_constraint = self.design['w_const'] if 'bn_w_const' in self.design else None
                    op_output = tf.layers.batch_normalization(
                        op_input, axis=axis,
                        center=center, scale=scale,  # control whether beta and gamma are used
                        beta_initializer=beta_init,  # initializer for beta and gamma
                        gamma_initializer=gamma_init,
                        gamma_constraint=gamma_constraint,  # constraint for gamma
                        training=is_training, renorm=False, fused=True, name='BN')
                elif self.design in {'lrn'}:
                    op_output = local_response_normalization(op_input, data_format=self.data_format)
                else:
                    raise AttributeError('{}: type {} not supported'.format(self.name_in_err, self.design['op']))

                self._output_check_(op_output)
                return op_output

    def apply(self, op_input, control_ops=None, is_training=True):
        """

        :param op_input:
        :param control_ops:
        :param is_training:
        :return:
        """
        return self.__call__(op_input, control_ops, is_training)


def spatial_shape_after_conv(input_spatial_shape, kernel_size, strides, dilation, padding):
    """ This function calculates the spatial shape after conv layer.

    The formula is obtained from: https://www.tensorflow.org/api_docs/python/tf/nn/convolution
    It should be note that current function assumes PS is done before conv

    :param input_spatial_shape:
    :param kernel_size:
    :param strides:
    :param dilation:
    :param padding:
    :return:
    """
    if padding in ['same', 'SAME']:
        return np.int(np.ceil(input_spatial_shape / strides))
    else:
        return np.int(np.ceil((input_spatial_shape - (kernel_size - 1) * dilation) / strides))


def spatial_shape_after_transpose_conv(input_spatial_shape, kernel_size, strides, dilation, padding):
    """ This function calculates the spatial shape after conv layer.

    Since transpose conv is often used in upsampling, scale_factor is not used here.

    This function has not been fully tested, and may be wrong in some cases.

    :param input_spatial_shape:
    :param kernel_size:
    :param strides:
    :param dilation:
    :param padding:
    :return:
    """
    if padding in ['same', 'SAME']:
        return np.int(input_spatial_shape * strides)
    else:
        return np.int(input_spatial_shape * strides + (kernel_size - 1) * dilation)


class ImageScaling(object):
    """ This class defines the sampling operation applied to the input

    """
    def __init__(self, design, input_shape, name_scope=None, scope_prefix='', data_format=None):
        """ This function initialize the operation

        :param design: dict with keys:
            'method': 'ps' - periodic shuffling, 'bil' - bilinear, 'bic' - bicubic,
                'avg' or 'max' - pooling (can only be used in downsampling)
                'unpool' - unpooling (can only be used in upsampling)
            'factor': scalar, positive - upsampling, negative - downsampling
            'size': target size
        :param input_shape:
        :param name_scope:
        :param scope_prefix:
        :param data_format:
        """
        self.method = design['method']
        self.name_scope = 'scale' if name_scope is None else name_scope
        self.name_in_err = scope_prefix + self.name_scope
        # IO
        self.input_shape = input_shape
        self.data_format = data_format
        if self.data_format == 'channels_first':
            self.data_format_alias = 'NCHW'
        elif self.data_format == 'channels_last':
            self.data_format_alias = 'NHWC'
        else:
            self.data_format_alias = self.data_format

        # calculate output_shape
        self._get_shape_(design)

    def _get_shape_(self, design):
        """ This function calculates the output shape

        :param design:
        :return:
        """
        self.factor = design['factor'] if 'factor' in design else 2
        new_h, new_w = design['size'] if 'size' in design else (None, None)
        if self.data_format == 'channels_first':
            fan_in, h, w = self.input_shape[1:]
        else:
            h, w, fan_in = self.input_shape[1:]
        if self.factor is None:
            assert new_h / h == new_w / w, 'the factors on height and width do not equal.'
            self.factor = int(new_h / h) if new_h > h else -int(h / new_h)
        elif self.factor > 0:
            new_h = int(h * self.factor)
            new_w = int(w * self.factor)
        elif self.factor < 0:
            new_h = int(-h / self.factor)
            new_w = int(-w / self.factor)
        else:
            raise AttributeError('{}: factor cannot be zero.'.format(self.name_in_err))
        # check if self.method is valid
        if self.factor > 0 and self.method in {'avg', 'max'}:
            raise AttributeError('{}: {} can only be used for downsampling'.format(self.name_in_err, self.method))
        if self.factor < 0 and self.method in {'unpool'}:
            raise AttributeError('{}: {} can only be used for upsampling'.format(self.name_in_err, self.method))
        if not self.factor == 2 and self.method in {'unpool'}:
            raise AttributeError('{}: {} can only deal with factor = 2'.format(self.name_in_err, self.method))
        # periodic shuffling will change the number of channels
        if self.method == 'ps':
            fan_out = int(fan_in * h * w / new_h / new_w)
        else:
            fan_out = fan_in
        self.output_shape = [self.input_shape[0], fan_out, new_h, new_w] \
            if self.data_format == 'channels_first' else [self.input_shape[0], new_h, new_w, fan_out]

    def _output_check_(self, op_output):
        """ Check the shape of the output

        :param op_output:
        :return:
        """
        output_shape = op_output.get_shape().as_list()
        assert self.output_shape[1:] == output_shape[1:], \
            '{}: the output shape {} does not match existed shape {}.'.format(
                self.name_in_err, output_shape[1:], self.output_shape[1:])

    def __call__(self, op_input, control_ops=None):
        """

        :param op_input:
        :param control_ops:
        :return:
        """
        with tf.variable_scope(self.name_scope, reuse=tf.AUTO_REUSE):
            with tf.control_dependencies(control_ops):
                if self.method == 'ps':
                    scale_up = self.factor > 0
                    op_output = periodic_shuffling(op_input, abs(self.factor), scale_up, data_format=self.data_format)
                elif self.method == 'bil':
                    # tf.resize_bilinear only supports NHWC
                    if self.data_format == 'channels_first':
                        size = self.output_shape[2:4]
                        op_input = tf.transpose(op_input, perm=(0, 2, 3, 1))  # NCHW to NHWC
                        op_output = tf.image.resize_bilinear(op_input, size, align_corners=True)
                        op_output = tf.transpose(op_output, perm=(0, 3, 1, 2))  # NHWC to NCHW
                    else:
                        size = self.output_shape[1:3]
                        op_output = tf.image.resize_bilinear(op_input, size, align_corners=True)
                elif self.method == 'bic':
                    # tf.resize_bicubic only supports NHWC
                    if self.data_format == 'channels_first':
                        size = self.output_shape[2:4]
                        op_input = tf.transpose(op_input, perm=(0, 2, 3, 1))  # NCHW to NHWC
                        op_output = tf.image.resize_bicubic(op_input, size, align_corners=True)
                        op_output = tf.transpose(op_output, perm=(0, 3, 1, 2))  # NHWC to NCHW
                    else:
                        size = self.output_shape[1:3]
                        op_output = tf.image.resize_bicubic(op_input, size, align_corners=True)
                elif self.method == 'max':  # max pool
                    factor = -self.factor
                    kernel = [1, 1, factor, factor] if self.data_format == 'channels_first' else [1, factor, factor, 1]
                    strides = kernel
                    op_output = tf.nn.max_pool(op_input, kernel, strides, 'SAME', data_format=self.data_format_alias)
                elif self.method == 'avg':  # ave pool
                    factor = -self.factor
                    kernel = [1, 1, factor, factor] if self.data_format == 'channels_first' else [1, factor, factor, 1]
                    strides = kernel
                    op_output = tf.nn.avg_pool(op_input, kernel, strides, 'SAME', data_format=self.data_format_alias)
                elif self.method == 'unpool':
                    concat_axis = 1 if self.data_format == 'channels_first' else 3
                    op_output = tf.concat([op_input, op_input, op_input, op_input], axis=concat_axis)
                    op_output = periodic_shuffling(op_output, 2, True, data_format=self.data_format)
                else:
                    raise NotImplementedError('{}: Method {} not implemented.'.format(
                        self.name_in_err, self.method))

                self._output_check_(op_output)
                return op_output

    def apply(self, op_input, control_ops=None):
        """

        :param op_input:
        :param control_ops:
        :return:
        """
        return self.__call__(op_input, control_ops)


########################################################################
def update_layer_design(layer_design):
    """ This function reads layer_design and outputs an universal layer design dictionary

    :param layer_design: layer_design may have following keys:
        'name': scope of the layer
        'type': 'default' - default layer layout; 'res' - residual block with scaling supported;
            'res_i' - residual block with identity shortcut
        'op': 'c' - conv layer; 'd' - dense layer; 'sc' - separable conv layer; 'i' - identity layer;
            'tc' - transpose conv; a list of strings
        'out': number of output channels or features
        'bias': if bias should be used. When 'w_nm' is 'b', bias is not used
        'act': activation function
        'act_nm': activation normalization method.
            'lrn' - local response normalization
            'b' - batch normalization; 'bns' - batch normalization with no scale
        'act_k': activation multiplier. When 'w_nm' is 's', the user can choose a multiply the activation with a
            constant to reimburse the norm loss caused by the activation.
        'w_nm': kernel normalization method.
            's' - spectral normalization;
            'h' - he normalization;
            None - no normalization is used.
        'w_p': kernel penalization method.
            's' - spectral penalization;
            None - no layer-wise penalty is used.
        'kernel': kernel size for conv layer; integer, or list/tuple of integers for multiple conv ops
        'strides': strides for conv layer; integer, or list/tuple of integers for multiple conv ops
        'dilation': dilation for conv layer; integer, or list/tuple of integers for multiple conv ops
        'padding': 'SAME' or 'VALID'; padding for conv layer; string, or list/tuple of strings for multiple conv ops
        'scale': a list containing the method used for upsampling and the scale factor;
            a positive scale factor means upsampling, a negative means downsampling
            None: do not apply scaling
            'ps': periodic shuffling, the factor can only be int
            'bil': bilinear sampling
            'bic': bicubic sampling
            'avg' or 'max': average pool - can only be used in downsampling
        'in_reshape': a shape list, reshape the input before passing it to kernel
        'out_reshape': a shape list, reshape the output before passing it to next layer
        'aux': auxiliary values and commands
    :return:
    """
    template = {'name': None, 'type': 'default', 'op': 'c', 'out': None, 'bias': True,
                'act': 'linear', 'act_nm': None, 'act_k': False,
                'w_nm': None, 'w_p': None,
                'kernel': 3, 'strides': 1, 'dilation': 1, 'padding': 'SAME', 'scale': None,
                'in_reshape': None, 'out_reshape': None, 'aux': None}
    # update template with parameters from layer_design
    for key in layer_design:
        template[key] = layer_design[key]
    # check template values to avoid error
    # if (template['act_nm'] in ['b', 'bn', 'BN']) and (template['act'] == 'linear'):
    #     template['act_nm'] = None  # batch normalization is not used with linear activation
    if template['act_nm'] in ['b', 'bn', 'BN']:
        template['bias'] = False  # batch normalization is not used with bias
    if template['op'] in {'tc'}:
        # transpose conv is usually used as upsampling method
        template['scale'] = None
    if template['w_nm'] is not None:  # weight normalization and act normalization cannot be used together
        template['act_nm'] = None
    if template['scale'] is not None:
        assert isinstance(template['scale'], (list, tuple)), \
            'Value for key "scale" must be list or tuple.'
    if template['w_nm'] is not None:  # This is because different normalization methods do not work in the same layer
        assert not isinstance(template['w_nm'], (list, tuple)), \
            'Value for key "w_nm" must not be list or tuple.'

    # output template
    if template['op'] == 'd':
        return {key: template[key]
                for key in ['name', 'op', 'type', 'out', 'bias',
                            'act', 'act_nm', 'act_k',
                            'w_nm', 'w_p',
                            'in_reshape', 'out_reshape', 'aux']}
    elif template['op'] in ['sc', 'c', 'tc', 'avg', 'max', 'sum']:
        return {key: template[key]
                for key in ['name', 'op', 'type', 'out', 'bias',
                            'act', 'act_nm', 'act_k',
                            'w_nm', 'w_p',
                            'kernel', 'strides', 'dilation', 'padding', 'scale',
                            'in_reshape', 'out_reshape', 'aux']}
    elif template['op'] in {'i'}:
        return {key: template[key]
                for key in ['name', 'op', 'act', 'act_nm', 'type', 'in_reshape', 'out_reshape']}
    else:
        raise AttributeError('layer op {} not supported.'.format(template['op']))


class Layer(object):
    """ This class defines the operations applied to the input

    """
    def __init__(self, design, input_shape=None, name_prefix='', data_format=None):
        """ This function defines the structure of layer

        :param design: see update_layer_design function
        :param input_shape:
        :param name_prefix:
        :param data_format:
        """
        # layer definition as dictionary
        self.design = design
        # scope
        self.layer_scope = name_prefix + self.design['name']
        # IO
        if input_shape is None:
            self.input_shape = None
        else:
            self.input_shape = input_shape if isinstance(input_shape, list) else list(input_shape)
        self.output_shape = None
        if data_format in {'channels_first', 'NCHW'}:
            self.data_format = 'channels_first'
            self.data_format_alias = 'NCHW'
        elif data_format in {'channels_last', 'NHWC'}:
            self.data_format = 'channels_last'
            self.data_format_alias = 'NHWC'
        else:
            self.data_format_alias = self.data_format = data_format
        # set up other values
        self._layer_output_ = None
        self._register_ = None  # _register_ is used to store info other than _layer_out_ that would be used later
        # layer status
        self.is_layer_built = False
        # ops
        self.ops = {}

    def _input_(self, layer_input):
        """ This function initializes layer_output.

        :param layer_input:
        :return:
        """
        if layer_input is None:  # Layer object always receives new data
            raise AttributeError('{}: input is not given.'.format(self.layer_scope))
        input_shape = layer_input.get_shape().as_list()
        if self.input_shape is None:
            self.input_shape = input_shape
        else:
            assert self.input_shape[1:] == input_shape[1:], \
                '{}: the input shape {} does not match existed shape {}.'.format(
                    self.layer_scope, input_shape[1:], self.input_shape[1:])
        self._layer_output_ = layer_input

    def _output_(self):
        """ This function returns the output
        :return:
        """
        output_shape = self._layer_output_.get_shape().as_list()
        if self.output_shape is None:
            self.output_shape = output_shape
        else:
            assert self.output_shape[1:] == output_shape[1:], \
                '{}: the output shape {} does not match existed shape {}.'.format(
                    self.layer_scope, output_shape[1:], self.output_shape[1:])
        # Layer object always forgets self._layer_output_ after its value returned
        layer_output = self._layer_output_
        self._layer_output_ = None
        return layer_output

    def get_register(self):
        """ This function returns the registered tensor

        :return:
        """
        # Layer object always forgets self._register_ after its value returned
        registered_info = self._register_
        self._register_ = None
        return registered_info

    def _add_image_scaling_(self, input_shape, name_scope='sampling'):
        """ This function registers a sampling process

        :param input_shape:
        :param name_scope:
        :return:
        """
        design = {'method': self.design['scale'][0], 'factor': self.design['scale'][1]}
        op = ImageScaling(
            design, input_shape, name_scope=name_scope,
            scope_prefix=self.layer_scope + '/', data_format=self.data_format)
        self.ops[name_scope] = op

        return op.output_shape

    def _add_kernel_(self, input_shape, name_scope='kernel', index=None, op=None, init_w_scale=None):
        """ This function registers a kernel

        :param input_shape:
        :param name_scope:
        :param index: in case multiple kernels of different shapes are used, 'kernel', 'strides',
            'dilation', 'padding' could be provided as a list.
        :param op: if self.design['op'] should not be used, provide an op here
        :param init_w_scale: some op requires different weight initialization scheme
        :return:
        """
        design = {'op': self.design['op']} if op is None else {'op': op}
        if init_w_scale is not None:
            design['init_w_scale'] = init_w_scale
        # acquire kernel definition from self.design
        target_keys = {'out', 'act', 'act_k', 'w_nm', 'kernel', 'strides', 'dilation', 'padding'}
        for key in target_keys:
            if key in self.design:
                if index is not None and isinstance(self.design[key], (list, tuple)):
                    design[key] = self.design[key][index]
                else:
                    design[key] = self.design[key]

        # register the op
        op = ParametricOperation(
            design, input_shape, name_scope=name_scope,
            scope_prefix=self.layer_scope + '/', data_format=self.data_format)
        self.ops[name_scope] = op

        return op.output_shape

    def _add_scalar_kernel_(self, input_shape, name_scope='scalar_kernel', init_w_scale=None, bound=None):
        """ This function registers an operation called multiplication by trainable scalar kernel.
        This op is required in only a few layers like non-local layer (type nl_0). Thus, we do not include it in
        update_layer_design.

        :param input_shape:
        :param name_scope:
        :param bound:
        :param init_w_scale:
        :return:
        """
        design = {'op': 'k'}
        if init_w_scale is not None:
            design['init_w_scale'] = init_w_scale
        if bound is not None:
            assert isinstance(bound, (list, tuple)), \
                '{}: Bound must be list or tuple. Got {}'.format(self.layer_scope, type(bound))
            design['bound'] = bound

        # register the op
        op = ParametricOperation(
            design, input_shape, name_scope=name_scope,
            scope_prefix=self.layer_scope + '/', data_format=self.data_format)
        self.ops[name_scope] = op

        return op.output_shape

    def _add_bias_(self, input_shape, name_scope='bias'):
        """ This function registers a bias-adding process

        :param input_shape:
        :param name_scope:
        :return:
        """
        design = {'op': 'b'}
        op = ParametricOperation(
            design, input_shape, name_scope=name_scope,
            scope_prefix=self.layer_scope + '/', data_format=self.data_format)
        self.ops[name_scope] = op

        return op.output_shape

    def _add_bn_(self, input_shape, name_scope='bias', center=None, scale=None, b_init=None, w_init=None, w_const=None):
        """ This function registers a batch normalization process

        :param input_shape:
        :param name_scope:
        :param center: whether to add offset beta
        :param scale: whether to multiply by gamma
        :param b_init: initializer for beta; if not provided, tf.zeros_initializer()
        :param w_init: initializer for gamma; if not provided, tf.ones_initializer()
        :param w_const: constraint for gamma; if not provided, None
        :return:
        """
        design = {'op': 'bn'}
        if center is not None:
            design['bn_center'] = center
        if scale is not None:
            design['bn_scale'] = scale
        if b_init is not None:
            design['bn_b_init'] = b_init
        if w_init is not None:
            design['bn_w_init'] = w_init
        if w_const is not None:
            design['bn_w_const'] = w_const

        op = ParametricOperation(
            design, input_shape, name_scope=name_scope,
            scope_prefix=self.layer_scope + '/', data_format=self.data_format)
        self.ops[name_scope] = op

        return op.output_shape

    def _apply_activation_(self, layer_input):
        return apply_activation(layer_input, self.design['act'])

    def _apply_input_reshape_(self):
        if self.design['in_reshape'] is not None:
            batch_size = self._layer_output_.get_shape().as_list()[0]
            self._layer_output_ = tf.reshape(self._layer_output_, shape=[batch_size] + self.design['in_reshape'])

    def _apply_output_reshape_(self):
        if self.design['out_reshape'] is not None:
            batch_size = self._layer_output_.get_shape().as_list()[0]
            self._layer_output_ = tf.reshape(self._layer_output_, shape=[batch_size] + self.design['out_reshape'])

    def _add_layer_default_(self, input_shape):
        """ This function adds the default layer

        The order of operations:
        upsampling - kernel - bias - BN - downsampling

        :param input_shape:
        :return:
        """
        # upsampling
        if 'scale' in self.design and self.design['scale'] is not None:
            if self.design['scale'][1] > 0:  # upsampling
                input_shape = self._add_image_scaling_(input_shape, 'upsampling')
        # kernel
        input_shape = self._add_kernel_(input_shape, 'kernel')
        # bias
        if 'bias' in self.design and self.design['bias']:
            input_shape = self._add_bias_(input_shape, 'bias')
        # batch normalization
        if self.design['act_nm'] in {'b', 'bn', 'BN'}:
            input_shape = self._add_bn_(input_shape, 'BN')
        # activation
        # donwsampling
        if 'scale' in self.design and self.design['scale'] is not None:
            if self.design['scale'][1] < 0:  # downsampling
                input_shape = self._add_image_scaling_(input_shape, 'downsampling')

        return input_shape

    def _apply_layer_default_(self, is_training=True):
        """ This function applies the default layer

        The order of operations:
        upsampling - kernel - bias - BN - downsampling

        :param is_training:
        :return:
        """
        # upsampling
        if 'upsampling' in self.ops:
            self._layer_output_ = self.ops['upsampling'].apply(self._layer_output_)
        # kernel
        self._layer_output_ = self.ops['kernel'].apply(self._layer_output_)
        # bias
        if 'bias' in self.ops:
            self._layer_output_ = self.ops['bias'].apply(self._layer_output_)
        # batch normalization
        if 'BN' in self.ops:
            self._layer_output_ = self.ops['BN'].apply(self._layer_output_, is_training=is_training)
        # activation
        self._layer_output_ = self._apply_activation_(self._layer_output_)
        # downsampling
        if 'downsampling' in self.ops:
            self._layer_output_ = self.ops['downsampling'].apply(self._layer_output_)

    def _add_layer_res_(self, input_shape):
        """ This function adds resnet block

        The order of operations:
        res branch: BN_0 - act - upsampling_0 - kernel_0 - bias_0 - BN_1 - act - kernel_1 - bias_1 - downsampling_0
        sc branch: upsampling_1 - kernel_sc - bias_sc - downsampling_1

        :param input_shape:
        :return:
        """
        # res branch
        # BN
        if self.design['act_nm'] in {'b', 'bn', 'BN'}:
            res_shape = self._add_bn_(input_shape, 'BN_0')
        else:
            res_shape = input_shape
        # activation
        # upsampling
        if 'scale' in self.design and self.design['scale'] is not None:
            if self.design['scale'][1] > 0:  # upsampling
                res_shape = self._add_image_scaling_(res_shape, 'upsampling_0')
        # kernel
        # print('kernel_0_in {}'.format(res_shape))
        res_shape = self._add_kernel_(res_shape, 'kernel_0', 0)
        # print('kernel_0_out {}'.format(res_shape))
        # bias
        if 'bias' in self.design and self.design['bias']:
            res_shape = self._add_bias_(res_shape, 'bias_0')
        # BN
        if self.design['act_nm'] in {'b', 'bn', 'BN'}:
            res_shape = self._add_bn_(res_shape, 'BN_1')
        # activation
        # kernel
        if self.design['op'] == 'tc':  # in res block that uses tc, the second conv is c
            res_shape = self._add_kernel_(res_shape, 'kernel_1', 1, op='c')
        else:
            res_shape = self._add_kernel_(res_shape, 'kernel_1', 1)
        # print('kernel_1_out {}'.format(res_shape))
        # bias
        if 'bias' in self.design:  # here I guess the bias should be kept
            res_shape = self._add_bias_(res_shape, 'bias_1')
        # downsampling
        if 'scale' in self.design and self.design['scale'] is not None:
            if self.design['scale'][1] < 0:  # downsampling
                res_shape = self._add_image_scaling_(res_shape, 'downsampling_0')
        # print('downsampling_0_out {}'.format(res_shape))

        # shortcut branch
        sc_shape = input_shape
        if self.design['type'] == 'res':  # for 'res_i', the shortcut branch is linear
            # upsampling
            if 'scale' in self.design and self.design['scale'] is not None:
                if self.design['scale'][1] > 0:  # upsampling
                    sc_shape = self._add_image_scaling_(sc_shape, 'upsampling_1')
            # kernel
            sc_shape = self._add_kernel_(sc_shape, 'kernel_sc', 2)
            # print('kernel_2_out {}'.format(sc_shape))
            # bias
            if 'bias' in self.design:  # here I guess the bias should be kept
                sc_shape = self._add_bias_(sc_shape, 'bias_sc')
            # downsampling
            if 'scale' in self.design and self.design['scale'] is not None:
                if self.design['scale'][1] < 0:  # downsampling
                    sc_shape = self._add_image_scaling_(sc_shape, 'downsampling_1')
        elif self.design['type'] == 'res_v1':
            # in wgan-gp paper, the shortcut in the first resnet block in discriminator has a downsample - conv order
            # with kernel size 1 and it has bias
            # downsampling
            if 'scale' in self.design and self.design['scale'] is not None:
                if self.design['scale'][1] < 0:  # downsampling
                    sc_shape = self._add_image_scaling_(sc_shape, 'downsampling_1')
                else:
                    raise AttributeError('{}: res_v1 is only used with downsampling.'.format(self.layer_scope))
            # kernel
            sc_shape = self._add_kernel_(sc_shape, 'kernel_sc', 2)
            # bias
            if 'bias' in self.design:  # here I guess the bias should be kept
                sc_shape = self._add_bias_(sc_shape, 'bias_sc')

        # check if sc_shape equals res_shape
        assert sc_shape == res_shape, \
            '{}: Resnet shape {} and shortcut shape {} do not match.'.format(self.layer_scope, res_shape, sc_shape)
        output_shape = sc_shape

        return output_shape

    def _apply_layer_res_(self, is_training=True):
        """ This function applies resnet block

        The order of operations:
        res branch: BN_0 - act - upsampling_0 - kernel_0 - bias_0 - BN_1 - act - kernel_1 - bias_1 - downsampling_0
        sc branch: upsampling_1 - kernel_sc - bias_sc - downsampling_1

        :param is_training:
        :return:
        """
        # res branch
        res_out = self._layer_output_
        if not self.design['type'] == 'res_v1':
            # batch normalization BN_0
            if 'BN_0' in self.ops:
                res_out = self.ops['BN_0'].apply(res_out, is_training=is_training)
            # activation
            res_out = self._apply_activation_(res_out)
        # upsampling_0
        if 'upsampling_0' in self.ops:
            res_out = self.ops['upsampling_0'].apply(res_out)
        # kernel_0
        res_out = self.ops['kernel_0'].apply(res_out)
        # bias_0
        if 'bias_0' in self.ops:
            res_out = self.ops['bias_0'].apply(res_out)
        # batch normalization
        if 'BN_1' in self.ops:
            res_out = self.ops['BN_1'].apply(res_out, is_training=is_training)
        # activation
        res_out = self._apply_activation_(res_out)
        # kernel_1
        res_out = self.ops['kernel_1'].apply(res_out)
        # bias_1
        if 'bias_1' in self.ops:
            res_out = self.ops['bias_1'].apply(res_out)
        # downsampling_0
        if 'downsampling_0' in self.ops:
            res_out = self.ops['downsampling_0'].apply(res_out)

        # shortcut branch
        sc_out = self._layer_output_
        if self.design['type'] == 'res':  # for 'res_i', the shortcut branch is linear
            # upsampling_1
            if 'upsampling_1' in self.ops:
                sc_out = self.ops['upsampling_1'].apply(sc_out)
            # kernel_sc
            sc_out = self.ops['kernel_sc'].apply(sc_out)
            # bias_1
            if 'bias_sc' in self.ops:
                sc_out = self.ops['bias_sc'].apply(sc_out)
            # downsampling_1
            if 'downsampling_1' in self.ops:
                sc_out = self.ops['downsampling_1'].apply(sc_out)
        elif self.design['type'] == 'res_v1':
            # in wgan-gp paper, the shortcut in the first resnet block in discriminator has a downsample - conv order
            if 'downsampling_1' in self.ops:
                sc_out = self.ops['downsampling_1'].apply(sc_out)
            # kernel_sc
            sc_out = self.ops['kernel_sc'].apply(sc_out)
            # bias_sc
            if 'bias_sc' in self.ops:
                sc_out = self.ops['bias_sc'].apply(sc_out)

        # layer_out
        self._layer_output_ = res_out + sc_out

    def _add_layer_nonlocal_(self, input_shape):
        """ This function adds non-local block

        There are two non-local block implementations. One from
        Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2018).
        Self-Attention Generative Adversarial Networks.
        The other from
        Wang, X., Girshick, R., Gupta, A., & He, K. (2017).
        Non-local Neural Networks.
        The difference is in the operations:
        (SAGAN) y = gamma * softmax(conv_f(x)' * conv_g(x)) * conv_h(x) + x
        (Nonlocal) y = BN(conv_e(softmax(conv_f(x)' * conv_g(x)) * conv_h(x))) + x;
        I shift the BN with conv_e to produce: y = conv_e(BN(softmax(conv_f(x)' * conv_g(x)) * conv_h(x))) + x;
        Thus, 'nl' or 'nl_0' is used to refer to SAGAN implementation, 'nl_1' to Non-local one, 'nl_2' to mine.

        :param input_shape:
        :return:
        """
        # attention map branch
        att_shape = input_shape  # NxH1xW1xC2 or NxC2xH1xW1
        # kernel
        att_shape_f = self._add_kernel_(att_shape, 'f_x', index=0)  # NxH1xW1xC1 or NxC1xH1xW1
        att_shape_g = self._add_kernel_(att_shape, 'g_x', index=1)  # NxH2xW2xC1 or NxC1xH2xW2
        # bias; the softmax operation makes it useless to add bias to g(x)
        att_shape_f = self._add_bias_(att_shape_f, 'bias_f')
        # if 'bias' in self.design and self.design['bias']:
        #     att_shape_f = self._add_bias_(att_shape_f, 'bias_f')
        #     att_shape_g = self._add_bias_(att_shape_g, 'bias_g')
        # reshape and matrix multiplication b = f'*g
        # activation softmax
        # feature map
        att_shape = self._add_kernel_(att_shape, 'h_x', index=2)  # NxH2xW2xC2 or NxC2xH2xW2
        # check shape
        if self.data_format == 'channels_first':
            assert att_shape_f[1] == att_shape_g[1], \
                '{}: f(x) channel {} does not match g(x) channel {}'.format(  # f.C1 == g.C1
                    self.layer_scope, att_shape_f[1], att_shape_g[1])
            assert att_shape_g[2:4] == att_shape[2:4], \
                '{}: g(x) size {} does not match h(x) size {}'.format(  # g.H2 == h.H2, g.W2 == h.W2
                    self.layer_scope, att_shape_g[2:4], att_shape[2:4])
        elif self.data_format == 'channels_last':
            assert att_shape_f[-1] == att_shape_g[-1], \
                '{}: f(x) channel {} does not match g(x) channel {}'.format(
                    self.layer_scope, att_shape_f[-1], att_shape_g[-1])
            assert att_shape_g[1:3] == att_shape[1:3], \
                '{}: g(x) size {} does not match h(x) size {}'.format(
                    self.layer_scope, att_shape_g[1:3], att_shape[1:3])
        else:
            raise AttributeError(
                '{}: the non-local block only supports channels_first or channels_first data format. Got {}'.format(
                    self.layer_scope, self.data_format))
        # reshape and matrix multiplication o = beta*h

        # add following kernel or scale kernel
        if self.design['type'] in {'nl', 'nl_0'}:
            if self.design['act_nm'] in {'b', 'bn', 'BN'}:  # scaling is learnt by next scale kernel
                att_shape = self._add_bn_(att_shape, name_scope='BN_0', scale=False)
            # add scale kernel
            bound = [-1.0, 1.0] if self.design['w_nm'] == 's' else None
            att_shape = self._add_scalar_kernel_(att_shape, 'k_x', init_w_scale=0.0, bound=bound)
        elif self.design['type'] in {'nl_1'}:
            # kernel
            att_shape = self._add_kernel_(att_shape, 'k_x', index=3)
            # batch normalization
            if self.design['type'] == 'nl_1' and self.design['act_nm'] in {'b', 'bn', 'BN'}:
                att_shape = self._add_bn_(  # when spectral normalization is applied, the network must be bounded
                    att_shape, 'BN_0', w_init=tf.zeros_initializer(),
                    w_const=lambda gamma: tf.clip_by_value(gamma, -1.0, 1.0) if self.design['w_nm'] == 's' else None)
        elif self.design['type'] in {'nl_2'}:
            # batch normalization
            if self.design['act_nm'] in {'b', 'bn', 'BN'}:  # scaling is learnt by next kernel
                att_shape = self._add_bn_(att_shape, name_scope='BN_0', scale=False)
            # add kernel
            att_shape = self._add_kernel_(att_shape, 'k_x', index=3, init_w_scale=0.0)
        # add bias to final attention map
        if 'bias' in self.design and self.design['bias']:
            att_shape = self._add_bias_(att_shape, 'bias_k')

        # shortcut branch
        output_shape = input_shape
        # check shape
        assert output_shape == att_shape, \
            '{}: attention map shape {} does not match input shape {}'.format(
                self.layer_scope, att_shape, input_shape)
        output_shape = input_shape

        return output_shape

    def _apply_layer_nonlocal_(self, is_training=True):
        """ This function applies non-local block

        There are two non-local block implementations. One from
        Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2018).
        Self-Attention Generative Adversarial Networks.
        The other from
        Wang, X., Girshick, R., Gupta, A., & He, K. (2017).
        Non-local Neural Networks.
        The difference is in the operations:
        (SAGAN) y = gamma * softmax(conv_f(x)' * conv_g(x)) * conv_h(x) + x
        (Nonlocal) y = BN(conv_e(softmax(conv_f(x)' * conv_g(x)) * conv_h(x))) + x;
        I shift the BN with conv_e to produce: y = conv_e(BN(softmax(conv_f(x)' * conv_g(x)) * conv_h(x))) + x;
        Thus, 'nl' or 'nl_0' is used to refer to SAGAN implementation, 'nl_1' to Non-local one, 'nl_2' to mine.

        :param is_training:
        :return:
        """
        att_out = self._layer_output_
        input_shape = self._layer_output_.get_shape().as_list()
        # apply kernel to get attention map
        # get embedding
        att_out_f = self.ops['f_x'].apply(att_out)  # NxH1xW1xC1 or NxC1xH1xW1
        att_out_f = self.ops['bias_f'].apply(att_out_f)
        att_out_g = self.ops['g_x'].apply(att_out)  # NxH2xW2xC1 or NxC1xH2xW2
        # flatten the tensor and do batch multiplication
        att_shape_f = tf.shape(att_out_f)  # att_out_f and att_out_g have the same batch_size and channel
        with tf.name_scope('att_map'):
            if self.data_format == 'channels_first':
                att_out_f = tf.reshape(att_out_f, shape=(att_shape_f[0], att_shape_f[1], -1))  # NxC1xHW1
                att_out_g = tf.reshape(att_out_g, shape=(att_shape_f[0], att_shape_f[1], -1))  # NxC1xHW2
                att_map_logits = tf.matmul(tf.transpose(att_out_f, [0, 2, 1]), att_out_g)  # NxHW1xHW2
            else:  # channels_last
                att_out_f = tf.reshape(att_out_f, shape=(att_shape_f[0], -1, att_shape_f[-1]))  # NxHW1xC1
                att_out_g = tf.reshape(att_out_g, shape=(att_shape_f[0], -1, att_shape_f[-1]))  # NxHW2xC1
                att_map_logits = tf.matmul(att_out_f, tf.transpose(att_out_g, [0, 2, 1]))  # NxHW1xHW2
            # apply softmax to each row of att_map
            att_map = tf.nn.softmax(att_map_logits, axis=2)  # NxHW1xHW2

        # get attention feature map
        # get embedding
        att_out_h = self.ops['h_x'].apply(att_out)  # NxH2xW2xC2 or NxC2xH2xW2
        att_shape_h = tf.shape(att_out_h)
        with tf.name_scope('att_features'):
            if self.data_format == 'channels_first':
                att_out_h = tf.reshape(att_out_h, shape=(att_shape_h[0], att_shape_h[1], -1))  # NxC2xHW2
                att_out = tf.matmul(att_out_h, tf.transpose(att_map, [0, 2, 1]))  # NxC2xHW1
                att_out = tf.reshape(  # NxC2xH1xW1
                    att_out,
                    shape=(att_shape_h[0], att_shape_h[1], att_shape_f[2], att_shape_f[3]))
            else:  # channels_last
                att_out_h = tf.reshape(att_out_h, shape=(att_shape_h[0], -1, att_shape_h[-1]))  # NxHW2xC2
                att_out = tf.matmul(att_map, att_out_h)  # NxHW1xC2
                att_out = tf.reshape(  # NxH1xW1xC2
                    att_out,
                    shape=(att_shape_h[0], att_shape_f[0], att_shape_f[1], att_shape_h[-1]))
        # this is done to pass shape check in ParametricOperation
        att_out.set_shape(input_shape)  # NxC2xH1xW1 or NxH1xW1xC2

        # apply following kernel or scale kernel
        if self.design['type'] in {'nl', 'nl_0', 'nl_2'}:
            if 'BN_0' in self.ops:
                att_out = self.ops['BN_0'].apply(att_out, is_training=is_training)
            att_out = self.ops['k_x'].apply(att_out)
        elif self.design['type'] in {'nl_1'}:
            att_out = self.ops['k_x'].apply(att_out)
            if 'BN_0' in self.ops:
                att_out = self.ops['BN_0'].apply(att_out, is_training=is_training)
        if 'bias_k' in self.ops:
            att_out = self.ops['bias_k'].apply(att_out)
        # layer_out
        self._layer_output_ = att_out + self._layer_output_

    def build_layer(self):
        """ This function builds the layer by adding all operations

        There are two static operations: in_reshape and out_reshape
        Other operations are defined by design['type']:
        'default': upsampling - kernel - add bias / batch_norm - act - downsampling

        :return:
        """
        if not self.is_layer_built:
            # in case input is reshaped, the new shape is used
            if self.design['in_reshape'] is None:
                input_shape = self.input_shape
            else:
                input_shape = [self.input_shape[0]] + self.design['in_reshape']

            # register ops
            if self.design['type'] == 'default':
                input_shape = self._add_layer_default_(input_shape)
            elif self.design['type'] in {'res', 'res_i', 'res_v1'}:
                input_shape = self._add_layer_res_(input_shape)
            elif self.design['type'] in {'nl', 'nl_0', 'nl_1', 'nl_2'}:  # nl is the same as nl_0
                input_shape = self._add_layer_nonlocal_(input_shape)
            else:
                raise NotImplementedError(
                    '{}: {} is not implemented.'.format(self.layer_scope, self.design['type']))

            # in case output is reshaped, the new shape is used
            if self.design['out_reshape'] is None:
                self.output_shape = input_shape
            else:
                self.output_shape = [input_shape[0]] + self.design['out_reshape']

        self.is_layer_built = True

    def __call__(self, layer_input, is_training=True):
        """ This function calculates layer_output

        :param layer_input:
        :param is_training:
        :return:
        """
        self.build_layer()  # in case layer has not been build
        with tf.variable_scope(self.layer_scope, reuse=tf.AUTO_REUSE):
            self._input_(layer_input)
            self._apply_input_reshape_()

            # register ops
            if self.design['type'] == 'default':
                self._apply_layer_default_(is_training)
            elif self.design['type'] in {'res', 'res_i', 'res_v1'}:
                self._apply_layer_res_(is_training)
            elif self.design['type'] in {'nl', 'nl_0', 'nl_1', 'nl_2'}:  # nl is the same as nl_0
                self._apply_layer_nonlocal_(is_training)

            self._apply_output_reshape_()
            return self._output_()

    def apply(self, layer_input, is_training=True):
        return self.__call__(layer_input, is_training)


class Net(object):
    """ This class is designed to:
        1. allow constraints on layer weights and bias,
        2. ease construction of networks with complex structure that need to refer to specific layers

    """

    def __init__(
            self, net_design, net_name='net', data_format=None):
        """ This function initializes a network

        :param net_design: [(layer_type, channel_out, act_fun_name(, normalization_method, kernel_size, strides,
        dilation, padding, scale_factor))].
        The first five parameters are channel_out, kernel_size, stride, dilation, activation.
        The rest one is scale_factor, which is optional. When doing up-scaling, one extra conv
        layer is added.
        :param net_name:
        :param data_format:
        """
        # net definition
        self.net_def = net_design
        self.num_layers = len(net_design)
        # scope
        self.net_name = net_name  # get the parent scope

        # initialize the layers
        self.layers = []
        for i in range(self.num_layers):
            layer_design = update_layer_design(self.net_def[i])
            if layer_design['op'] in {'d', 'i'}:
                layer_data_format = None
            else:
                layer_data_format = data_format
            self.layers.append(
                Layer(
                    layer_design, name_prefix=self.net_name + '/',
                    data_format=layer_data_format))
        # self.layer_names = [layer.layer_scope for layer in self.layers]

    def get_layer_kernel_norm(self):
        """ This function gets the kernel norm for each kernel in each layer of the net

        :return:
        """
        layer_norms = {}
        for layer in self.layers:
            for key in layer.ops:
                kernel_norm = getattr(layer.ops[key], 'kernel_norm', None)
                if kernel_norm is not None:
                    layer_norms[layer.ops[key].name_in_err] = kernel_norm

        return layer_norms

    def get_layer_register(self, dict_form=False):
        """ This function collects the registered tensors for each layer in the net

        :param dict_form:
        :return:
        """
        if dict_form:
            registered_info = {}
            for layer in self.layers:
                with tf.variable_scope(layer.layer_scope, reuse=tf.AUTO_REUSE):
                    registered_info[layer.layer_scope] = layer.get_register()
        else:
            registered_info = []
            for layer in self.layers:
                with tf.variable_scope(layer.layer_scope, reuse=tf.AUTO_REUSE):
                    registered_info.append(layer.get_register())

        return registered_info


class Routine(object):
    """ This class initializes a tensor flow and infers the input / output shapes of each layer

    """

    def __init__(self, net_object):
        self.net = net_object
        self.operations = []
        self.layer_indices = []
        self.output_layer_indices = []
        self._out_temp_ = {}
        self.del_inserted = False
        self.output_added = False

    def add_input_layers(self, input_shape, out_layer_indices):
        """ This function adds input layers

        :param out_layer_indices: a list, e.g. [1, 2, 3], [1]
        :param input_shape: a list, e.g. [64, 256], [64, 3, 64, 64]
        :return:
        """
        for out_index in out_layer_indices:
            # check output layers and force each layer to have unique output
            if out_index in self.layer_indices:
                raise AttributeError('Layer {} has already been added.'.format(out_index))
            else:
                self.layer_indices.append(out_index)
            # register the input shape
            layer = self.net.layers[out_index]
            layer.input_shape = input_shape
            # infer the output shape
            layer.build_layer()
            # register operation
            self.operations.append([None, None, layer.apply, [out_index]])

    def link(self, in_layer_indices, out_layer_indices, input_fun=None):
        """ This layer links layers in in_layer_indices to layers in out_layer_indices.
        The link layers are designed such that each output layer has only one output.

        :param in_layer_indices: a list, e.g. [1], [1, 2]
        :param out_layer_indices: a list, e.g. [1], [1, 2]
        :param input_fun:
            <1 one input and multiple outputs: 'split' - split in channel dimension;
                None - input is passed to each output
            <2 multiple inputs and one output: 'concat' - concatenate in channel dimension;
                'sum' - sum up each inputs.
            <3 multiple inputs and multiple outputs: input_op is ignored and each input is passed to each output
            <4 one input and one output: None - input_op is ignored
        :return:
        """
        # check input layers and force layer dependency
        for in_index in in_layer_indices:
            if self.net.layers[in_index].output_shape is None:
                raise NotImplementedError('Input layer {} has not been defined yet.'.format(in_index))
        # check output layers and force each layer to have unique output
        for out_index in out_layer_indices:
            if out_index in self.layer_indices:
                raise AttributeError('Layer {} has already been linked.'.format(out_index))
            else:
                self.layer_indices.append(out_index)
        # get basic info
        num_in_layer = len(in_layer_indices)
        num_out_layer = len(out_layer_indices)

        if num_in_layer == num_out_layer:
            # assign each input layer to each output layer
            for index in range(num_in_layer):
                in_shape = self.net.layers[in_layer_indices[index]].output_shape[:]
                # define layer input shape
                layer = self.net.layers[out_layer_indices[index]]
                layer.input_shape = in_shape
                # infer the output shape
                layer.build_layer()
                # register operation
                self.operations.append([[in_layer_indices[index]], None, layer.apply, [out_layer_indices[index]]])

        elif num_in_layer > 1 and num_out_layer == 1:
            in_shape = self.net.layers[in_layer_indices[0]].output_shape[:]
            data_format = self.net.layers[in_layer_indices[0]].data_format
            # get input handler and input shape
            if input_fun == 'concat':
                if data_format == 'channels_first':
                    input_handler = lambda layer_inputs: tf.concat(layer_inputs, axis=1)
                    for in_index in in_layer_indices[1:]:
                        in_shape[1] = in_shape[1] + self.net.layers[in_index].output_shape[1]
                else:
                    input_handler = lambda layer_inputs: tf.concat(layer_inputs, axis=-1)
                    for in_index in in_layer_indices[1:]:
                        in_shape[-1] = in_shape[-1] + self.net.layers[in_index].output_shape[-1]
            elif input_fun == 'sum':
                input_handler = lambda layer_inputs: tf.add_n(layer_inputs)
            else:
                raise AttributeError('{}: input function {} is not supported.'.format(in_layer_indices, input_fun))
            # define layer input shape
            layer = self.net.layers[out_layer_indices[0]]
            layer.input_shape = in_shape
            # infer the output shape
            layer.build_layer()
            # register operation
            self.operations.append([in_layer_indices, input_handler, layer.apply, out_layer_indices])

        elif num_in_layer == 1 and num_out_layer > 1:
            # a_list[:] forces pass_by_value as it creates a new instance of a_list
            # otherwise, change of in_shape[1] will also affects self.net.layers[in_layer_indices[0]].output_shape
            in_shape = self.net.layers[in_layer_indices[0]].output_shape[:]
            if input_fun == 'split':  # input is split and passed to each output
                data_format = self.net.layers[in_layer_indices[0]].data_format
                # get input handler
                if data_format == 'channels_first':
                    in_shape[1] = int(in_shape[1] / num_out_layer)
                    input_handler = lambda layer_input: tf.split(
                        layer_input, num_or_size_splits=num_out_layer, axis=1)
                else:
                    in_shape[-1] = int(in_shape[-1] / num_out_layer)
                    input_handler = lambda layer_input: tf.split(
                        layer_input, num_or_size_splits=num_out_layer, axis=-1)
                # group layer.apply
                layer_apply_group = []
                for out_index in out_layer_indices:
                    # define layer input shape
                    layer = self.net.layers[out_index]
                    layer.input_shape = in_shape
                    # infer the output shape
                    layer.build_layer()
                    layer_apply_group.append(layer.apply)
                # register operation
                self.operations.append([in_layer_indices, input_handler, layer_apply_group, out_layer_indices])
            else:  # input is passed to each output
                for out_index in out_layer_indices:
                    # define layer input shape
                    layer = self.net.layers[out_index]
                    layer.input_shape = in_shape
                    # infer the output shape
                    layer.build_layer()
                    # register operation
                    self.operations.append([in_layer_indices, None, layer.apply, [out_index]])

        else:
            raise AttributeError(
                '{}: input has {} layers which do not match the output with {} layers.'.format(
                    in_layer_indices, num_in_layer, num_out_layer))

    def seq_links(self, in_layer_indices):
        """ This layer links layers in in_layer_indices sequentially.
        The link layers are designed such that each output layer has only one output.

        :param in_layer_indices:
        :return:
        """
        # check input layers and force layer dependency
        if self.net.layers[in_layer_indices[0]].output_shape is None:
            raise NotImplementedError('Input layer {} has not been defined yet.'.format(in_layer_indices[0]))
        # check output layers and force each layer to have unique output
        for out_index in in_layer_indices[1:]:
            if out_index in self.layer_indices:
                raise AttributeError('Layer {} has already been linked.'.format(out_index))
            else:
                self.layer_indices.append(out_index)
        # get basic info
        num_in_layer = len(in_layer_indices)
        # link in_layer_indices sequentially
        for index in range(num_in_layer - 1):
            in_shape = self.net.layers[in_layer_indices[index]].output_shape[:]
            # define layer input shape
            layer = self.net.layers[in_layer_indices[index + 1]]
            layer.input_shape = in_shape
            # infer the output shape
            layer.build_layer()
            # register operation
            self.operations.append([[in_layer_indices[index]], None, layer.apply, [in_layer_indices[index + 1]]])

    def add_output_layers(self, in_layer_indices):
        """ This function adds output layers

        :param in_layer_indices:
        :return:
        """
        # check input layers and force layer dependency
        for out_index in in_layer_indices:
            # check output layers and force each layer to have unique output
            if out_index in self.output_layer_indices:
                raise AttributeError('Layer {} has already been added as output layer.'.format(out_index))
            else:
                self.output_layer_indices.append(out_index)
            if self.net.layers[out_index].output_shape is None:
                raise NotImplementedError('Output layer {} has not been linked yet.'.format(out_index))
            # register operations
        self.operations.append([in_layer_indices, None, None, None])
        self.output_added = True

    def _del_layer_output_(self, layer_index):
        del self._out_temp_[layer_index]

    def _insert_del_(self):
        """ This function inserts operations between layer operations

        There are six types of layer operations:
        [None, None, layer.apply, [out_index]]
        [[in_index], None, layer.apply, [out_index]]
        [in_indices, tf.concat, layer.apply, [out_index]], or [in_indices, tf.add_n, layer.apply, [out_index]]
        [[in_index], tf.split, layer_apply_group, out_indices]
        [in_indices, None, None, None]

        This function inserts:
        [[in_index], del_layer_output, None, None]

        :return:
        """
        # get the locations to insert del_layer_output
        insert_list = []
        num_ops = len(self.operations)
        for i_query in range(num_ops - 1):
            out_indices = self.operations[i_query][3]
            if out_indices is not None:
                for out_index in out_indices:
                    if out_index not in self.output_layer_indices:
                        insert_index = i_query + 1
                        for i_ref in range(i_query + 1, num_ops):
                            in_indices = self.operations[i_ref][0]
                            if in_indices is not None:
                                if out_index in in_indices:
                                    insert_index = i_ref + 1
                        insert_list.append([insert_index, [[out_index], self._del_layer_output_, None, None]])
        # insert del_layer_output
        for insert_op in reversed(insert_list):
            self.operations.insert(insert_op[0], insert_op[1])
        self.del_inserted = True

    def _output_(self):
        """ This function deletes self._out_temp_ and returns the output

        :return:
        """
        if len(self._out_temp_) == 1:
            _out_ = list(self._out_temp_.values())[0]
        else:
            _out_ = self._out_temp_
        # here clear() should not be used because it will also affect _out_ because
        # dictionary is passed by reference. Instead, set self._out_temp_ = {} will
        # only affect self._out_temp_
        # self._out_temp_.clear()  # delete all items in dict
        self._out_temp_ = {}
        return _out_

    def __call__(self, routine_inputs, is_training=True):
        """ This function calculate routine outputs

        :param routine_inputs:
        :param is_training:
        :return:
        """
        if not self.del_inserted:
            self._insert_del_()
        if not self.output_added:
            raise NotImplementedError('Output layer has not been defined.')
        # run operations
        for current_op in self.operations:
            if current_op[0] is None:  # [None, None, layer.apply, [out_index]]
                self._out_temp_[current_op[3][0]] = current_op[2](routine_inputs, is_training)
            else:
                if current_op[1] is None:
                    if current_op[2] is not None:  # [[in_index], None, layer.apply, [out_index]]
                        self._out_temp_[current_op[3][0]] = current_op[2](
                            self._out_temp_[current_op[0][0]], is_training)
                    # if current_op[2] is None:  # [in_indices, None, None, None]
                    #     for out_index in current_op[0]:
                    #         _out_[out_index] = self._temp_[out_index]
                    # else:  # [[in_index], None, layer.apply, [out_index]]
                    #     self._temp_[current_op[3][0]] = current_op[2](self._temp_[current_op[0][0]])
                else:
                    if current_op[2] is None:  # [[in_index], del_layer_output, None, None]
                        current_op[1](current_op[0][0])
                    else:  # current_op has no None elements
                        if isinstance(current_op[2], list):  # [[in_index], tf.split, layer_apply_group, out_indices]
                            layer_split = current_op[1](self._out_temp_[current_op[0][0]])
                            for i_out in range(len(current_op[2])):
                                self._out_temp_[current_op[3][i_out]] = current_op[2][i_out](
                                    layer_split[i_out], is_training)
                        else:
                            # [in_indices, tf.concat, layer.apply, [out_index]], or
                            # [in_indices, tf.add_n, layer.apply, [out_index]]
                            layer_group = [self._out_temp_[i_in] for i_in in current_op[0]]
                            self._out_temp_[current_op[3][0]] = current_op[2](current_op[1](layer_group), is_training)

        return self._output_()

    def apply(self, routine_inputs, is_training=True):
        return self.__call__(routine_inputs, is_training)

    def get_layer_kernel_norm(self):
        """ This function gets the kernel norm for each kernel in each layer of the net

        :return:
        """
        layer_norms = {}
        for layer_index in self.layer_indices:
            layer = self.net.layers[layer_index]
            for key in layer.ops:
                kernel_norm = getattr(layer.ops[key], 'kernel_norm', None)
                if kernel_norm is not None:
                    layer_norms[layer.ops[key].name_in_err] = kernel_norm

        return layer_norms

    def get_layer_register(self, dict_form=False):
        """ This function collects the registered tensors for each layer in current routine

        :param dict_form:
        :return:
        """
        if dict_form:
            registered_info = {}
            for layer_index in self.layer_indices:
                layer = self.net.layers[layer_index]
                with tf.variable_scope(layer.layer_scope, reuse=tf.AUTO_REUSE):
                    registered_info[layer.layer_scope] = layer.get_register()
        else:
            registered_info = []
            for layer_index in self.layer_indices:
                layer = self.net.layers[layer_index]
                with tf.variable_scope(layer.layer_scope, reuse=tf.AUTO_REUSE):
                    registered_info.append(layer.get_register())

        return registered_info
