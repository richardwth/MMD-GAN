""" This code contains general functions that may help build other models

"""

# default modules
import numpy as np
import tensorflow as tf
from GeneralTools.misc_fun import FLAGS
from GeneralTools.math_func import SpectralNorm, get_batch_squared_dist, spatial_shape_after_conv, \
    spatial_shape_after_transpose_conv


########################################################################
def weight_initializer(act_fun, init_w_scale=1.0):
    """ This function includes initializer for several common activation functions.
    The initializer will be passed to tf.layers.

    Notes
    FAN_AVG: for tensor shape (a, b, c), fan_avg = a * (b + c) / 2

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
            if act_fun == 'relu':
                initializer = tf.variance_scaling_initializer(
                    scale=2.0 * init_w_scale, mode='fan_in', distribution='normal')
            # elif act_fun == 'tanh':
            #     initializer = tf.variance_scaling_initializer(
            #         scale=1.0 * init_w_scale, mode='fan_avg', distribution='uniform')
            #     initializer = tf.contrib.layers.variance_scaling_initializer(
            #         factor=3.0 * init_w_scale, mode='FAN_AVG', uniform=True)
            elif act_fun == 'lrelu':  # assume alpha = 0.1
                initializer = tf.variance_scaling_initializer(
                    scale=2.0 / 1.01 * init_w_scale, mode='fan_in', distribution='normal')
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
        FLAGS.print('You are using custom initializer.')
        initializer = tf.truncated_normal_initializer(stddev=0.02)
    elif FLAGS.WEIGHT_INITIALIZER == 'pg_paper':
        # paper on progressively growing gan adjust the weight on runtime.
        FLAGS.print('You are using custom initializer.')
        initializer = tf.truncated_normal_initializer()
    else:
        raise NotImplementedError('The initializer {} is not implemented.'.format(FLAGS.WEIGHT_INITIALIZER))

    return initializer


def bias_initializer(init_b_scale=0.0):
    """ This function includes initializer for bias

    :param init_b_scale:
    :return:
    """
    if init_b_scale == 0.0:
        initializer = tf.zeros_initializer()
    else:  # a very small initial bias can avoid the zero outputs that may cause problems
        initializer = tf.truncated_normal_initializer(stddev=init_b_scale)

    return initializer


#######################################################################
def spectral_norm_variable_initializer(shape, dtype=tf.float32, partition_info=None):
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
                'b', 'bias' - add bias
                'cb', 'c_bias' - conditional bias
                'bcb' - bias + conditional bias
                'bn', 'BN' - batch normalization
                'cbn', 'CBN' - conditional batch normalization
                'lrn' - local response normalization,
                'sum' - sum pool (used in sn_paper)
                'project' - label projection
                'dcd' - dense + conditional dense
                'dck' - dense * (1 + conditional scale)
                'cck' - conv * (1 + conditional scale)
                'tcck' - transpose conv * (1 + conditional scale)
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
            'num_class': number of classes
        :param input_shape:
        :param name_scope:
        :param scope_prefix: used to indicate layer/net info when print error
        :param data_format: 'channels_first' or 'channels_last' or None
        """
        self.design = design
        self.name_scope = 'kernel' if name_scope is None else name_scope
        self.name_in_err = scope_prefix + self.name_scope
        # IO
        self.input_shape = input_shape if isinstance(input_shape, list) else list(input_shape)
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
                if self.design['op'] in {'c', 'cck'}:
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
        self.is_kernel_initialized = False

    def _get_shape_(self):
        """ This function calculates the kernel shape and the output shape

        :return:
        """
        if self.design['op'] == 'i':
            self.output_shape = self.input_shape
        elif self.design['op'] == 'k':
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
            h, w = spatial_shape_after_conv(
                [h, w], self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            self.output_shape = [self.input_shape[0], self.design['out'], h, w] \
                if self.data_format == 'channels_first' else [self.input_shape[0], h, w, self.design['out']]
        elif self.design['op'] == 'tc':  # transpose conv layer
            if self.data_format == 'channels_first':
                fan_in, h, w = self.input_shape[1:]
            else:
                h, w, fan_in = self.input_shape[1:]
            self.kernel_shape = [self.design['kernel'], self.design['kernel'], self.design['out'], fan_in]
            h, w = spatial_shape_after_transpose_conv(
                [h, w], self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            self.output_shape = [self.input_shape[0], self.design['out'], h, w] \
                if self.data_format == 'channels_first' else [self.input_shape[0], h, w, self.design['out']]
        elif self.design['op'] in {'cck'}:
            if self.data_format == 'channels_first':
                fan_in, h, w = self.input_shape[1:]
                ck_shape = [self.design['num_class'], self.design['out'], 1, 1]
            else:
                h, w, fan_in = self.input_shape[1:]
                ck_shape = [self.design['num_class'], 1, 1, self.design['out']]
            h, w = spatial_shape_after_conv(
                [h, w], self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            c_shape = [self.design['kernel'], self.design['kernel'], fan_in, self.design['out']]
            self.kernel_shape = [c_shape, ck_shape]
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
            h, w = spatial_shape_after_conv(
                [h, w], self.design['kernel'], self.design['strides'], self.design['dilation'],
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
            h, w = spatial_shape_after_conv(
                [h, w], self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            self.output_shape = [self.input_shape[0], self.design['out'], h, w] \
                if self.data_format == 'channels_first' else [self.input_shape[0], h, w, self.design['out']]
        elif self.design['op'] in {'b', 'bias'}:
            self.kernel_shape = self.input_shape[1] if self.data_format == 'channels_first' else self.input_shape[-1]
            self.output_shape = self.input_shape
        elif self.design['op'] in {'bn', 'BN', 'lrn'}:
            self.output_shape = self.input_shape
        elif self.design['op'] in {'cbn', 'CBN', 'c_bias', 'cb'}:
            # conditional batch normalization, conditional bias, conditional kernel and bias
            self.output_shape = self.input_shape
            if self.data_format == 'channels_first':
                self.kernel_shape = [self.design['num_class'], self.input_shape[1], 1, 1]
            elif self.data_format == 'channels_last':
                self.kernel_shape = [self.design['num_class'], 1, 1, self.input_shape[-1]]
            else:
                self.kernel_shape = [self.design['num_class'], self.input_shape[-1]]
            if 'bn_center' in self.design:
                self.design['bn_center'] = False
            if 'bn_scale' in self.design:
                self.design['bn_scale'] = False
        elif self.design['op'] in {'bcb'}:  # bias + conditional bias
            b_shape = self.input_shape[1] if self.data_format == 'channels_first' else self.input_shape[-1]
            if self.data_format == 'channels_first':
                cb_shape = [self.design['num_class'], self.input_shape[1], 1, 1]
            elif self.data_format == 'channels_last':
                cb_shape = [self.design['num_class'], 1, 1, self.input_shape[-1]]
            else:
                cb_shape = [self.design['num_class'], self.input_shape[-1]]
            self.kernel_shape = [b_shape, cb_shape]
            self.output_shape = self.input_shape
        elif self.design['op'] in {'project'}:
            self.output_shape = [self.input_shape[0], 1]
            self.kernel_shape = [self.design['num_class'], self.input_shape[1]]
        elif self.design['op'] in {'dcd'}:  # dense kernel + conditional dense kernel
            # dense and conditional dense is a generalization of projection for output dimension other than 1
            self.output_shape = [self.input_shape[0], self.design['out']]
            d_shape = [self.input_shape[1], self.design['out']]
            cd_shape = [self.design['num_class'], self.input_shape[1], self.design['out']]
            self.kernel_shape = [d_shape, cd_shape]
        elif self.design['op'] in {'dck'}:  # dense * (1 + conditional scale)
            self.output_shape = [self.input_shape[0], self.design['out']]
            d_shape = [self.input_shape[1], self.design['out']]
            ck_shape = [self.design['num_class'], self.design['out']]
            self.kernel_shape = [d_shape, ck_shape]
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
        if not self.is_kernel_initialized:
            # some ops may require different initial scale for kernel
            if self.design['op'] in {'d', 'c', 'tc', 'sc', 'project', 'dcd', 'dck', 'cck'}:
                kernel_init = weight_initializer(self.design['act'], self.design['init_w_scale']) \
                    if self.design.get('init_w_scale') is not None else weight_initializer(self.design['act'])
            elif self.design['op'] in {'k'}:
                kernel_init = tf.zeros_initializer() \
                    if self.design.get('init_w_scale') == 0.0 else tf.ones_initializer()
            else:
                kernel_init = None

            if self.design['op'] in {'d', 'c', 'tc', 'k', 'project'}:
                # dense, conv, transpose conv, scalar weight, projection weight
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
                # if bias is initialized as 0, the CNN output at 1st iter could be at e-15 level.
                # otherwise, the CNN output would be dominated by bias, which may cause problem if
                # some operation depends on the
                self.kernel = tf.get_variable(
                    'bias', shape=self.kernel_shape, dtype=tf.float32, trainable=True,
                    initializer=bias_initializer(1e-5))
            elif self.design['op'] in {'c_bias', 'cb'}:
                # class_wise bias
                self.kernel = tf.get_variable(
                    'c_bias', shape=self.kernel_shape, dtype=tf.float32, trainable=True,
                    initializer=bias_initializer(1e-5))
            elif self.design['op'] in {'bcb'}:
                b_kernel = tf.get_variable(
                    'bias', shape=self.kernel_shape[0], dtype=tf.float32, trainable=True,
                    initializer=bias_initializer(1e-5))
                # class_wise bias
                cb_kernel = tf.get_variable(
                    'c_bias', shape=self.kernel_shape[1], dtype=tf.float32, trainable=True,
                    initializer=tf.zeros_initializer())
                self.kernel = [b_kernel, cb_kernel]
            elif self.design['op'] in {'cbn'}:
                # conditional batch normalization and conditional kernel and bias
                self.kernel = [
                    tf.get_variable(
                        'scale', self.kernel_shape,
                        dtype=tf.float32, initializer=tf.ones_initializer, trainable=True),
                    tf.get_variable(
                        'offset', self.kernel_shape,
                        dtype=tf.float32, initializer=bias_initializer(1e-5), trainable=True)]
            elif self.design['op'] in {'dcd', 'dck', 'cck'}:
                self.kernel = [
                    tf.get_variable(
                        'kernel', self.kernel_shape[0], dtype=tf.float32, initializer=kernel_init, trainable=True),
                    tf.get_variable(
                        'c_kernel', self.kernel_shape[1], dtype=tf.float32,
                        initializer=tf.zeros_initializer(), trainable=True)]

            # initialize kernel norm and multiplier
            self._get_weight_norm_()
            self._get_multiplier_()

            self.is_kernel_initialized = True

    def _get_weight_norm_(self):
        """ This function get the weight of the norm

        :return:
        """
        if not self.is_kernel_initialized and self.design.get('w_nm') is not None:
            if self.design['w_nm'] in {'s'}:
                if self.design['op'] in {'d', 'c', 'tc', 'project', 'dcd', 'dck', 'cck'}:
                    # apply spectral normalization to get w_norm
                    if self.design['op'] in {'d', 'project', 'dcd', 'dck'}:
                        # for d, project, self.kernel_norm is scalar
                        # for dcd, dck, self.kernel_norm shape should be [num_class, 1]
                        sn_def = {'op': self.design['op']}
                        self.kernel_norm = SpectralNorm(sn_def, 'SN', num_iter=1).apply(self.kernel)
                        if self.design['op'] in {'dcd'}:
                            self.kernel_norm = tf.squeeze(self.kernel_norm, axis=2)  # [num_class, 1]
                    elif self.design['op'] in {'c', 'tc', 'cck'}:
                        # for c, tc, self.kernel_norm is scalar
                        # for cck, self.kernel_norm shape should be [num_class, 1, 1, 1]
                        sn_def = {'op': self.design['op']}
                        sn_def.update({key: self.design[key] for key in ['strides', 'dilation', 'padding']})
                        sn_def['input_shape'] = self.input_shape
                        sn_def['data_format'] = self.data_format_alias
                        sn_def['output_shape'] = self.output_shape
                        self.kernel_norm = SpectralNorm(sn_def, 'SN', num_iter=1).apply(self.kernel)
                else:  # kernel normalization has not been implemented for sc kernels
                    raise NotImplementedError(
                        '{}: spectral norm for {} has not been implemented.'.format(
                            self.name_in_err, self.design['op']))
            elif self.design['w_nm'] in {'l2'}:
                raise NotImplementedError('{}: l2 norm not finished yet'.format(self.name_in_err))
                # apply he normalization to get w_norm
                # self.kernel_norm = l2_norm(self.kernel)
            else:
                raise NotImplementedError(
                    '{}: {} method not implemented'.format(self.name_in_err, self.design['w_nm']))

    def _get_multiplier_(self):
        """ This function reimburse the norm loss caused by the activation function

        :return:
        """
        if self.design['op'] in {'d', 'c', 'tc', 'dcd', 'dck', 'cck'}:
            # project is excluded as it can only be added at the last layer (not exactly but...)
            if self.design.get('w_nm') in ['spectral', 's']:
                if self.design['act_k'] is True:
                    # self.multiplier = None equals self.multiplier = 1
                    if self.design['act'] == 'lrelu':
                        # self.multiplier = 1.0 / 0.55
                        # self.multiplier = np.sqrt(2.0)
                        # self.multiplier = 1.6
                        self.multiplier = 1.5
                        # self.multiplier = 2.0
                    elif self.design['act'] == 'relu':
                        # self.multiplier = 1.6
                        # self.multiplier = 2.0
                        # self.multiplier = np.sqrt(2.0)
                        self.multiplier = 1.5
                        # self.multiplier = 2.0
            elif FLAGS.WEIGHT_INITIALIZER == 'pg_paper':
                if self.design['op'] in {'d', 'c', 'tc'}:
                    fan_in = np.prod(self.kernel_shape[:-1], dtype=np.float32)
                    fan_out = self.kernel_shape[-1]
                    if self.design['act'] in {'relu'}:
                        self.multiplier = np.sqrt(2.0 / fan_in)
                    elif self.design['act'] in {'lrelu'}:  # assume alpha = 0.1
                        self.multiplier = np.sqrt(2.0 / 1.01 / fan_in)
                    elif self.design['act'] in {'sigmoid'}:
                        self.multiplier = np.sqrt(32.0 / (fan_in + fan_out))
                    else:
                        self.multiplier = np.sqrt(2.0 / (fan_in + fan_out))
                elif self.design['op'] in {'sc'}:
                    # multiplier has not been implemented for sc kernels
                    raise NotImplementedError(
                        '{}: kernel norm for {} has not been implemented.'.format(
                            self.name_in_err + '/' + self.name_scope, self.design['op']))

    def __call__(self, op_input, control_ops=None, is_training=True, **kwargs):
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
                if self.design.get('w_nm') in ['s', 'l2'] and \
                        self.design['op'] in {'d', 'c', 'tc', 'dcd', 'dck', 'cck'}:
                    multiplier = 1.0 / self.kernel_norm \
                        if self.multiplier is None else self.multiplier / self.kernel_norm

                elif FLAGS.WEIGHT_INITIALIZER == 'pg_paper':
                    multiplier = self.multiplier
                else:
                    multiplier = None

                # check label info
                if self.design['op'] in {'c_bias', 'cb', 'bcb', 'cbn', 'project', 'dcd', 'dck', 'cck'}:
                    assert 'label' in kwargs and isinstance(kwargs['label'], tf.Tensor), \
                        '{}: labels must be provided for op: {}.'.format(self.name_in_err, self.design['op'])

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
                    kernel = self.kernel if multiplier is None or multiplier == 1.0 else self.kernel * multiplier
                    if self.design['dilation'] > 1:  # atrous_conv2d_transpose does not support NCHW
                        raise NotImplementedError(
                            '{}: atrous_conv2d_transpose has not been implemented.'.format(self.name_in_err))
                    else:
                        # conv2d_transpose needs specified output_shape including the batch size, which
                        # may change during training and test
                        output_shape = [op_input.get_shape().as_list()[0]] + self.output_shape[1:]
                        op_output = tf.nn.conv2d_transpose(
                            op_input, kernel, output_shape,
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
                elif self.design['op'] in {'bn', 'BN', 'cbn', 'CBN'}:
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
                    if self.design['op'] in {'cbn'}:  # add class-wise scale and offset
                        label = tf.squeeze(kwargs['label'], axis=1)  # here label is required to be [N,] tensor
                        scale = tf.gather(self.kernel[0], label)  # NC11 or N11C
                        offset = tf.gather(self.kernel[1], label)  # NC11 or N11C
                        op_output = op_output * scale + offset
                elif self.design['op'] in {'lrn'}:
                    op_output = local_response_normalization(op_input, data_format=self.data_format)
                elif self.design['op'] in {'project'}:
                    label = tf.squeeze(kwargs['label'], axis=1)  # here label is required to be [N,] tensor
                    kernel = self.kernel if multiplier is None or multiplier == 1.0 else self.kernel * multiplier
                    selected_kernel = tf.gather(kernel, label)  # N-by-D
                    op_output = tf.reduce_sum(selected_kernel * op_input, axis=1, keepdims=True)  # N-by-1
                elif self.design['op'] in {'c_bias', 'cb'}:
                    label = tf.squeeze(kwargs['label'], axis=1)  # here label is required to be [N,] tensor
                    selected_kernel = tf.gather(self.kernel, label)  # N-by-d or NCHW or NHWC
                    op_output = tf.add(op_input, selected_kernel)
                elif self.design['op'] in {'bcb'}:
                    if len(self.input_shape) == 2:
                        op_output = tf.add(op_input, self.kernel[0])
                    elif len(self.input_shape) == 4:
                        op_output = tf.nn.bias_add(op_input, self.kernel[0], self.data_format_alias)
                    else:
                        raise AttributeError('{}: does not support bias_add operation.'.format(self.name_in_err))
                    label = tf.squeeze(kwargs['label'], axis=1)  # here label is required to be [N,] tensor
                    selected_kernel = tf.gather(self.kernel[1], label)  # N-by-d or NCHW or NHWC
                    op_output = tf.add(op_output, selected_kernel)
                elif self.design['op'] == 'dcd':  # dense layer + conditional dense layer
                    # dense
                    op_output_d = tf.matmul(op_input, self.kernel[0])  # N-S
                    # conditional dense
                    label = tf.squeeze(kwargs['label'], axis=1)  # [N,]
                    kernel_cd = tf.gather(self.kernel[1], label)  # N-D-S
                    op_output_cd = tf.squeeze(  # N-S
                        tf.matmul(tf.expand_dims(op_input, axis=1), kernel_cd), axis=1)
                    op_output = op_output_d + op_output_cd  # N-S
                    if multiplier is not None:
                        op_output = op_output * tf.gather(multiplier, label)
                elif self.design['op'] in {'dck'}:
                    # dense
                    op_output_d = tf.matmul(op_input, self.kernel[0])  # N-S
                    # conditional scale
                    label = tf.squeeze(kwargs['label'], axis=1)  # [N,]
                    kernel_ck = tf.gather(1.0 + self.kernel[1], label)  # N-S
                    op_output = op_output_d * kernel_ck
                    if multiplier is not None:
                        op_output = op_output * tf.gather(multiplier, label)
                elif self.design['op'] in {'cck'}:
                    # conv
                    op_output_c = tf.nn.conv2d(
                        op_input, self.kernel[0], self.strides, self.design['padding'],
                        data_format=self.data_format_alias, dilations=self.dilation)
                    # conditional scale
                    label = tf.squeeze(kwargs['label'], axis=1)  # [N,]
                    kernel_ck = tf.gather(1.0 + self.kernel[1], label)  # N-C-1-1 or N-1-1-C
                    op_output = op_output_c * kernel_ck
                    if multiplier is not None:
                        op_output = op_output * tf.gather(multiplier, label)
                else:
                    raise AttributeError('{}: type {} not supported'.format(self.name_in_err, self.design['op']))

                self._output_check_(op_output)
                return op_output

    def apply(self, op_input, control_ops=None, is_training=True, **kwargs):
        """

        :param op_input:
        :param control_ops:
        :param is_training:
        :return:
        """
        return self.__call__(op_input, control_ops, is_training, **kwargs)


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
            'b', 'bn', 'BN' - batch normalization;
            'cbn', 'CBN' - conditional batch normalization
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
    template = {'name': None, 'type': 'default', 'op': 'c', 'out': None, 'bias': 'b',
                'act': 'linear', 'act_nm': None, 'act_k': False,
                'w_nm': None, 'w_p': None,
                'kernel': 3, 'strides': 1, 'dilation': 1, 'padding': 'SAME', 'scale': None,
                'in_reshape': None, 'out_reshape': None, 'aux': None}
    # update template with parameters from layer_design
    for key in layer_design:
        template[key] = layer_design[key]
    # check template values to avoid error
    # if (template['act_nm'] in {'bn', 'BN', 'cbn', 'CBN'}) and (template['act'] == 'linear'):
    #     template['act_nm'] = None  # batch normalization is not used with linear activation
    if template['act_nm'] in {'bn', 'BN'} and template['bias'] in {'b', 'bias'}:
        template['bias'] = None  # batch normalization is not used with common bias, but conditional bias may be used
    if template['act_nm'] in {'cbn', 'CBN'}:
        template['bias'] = None  # conditional batch normalization is not used with any bias
    if template['op'] in {'tc'}:
        # transpose conv is usually used as upsampling method
        template['scale'] = None
    # if template['w_nm'] is not None:  # weight normalization and act normalization cannot be used together
    #     template['act_nm'] = None
    if template['scale'] is not None:
        assert isinstance(template['scale'], (list, tuple)), \
            'Value for key "scale" must be list or tuple.'
    if template['w_nm'] is not None:  # This is because different normalization methods do not work in the same layer
        assert not isinstance(template['w_nm'], (list, tuple)), \
            'Value for key "w_nm" must not be list or tuple.'

    # output template
    if template['op'] in {'d', 'dcd', 'dck'}:
        return {key: template[key]
                for key in ['name', 'op', 'type', 'out', 'bias',
                            'act', 'act_nm', 'act_k',
                            'w_nm', 'w_p',
                            'in_reshape', 'out_reshape', 'aux']}
    elif template['op'] in ['sc', 'c', 'tc', 'avg', 'max', 'sum', 'cck', 'tcck']:
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

    def __init__(self, design, input_shape=None, name_prefix='', data_format=None, num_class=0):
        """ This function defines the structure of layer

        :param design: see update_layer_design function
        :param input_shape:
        :param name_prefix: net name
        :param data_format: for image data, 'channels_first' or 'channels_last'; for other data, None
        :param num_class: number of data classes
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
        self.num_class = num_class
        if self.num_class < 2:  # in unsupervised cases, some methods cannot be used
            assert not self.design['type'] in {'project'}, \
                '{}: cannot use {} for one class'.format(self.layer_scope, self.design['type'])
            assert not self.design['act_nm'] in {'cbn', 'CBN'}, \
                '{}: cannot use {} for one class'.format(self.layer_scope, self.design['act_nm'])
        # set up other values
        self._layer_output_ = None
        # debug_register is used to store info other than _layer_out_ that would be used for debug
        self.debug_register = None
        # layer status
        self.is_layer_built = False
        # ops
        self.ops = {}

    def _input_(self, layer_input):
        """ This function initializes layer_output.

        :param layer_input: a tensor x or a dictionary {'x': x, 'y': y}, typically y is a N-by-1 tensor
        :return:
        """
        # check input
        assert isinstance(layer_input, (dict, tf.Tensor)), \
            '{}: input type must be dict or tf.Tensor. Got {}.'.format(self.layer_scope, type(layer_input))
        if isinstance(layer_input, tf.Tensor):
            layer_input = {'x': layer_input, 'y': None}

        # check input shape
        input_shape = layer_input['x'].get_shape().as_list()
        if self.input_shape is None:
            self.input_shape = input_shape
        else:
            assert self.input_shape[1:] == input_shape[1:], \
                '{}: the actual input shape {} does not match theoretic shape {}.'.format(
                    self.layer_scope, input_shape[1:], self.input_shape[1:])

        # copy input;
        # It should be noted that we do not want any ops in current layer to change the layer_input in memory
        # This is because the layer_input may be fed into other layers.
        # However, if we use self._layer_output_ = layer_input directly, ops like partial assignment
        # self._layer_output_['x'] = self._layer_output_['x'] + 0.1 will change layer_input
        self._layer_output_ = dict(layer_input)
        # print('{} input has y? {}'.format(self.layer_scope, 'y' in layer_input))

    def _output_(self):
        """ This function returns the output
        :return layer_output: a tensor x or a dictionary {'x': x, 'y': y}
        """
        output_shape = self._layer_output_['x'].get_shape().as_list()
        if self.output_shape is None:
            self.output_shape = output_shape
        else:
            assert self.output_shape[1:] == output_shape[1:], \
                '{}: the actual output shape {} does not match theoretic shape {}.'.format(
                    self.layer_scope, output_shape[1:], self.output_shape[1:])
        # Layer object always forgets self._layer_output_ after its value returned
        layer_output = dict(self._layer_output_)
        self._layer_output_ = None
        return layer_output

    def get_register(self):
        """ This function returns the registered tensor

        :return:
        """
        # Layer object always forgets self._register_ after its value returned
        registered_info = self.debug_register
        self.debug_register = None
        return registered_info

    def _update_design_(self, design, target_keys, index=None):
        """ This function updates design with self.design

        :param design: a dictionary
        :param target_keys:
        :return:
        """
        for key in target_keys:
            if key in self.design:
                if index is not None and isinstance(self.design[key], (list, tuple)):
                    design[key] = self.design[key][index]
                else:
                    design[key] = self.design[key]

        # there is no need actually to return design, but anyway...
        return design

    def _add_image_scaling_(self, input_shape, name_scope='sampling', scale_design=None):
        """ This function registers a sampling process

        :param input_shape:
        :param name_scope:
        :param scale_design: if self.design['scale'] should not be used, provide a scale design here, e.g. ['max', -2]
        :return:
        """
        if scale_design is None:
            scale_design = self.design['scale']
        design = {'method': scale_design[0], 'factor': scale_design[1]}
        op = ImageScaling(
            design, input_shape, name_scope=name_scope,
            scope_prefix=self.layer_scope + '/', data_format=self.data_format)
        self.ops[name_scope] = op

        return op.output_shape

    def _add_kernel_(self, input_shape, name_scope='kernel', index=None, op_design=None, kernel_in=None,
                     kernel_out=None, kernel_init_scale=None):
        """ This function registers a kernel

        :param kernel_out:
        :param input_shape:
        :param name_scope:
        :param index: in case multiple kernels of different shapes are used, 'kernel', 'strides',
            'dilation', 'padding' could be provided as a list.
        :param op_design: if self.design['op'] should not be used, provide an op here
        :param kernel_init_scale: some op requires different weight initialization scheme
        :param kernel_in: some op requires different input dimension/channels
        :param kernel_out: some op requires different output dimension/channels
        :return:
        """
        design = {'op': self.design['op']} if op_design is None else {'op': op_design}
        # acquire kernel definition from self.design
        target_keys = {'out', 'act', 'act_k', 'w_nm', 'kernel', 'strides', 'dilation', 'padding'}
        design = self._update_design_(design, target_keys, index)
        # check for exceptional cases
        if design['op'] in {'dcd', 'dck', 'cck', 'tcck'}:  # some ops require class info
            design['num_class'] = self.num_class
        # check inputs
        if kernel_init_scale is not None:
            design['init_w_scale'] = kernel_init_scale
        if kernel_in is not None:
            input_shape = input_shape.copy()
            if self.data_format == 'channels_first':
                input_shape[1] = kernel_in
            else:
                input_shape[-1] = kernel_in
        if kernel_out is not None:
            design['out'] = kernel_out

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

    def _add_projection_kernel_(self, input_shape, name_scope='project'):
        """ This function registers a label projection op that does:
        o = y*W*x, where y is one-hot label vector, x is layer input

        The ops is proposed by the following paper:
        Miyato, T., & Koyama, M. (2018). cGANs with Projection Discriminator. In ICLR.

        :param input_shape: shape of x, [N, D]
        :param name_scope:
        :return:
        """
        design = {'op': 'project', 'num_class': self.num_class, 'act': 'linear'}
        target_keys = {'act_k', 'w_nm'}
        design = self._update_design_(design, target_keys)
        # register the op
        op = ParametricOperation(
            design, input_shape, name_scope=name_scope,
            scope_prefix=self.layer_scope + '/', data_format=self.data_format)
        self.ops[name_scope] = op

        # return op.output_shape

    def _add_bias_(self, input_shape, name_scope='bias', op_design=None):
        """ This function registers a bias-adding process. The bias may be conditional on class labels

        :param input_shape:
        :param name_scope:
        :param op_design:
        :return:
        """
        if op_design is None:
            op_design = self.design['bias']
        if op_design in {'bias', 'b'} or op_design is None:
            # here None is also considered as normal bias
            # the idea is that in some cases bias is needed even bn is used.
            design = {'op': 'bias'}
        elif op_design in {'cb', 'c_bias'}:  # conditional bias
            design = {'op': 'c_bias', 'num_class': self.num_class}
        elif op_design in {'bcb'}:  # common bias and conditional bias
            design = {'op': 'bcb', 'num_class': self.num_class}
        elif op_design is False:  # do not use bias
            design = None
        else:
            raise NotImplementedError(
                '{}: bias option {} not implemented.'.format(self.layer_scope, op_design))

        if design is None:
            return input_shape
        else:
            op = ParametricOperation(
                design, input_shape, name_scope=name_scope,
                scope_prefix=self.layer_scope + '/', data_format=self.data_format)
            self.ops[name_scope] = op

            return op.output_shape

    def _add_bn_(self, input_shape, name_scope='bias', offset=None, offset_init=None, scale=None, scale_init=None,
                 scale_const=None):
        """ This function registers a batch normalization process

        :param input_shape:
        :param name_scope:
        :param offset: whether to add offset beta
        :param scale: whether to multiply by gamma
        :param offset_init: initializer for beta; if not provided, tf.zeros_initializer()
        :param scale_init: initializer for gamma; if not provided, tf.ones_initializer()
        :param scale_const: constraint for gamma; if not provided, None
        :return:
        """
        if self.design['act_nm'] in {'cbn', 'CBN'}:
            assert self.num_class is not None, '{}: C must be provided for CBN.'.format(self.layer_scope)
            design = {'op': 'cbn', 'num_class': self.num_class}
            offset = False  # class-wise offset and scale will be used
            scale = False
        elif self.design['act_nm'] in {'b', 'bn', 'BN'}:
            design = {'op': 'bn'}
        else:
            raise NotImplementedError('{}: {} not implemented'.format(self.layer_scope, self.design['act_nm']))
        if offset is not None:
            design['bn_center'] = offset
        if scale is not None:
            design['bn_scale'] = scale
        if offset_init is not None:
            design['bn_b_init'] = offset_init
        if scale_init is not None:
            design['bn_w_init'] = scale_init
        if scale_const is not None:
            design['bn_w_const'] = scale_const

        op = ParametricOperation(
            design, input_shape, name_scope=name_scope,
            scope_prefix=self.layer_scope + '/', data_format=self.data_format)
        self.ops[name_scope] = op

        return op.output_shape

    def _apply_activation_(self, layer_input, index=None, act_fun=None):
        """ This function applies activation function on the input

        :param layer_input:
        :param index: if there are multiple activation functions in self.design['act'], provide the index
        :param act_fun:
        :return:
        """
        if act_fun is None:
            if isinstance(self.design['act'], str):
                act_fun = self.design['act']
            elif isinstance(self.design['act'], (list, tuple)) and index is not None:
                act_fun = self.design['act'][index]
            else:
                raise AttributeError('{}: activation setup is wrong'.format(self.layer_scope))

        return apply_activation(layer_input, act_fun)

    def _apply_input_reshape_(self):
        if self.design['in_reshape'] is not None:
            batch_size = self._layer_output_['x'].get_shape().as_list()[0]
            self._layer_output_['x'] = tf.reshape(
                self._layer_output_['x'], shape=[batch_size] + self.design['in_reshape'])

    def _apply_output_reshape_(self):
        if self.design['out_reshape'] is not None:
            batch_size = self._layer_output_['x'].get_shape().as_list()[0]
            # print(self._layer_output_['x'].get_shape().as_list())
            self._layer_output_['x'] = tf.reshape(
                self._layer_output_['x'], shape=[batch_size] + self.design['out_reshape'])

    def _add_layer_default_(self, input_shape):
        """ This function adds the default layer, optionally with projection y*V

        The order of operations:
        upsampling - kernel - bias - BN - downsampling

        :param input_shape:
        :return:
        """
        # label projection
        if self.design['type'] in {'project'}:  # register a kernel with shape [num_class, num_input_features]
            assert len(input_shape) == 2 and self.design['out'] == 1, \
                '{}: currently projection only applies to dense layer with one output'.format(self.layer_scope)
            self._add_projection_kernel_(input_shape, 'project')

        # upsampling
        if self.design.get('scale') is not None:
            if self.design['scale'][1] > 0:  # upsampling
                input_shape = self._add_image_scaling_(input_shape, 'upsampling')
        # kernel
        input_shape = self._add_kernel_(input_shape, 'kernel')
        # bias
        if self.design.get('bias') is not None:
            input_shape = self._add_bias_(input_shape, 'bias')
        # batch normalization
        if self.design['act_nm'] in {'bn', 'BN', 'cbn', 'CBN'}:
            input_shape = self._add_bn_(input_shape, 'BN')
        # activation
        # donwsampling
        if self.design.get('scale') is not None:
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
        layer_out = self._layer_output_['x']
        # upsampling
        if 'upsampling' in self.ops:
            layer_out = self.ops['upsampling'].apply(layer_out)
        # kernel
        layer_out = self.ops['kernel'].apply(
            layer_out,
            label=self._layer_output_['y'] if 'y' in self._layer_output_ else None)
        # bias
        if 'bias' in self.ops:
            layer_out = self.ops['bias'].apply(
                layer_out,
                label=self._layer_output_['y'] if 'y' in self._layer_output_ else None)
        # batch normalization
        if 'BN' in self.ops:
            layer_out = self.ops['BN'].apply(
                layer_out, is_training=is_training,
                label=self._layer_output_['y'] if 'y' in self._layer_output_ else None)
        # activation
        layer_out = self._apply_activation_(layer_out)
        # downsampling
        if 'downsampling' in self.ops:
            layer_out = self.ops['downsampling'].apply(layer_out)

        if 'project' in self.ops:
            project_out = self.ops['project'].apply(
                self._layer_output_['x'], label=self._layer_output_['y'])
            # print(project_out.get_shape().as_list())
            self._layer_output_['x'] = layer_out + project_out
        else:
            self._layer_output_['x'] = layer_out

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
        if (not self.design['type'] == 'res_v1') and self.design['act_nm'] in {'bn', 'BN', 'cbn', 'CBN'}:
            res_shape = self._add_bn_(input_shape, 'BN_0')
        else:
            res_shape = input_shape
        # activation
        # upsampling
        if self.design.get('scale') is not None:
            if self.design['scale'][1] > 0:  # upsampling
                res_shape = self._add_image_scaling_(res_shape, 'upsampling_0')
        # kernel
        # print('kernel_0_in {}'.format(res_shape))
        res_shape = self._add_kernel_(res_shape, 'kernel_0', index=0)
        # print('kernel_0_out {}'.format(res_shape))
        # bias
        if self.design.get('bias') is not None:
            res_shape = self._add_bias_(res_shape, 'bias_0')
        # BN
        if self.design['act_nm'] in {'bn', 'BN', 'cbn', 'CBN'}:
            res_shape = self._add_bn_(res_shape, 'BN_1')
        # activation
        # kernel
        if self.design['op'] == 'tc':  # in res block that uses tc, the second conv is c
            res_shape = self._add_kernel_(res_shape, 'kernel_1', index=1, op_design='c')
        else:
            res_shape = self._add_kernel_(res_shape, 'kernel_1', index=1)
        # print('kernel_1_out {}'.format(res_shape))
        # bias
        if self.design.get('bias') is not None:
            res_shape = self._add_bias_(res_shape, 'bias_1')
        # downsampling
        if self.design.get('scale') is not None:
            if self.design['scale'][1] < 0:  # downsampling
                res_shape = self._add_image_scaling_(res_shape, 'downsampling_0')
        # print('downsampling_0_out {}'.format(res_shape))

        # shortcut branch
        sc_shape = input_shape
        if self.design['type'] == 'res':  # for 'res_i', the shortcut branch is linear
            # upsampling
            if self.design.get('scale') is not None:
                if self.design['scale'][1] > 0:  # upsampling
                    sc_shape = self._add_image_scaling_(sc_shape, 'upsampling_1')
            # kernel
            sc_shape = self._add_kernel_(sc_shape, 'kernel_sc', index=2)
            # print('kernel_2_out {}'.format(sc_shape))
            # bias
            if 'bias' in self.design:  # here I guess the bias should be kept
                sc_shape = self._add_bias_(sc_shape, 'bias_sc')
            # downsampling
            if self.design.get('scale') is not None:
                if self.design['scale'][1] < 0:  # downsampling
                    sc_shape = self._add_image_scaling_(sc_shape, 'downsampling_1')
        elif self.design['type'] == 'res_v1':
            # in wgan-gp paper, the shortcut in the first resnet block in discriminator has a downsample - conv order
            # with kernel size 1 and it has bias
            # downsampling
            if self.design.get('scale') is not None:
                if self.design['scale'][1] < 0:  # downsampling
                    sc_shape = self._add_image_scaling_(sc_shape, 'downsampling_1')
                else:
                    raise AttributeError('{}: res_v1 is only used with downsampling.'.format(self.layer_scope))
            # kernel
            sc_shape = self._add_kernel_(sc_shape, 'kernel_sc', index=2)
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
        res_out = self._layer_output_['x']
        if not self.design['type'] == 'res_v1':
            # batch normalization BN_0
            if 'BN_0' in self.ops:
                res_out = self.ops['BN_0'].apply(
                    res_out, is_training=is_training,
                    label=self._layer_output_['y'] if 'y' in self._layer_output_ else None)
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
            res_out = self.ops['BN_1'].apply(
                res_out, is_training=is_training,
                label=self._layer_output_['y'] if 'y' in self._layer_output_ else None)
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
        sc_out = self._layer_output_['x']
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
        self._layer_output_['x'] = res_out + sc_out

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

        I rearrange the nonlocal block in the resnet style
            z = act(bn(x))
            m = softmax(conv_f(z)' * conv_g(max_pool(z)))
            o = m * conv_h(max_pool(z))
            y = conv_k(o) + x
        For nl_pool, max_pool is used, otherwise ignored.

        :param input_shape:
        :return:
        """
        # att branch
        # BN
        # if self.design['act_nm'] in {'bn', 'BN', 'cbn', 'CBN'}:
        #     att_shape = self._add_bn_(input_shape, 'BN_0')
        # else:
        #     att_shape = input_shape  # NxH1xW1xC1 or NxC1xH1xW1
        att_shape = input_shape
        # activation

        # attention map kernel f
        att_shape_f = self._add_kernel_(att_shape, 'f_x', index=0)  # NxH1xW1xC2 or NxC2xH1xW1
        # bias;
        att_shape_f = self._add_bias_(att_shape_f, 'bias_f')

        # attention map kernel g and h
        # the softmax operation makes it useless to add bias to g(x)
        if self.design['type'] in {'nl_pool', 'nl_pool_dist'}:
            att_shape_gh = self._add_image_scaling_(att_shape, 'downsampling', ['max', -2])  # NxH2xW2xC1 or NxC1xH2xW2
        else:
            att_shape_gh = att_shape
        att_shape_g = self._add_kernel_(att_shape_gh, 'g_x', index=1)  # NxH2xW2xC2 or NxC2xH2xW2
        att_shape_h = self._add_kernel_(att_shape_gh, 'h_x', index=2)  # NxH2xW2xC1 or NxC1xH2xW2
        # beta = f * g: NxHW1xHW2
        # beta = softmax(beta)
        # o = beta*h: NxH1xW1xC1 or NxC1xH1xW1

        # check shape
        if self.data_format == 'channels_first':
            assert att_shape_f[1] == att_shape_g[1], \
                '{}: f(x) channel {} does not match g(x) channel {}'.format(  # f.C1 == g.C1
                    self.layer_scope, att_shape_f[1], att_shape_g[1])
            assert att_shape_g[2:4] == att_shape_h[2:4], \
                '{}: g(x) size {} does not match h(x) size {}'.format(  # g.H2 == h.H2, g.W2 == h.W2
                    self.layer_scope, att_shape_g[2:4], att_shape_h[2:4])
            att_shape = [att_shape[0], att_shape_h[1], att_shape_f[2], att_shape_f[3]]
        elif self.data_format == 'channels_last':
            assert att_shape_f[-1] == att_shape_g[-1], \
                '{}: f(x) channel {} does not match g(x) channel {}'.format(
                    self.layer_scope, att_shape_f[-1], att_shape_gh[-1])
            assert att_shape_g[1:3] == att_shape_h[1:3], \
                '{}: g(x) size {} does not match h(x) size {}'.format(
                    self.layer_scope, att_shape_g[1:3], att_shape_h[1:3])
            att_shape = [att_shape[0], att_shape_f[1], att_shape_f[2], att_shape_h[3]]
        else:
            raise AttributeError(
                '{}: the non-local block only supports channels_first or channels_first data format. Got {}'.format(
                    self.layer_scope, self.data_format))

        # add kernel
        att_shape = self._add_bn_(att_shape, name_scope='BN_1', scale=False)
        bound = [-1.0, 1.0] if self.design['w_nm'] == 's' else None
        att_shape = self._add_scalar_kernel_(att_shape, 'k_x', init_w_scale=0.0, bound=bound)
        # att_shape = self._add_kernel_(att_shape, 'k_x', index=3, init_w_scale=0.0)
        # add bias to final attention map
        # if self.design['bias']:
        #     att_shape = self._add_bias_(att_shape, 'bias_k')

        # shortcut branch
        sc_shape = input_shape
        # check shape
        assert sc_shape == att_shape, \
            '{}: attention map shape {} does not match input shape {}'.format(
                self.layer_scope, att_shape, input_shape)
        output_shape = sc_shape

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

        I rearrange the nonlocal block in the resnet style
            z = act(bn(x))
            m = softmax(conv_f(z)' * conv_g(max_pool(z)))
            o = m * conv_h(max_pool(z))
            y = conv_k(o) + x
        For nl_pool, max_pool is used, otherwise ignored.

        :param is_training:
        :return:
        """
        # batch normalization and activation
        att_out = self._layer_output_['x']
        # if 'BN_0' in self.ops:
        #     att_out = self.ops['BN_0'].apply(
        #               att_out, is_training=is_training, label=self._layer_output_['y'])
        # att_out = self._apply_activation_(att_out)

        # attention map kernel f
        att_out_f = self.ops['f_x'].apply(att_out)  # NxH1xW1xC2 or NxC2xH1xW1
        att_out_f = self.ops['bias_f'].apply(att_out_f)

        # attention map kernel g and h
        if 'downsampling' in self.ops:
            att_out_gh = self.ops['downsampling'].apply(att_out)
        else:
            att_out_gh = att_out
        att_out_g = self.ops['g_x'].apply(att_out_gh)  # NxH2xW2xC2 or NxC2xH2xW2
        att_out_h = self.ops['h_x'].apply(att_out_gh)  # NxH2xW2xC1 or NxC1xH2xW2

        # flatten the tensor and do batch multiplication
        att_shape_f = att_out_f.get_shape().as_list()  # NxH1xW1xC2 or NxC2xH1xW1
        att_shape_g = att_out_g.get_shape().as_list()  # NxH2xW2xC2 or NxC2xH2xW2
        with tf.name_scope('att_map'):
            if self.data_format == 'channels_first':
                c_float = np.float32(att_shape_g[1])
                att_out_f = tf.reshape(
                    att_out_f, shape=(-1, att_shape_f[1], att_shape_f[2] * att_shape_f[3]))  # NxC2xHW1
                att_out_g = tf.reshape(
                    att_out_g, shape=(-1, att_shape_g[1], att_shape_g[2] * att_shape_g[3]))  # NxC2xHW2
                if self.design['type'] in {'nl_dist', 'nl_pool_dist'}:
                    dist_fg = get_batch_squared_dist(
                        att_out_f, att_out_g, axis=1, mode='xy', name='squared_dist')
                    att_map_logits = -dist_fg / c_float  # NxHW1xHW2
                else:
                    sqrt_channel = np.sqrt(c_float, dtype=np.float32)
                    att_map_logits = tf.matmul(
                        tf.transpose(att_out_f, [0, 2, 1]), att_out_g) / sqrt_channel  # NxHW1xHW2
                    # att_map_logits = tf.matmul(
                    #     tf.transpose(att_out_f, [0, 2, 1]), att_out_g)  # NxHW1xHW2
            else:  # channels_last
                c_float = np.float32(att_shape_g[3])
                att_out_f = tf.reshape(
                    att_out_f, shape=(-1, att_shape_f[1] * att_shape_f[2], att_shape_f[3]))  # NxHW1xC2
                att_out_g = tf.reshape(
                    att_out_g, shape=(-1, att_shape_g[1] * att_shape_g[2], att_shape_g[3]))  # NxHW2xC2
                if self.design['type'] in {'nl_dist', 'nl_pool_dist'}:
                    dist_fg = get_batch_squared_dist(
                        att_out_f, att_out_g, axis=2, mode='xy', name='squared_dist')
                    att_map_logits = -dist_fg / c_float  # NxHW1xHW2
                else:
                    sqrt_channel = np.sqrt(c_float, dtype=np.float32)
                    att_map_logits = tf.matmul(
                        att_out_f, tf.transpose(att_out_g, [0, 2, 1])) / sqrt_channel  # NxHW1xHW2
            # apply softmax to each row of att_map
            att_map = tf.nn.softmax(att_map_logits, axis=2)  # NxHW1xHW2
            # att_map = att_map_logits / hw  # NxHW1xHW2

        # get final attention feature map
        att_shape_h = att_out_h.get_shape().as_list()  # NxH2xW2xC1 or NxC1xH2xW2
        with tf.name_scope('att_features'):
            if self.data_format == 'channels_first':
                att_out_h = tf.reshape(
                    att_out_h, shape=(-1, att_shape_h[1], att_shape_h[2] * att_shape_h[3]))  # NxC1xHW2
                att_out_o = tf.matmul(att_out_h, tf.transpose(att_map, [0, 2, 1]))  # NxC1xHW1
                att_out_o = tf.reshape(  # NxC1xH1xW1
                    att_out_o,
                    shape=(-1, att_shape_h[1], att_shape_f[2], att_shape_f[3]))
            else:  # channels_last
                att_out_h = tf.reshape(
                    att_out_h, shape=(-1, att_shape_h[1] * att_shape_h[2], att_shape_h[3]))  # NxHW2xC1
                att_out_o = tf.matmul(att_map, att_out_h)  # NxHW1xC1
                att_out_o = tf.reshape(  # NxH1xW1xC1
                    att_out_o,
                    shape=(-1, att_shape_f[1], att_shape_f[2], att_shape_h[3]))

        # scalar kernel
        if 'BN_1' in self.ops:  # conditional batch normalization for scalar has not been implemented
            att_out_o = self.ops['BN_1'].apply(
                att_out_o, is_training=is_training)
        att_out_o = self.ops['k_x'].apply(att_out_o)
        # if 'bias_k' in self.ops:
        #     att_out = self.ops['bias_k'].apply(att_out)

        # layer_out
        self._layer_output_['x'] = att_out_o + self._layer_output_['x']

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
            if self.design['type'] in {'default', 'project', 'c_bias'}:
                input_shape = self._add_layer_default_(input_shape)
            elif self.design['type'] in {'res', 'res_i', 'res_v1'}:
                input_shape = self._add_layer_res_(input_shape)
            elif self.design['type'] in {'nl', 'nl_dist', 'nl_pool', 'nl_pool_dist'}:
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

        :param layer_input: a tensor x or a dictionary {'x': x, 'y': y}
        :param is_training:
        :return:
        """
        self.build_layer()  # in case layer has not been build
        with tf.variable_scope(self.layer_scope, reuse=tf.AUTO_REUSE):
            self._input_(layer_input)
            self._apply_input_reshape_()

            # register ops
            if self.design['type'] in {'default', 'project', 'c_bias'}:
                self._apply_layer_default_(is_training)
            elif self.design['type'] in {'res', 'res_i', 'res_v1'}:
                self._apply_layer_res_(is_training)
            elif self.design['type'] in {'nl', 'nl_dist', 'nl_pool', 'nl_pool_dist'}:
                self._apply_layer_nonlocal_(is_training)

            self._apply_output_reshape_()
            return self._output_()

    def apply(self, layer_input, is_training=True):
        """ This function calls self.__call__

        :param layer_input: a tensor x or a dictionary {'x': x, 'y': y}
        :param is_training:
        :return:
        """
        return self.__call__(layer_input, is_training)


class Net(object):
    """ This class is designed to:
        1. allow constraints on layer weights and bias,
        2. ease construction of networks with complex structure that need to refer to specific layers

    """

    def __init__(
            self, net_design, net_name='net', data_format=None, num_class=0):
        """ This function initializes a network

        :param net_design: [(layer_type, channel_out, act_fun_name(, normalization_method, kernel_size, strides,
        dilation, padding, scale_factor))].
        The first five parameters are channel_out, kernel_size, stride, dilation, activation.
        The rest one is scale_factor, which is optional. When doing up-scaling, one extra conv
        layer is added.
        :param net_name:
        :param data_format:
        :param num_class:
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
            if layer_design['op'] in {'d', 'dcd', 'dck'}:
                layer_data_format = None
            elif layer_design['op'] in {'i'} and self.layers[i-1].design['op'] in {'d', 'dcd', 'dck'}:
                layer_data_format = None
            else:
                layer_data_format = data_format
            self.layers.append(
                Layer(
                    layer_design, name_prefix=self.net_name + '/',
                    data_format=layer_data_format, num_class=num_class))
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

    def add_summary(self, attribute):
        """ This function adds the attribute of each layer into summary

        :param attribute: string, e.g. 'kernel_norm'
        :return:
        """
        for layer in self.layers:
            for key in layer.ops:
                values = getattr(layer.ops[key], attribute, None)
                if values is not None:
                    prefix = attribute + '/' + layer.ops[key].name_in_err
                    if not isinstance(values, (list, tuple)):
                        values = [values]
                    for count, value in enumerate(values):
                        summary_name = prefix if count == 0 else prefix + '_{}'.format(count)
                        if len(value.get_shape().as_list()) == 0:
                            tf.summary.scalar(summary_name, value)
                        else:
                            tf.summary.histogram(summary_name, value)


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
