# default modules
import numpy as np
import tensorflow as tf
import warnings
from GeneralTools.misc_fun import FLAGS


########################################################################
def kron_by_reshape(mat1, mat2, mat_shape=None):
    """ This function does kronecker product through reshape and perm

    :param mat1: 2-D tensor
    :param mat2: 2-D tensor
    :param mat_shape: shape of mat1 and mat2
    :return mat3: mat3 = kronecker(mat1, mat2)
    """
    if mat_shape is None:
        a, b = mat1.shape
        c, d = mat2.shape
    else:  # in case of tensorflow, mat_shape must be provided
        a, b, c, d = mat_shape

    if isinstance(mat1, np.ndarray) and isinstance(mat2, np.ndarray):
        mat3 = np.matmul(np.reshape(mat1, [-1, 1]), np.reshape(mat2, [1, -1]))  # (axb)-by-(cxd)
        mat3 = np.reshape(mat3, [a, b, c, d])  # a-by-b-by-c-by-d
        mat3 = np.transpose(mat3, axes=[0, 2, 1, 3])  # a-by-c-by-b-by-d
        mat3 = np.reshape(mat3, [a * c, b * d])  # (axc)-by-(bxd)
    elif isinstance(mat1, tf.Tensor) and isinstance(mat2, tf.Tensor):
        mat3 = tf.matmul(tf.reshape(mat1, [-1, 1]), tf.reshape(mat2, [1, -1]))  # (axb)-by-(cxd)
        mat3 = tf.reshape(mat3, [a, b, c, d])  # a-by-b-by-c-by-d
        mat3 = tf.transpose(mat3, perm=[0, 2, 1, 3])  # a-by-c-by-b-by-d
        mat3 = tf.reshape(mat3, [a * c, b * d])  # (axc)-by-(bxd)
    else:
        raise AttributeError('Input should be numpy array or tensor')

    return mat3


########################################################################
def scale_range(x, scale_min=-1.0, scale_max=1.0, axis=1):
    """ This function scales numpy matrix to range [scale_min, scale_max]

    """
    x_min = np.amin(x, axis=axis, keepdims=True)
    x_range = np.amax(x, axis=axis, keepdims=True) - x_min
    x_range[x_range == 0.0] = 1.0
    # scale to [0,1]
    x = (x - x_min) / x_range
    # scale to [scale_min, scale_max]
    x = x * (scale_max - scale_min) + scale_min

    return x


########################################################################
def mean_cov_np(x):
    """ This function calculates mean and covariance for 2d array x.
    This function is faster than separately running np.mean and np.cov

    :param x: 2D array, columns of x represents variables.
    :return:
    """
    mu = np.mean(x, axis=0)
    x_centred = x - mu
    cov = np.matmul(x_centred.transpose(), x_centred) / (x.shape[0] - 1.0)

    return mu, cov


########################################################################
def mean_cov_tf(x):
    """ This function calculates mean and covariance for 2d array x.

    :param x: 2D array, columns of x represents variables.
    :return:
    """
    mu = tf.reduce_mean(x, axis=0, keepdims=True)  # 1-D
    x_centred = x - mu
    cov = tf.matmul(x_centred, x_centred, transpose_a=True) / (x.get_shape().as_list()[0] - 1.0)

    return mu, cov


########################################################################
def scale_image_range(image, scale_min=-1.0, scale_max=1.0, image_format='channels_last'):
    """ This function scales images per channel to [-1,1]. The max and min are calculated over all samples.

    Note that, in batch normalization, they also calculate the mean and std for each feature map.

    :param image: 4-D numpy array, either in channels_first format or channels_last format
    :param scale_min:
    :param scale_max:
    :param image_format
    :return:
    """
    if len(image.shape) != 4:
        raise AttributeError('Input must be 4-D tensor.')

    if image_format == 'channels_last':
        num_instance, height, width, num_channel = image.shape
        pixel_channel = image.reshape((-1, num_channel))  # [pixels, channel]
        pixel_channel = scale_range(pixel_channel, scale_min=scale_min, scale_max=scale_max, axis=0)
        image = pixel_channel.reshape((num_instance, height, width, num_channel))
    elif image_format == 'channels_first':
        # scale_range works faster when axis=1, work on this
        image = np.transpose(image, axes=(1, 0, 2, 3))
        num_channel, num_instance, height, width = image.shape
        pixel_channel = image.reshape((num_channel, -1))  # [channel, pixels]
        pixel_channel = scale_range(pixel_channel, scale_min=scale_min, scale_max=scale_max, axis=1)
        image = pixel_channel.reshape((num_channel, num_instance, height, width))
        image = np.transpose(image, axes=(1, 0, 2, 3))  # convert back to channels_first

    return image


########################################################################
def pairwise_dist(mat1, mat2=None):
    """ This function calculates the pairwise distance matrix dist. If mat2 is not provided,
    dist is defined among row vectors of mat1.

    The distance is formed as sqrt(mat1*mat1' - 2*mat1*mat2' + mat2*mat2')

    :param mat1:
    :param mat2:
    :return:
    """
    # tf.reduce_sum() will produce result of shape (N,), which, when transposed, is still (N,)
    # Thus, to force mm1 and mm2 (or mm1') to have different shape, tf.expand_dims() is used
    mm1 = tf.expand_dims(tf.reduce_sum(tf.multiply(mat1, mat1), axis=1), axis=1)
    if mat2 is None:
        mmt = tf.multiply(tf.matmul(mat1, mat1, transpose_b=True), -2)
        dist = tf.sqrt(tf.add(tf.add(tf.add(mm1, tf.transpose(mm1)), mmt), FLAGS.EPSI))
    else:
        mm2 = tf.expand_dims(tf.reduce_sum(tf.multiply(mat2, mat2), axis=1), axis=0)
        mrt = tf.multiply(tf.matmul(mat1, mat2, transpose_b=True), -2)
        dist = tf.sqrt(tf.add(tf.add(tf.add(mm1, mm2), mrt), FLAGS.EPSI))
        # dist = tf.sqrt(tf.add(tf.add(mm1, mm2), mrt))

    return dist


########################################################################
def slerp(p0, p1, t):
    """ This function calculates the spherical linear interpolation of p0 and p1

    :param p0: a vector of shape (d, )
    :param p1: a vector of shape (d, )
    :param t: a scalar, or a vector of shape (n, )
    :return:

    Numeric instability may occur when theta is close to zero or pi. In these cases,
    sin(t * theta) >> sin(theta). These cases are common, e.g. p0 = -p1.

    """
    from numpy.linalg import norm

    theta = np.arccos(np.dot(p0 / norm(p0), p1 / norm(p1)), dtype=np.float32)
    st = np.sin(theta)  # there is no dtype para for np.sin
    # in case t is a vector, output is a row matrix
    if not np.isscalar(t):
        p0 = np.expand_dims(p0, axis=0)
        p1 = np.expand_dims(p1, axis=0)
        t = np.expand_dims(t, axis=1)
    if st > 0.1:
        p2 = np.sin((1.0 - t) * theta) / st * p0 + np.sin(t * theta) / st * p1
    else:
        p2 = (1.0 - t) * p0 + t * p1

    return p2


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
    if isinstance(input_spatial_shape, (list, tuple)):
        return [spatial_shape_after_conv(
            one_shape, kernel_size, strides, dilation, padding) for one_shape in input_spatial_shape]
    else:
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
    if isinstance(input_spatial_shape, (list, tuple)):
        return [spatial_shape_after_transpose_conv(
            one_shape, kernel_size, strides, dilation, padding) for one_shape in input_spatial_shape]
    else:
        if padding in ['same', 'SAME']:
            return np.int(input_spatial_shape * strides)
        else:
            return np.int(input_spatial_shape * strides + (kernel_size - 1) * dilation)


########################################################################
class MeshCode(object):
    def __init__(self, code_length, mesh_num=None):
        """ This function creates meshed code for generative models

        :param code_length:
        :param mesh_num:
        :return:
        """
        self.D = code_length
        if mesh_num is None:
            self.mesh_num = (10, 10)
        else:
            self.mesh_num = mesh_num

    def get_batch(self, mesh_mode, name=None):
        if name is None:
            name = 'Z'
        if mesh_mode == 0 or mesh_mode == 'random':
            z_batch = self.by_random(name)
        elif mesh_mode == 1 or mesh_mode == 'sine':
            z_batch = self.by_sine(name)
        elif mesh_mode == 2 or mesh_mode == 'feature':
            z_batch = self.by_feature(name)
        else:
            raise AttributeError('mesh_mode is not supported.')
        return z_batch

    def by_random(self, name=None):
        """ This function generates mesh code randomly

        :param name:
        :return:
        """
        return tf.random_normal(
            [self.mesh_num[0] * self.mesh_num[1], self.D],
            mean=0.0,
            stddev=1.0,
            name=name)

    def by_sine(self, z_support=None, name=None):
        """ This function creates mesh code by interpolating between four supporting codes

        :param z_support:
        :param name: list or tuple of two elements
        :return:
        """
        if z_support is None:
            z_support = tf.random_normal(
                [4, self.D],
                mean=0.0,
                stddev=1.0)
        elif isinstance(z_support, np.ndarray):
            z_support = tf.constant(z_support, dtype=tf.float32)
        z0 = tf.expand_dims(z_support[0], axis=0)  # create 1-by-D vector
        z1 = tf.expand_dims(z_support[1], axis=0)
        z2 = tf.expand_dims(z_support[2], axis=0)
        z3 = tf.expand_dims(z_support[3], axis=0)
        # generate phi and psi from 0 to 90 degrees
        mesh_phi = np.float32(  # mesh_num[0]-by-1 vector
            np.expand_dims(np.pi / 4.0 * np.linspace(0.0, 1.0, self.mesh_num[0]), axis=1))
        mesh_psi = np.float32(
            np.expand_dims(np.pi / 4.0 * np.linspace(0.0, 1.0, self.mesh_num[1]), axis=1))
        # sample instances on the manifold
        z_batch = tf.identity(  # mesh_num[0]*mesh_num[1]-by-1 vector
            kron_by_reshape(  # do kronecker product
                tf.matmul(tf.cos(mesh_psi), z0) + tf.matmul(tf.sin(mesh_psi), z1),
                tf.cos(mesh_phi),
                mat_shape=[self.mesh_num[1], self.D, self.mesh_num[0], 1])
            + kron_by_reshape(
                tf.matmul(tf.cos(mesh_psi), z2) + tf.matmul(tf.sin(mesh_psi), z3),
                tf.sin(mesh_phi),
                mat_shape=[self.mesh_num[1], self.D, self.mesh_num[0], 1]),
            name=name)

        return z_batch

    def by_feature(self, grid=2.0, name=None):
        """ This function creates mesh code by varying a single feature. In this case,
        mesh_num[0] refers to the number of features to mesh, mesh[1] refers to the number
        of variations in one feature

        :param grid:
        :param name: string
        :return:
        """
        mesh = np.float32(  # mesh_num[0]-by-1 vector
            np.expand_dims(np.linspace(-grid, grid, self.mesh_num[1]), axis=1))
        # sample instances on the manifold
        z_batch = kron_by_reshape(  # mesh_num[0]*mesh_num[1]-by-1 vector
            tf.eye(num_rows=self.mesh_num[0], num_columns=self.D),
            tf.constant(mesh),
            mat_shape=[self.mesh_num[0], self.D, self.mesh_num[1], 1])
        # shuffle the columns of z_batch
        z_batch = tf.identity(
            tf.transpose(tf.random_shuffle(tf.transpose(z_batch, perm=[1, 0])), perm=[1, 0]),
            name=name)

        return z_batch

    def simple_grid(self, grid=None):
        """ This function creates simple grid meshes

        Note: this function returns np.ndarray

        :param grid:
        :return:
        """
        if self.D != 2:
            raise AttributeError('Code length has to be two')
        if grid is None:
            grid = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        x = np.linspace(grid[0][0], grid[0][1], self.mesh_num[0])
        y = np.linspace(grid[1][0], grid[1][1], self.mesh_num[1])
        z0 = np.reshape(np.transpose(np.tile(x, (self.mesh_num[1], 1))), [-1, 1])
        z1 = np.reshape(np.tile(y, (1, self.mesh_num[0])), [-1, 1])
        z = np.concatenate((z0, z1), axis=1)

        return z, x, y

    def j_diagram(self, name=None):
        """ This function creates a j diagram using slerp

        This function is not finished as there is a problem with the slerp idea.

        :param name:
        :return:
        """
        raise NotImplementedError('This function has not been implemented.')
        # z_support = np.random.randn(4, self.D)
        # z0 = tf.expand_dims(z_support[0], axis=0)  # create 1-by-D vector
        # z1 = tf.expand_dims(z_support[1], axis=0)
        # z2 = tf.expand_dims(z_support[2], axis=0)
        # pass


########################################################################
def mat_slice(mat, row_index, col_index=None, name='slice'):
    """ This function gets mat[index, index] where index is either bool or int32.

    Note that:
        if index is bool, output size is typically smaller than mat unless each element in index is True
        if index is int32, output can be any size.

    :param mat:
    :param row_index:
    :param col_index:
    :param name;
    :return:
    """
    if col_index is None:
        col_index = row_index

    with tf.name_scope(name):
        if row_index.dtype != col_index.dtype:
            raise AttributeError('dtype of row-index and col-index do not match.')
        if row_index.dtype == tf.int32:
            return tf.gather(tf.gather(mat, row_index, axis=0), col_index, axis=1)
        elif row_index.dtype == tf.bool:
            return tf.boolean_mask(tf.boolean_mask(mat, row_index, axis=0), col_index, axis=1)
        else:
            raise AttributeError('Type of index is: {}; expected either tf.int32 or tf.bool'.format(row_index.dtype))


########################################################################
def l2normalization(w):
    """ This function applies l2 normalization to the input vector.
    If w is a matrix / tensor, the Frobenius norm is used for normalization.

    :param w:
    :return:
    """

    # tf.norm is slightly faster than tf.sqrt(tf.reduce_sum(tf.square()))
    # it is important that axis=None; in this case, norm(w) = norm(vec(w))
    return w / (tf.norm(w, ord='euclidean', axis=None) + FLAGS.EPSI)


class SpectralNorm(object):
    def __init__(self, sn_def, name_scope='SN', scope_prefix='', num_iter=1):
        """ This class contains functions to calculate the spectral normalization of the weight matrix
        using power iteration.

        The application of spectral normal to NN is proposed in following papers:
        Yoshida, Y., & Miyato, T. (2017).
        Spectral Norm Regularization for Improving the Generalizability of Deep Learning.
        Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2017).
        Spectral Normalization for Generative Adversarial Networks,
        Here spectral normalization is generalized for any linear ops or combination of linear ops

        Example of usage:
        Example 1.
        w = tf.constant(np.random.randn(3, 3, 128, 64).astype(np.float32))
        sn_def = {'op': 'tc', 'input_shape': [10, 64, 64, 64],
                  'output_shape': [10, 128, 64, 64],
                  'strides': 1, 'dilation': 1, 'padding': 'SAME',
                  'data_format': 'NCHW'}
        sigma = SpectralNorm(sn_def, name_scope='SN1', num_iter=20).apply(w)

        Example 2.
        w = tf.constant(np.random.randn(3, 3, 128, 64).astype(np.float32))
        w2 = tf.constant(np.random.randn(3, 3, 128, 64).astype(np.float32))
        sn_def = {'op': 'tc', 'input_shape': [10, 64, 64, 64],
                  'output_shape': [10, 128, 64, 64],
                  'strides': 1, 'dilation': 1, 'padding': 'SAME',
                  'data_format': 'NCHW'}

        SN = SpectralNorm(sn_def, num_iter=20)
        sigma1 = SN.apply(w)
        sigma2 = SN.apply(w2, name_scope='SN2', num_iter=30)


        :param sn_def: a dictionary with keys depending on the type of kernel:
            type     keys   value options
            dense:    'op'    'd' - common dense layer; 'cd' - conditional dense layers;
                            'dcd' - dense + conditional dense; 'dck' - dense * conditional scale
                            'project' - same to cd, except num_out is 1
            conv:    'op'    'c' - convolution; 'tc' - transpose convolution;
                            'cck' - convolution * conditional scale; 'tcck' - t-conv * conditional scale
                     'strides'    integer
                     'dilation'    integer
                     'padding'    'SAME' or 'VALID'
                     'data_format'    'NCHW' or 'NHWC'
                     'input_shape'    list of integers in format NCHW or NHWC
                     'output_shape'    for 'tc', output shape must be provided
        :param name_scope:
        :param scope_prefix:
        :param num_iter: number of power iterations per run
        """
        self.sn_def = sn_def.copy()
        self.name_scope = name_scope
        self.scope_prefix = scope_prefix
        self.name_in_err = self.scope_prefix + self.name_scope
        self.num_iter = num_iter
        # initialize
        self.w = None
        self.x = None
        self.use_u = None
        self.is_initialized = False
        self.forward = None
        self.backward = None

        # format stride
        if self.sn_def['op'] in {'c', 'tc', 'cck', 'tcck'}:
            if self.sn_def['data_format'] in ['NCHW', 'channels_first']:
                self.sn_def['strides'] = (1, 1, self.sn_def['strides'], self.sn_def['strides'])
            else:
                self.sn_def['strides'] = (1, self.sn_def['strides'], self.sn_def['strides'], 1)
            assert 'output_shape' in self.sn_def, \
                '{}: for conv, output_shape must be provided.'.format(self.name_in_err)

    def _init_routine(self):
        """ This function decides the routine to minimize memory usage

        :return:
        """
        if self.is_initialized is False:
            # decide the routine
            if self.sn_def['op'] in {'d', 'project'}:
                # for d kernel_shape [num_in, num_out]; for project, kernel shape [num_class, num_in]
                assert len(self.kernel_shape) == 2, \
                    '{}: kernel shape {} does not have length 2'.format(self.name_in_err, self.kernel_shape)
                num_in, num_out = self.kernel_shape
                # self.use_u = True
                self.use_u = True if num_in <= num_out else False
                x_shape = [1, num_in] if self.use_u else [1, num_out]
                self.forward = self._dense_ if self.use_u else self._dense_t_
                self.backward = self._dense_t_ if self.use_u else self._dense_
            elif self.sn_def['op'] in {'cd'}:  # kernel_shape [num_class, num_in, num_out]
                assert len(self.kernel_shape) == 3, \
                    '{}: kernel shape {} does not have length 3'.format(self.name_in_err, self.kernel_shape)
                num_class, num_in, num_out = self.kernel_shape
                self.use_u = True if num_in <= num_out else False
                x_shape = [num_class, 1, num_in] if self.use_u else [num_class, 1, num_out]
                self.forward = self._dense_ if self.use_u else self._dense_t_
                self.backward = self._dense_t_ if self.use_u else self._dense_
            elif self.sn_def['op'] in {'dck'}:  # convolution * conditional scale
                assert isinstance(self.kernel_shape, (list, tuple)) and len(self.kernel_shape) == 2, \
                    '{}: kernel shape must be a list of length 2. Got {}'.format(self.name_in_err, self.kernel_shape)
                assert len(self.kernel_shape[0]) == 2 and len(self.kernel_shape[1]) == 2, \
                    '{}: kernel shape {} does not have length 2'.format(self.name_in_err, self.kernel_shape)
                num_in, num_out = self.kernel_shape[0]
                num_class = self.kernel_shape[1][0]
                self.use_u = True if num_in <= num_out else False
                x_shape = [num_class, num_in] if self.use_u else [num_class, num_out]
                self.forward = (lambda x: self._scalar_(self._dense_(x, index=0), index=1, offset=1.0)) \
                    if self.use_u else (lambda y: self._dense_t_(self._scalar_(y, index=1, offset=1.0), index=0))
                self.backward = (lambda y: self._dense_t_(self._scalar_(y, index=1, offset=1.0), index=0)) \
                    if self.use_u else (lambda x: self._scalar_(self._dense_(x, index=0), index=1, offset=1.0))
            elif self.sn_def['op'] in {'c', 'tc'}:
                assert len(self.kernel_shape) == 4, \
                    '{}: kernel shape {} does not have length 4'.format(self.name_in_err, self.kernel_shape)
                # self.use_u = True
                self.use_u = True \
                    if np.prod(self.sn_def['input_shape'][1:]) <= np.prod(self.sn_def['output_shape'][1:]) \
                    else False
                if self.sn_def['op'] in {'c'}:  # input / output shape NCHW or NHWC
                    x_shape = self.sn_def['input_shape'].copy() if self.use_u else self.sn_def['output_shape'].copy()
                    x_shape[0] = 1
                    y_shape = self.sn_def['input_shape'].copy()
                    y_shape[0] = 1
                elif self.sn_def['op'] in {'tc'}:  # tc
                    x_shape = self.sn_def['output_shape'].copy() if self.use_u else self.sn_def['input_shape'].copy()
                    x_shape[0] = 1
                    y_shape = self.sn_def['output_shape'].copy()
                    y_shape[0] = 1
                else:
                    raise NotImplementedError('{}: {} not implemented.'.format(self.name_in_err, self.sn_def['op']))
                self.forward = self._conv_ if self.use_u else (lambda y: self._conv_t_(y, x_shape=y_shape))
                self.backward = (lambda y: self._conv_t_(y, x_shape=y_shape)) if self.use_u else self._conv_
            elif self.sn_def['op'] in {'cck', 'tcck'}:  # convolution * conditional scale
                assert isinstance(self.kernel_shape, (list, tuple)) and len(self.kernel_shape) == 2, \
                    '{}: kernel shape must be a list of length 2. Got {}'.format(self.name_in_err, self.kernel_shape)
                assert len(self.kernel_shape[0]) == 4 and len(self.kernel_shape[1]) == 4, \
                    '{}: kernel shape {} does not have length 4'.format(self.name_in_err, self.kernel_shape)
                self.use_u = True \
                    if np.prod(self.sn_def['input_shape'][1:]) <= np.prod(self.sn_def['output_shape'][1:]) \
                    else False
                num_class = self.kernel_shape[1][0]
                if self.sn_def['op'] in {'cck'}:  # input / output shape NCHW or NHWC
                    x_shape = self.sn_def['input_shape'].copy() if self.use_u else self.sn_def['output_shape'].copy()
                    x_shape[0] = num_class
                    y_shape = self.sn_def['input_shape'].copy()
                    y_shape[0] = num_class
                    self.forward = (lambda x: self._scalar_(self._conv_(x, index=0), index=1, offset=1.0)) \
                        if self.use_u \
                        else (lambda y: self._conv_t_(self._scalar_(y, index=1, offset=1.0), x_shape=y_shape, index=0))
                    self.backward = (lambda y: self._conv_t_(
                        self._scalar_(y, index=1, offset=1.0), x_shape=y_shape, index=0)) \
                        if self.use_u else (lambda x: self._scalar_(self._conv_(x, index=0), index=1, offset=1.0))
                elif self.sn_def['op'] in {'tcck'}:  # tcck
                    x_shape = self.sn_def['output_shape'].copy() if self.use_u else self.sn_def['input_shape'].copy()
                    x_shape[0] = num_class
                    y_shape = self.sn_def['output_shape'].copy()
                    y_shape[0] = num_class
                    self.forward = (lambda x: self._conv_(self._scalar_(x, index=1, offset=1.0), index=0)) \
                        if self.use_u \
                        else (lambda y: self._scalar_(self._conv_t_(y, x_shape=y_shape, index=0), index=1, offset=1.0))
                    self.backward = (lambda y: self._scalar_(
                        self._conv_t_(y, x_shape=y_shape, index=0), index=1, offset=1.0)) \
                        if self.use_u else (lambda x: self._conv_(self._scalar_(x, index=1, offset=1.0), index=0))
                else:
                    raise NotImplementedError('{}: {} not implemented.'.format(self.name_in_err, self.sn_def['op']))
            else:
                raise NotImplementedError('{}: {} is not implemented.'.format(self.name_in_err, self.sn_def['op']))

            self.x = tf.get_variable(
                'in_rand', shape=x_shape, dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(), trainable=False)

            self.is_initialized = True

    def _scalar_(self, x, index=None, offset=0.0):
        """ This function defines a elementwise multiplication op: y = x * w, where x shape [N, C, ...] or [N, ..., C],
        w shape [N, C, 1,..,1] or [N, 1,...,1, C], y shape [N, C, ...] or [N, ..., C]

        :param x:
        :param index: if index is provided, self.w is a list or tuple
        :param offset: add a constant offset
        :return:
        """
        w = self.w if index is None else self.w[index]
        return tf.multiply(x, w, name='scalar') if offset == 0.0 else tf.multiply(x, w + offset, name='scalar')

    def _dense_(self, x, index=None):
        """ This function defines a dense op: y = x * w, where x shape [..., a, b], w shape [..., b, c],
        y shape [..., a, c]

        :param x:
        :param index: if index is provided, self.w is a list or tuple
        :return:
        """
        w = self.w if index is None else self.w[index]
        return tf.matmul(x, w, name='dense')

    def _dense_t_(self, y, index=None):
        """ Transpose version of self._dense_

        :param y:
        :param index: if index is provided, self.w is a list or tuple
        :return:
        """
        w = self.w if index is None else self.w[index]
        return tf.matmul(y, w, transpose_b=True, name='dense_t')

    def _conv_(self, x, index=None):
        """ This function defines a conv op: y = x \otimes w, where x shape NCHW or NHWC, w shape kkhw,
        y shape NCHW or NHWC

        :param x:
        :param index: if index is provided, self.w is a list or tuple
        :return:
        """
        w = self.w if index is None else self.w[index]
        if self.sn_def['dilation'] > 1:
            return tf.nn.atrous_conv2d(
                x, w, rate=self.sn_def['dilation'], padding=self.sn_def['padding'], name='conv')
        else:
            return tf.nn.conv2d(
                x, w, strides=self.sn_def['strides'], padding=self.sn_def['padding'],
                data_format=self.sn_def['data_format'], name='conv')

    def _conv_t_(self, y, x_shape, index=None):
        """ Transpose version of self._conv_
        
        :param y: 
        :param x_shape:
        :param index: 
        :return: 
        """
        w = self.w if index is None else self.w[index]
        if self.sn_def['dilation'] > 1:
            return tf.nn.atrous_conv2d_transpose(
                y, w, output_shape=x_shape, rate=self.sn_def['dilation'], padding=self.sn_def['padding'],
                name='conv_t')
        else:
            return tf.nn.conv2d_transpose(
                y, w, output_shape=x_shape, strides=self.sn_def['strides'], padding=self.sn_def['padding'],
                data_format=self.sn_def['data_format'], name='conv_t')

    def _l2_norm(self, x):
        if self.sn_def['op'] in {'cd'}:  # x shape [num_class, 1, num_in or num_out]
            return tf.norm(x, ord='euclidean', axis=2, keepdims=True)  # return [num_class, 1, 1]
        elif self.sn_def['op'] in {'dck'}:  # x shape [num_class, num_in or num_out]
            return tf.norm(x, ord='euclidean', axis=1, keepdims=True)  # return [num_class, 1]
        elif self.sn_def['op'] in {'cck', 'tcck'}:
            # x shape [num_class, num_in or num_out, H, W] or [num_class, H, W, num_in or num_out]
            # here i did not use tf.norm because axis cannot be (1, 2, 3)
            return tf.sqrt(
                tf.reduce_sum(tf.square(x), axis=(1, 2, 3), keepdims=True), name='norm')  # return [num_class, 1, 1, 1]
        elif self.sn_def['op'] in {'d', 'c', 'tc', 'project'}:
            # x shape [1, num_in or num_out], or [1, num_in or num_out, H, W] or [1, H, W, num_in or num_out]
            return tf.norm(x, ord='euclidean', axis=None)  # return scalar

    def _l2_normalize_(self, w):
        """

        :param w:
        :return:
        """
        return w / (self._l2_norm(w) + FLAGS.EPSI)

    def _power_iter_(self, x, step):
        """ This function does power iteration for one step

        :param x:
        :param step:
        :return:
        """
        y = self._l2_normalize_(self.forward(x))
        x_update = self._l2_normalize_(self.backward(y))
        sigma = self._l2_norm(self.forward(x))

        return sigma, x_update, step + 1

    def __call__(self, kernel, **kwargs):
        """ This function calculates spectral normalization for kernel

        :param kernel:
        :param kwargs:
        :return:
        """
        # check inputs
        if 'name_scope' in kwargs and kwargs['name_scope'] != self.name_scope:
            # different name_scope will initialize another SN process
            self.name_scope = kwargs['name_scope']
            self.name_in_err = self.scope_prefix + self.name_scope
            if self.is_initialized:
                warnings.warn(
                    '{}: a new SN process caused lost of links to the previous one.'.format(self.name_in_err))
                self.is_initialized = False
            self.use_u = None
        if 'num_iter' in kwargs:
            self.num_iter = kwargs['num_iter']
        if isinstance(kernel, (list, tuple)):
            # for dcd, cck, the kernel is a list of two kernels
            kernel_shape = [k.get_shape().as_list() for k in kernel]
        else:
            kernel_shape = kernel.get_shape().as_list()

        with tf.variable_scope(self.name_scope, reuse=tf.AUTO_REUSE):
            # In some cases, the spectral norm can be easily calculated.
            sigma = None
            if self.sn_def['op'] in {'d', 'project'} and 1 in kernel_shape:
                # for project op. kernel_shape = [num_class, num_in]
                sigma = tf.norm(kernel, ord='euclidean')
            elif self.sn_def['op'] in {'cd'}:
                if len(kernel_shape) == 2:  # equivalent to [num_class, num_in, 1]
                    sigma = tf.norm(kernel, ord='euclidean', axis=1, keepdims=True)
                elif kernel_shape[1] == 1 or kernel_shape[2] == 1:
                    sigma = tf.norm(kernel, ord='euclidean', axis=(1, 2), keepdims=True)
            elif self.sn_def['op'] in {'dcd'}:  # dense + conditional dense
                # kernel_cd [num_class, num_in, num_out]
                kernel_cd = tf.expand_dims(kernel[1], axis=2) if len(kernel_shape[1]) == 2 else kernel[1]
                kernel = tf.expand_dims(kernel[0], axis=0) + kernel_cd  # [num_class, num_in, num_out]
                if 1 in kernel_shape[0]:  # kernel_d shape [1, num_out] or [num_in, 1]
                    sigma = tf.norm(kernel, ord='euclidean', axis=(1, 2), keepdims=True)  # [num_class, 1, 1]
                else:  # convert dcd to cd
                    kernel_shape = kernel.get_shape().as_list()
                    self.sn_def['op'] = 'cd'
            elif self.sn_def['op'] in {'dck'}:  # dense * conditional scales
                if kernel_shape[0][1] == 1:
                    sigma = tf.norm(kernel[0], ord='euclidean') * tf.abs(kernel[1])  # [num_class, 1]

            # initialize a random input and calculate spectral norm
            if sigma is None:
                # decide the routine
                self.w = kernel
                self.kernel_shape = kernel_shape
                self._init_routine()
                # initialize sigma
                if self.sn_def['op'] in {'dck'}:
                    sigma_init = tf.zeros((self.kernel_shape[1][0], 1), dtype=tf.float32)
                elif self.sn_def['op'] in {'cd'}:  # for cd, the sigma is a [num_class, 1, 1]
                    sigma_init = tf.zeros((self.kernel_shape[0], 1, 1), dtype=tf.float32)
                elif self.sn_def['op'] in {'cck', 'tcck'}:
                    sigma_init = tf.zeros((self.kernel_shape[1][0], 1, 1, 1), dtype=tf.float32)
                else:
                    sigma_init = tf.constant(0.0, dtype=tf.float32)
                # do power iterations
                sigma, x_update, _ = tf.while_loop(
                    cond=lambda _1, _2, i: i < self.num_iter,
                    body=lambda _1, x, i: self._power_iter_(x, step=i),
                    loop_vars=(sigma_init, self.x, tf.constant(0, dtype=tf.int32)))
                # update the random input
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(self.x, x_update))

        return sigma

    def apply(self, kernel, **kwargs):
        return self.__call__(kernel, **kwargs)


########################################################################
def batch_norm(tensor, axis=None, keepdims=False, name='norm'):
    """ This function calculates the l2 norm for each instance in a batch

    :param tensor: shape [batch_size, ...]
    :param axis: the axis to calculate norm, could be integer or list/tuple of integers
    :param keepdims: whether to keep dimensions
    :param name:
    :return:
    """
    with tf.name_scope(name):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=axis, keepdims=keepdims))


########################################################################
def get_squared_dist(
        x, y=None, scale=None, z_score=False, mode='xxxyyy', name='squared_dist',
        do_summary=False, scope_prefix=''):
    """ This function calculates the pairwise distance between x and x, x and y, y and y

    Warning: when x, y has mean far away from zero, the distance calculation is not accurate; use get_dist_ref instead

    :param x: batch_size-by-d matrix
    :param y: batch_size-by-d matrix
    :param scale: 1-by-d vector, the precision vector. dxy = x*scale*y
    :param z_score:
    :param mode: 'xxxyyy', 'xx', 'xy', 'xxxy'
    :param name:
    :param do_summary:
    :param scope_prefix: summary scope prefix
    :return:
    """
    with tf.name_scope(name):
        # check inputs
        if len(x.get_shape().as_list()) > 2:
            raise AttributeError('get_dist: Input must be a matrix.')
        if y is None:
            mode = 'xx'
        if z_score:
            if y is None:
                mu = tf.reduce_mean(x, axis=0, keepdims=True)
                x = x - mu
            else:
                mu = tf.reduce_mean(tf.concat((x, y), axis=0), axis=0, keepdims=True)
                x = x - mu
                y = y - mu

        if mode in ['xx', 'xxxy', 'xxxyyy']:
            if scale is None:
                xxt = tf.matmul(x, x, transpose_b=True)  # [xi_xi, xi_xj; xj_xi, xj_xj], batch_size-by-batch_size
            else:
                xxt = tf.matmul(x * scale, x, transpose_b=True)
            dx = tf.diag_part(xxt)  # [xxt], [batch_size]
            dist_xx = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xxt + tf.expand_dims(dx, axis=0), 0.0)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                    tf.summary.histogram(scope_prefix + name + '/dxx', dist_xx)

            if mode == 'xx':
                return dist_xx
            elif mode == 'xxxy':  # estimate dy without yyt
                if scale is None:
                    xyt = tf.matmul(x, y, transpose_b=True)
                    dy = tf.reduce_sum(tf.multiply(y, y), axis=1)
                else:
                    xyt = tf.matmul(x * scale, y, transpose_b=True)
                    dy = tf.reduce_sum(tf.multiply(y * scale, y), axis=1)
                dist_xy = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xyt + tf.expand_dims(dy, axis=0), 0.0)
                if do_summary:
                    with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                        tf.summary.histogram(scope_prefix + name + '/dxy', dist_xy)

                return dist_xx, dist_xy
            elif mode == 'xxxyyy':
                if scale is None:
                    xyt = tf.matmul(x, y, transpose_b=True)
                    yyt = tf.matmul(y, y, transpose_b=True)
                else:
                    xyt = tf.matmul(x * scale, y, transpose_b=True)
                    yyt = tf.matmul(y * scale, y, transpose_b=True)
                dy = tf.diag_part(yyt)
                dist_xy = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xyt + tf.expand_dims(dy, axis=0), 0.0)
                dist_yy = tf.maximum(tf.expand_dims(dy, axis=1) - 2.0 * yyt + tf.expand_dims(dy, axis=0), 0.0)
                if do_summary:
                    with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                        tf.summary.histogram(scope_prefix + name + '/dxy', dist_xy)
                        tf.summary.histogram(scope_prefix + name + '/dyy', dist_yy)

                return dist_xx, dist_xy, dist_yy

        elif mode == 'xy':
            if scale is None:
                dx = tf.reduce_sum(tf.multiply(x, x), axis=1)
                dy = tf.reduce_sum(tf.multiply(y, y), axis=1)
                xyt = tf.matmul(x, y, transpose_b=True)
            else:
                dx = tf.reduce_sum(tf.multiply(x * scale, x), axis=1)
                dy = tf.reduce_sum(tf.multiply(y * scale, y), axis=1)
                xyt = tf.matmul(x * scale, y, transpose_b=True)
            dist_xy = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xyt + tf.expand_dims(dy, axis=0), 0.0)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                    tf.summary.histogram(scope_prefix + name + '/dxy', dist_xy)

            return dist_xy
        else:
            raise AttributeError('Mode {} not supported'.format(mode))


def get_squared_dist_ref(x, y):
    """ This function calculates the pairwise distance between x and x, x and y, y and y.
    It is more accurate than get_dist at the cost of higher memory and complexity.

    :param x:
    :param y:
    :return:
    """
    with tf.name_scope('squared_dist_ref'):
        if len(x.get_shape().as_list()) > 2:
            raise AttributeError('get_dist: Input must be a matrix.')

        x_expand = tf.expand_dims(x, axis=2)  # m-by-d-by-1
        x_permute = tf.transpose(x_expand, perm=(2, 1, 0))  # 1-by-d-by-m
        dxx = x_expand - x_permute  # m-by-d-by-m, the first page is ai - a1
        dist_xx = tf.reduce_sum(tf.multiply(dxx, dxx), axis=1)  # m-by-m, the first column is (ai-a1)^2

        if y is None:
            return dist_xx
        else:
            y_expand = tf.expand_dims(y, axis=2)  # m-by-d-by-1
            y_permute = tf.transpose(y_expand, perm=(2, 1, 0))
            dxy = x_expand - y_permute  # m-by-d-by-m, the first page is ai - b1
            dist_xy = tf.reduce_sum(tf.multiply(dxy, dxy), axis=1)  # m-by-m, the first column is (ai-b1)^2
            dyy = y_expand - y_permute  # m-by-d-by-m, the first page is ai - b1
            dist_yy = tf.reduce_sum(tf.multiply(dyy, dyy), axis=1)  # m-by-m, the first column is (ai-b1)^2

            return dist_xx, dist_xy, dist_yy


########################################################################
def squared_dist_triplet(x, y, z, name='squared_dist', do_summary=False, scope_prefix=''):
    """ This function calculates the pairwise distance between x and x, x and y, y and y, y and z, z and z in 'seq'
    mode, or any two pairs in 'all' mode

    :param x:
    :param y:
    :param z:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    with tf.name_scope(name):
        x_x = tf.matmul(x, x, transpose_b=True)
        y_y = tf.matmul(y, y, transpose_b=True)
        z_z = tf.matmul(z, z, transpose_b=True)
        x_y = tf.matmul(x, y, transpose_b=True)
        y_z = tf.matmul(y, z, transpose_b=True)
        x_z = tf.matmul(x, z, transpose_b=True)
        d_x = tf.diag_part(x_x)
        d_y = tf.diag_part(y_y)
        d_z = tf.diag_part(z_z)

        d_x_x = tf.maximum(tf.expand_dims(d_x, axis=1) - 2.0 * x_x + tf.expand_dims(d_x, axis=0), 0.0)
        d_y_y = tf.maximum(tf.expand_dims(d_y, axis=1) - 2.0 * y_y + tf.expand_dims(d_y, axis=0), 0.0)
        d_z_z = tf.maximum(tf.expand_dims(d_z, axis=1) - 2.0 * z_z + tf.expand_dims(d_z, axis=0), 0.0)
        d_x_y = tf.maximum(tf.expand_dims(d_x, axis=1) - 2.0 * x_y + tf.expand_dims(d_y, axis=0), 0.0)
        d_y_z = tf.maximum(tf.expand_dims(d_y, axis=1) - 2.0 * y_z + tf.expand_dims(d_z, axis=0), 0.0)
        d_x_z = tf.maximum(tf.expand_dims(d_x, axis=1) - 2.0 * x_z + tf.expand_dims(d_z, axis=0), 0.0)

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.histogram(scope_prefix + name + '/dxx', d_x_x)
                tf.summary.histogram(scope_prefix + name + '/dyy', d_y_y)
                tf.summary.histogram(scope_prefix + name + '/dzz', d_z_z)
                tf.summary.histogram(scope_prefix + name + '/dxy', d_x_y)
                tf.summary.histogram(scope_prefix + name + '/dyz', d_y_z)
                tf.summary.histogram(scope_prefix + name + '/dxz', d_x_z)

        return d_x_x, d_y_y, d_z_z, d_x_y, d_x_z, d_y_z


########################################################################
def get_dist_np(x, y):
    """ This function calculates the pairwise distance between x and y using numpy

    :param x: m-by-d array
    :param y: n-by-d array
    :return:
    """
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    x_expand = np.expand_dims(x, axis=2)  # m-by-d-by-1
    y_expand = np.expand_dims(y, axis=2)  # n-by-d-by-1
    y_permute = np.transpose(y_expand, axes=(2, 1, 0))  # 1-by-d-by-n
    dxy = x_expand - y_permute  # m-by-d-by-n, the first page is ai - b1
    dist_xy = np.sqrt(np.sum(np.multiply(dxy, dxy), axis=1, dtype=np.float32))  # m-by-n, the first column is (ai-b1)^2

    return dist_xy


#########################################################################
def get_batch_squared_dist(x_batch, y_batch=None, axis=1, mode='xx', name='squared_dist'):
    """ This function calculates squared pairwise distance for vectors under xi or between xi and yi
    where i refers to the samples in the batch

    :param x_batch: batch_size-a-b tensor
    :param y_batch: batch_size-c-d tensor
    :param axis: the axis to be considered as features; if axis==1, a=c; if axis=2, b=d
    :param mode: 'xxxyyy', 'xx', 'xy', 'xxxy'
    :param name:
    :return: dist tensor(s)
    """
    # check inputs
    assert axis in [1, 2], 'axis has to be 1 or 2.'
    batch, a, b = x_batch.get_shape().as_list()
    if y_batch is not None:
        batch_y, c, d = y_batch.get_shape().as_list()
        assert batch == batch_y, 'Batch sizes do not match.'
        if axis == 1:
            assert a == c, 'Feature sizes do not match.'
        elif axis == 2:
            assert b == d, 'Feature sizes do not match.'
        if mode == 'xx':
            mode = 'xy'

    with tf.name_scope(name):
        if mode in {'xx', 'xxxyyy', 'xxxy'}:
            # xxt is batch-a-a if axis is 2 else batch-b-b
            xxt = tf.matmul(x_batch, tf.transpose(x_batch, [0, 2, 1])) \
                if axis == 2 else tf.matmul(tf.transpose(x_batch, [0, 2, 1]), x_batch)
            # dx is batch-a if axis is 2 else batch-b
            dx = tf.matrix_diag_part(xxt)
            dist_xx = tf.maximum(tf.expand_dims(dx, axis=2) - 2.0 * xxt + tf.expand_dims(dx, axis=1), 0.0)
            if mode == 'xx':
                return dist_xx
            elif mode == 'xxxy':
                # xyt is batch-a-c if axis is 2 else batch-b-d
                xyt = tf.matmul(x_batch, tf.transpose(y_batch, [0, 2, 1])) \
                    if axis == 2 else tf.matmul(tf.transpose(x_batch, [0, 2, 1]), y_batch)
                # dy is batch-c if axis is 2 else batch-d
                dy = tf.reduce_sum(tf.multiply(y_batch, y_batch), axis=axis)
                dist_xy = tf.maximum(tf.expand_dims(dx, axis=2) - 2.0 * xyt + tf.expand_dims(dy, axis=1), 0.0)

                return dist_xx, dist_xy
            elif mode == 'xxxyyy':
                # xyt is batch-a-c if axis is 2 else batch-b-d
                xyt = tf.matmul(x_batch, tf.transpose(y_batch, [0, 2, 1])) \
                    if axis == 2 else tf.matmul(tf.transpose(x_batch, [0, 2, 1]), y_batch)
                # yyt is batch-c-c if axis is 2 else batch-d-d
                yyt = tf.matmul(y_batch, tf.transpose(y_batch, [0, 2, 1])) \
                    if axis == 2 else tf.matmul(tf.transpose(y_batch, [0, 2, 1]), y_batch)
                # dy is batch-c if axis is 2 else batch-d
                dy = tf.reduce_sum(tf.multiply(y_batch, y_batch), axis=axis)
                dist_xy = tf.maximum(tf.expand_dims(dx, axis=2) - 2.0 * xyt + tf.expand_dims(dy, axis=1), 0.0)
                dist_yy = tf.maximum(tf.expand_dims(dy, axis=2) - 2.0 * yyt + tf.expand_dims(dy, axis=1), 0.0)

                return dist_xx, dist_xy, dist_yy

        elif mode == 'xy':
            # dx is batch-a if axis is 2 else batch-b
            dx = tf.reduce_sum(tf.multiply(x_batch, x_batch), axis=axis)
            # dy is batch-c if axis is 2 else batch-d
            dy = tf.reduce_sum(tf.multiply(y_batch, y_batch), axis=axis)
            # xyt is batch-a-c if axis is 2 else batch-b-d
            xyt = tf.matmul(x_batch, tf.transpose(y_batch, [0, 2, 1])) \
                if axis == 2 else tf.matmul(tf.transpose(x_batch, [0, 2, 1]), y_batch)
            dist_xy = tf.maximum(tf.expand_dims(dx, axis=2) - 2.0 * xyt + tf.expand_dims(dy, axis=1), 0.0)

            return dist_xy
        else:
            raise AttributeError('Mode {} not supported'.format(mode))


#######################################################################
def newton_root(x, f, df, step=None):
    """ This function does one iteration update on x to find the root f(x)=0. It is primarily used as the body of
    tf.while_loop.

    :param x:
    :param f: a function that receives x as input and outputs f(x) and other info for gradient calculation
    :param df: a function that receives info as inputs and outputs the gradient of f at x
    :param step:
    :return:
    """
    fx, info2grad = f(x)
    gx = df(info2grad)
    x = x - fx / (gx + FLAGS.EPSI)

    if step is None:
        return x
    else:
        return x, step + 1


#######################################################################
def matrix_mean_wo_diagonal(matrix, num_row, num_col=None, name='mu_wo_diag'):
    """ This function calculates the mean of the matrix elements not in the diagonal

    2018.4.9 - replace tf.diag_part with tf.matrix_diag_part
    tf.matrix_diag_part can be used for rectangle matrix while tf.diag_part can only be used for square matrix

    :param matrix:
    :param num_row:
    :type num_row: float
    :param num_col:
    :type num_col: float
    :param name:
    :return:
    """
    with tf.name_scope(name):
        if num_col is None:
            mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.matrix_diag_part(matrix))) / (num_row * (num_row - 1.0))
        else:
            mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.matrix_diag_part(matrix))) \
                 / (num_row * num_col - tf.minimum(num_col, num_row))

    return mu


########################################################################
def row_mean_wo_diagonal(matrix, num_col, name='mu_wo_diag'):
    """ This function calculates the mean of each row of the matrix elements excluding the diagonal
    
    :param matrix:
    :param num_col:
    :type num_col: float
    :param name:
    :return: 
    """
    with tf.name_scope(name):
        return (tf.reduce_sum(matrix, axis=1) - tf.matrix_diag_part(matrix)) / (num_col - 1.0)


#########################################################################
def mmd_t(
        dist_xx, dist_xy, dist_yy, batch_size, alpha=1.0, beta=2.0, var_target=None, name='mmd',
        do_summary=False, scope_prefix=''):
    """This function calculates the maximum mean discrepancy with t-distribution kernel

    The code is inspired by the Github page of following paper:
    Binkowski M., Sutherland D., Arbel M., Gretton A. (2018)
    Demystifying MMD GANs.

    :param dist_xx: batch_size-by-batch_size matrix
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param alpha:
    :param beta:
    :param var_target: if alpha is trainable, var_target contain the target for sigma
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """

    with tf.name_scope(name):
        log_k_xx = tf.log(dist_xx / (beta * alpha) + 1.0)  # use log for better condition
        log_k_xy = tf.log(dist_xy / (beta * alpha) + 1.0)
        log_k_yy = tf.log(dist_yy / (beta * alpha) + 1.0)

        k_xx = tf.exp(-alpha * log_k_xx)  # [1.0, k(xi, xj); k(xi, xj), 1.0]
        k_xy = tf.exp(-alpha * log_k_xy)
        k_yy = tf.exp(-alpha * log_k_yy)

        m = tf.constant(batch_size, tf.float32)
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)

        mmd = e_kxx + e_kyy - 2.0 * e_kxy

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)

        # return e_kxx, e_kxy, e_kyy
        if var_target is None:
            return mmd
        else:
            var = e_kxx + e_kyy + 2.0 * e_kxy
            loss_sigma = tf.square(var - var_target)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                    tf.summary.scalar(scope_prefix + name + '/loss_sigma', loss_sigma)

            return mmd, loss_sigma


#########################################################################
def mixture_mmd_t(
        dist_xx, dist_xy, dist_yy, batch_size, alpha=None, beta=2.0, var_targets=None, name='mmd',
        do_summary=False, scope_prefix=''):
    """ This function calculates the maximum mean discrepancy with a list of t-distribution kernels

    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param alpha: [0.2, 0.5, 1, 2, 25]
    :type alpha: list
    :param beta:
    :param var_targets: if alpha is trainable, var_targets contain the target for each alpha
    :type var_targets: list
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    num_alpha = len(alpha) if isinstance(alpha, list) else len(var_targets)
    with tf.name_scope(name):
        mmd = 0.0
        if var_targets is None:
            for i in range(num_alpha):
                mmd_i = mmd_t(
                    dist_xx, dist_xy, dist_yy, batch_size, alpha=alpha[i], beta=beta,
                    name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
                mmd = mmd + mmd_i

            return mmd
        else:
            loss_alpha = 0.0
            for i in range(num_alpha):
                mmd_i, loss_i = mmd_t(
                    dist_xx, dist_xy, dist_yy, batch_size, alpha=alpha[i], beta=beta, var_target=var_targets[i],
                    name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
                mmd = mmd + mmd_i
                loss_alpha = loss_alpha + loss_i

            return mmd, loss_alpha


#########################################################################
def witness_t(dist_zx, dist_zy, alpha=1.0, beta=2.0, name='witness', do_summary=False, scope_prefix=''):
    """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on t-distribution kernel

    :param dist_zx:
    :param dist_zy:
    :param alpha:
    :param beta:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    with tf.name_scope(name):
        # get dist between (x, z) and (y, z)
        # dist_zx = get_squared_dist(z, x, mode='xy', name='dist_zx', do_summary=do_summary)
        # dist_zy = get_squared_dist(z, y, mode='xy', name='dist_zy', do_summary=do_summary)

        log_k_zx = tf.log(dist_zx / (beta * alpha) + 1.0)
        log_k_zy = tf.log(dist_zy / (beta * alpha) + 1.0)

        k_zx = tf.exp(-alpha * log_k_zx)
        k_zy = tf.exp(-alpha * log_k_zy)

        e_kx = tf.reduce_mean(k_zx, axis=1)
        e_ky = tf.reduce_mean(k_zy, axis=1)

        witness = e_kx - e_ky

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.histogram(scope_prefix + name + '/kzx', e_kx)
                tf.summary.histogram(scope_prefix + name + '/kzy', e_ky)

        return witness


#########################################################################
def witness_mix_t(dist_zx, dist_zy, alpha=None, beta=2.0, name='witness', do_summary=False):
    """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on
    a list of t-distribution kernels.

    :param dist_zx:
    :param dist_zy:
    :param alpha:
    :param beta:
    :param name:
    :param do_summary:
    :return:
    """
    num_alpha = len(alpha)
    with tf.name_scope(name):
        witness = 0.0
        for i in range(num_alpha):
            wit_i = witness_t(
                dist_zx, dist_zy, alpha=alpha[i], beta=beta, name='d{}'.format(i), do_summary=do_summary)
            witness = witness + wit_i

        return witness


#########################################################################
def cramer(dist_xx, dist_xy, dist_yy, batch_size, name='mmd', epsi=1e-16, do_summary=False, scope_prefix=''):
    """ This function calculates the energy distance without the need of independent samples.

    The energy distance is taken originall from following paper:
    Bellemare1, M.G., Danihelka1, I., Dabney, W., Mohamed S., Lakshminarayanan B., Hoyer S., Munos R. (2017).
    The Cramer Distance as a Solution to Biased Wasserstein Gradients
    However, the original method requires two batches to calculate the kernel.

    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param name:
    :param epsi:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    with tf.name_scope(name):
        k_xx = -tf.sqrt(dist_xx + epsi)
        k_xy = -tf.sqrt(dist_xy + epsi)
        k_yy = -tf.sqrt(dist_yy + epsi)

        m = tf.constant(batch_size, tf.float32)
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)

        # return e_kxx, e_kxy, e_kyy
        return e_kxx + e_kyy - 2.0 * e_kxy


#########################################################################
def mmd_g(
        dist_xx, dist_xy, dist_yy, batch_size, sigma=1.0, var_target=None, upper_bound=None, lower_bound=None,
        name='mmd', do_summary=False, scope_prefix='', custom_weights=None):
    """This function calculates the maximum mean discrepancy with Gaussian distribution kernel

    The kernel is taken from following paper:
    Li, C.-L., Chang, W.-C., Cheng, Y., Yang, Y., & Póczos, B. (2017).
    MMD GAN: Towards Deeper Understanding of Moment Matching Network.

    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param sigma:
    :param var_target: if sigma is trainable, var_target contain the target for sigma
    :param upper_bound: bounds for pairwise distance in mmd-g.
    :param lower_bound:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :param custom_weights: weights for loss in mmd, default is [2.0, 1.0], custom[0] - custom[1] = 1.0
    :type custom_weights: list
    :return:
    """
    with tf.name_scope(name):
        if lower_bound is None:
            k_xx = tf.exp(-dist_xx / (2.0 * sigma**2), name='k_xx')
            k_yy = tf.exp(-dist_yy / (2.0 * sigma ** 2), name='k_yy')
        else:
            k_xx = tf.exp(-tf.maximum(dist_xx, lower_bound) / (2.0 * sigma ** 2), name='k_xx_lb')
            k_yy = tf.exp(-tf.maximum(dist_yy, lower_bound) / (2.0 * sigma ** 2), name='k_yy_lb')
        if upper_bound is None:
            k_xy = tf.exp(-dist_xy / (2.0 * sigma**2), name='k_xy')
        else:
            k_xy = tf.exp(-tf.minimum(dist_xy, upper_bound) / (2.0 * sigma ** 2), name='k_xy_ub')

        m = tf.constant(batch_size, tf.float32)
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)

        if var_target is None:
            if custom_weights is None:
                mmd = e_kxx + e_kyy - 2.0 * e_kxy
                return mmd
            else:  # note that here kyy is for the real data!
                assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
                mmd1 = e_kxx + e_kyy - 2.0 * e_kxy
                mmd2 = custom_weights[0] * e_kxy - e_kxx - custom_weights[1] * e_kyy
                return mmd1, mmd2
        else:
            mmd = e_kxx + e_kyy - 2.0 * e_kxy
            var = e_kxx + e_kyy + 2.0 * e_kxy
            loss_sigma = tf.square(var - var_target)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                    tf.summary.scalar(scope_prefix + name + '/loss_sigma', loss_sigma)

            return mmd, loss_sigma


#########################################################################
def mmd_g_bounded(
        dist_xx, dist_xy, dist_yy, batch_size, sigma=1.0, var_target=None, upper_bound=None, lower_bound=None,
        name='mmd', do_summary=False, scope_prefix='', custom_weights=None):
    """This function calculates the maximum mean discrepancy with Gaussian distribution kernel

    The kernel is taken from following paper:
    Li, C.-L., Chang, W.-C., Cheng, Y., Yang, Y., & Póczos, B. (2017).
    MMD GAN: Towards Deeper Understanding of Moment Matching Network.

    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param sigma:
    :param var_target: if sigma is trainable, var_target contain the target for sigma
    :param upper_bound:
    :param lower_bound:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :param custom_weights: weights for loss in mmd, default is [2.0, 1.0], custom[0] - custom[1] = 1.0
    :type custom_weights: list
    :return:
    """
    with tf.name_scope(name):
        k_xx = tf.exp(-dist_xx / (2.0 * sigma ** 2), name='k_xx')
        k_yy = tf.exp(-dist_yy / (2.0 * sigma ** 2), name='k_yy')
        k_xy = tf.exp(-dist_xy / (2.0 * sigma ** 2), name='k_xy')

        # in rep loss, custom_weights[0] - custom_weights[1] = 1
        k_xx_b = tf.exp(-tf.maximum(dist_xx, lower_bound) / (2.0 * sigma ** 2), name='k_xx_lb')
        if custom_weights[0] > 0:
            k_xy_b = tf.exp(-tf.minimum(dist_xy, upper_bound) / (2.0 * sigma ** 2), name='k_xy_ub')
        else:
            k_xy_b = k_xy  # no lower bound should be enforced as k_xy may be zero at equilibrium
        if custom_weights[1] > 0:  # the original mmd-g
            k_yy_b = tf.exp(-tf.maximum(dist_yy, lower_bound) / (2.0 * sigma ** 2), name='k_yy_ub')
        else:  # the repulsive mmd-g
            k_yy_b = tf.exp(-tf.minimum(dist_yy, upper_bound) / (2.0 * sigma ** 2), name='k_yy_ub')

        m = tf.constant(batch_size, tf.float32)
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)
        e_kxx_b = matrix_mean_wo_diagonal(k_xx_b, m)
        e_kyy_b = matrix_mean_wo_diagonal(k_yy_b, m)
        e_kxy_b = matrix_mean_wo_diagonal(k_xy_b, m) if custom_weights[0] < 0 else e_kxy

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)
                tf.summary.scalar(scope_prefix + name + '/kxx_b', e_kxx_b)
                tf.summary.scalar(scope_prefix + name + '/kyy_b', e_kyy_b)
                if custom_weights[0] > 0:
                    tf.summary.scalar(scope_prefix + name + '/kxy_b', e_kxy_b)

        if var_target is None:
            if custom_weights is None:
                mmd = e_kxx + e_kyy - 2.0 * e_kxy
                return mmd
            else:
                assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
                mmd1 = e_kxx + e_kyy - 2.0 * e_kxy
                mmd2 = custom_weights[0] * e_kxy_b - e_kxx_b - custom_weights[1] * e_kyy_b
                return mmd1, mmd2
        else:
            mmd = e_kxx + e_kyy - 2.0 * e_kxy
            var = e_kxx + e_kyy + 2.0 * e_kxy
            loss_sigma = tf.square(var - var_target)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.summary.scalar(scope_prefix + name + '/loss_sigma', loss_sigma)

            return mmd, loss_sigma


#########################################################################
def mixture_mmd_g(
        dist_xx, dist_xy, dist_yy, batch_size, sigma=None, var_targets=None, name='mmd_g',
        do_summary=False, scope_prefix=''):
    """ This function calculates the maximum mean discrepancy with a list of Gaussian distribution kernel

    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param sigma:
    :type sigma: list
    :param var_targets: if sigma is trainable, var_targets contain the target for each sigma
    :type var_targets: list
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    num_sigma = len(sigma) if isinstance(sigma, list) else len(var_targets)
    with tf.name_scope(name):
        mmd = 0.0
        if var_targets is None:
            for i in range(num_sigma):
                mmd_i = mmd_g(
                    dist_xx, dist_xy, dist_yy, batch_size, sigma=sigma[i],
                    name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
                mmd = mmd + mmd_i

            return mmd
        else:
            loss_sigma = 0.0
            for i in range(num_sigma):
                mmd_i, loss_i = mmd_g(
                    dist_xx, dist_xy, dist_yy, batch_size, sigma=sigma[i], var_target=var_targets[i],
                    name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
                mmd = mmd + mmd_i
                loss_sigma = loss_sigma + loss_i

            return mmd, loss_sigma


#########################################################################
def witness_g(dist_zx, dist_zy, sigma=2.0, name='witness', do_summary=False, scope_prefix=''):
    """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on Gaussian kernel

    :param dist_zx:
    :param dist_zy:
    :param sigma:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    with tf.name_scope(name):
        # get dist between (x, z) and (y, z)
        # dist_zx = get_squared_dist(z, x, mode='xy', name='dist_zx', do_summary=do_summary)
        # dist_zy = get_squared_dist(z, y, mode='xy', name='dist_zy', do_summary=do_summary)

        k_zx = tf.exp(-dist_zx / (2.0 * sigma), name='k_zx')
        k_zy = tf.exp(-dist_zy / (2.0 * sigma), name='k_zy')

        e_kx = tf.reduce_mean(k_zx, axis=1)
        e_ky = tf.reduce_mean(k_zy, axis=1)

        witness = e_kx - e_ky

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.histogram(scope_prefix + name + '/kzx', e_kx)
                tf.summary.histogram(scope_prefix + name + '/kzy', e_ky)

        return witness


#########################################################################
def witness_mix_g(dist_zx, dist_zy, sigma=None, name='witness', do_summary=False):
    """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on
    a list of t-distribution kernels.

    :param dist_zx:
    :param dist_zy:
    :param sigma:
    :param name:
    :param do_summary:
    :return:
    """
    num_sigma = len(sigma)
    with tf.name_scope(name):
        witness = 0.0
        for i in range(num_sigma):
            wit_i = witness_g(
                dist_zx, dist_zy, sigma=sigma[i], name='d{}'.format(i), do_summary=do_summary)
            witness = witness + wit_i

        return witness


def mmd_g_xn(
        batch_size, d, sigma, x, dist_xx=None, y_mu=0.0, y_var=1.0, name='mmd',
        do_summary=False, scope_prefix=''):
    """ This function calculates the mmd between two samples x and y. y is sampled from normal distribution
    with zero mean and specified variance.

    :param x:
    :param y_var:
    :param batch_size:
    :param d:
    :param sigma:
    :param y_mu:
    :param dist_xx:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    with tf.name_scope(name):
        # get dist_xx
        if dist_xx is None:
            xxt = tf.matmul(x, x, transpose_b=True)
            dx = tf.diag_part(xxt)
            dist_xx = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xxt + tf.expand_dims(dx, axis=0), 0.0)
        # get dist(x, Ey)
        dist_xy = tf.reduce_sum(tf.multiply(x - y_mu, x - y_mu), axis=1)

        k_xx = tf.exp(-dist_xx / (2.0 * sigma), name='k_xx')
        k_xy = tf.multiply(
            tf.exp(-dist_xy / (2.0 * (sigma + y_var))),
            tf.pow(sigma / (sigma + y_var), d / 2.0), name='k_xy')

        m = tf.constant(batch_size, tf.float32)
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = tf.reduce_mean(k_xy)
        e_kyy = tf.pow(sigma / (sigma + 2.0 * y_var), d / 2.0)

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)

        return e_kxx + e_kyy - 2.0 * e_kxy


def mixture_g_xn(batch_size, d, sigma, x, dist_xx=None, y_mu=0.0, y_var=1.0, name='mmd', do_summary=False):
    """ This function calculates the mmd between two samples x and y. y is sampled from normal distribution
    with zero mean and specified variance. A mixture of sigma is used.

    :param batch_size:
    :param d:
    :param sigma:
    :param x:
    :param dist_xx:
    :param y_mu:
    :param y_var:
    :param name:
    :param do_summary:
    :return:
    """
    num_sigma = len(sigma)
    with tf.name_scope(name):
        mmd = 0.0
        for i in range(num_sigma):
            mmd_i = mmd_g_xn(
                batch_size, d, sigma[i], x=x, dist_xx=dist_xx, y_mu=y_mu, y_var=y_var,
                name='d{}'.format(i), do_summary=do_summary)
            mmd = mmd + mmd_i

        return mmd


#########################################################################
def rand_mmd_g(dist_all, batch_size, omega=0.5, max_iter=0, name='mmd', do_summary=False, scope_prefix=''):
    """ This function uses a global sigma to make e_k match the given omega which is sampled uniformly. The sigma is
    initialized with geometric mean of pairwise distances and updated with Newton's method.

    :param dist_all:
    :param batch_size:
    :param omega:
    :param max_iter:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    with tf.name_scope(name):
        m = tf.constant(batch_size, tf.float32)

        def kernel(b):
            return tf.exp(-dist_all * b)

        def f(b):
            k = kernel(b)
            e_k = matrix_mean_wo_diagonal(k, 2 * m)
            return e_k - omega, k

        def df(k):
            kd = -k * dist_all  # gradient of exp(-d*w)
            e_kd = matrix_mean_wo_diagonal(kd, 2 * m)
            return e_kd

        # initialize sigma as the geometric mean of all pairwise distances
        dist_mean = matrix_mean_wo_diagonal(dist_all, 2 * m)
        beta = -tf.log(omega) / (dist_mean + FLAGS.EPSI)  # beta = 1/2/sigma
        # if max_iter is larger than one, do newton's update
        if max_iter > 0:
            beta, _ = tf.while_loop(
                cond=lambda _1, i: i < max_iter,
                body=lambda b, i: newton_root(b, f, df, step=i),
                loop_vars=(beta, tf.constant(0, dtype=tf.int32)))

        k_all = kernel(beta)
        k_xx = k_all[0:batch_size, 0:batch_size]
        k_xy_0 = k_all[0:batch_size, batch_size:]
        k_xy_1 = k_all[batch_size:, 0:batch_size]
        k_yy = k_all[batch_size:, batch_size:]

        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy_0 = matrix_mean_wo_diagonal(k_xy_0, m)
        e_kxy_1 = matrix_mean_wo_diagonal(k_xy_1, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.summary.scalar(scope_prefix + name + '/kxy_0', e_kxy_0)
                tf.summary.scalar(scope_prefix + name + '/kxy_1', e_kxy_1)
                # tf.summary.scalar(scope_prefix + name + 'omega', omega)

        return e_kxx + e_kyy - e_kxy_0 - e_kxy_1


def rand_mmd_g_xy(
        dist_xx, dist_xy, dist_yy, batch_size=None, dist_yx=None, omega=0.5, max_iter=3, name='mmd',
        do_summary=False, scope_prefix=''):
    """ This function calculates the mmd between two samples x and y. It uses a global sigma to make e_k match the
    given omega which is sampled uniformly. The sigma is initialized with geometric mean of dist_xy and updated
    with Newton's method.

    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param dist_yx: optional, if dist_xy and dist_yx are not the same
    :param batch_size: do not provide batch_size when the diagonal part of k** also need to be considered.
    :param omega:
    :param max_iter:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    with tf.name_scope(name):

        def kernel(dist, b):
            return tf.exp(-dist * b)

        def f(b):
            k = kernel(dist_xy, b)
            e_k = tf.reduce_mean(k)
            return e_k - omega, k

        def df(k):
            kd = -k * dist_xy  # gradient of exp(-d*w)
            e_kd = tf.reduce_mean(kd)
            return e_kd

        def f_plus(b):
            k0 = kernel(dist_xy, b)
            e_k0 = tf.reduce_mean(k0)
            k1 = kernel(dist_yx, b)
            e_k1 = tf.reduce_mean(k1)
            return e_k0 + e_k1 - 2.0 * omega, (k0, k1)

        def df_plus(k):
            kd0 = -k[0] * dist_xy  # gradient of exp(-d*w)
            kd1 = -k[1] * dist_yx  # gradient of exp(-d*w)
            e_kd = tf.reduce_mean(kd0) + tf.reduce_mean(kd1)
            return e_kd

        if dist_yx is None:
            # initialize sigma as the geometric mean of dist_xy
            beta = -tf.log(omega) / tf.reduce_mean(dist_xy + FLAGS.EPSI)  # beta = 1/2/sigma
            # if max_iter is larger than one, do newton's update
            if max_iter > 0:
                beta, _ = tf.while_loop(
                    cond=lambda _1, i: i < max_iter,
                    body=lambda b, i: newton_root(b, f, df, step=i),
                    loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
        else:
            # initialize sigma as the geometric mean of dist_xy and dist_yx
            # beta = 1/2/sigma
            beta = -2.0 * tf.log(omega) / (tf.reduce_mean(dist_xy) + tf.reduce_mean(dist_yx) + FLAGS.EPSI)
            # if max_iter is larger than one, do newton's update
            if max_iter > 0:
                beta, _ = tf.while_loop(
                    cond=lambda _1, i: i < max_iter,
                    body=lambda b, i: newton_root(b, f_plus, df_plus, step=i),
                    loop_vars=(beta, tf.constant(0, dtype=tf.int32)))

        k_xx = kernel(dist_xx, beta)
        k_xy = kernel(dist_xy, beta)
        k_yy = kernel(dist_yy, beta)

        if batch_size is None:  # include diagonal elements in k**
            e_kxx = tf.reduce_mean(k_xx)
            e_kxy = tf.reduce_mean(k_xy)
            e_kyy = tf.reduce_mean(k_yy)
        else:  # exclude diagonal elements in k**
            m = tf.constant(batch_size, tf.float32)
            e_kxx = matrix_mean_wo_diagonal(k_xx, m)
            e_kxy = matrix_mean_wo_diagonal(k_xy, m)
            e_kyy = matrix_mean_wo_diagonal(k_yy, m)

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)
                # tf.summary.scalar(scope_prefix + name + 'omega', omega)
                # tf.summary.histogram(scope_prefix + name + 'dxx', dist_xx)
                # tf.summary.histogram(scope_prefix + name + 'dxy', dist_xy)
                # tf.summary.histogram(scope_prefix + name + 'dyy', dist_yy)

        if dist_yx is None:
            return e_kxx + e_kyy - 2.0 * e_kxy
        else:
            k_yx = kernel(dist_yx, beta)
            if batch_size is None:
                e_kyx = tf.reduce_mean(k_yx)
            else:
                m = tf.constant(batch_size, tf.float32)
                e_kyx = matrix_mean_wo_diagonal(k_yx, m)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                    tf.summary.scalar(scope_prefix + name + 'kyx', e_kyx)
            return e_kxx + e_kyy - e_kxy - e_kyx


def rand_mmd_g_xy_bounded(
        dist_xx, dist_xy, dist_yy, batch_size=None, dist_yx=None, omega=0.5, max_iter=3, name='mmd',
        beta_lb=0.125, beta_ub=2.0, do_summary=False, scope_prefix=''):
    """ This function calculates the mmd between two samples x and y. It uses a global sigma to make e_k match the
    given omega which is sampled uniformly. The sigma is initialized with geometric mean of dist_xy and updated
    with Newton's method.

    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param dist_yx: optional, if dist_xy and dist_yx are not the same
    :param batch_size: do not provide batch_size when the diagonal part of k** also need to be considered.
    :param omega:
    :param max_iter:
    :param name:
    :param beta_lb: lower bound for beta (upper bound for sigma)
    :param beta_ub: upper bound for beta (lower bound for sigma)
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    with tf.name_scope(name):

        def kernel(dist, b):
            return tf.exp(-dist * b)

        def f(b):
            k = kernel(dist_xy, b)
            e_k = tf.reduce_mean(k)
            return e_k - omega, k

        def df(k):
            kd = -k * dist_xy  # gradient of exp(-d*w)
            e_kd = tf.reduce_mean(kd)
            return e_kd

        def f_plus(b):
            k0 = kernel(dist_xy, b)
            e_k0 = tf.reduce_mean(k0)
            k1 = kernel(dist_yx, b)
            e_k1 = tf.reduce_mean(k1)
            return e_k0 + e_k1 - 2.0 * omega, (k0, k1)

        def df_plus(k):
            kd0 = -k[0] * dist_xy  # gradient of exp(-d*w)
            kd1 = -k[1] * dist_yx  # gradient of exp(-d*w)
            e_kd = tf.reduce_mean(kd0) + tf.reduce_mean(kd1)
            return e_kd

        if dist_yx is None:
            # initialize sigma as the geometric mean of dist_xy
            beta = -tf.log(omega) / tf.reduce_mean(dist_xy + FLAGS.EPSI)  # beta = 1/2/sigma
            # if max_iter is larger than one, do newton's update
            if max_iter > 0:
                beta, _ = tf.while_loop(
                    cond=lambda _1, i: i < max_iter,
                    body=lambda b, i: newton_root(b, f, df, step=i),
                    loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
        else:
            # initialize sigma as the geometric mean of dist_xy and dist_yx
            # beta = 1/2/sigma
            beta = -2.0 * tf.log(omega) / (tf.reduce_mean(dist_xy) + tf.reduce_mean(dist_yx) + FLAGS.EPSI)
            # if max_iter is larger than one, do newton's update
            if max_iter > 0:
                beta, _ = tf.while_loop(
                    cond=lambda _1, i: i < max_iter,
                    body=lambda b, i: newton_root(b, f_plus, df_plus, step=i),
                    loop_vars=(beta, tf.constant(0, dtype=tf.int32)))

        beta = tf.clip_by_value(beta, beta_lb, beta_ub)
        k_xx = kernel(dist_xx, beta)
        k_xy = kernel(dist_xy, beta)
        k_yy = kernel(dist_yy, beta)
        k_xx_b = kernel(tf.maximum(dist_xx, 0.125/beta), beta)
        k_xy_b = kernel(tf.minimum(dist_xy, 2.0/beta), beta)
        k_yy_b = kernel(tf.maximum(dist_yy, 0.125/beta), beta)

        if batch_size is None:  # include diagonal elements in k**
            e_kxx = tf.reduce_mean(k_xx)
            e_kxy = tf.reduce_mean(k_xy)
            e_kyy = tf.reduce_mean(k_yy)
            e_kxx_b = tf.reduce_mean(k_xx_b)
            e_kxy_b = tf.reduce_mean(k_xy_b)
            e_kyy_b = tf.reduce_mean(k_yy_b)
        else:  # exclude diagonal elements in k**
            m = tf.constant(batch_size, tf.float32)
            e_kxx = matrix_mean_wo_diagonal(k_xx, m)
            e_kxy = matrix_mean_wo_diagonal(k_xy, m)
            e_kyy = matrix_mean_wo_diagonal(k_yy, m)
            e_kxx_b = matrix_mean_wo_diagonal(k_xx_b, m)
            e_kxy_b = matrix_mean_wo_diagonal(k_xy_b, m)
            e_kyy_b = matrix_mean_wo_diagonal(k_yy_b, m)

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)
                tf.summary.scalar(scope_prefix + name + '/beta', beta)
                tf.summary.scalar(scope_prefix + name + '/kxx_b', e_kxx_b)
                tf.summary.scalar(scope_prefix + name + '/kyy_b', e_kyy_b)
                tf.summary.scalar(scope_prefix + name + '/kxy_b', e_kxy_b)
                # tf.summary.scalar(scope_prefix + name + '/kxy_b', e_kxy_b)
                # tf.summary.scalar(scope_prefix + name + 'omega', omega)
                # tf.summary.histogram(scope_prefix + name + 'dxx', dist_xx)
                # tf.summary.histogram(scope_prefix + name + 'dxy', dist_xy)
                # tf.summary.histogram(scope_prefix + name + 'dyy', dist_yy)

        if dist_yx is None:
            return e_kxx + e_kyy - 2.0 * e_kxy, e_kxx_b - 2.0 * e_kyy_b + e_kxy_b
        else:
            k_yx = kernel(dist_yx, beta)
            # k_yx_b = kernel(tf.minimum(dist_yx, upper_bound), beta)
            if batch_size is None:
                e_kyx = tf.reduce_mean(k_yx)
                # e_kyx_b = tf.reduce_mean(k_yx_b)
            else:
                m = tf.constant(batch_size, tf.float32)
                e_kyx = matrix_mean_wo_diagonal(k_yx, m)
                # e_kyx_b = matrix_mean_wo_diagonal(k_yx_b, m)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.summary.scalar(scope_prefix + name + 'kyx', e_kyx)
                    # tf.summary.scalar(scope_prefix + name + 'kyx_b', e_kyx_b)
            return e_kxx + e_kyy - e_kxy - e_kyx


def rand_mmd_g_xn(
        x, y_rho, batch_size, d, y_mu=0.0, dist_xx=None, omega=0.5, max_iter=0, name='mmd',
        do_summary=False, scope_prefix=''):
    """ This function calculates the mmd between two samples x and y. y is sampled from normal distribution
    with zero mean and specified STD. This function uses a global sigma to make e_k match the given omega
    which is sampled uniformly. The sigma is initialized with geometric mean of dist_xy and updated with
    Newton's method.

    :param x:
    :param y_rho: y_std = sqrt(y_rho / 2.0 / d)
    :param batch_size:
    :param d: number of features in x
    :param y_mu:
    :param dist_xx:
    :param omega:
    :param max_iter:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    with tf.name_scope(name):
        # get dist_xx
        if dist_xx is None:
            xxt = tf.matmul(x, x, transpose_b=True)
            dx = tf.diag_part(xxt)
            dist_xx = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xxt + tf.expand_dims(dx, axis=0), 0.0)
        # get dist(x, Ey)
        dist_xy = tf.reduce_sum(tf.multiply(x - y_mu, x - y_mu), axis=1)

        def kernel(dist, b):
            return tf.exp(-dist * b)

        def f(b):
            const_f = d / (d + b * y_rho)
            k = tf.pow(const_f, d / 2.0) * tf.exp(-b * const_f * dist_xy)
            e_k = tf.reduce_mean(k)
            return e_k - omega, (const_f, k, e_k)

        def df(k):
            kd = -y_rho * k[0] / 2.0 * k[2] - tf.reduce_mean(tf.pow(k[0], 2) * dist_xy * k[1])  # gradient of exp(-d*w)
            e_kd = tf.reduce_mean(kd)
            return e_kd

        # initialize sigma as the geometric mean of dist_xy
        beta = -tf.log(omega) / (tf.reduce_mean(dist_xy) + y_rho / 2.0)  # beta = 1/2/sigma
        # if max_iter is larger than one, do newton's update
        if max_iter > 0:
            beta, _ = tf.while_loop(
                cond=lambda _1, i: i < max_iter,
                body=lambda b, i: newton_root(b, f, df, step=i),
                loop_vars=(beta, tf.constant(0, dtype=tf.int32)))

        const_0 = d / (d + beta * y_rho)
        k_xx = kernel(dist_xx, beta)
        k_xy = tf.pow(const_0, d / 2.0) * tf.exp(-beta * const_0 * dist_xy)

        e_kxx = matrix_mean_wo_diagonal(k_xx, tf.constant(batch_size, tf.float32))
        e_kxy = tf.reduce_mean(k_xy)
        e_kyy = tf.pow(d / (d + 2.0 * beta * y_rho), d / 2.0)

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)

        return e_kxx + e_kyy - 2.0 * e_kxy


def get_tensor_name(tensor):
    """ This function return tensor name without scope

    :param tensor:
    :return:
    """
    import re
    # split 'scope/name:0' into [scope, name, 0]
    return re.split('[/:]', tensor.name)[-2]


def moving_average_update(name, shape, tensor_update, rho=0.01, initializer=None, clip_values=None, dtype=tf.float32):
    """ This function creates a tensor that will be updated by tensor_update using moving average

    :param tensor_update: update at each iteration
    :param name: name for the tensor
    :param shape: shape of tensor
    :param rho:
    :param initializer:
    :param clip_values:
    :param dtype:
    :return:
    """
    if initializer is None:
        initializer = tf.zeros_initializer

    tensor = tf.get_variable(
        name, shape=shape, dtype=dtype, initializer=initializer, trainable=False)
    if clip_values is None:
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            tf.assign(tensor, tensor + rho * tensor_update))
    else:
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            tf.assign(
                tensor,
                tf.clip_by_value(
                    tensor + rho * tensor_update,
                    clip_value_min=clip_values[0], clip_value_max=clip_values[1])))

    return tensor


def moving_average_copy(tensor, name=None, rho=0.01, initializer=None, dtype=tf.float32):
    """ This function creates a moving average copy of tensor

    :param tensor:
    :param name: name for the moving average
    :param rho:
    :param initializer:
    :param dtype:
    :return:
    """
    if initializer is None:
        initializer = tf.zeros_initializer
    if name is None:
        name = get_tensor_name(tensor) + '_copy'

    tensor_copy = tf.get_variable(
        name, shape=tensor.get_shape().as_list(), dtype=dtype, initializer=initializer, trainable=False)
    tf.add_to_collection(
        tf.GraphKeys.UPDATE_OPS,
        tf.assign(tensor_copy, (1.0 - rho) * tensor_copy + rho * tensor))

    return tensor_copy


def slice_pairwise_distance(pair_dist, batch_size=None, indices=None):
    """ This function slice pair-dist into smaller pairwise distance matrices

    :param pair_dist: 2batch_size-by-2batch_size pairwise distance matrix
    :param batch_size:
    :param indices:
    :return:
    """
    with tf.name_scope('slice_dist'):
        if indices is None:
            dist_g1 = pair_dist[0:batch_size, 0:batch_size]
            dist_g2 = pair_dist[batch_size:, batch_size:]
            dist_g1g2 = pair_dist[0:batch_size, batch_size:]
        else:
            mix_group_1 = tf.concat((indices, tf.logical_not(indices)), axis=0)
            mix_group_2 = tf.concat((tf.logical_not(indices), indices), axis=0)
            dist_g1 = mat_slice(pair_dist, mix_group_1)
            dist_g2 = mat_slice(pair_dist, mix_group_2)
            dist_g1g2 = mat_slice(pair_dist, mix_group_1, mix_group_2)

    return dist_g1, dist_g1g2, dist_g2


def get_mix_coin(
        loss, loss_threshold, batch_size=None, loss_average_update=0.01, mix_prob_update=0.01,
        loss_average_name='loss_ave'):
    """ This function generate a mix_indices to mix data from two classes

    :param loss:
    :param loss_threshold:
    :param batch_size:
    :param loss_average_update:
    :param mix_prob_update:
    :param loss_average_name:
    :return:
    """
    with tf.variable_scope('coin', reuse=tf.AUTO_REUSE):
        # calculate moving average of loss
        loss_average = moving_average_copy(loss, loss_average_name, rho=loss_average_update)
        # update mixing probability
        mix_prob = moving_average_update(
            'prob', [], loss_average - loss_threshold, rho=mix_prob_update, clip_values=[0.0, 0.5])
        # sample mix_indices
        uni = tf.random_uniform([batch_size], 0.0, 1.0, dtype=tf.float32, name='uni')
        mix_indices = tf.greater(uni, mix_prob, name='mix_indices')  # mix_indices for using original data

    # loss_average and mix_prob is returned so that summary can be added outside of coin variable scope
    return mix_indices, loss_average, mix_prob


class GANLoss(object):
    def __init__(self, do_summary=False):
        """ This class defines all kinds of loss functions for generative adversarial nets

        Current losses include:

        """
        # IO
        self.do_summary = do_summary
        self.score_gen = None
        self.score_data = None
        self.batch_size = None
        self.num_scores = None
        # loss
        self.loss_gen = None
        self.loss_dis = None
        self.dis_penalty = None
        self.dis_scale = None
        self.debug_register = None  # output used for debugging
        # hyperparameters
        self.sigma = [1.0, np.sqrt(2.0), 2.0, np.sqrt(8.0), 4.0]
        # self.sigma = [1.0, 2.0, 4.0, 8.0, 16.0]  # mmd-g, kernel scales used in original paper
        self.alpha = [0.2, 0.5, 1, 2, 5.0]  # mmd-t, kernel scales used in original paper
        self.beta = 2.0  # mmd-t, kernel scales used in original paper
        self.omega_range = [0.05, 0.85]  # rand_g parameter
        self.ref_normal = 1.0  # rand_g parameter
        # weights[0] - weights[1] = 1.0
        self.repulsive_weights = [0.0, -1.0]  # weights for e_kxy and -e_kyy; note that kyy is for the real data!
        # self.repulsive_weights = [-1.0, -2.0]  # weights for e_kxy and -e_kyy

    def _add_summary_(self):
        """ This function adds summaries

        :return:
        """
        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar('GANLoss/gen', self.loss_gen)
                tf.summary.scalar('GANLoss/dis', self.loss_dis)

    def _logistic_(self):
        """ non-saturate logistic loss
        :return:
        """
        with tf.name_scope('logistic_loss'):
            self.loss_dis = tf.reduce_mean(tf.nn.softplus(self.score_gen) + tf.nn.softplus(-self.score_data))
            self.loss_gen = tf.reduce_mean(tf.nn.softplus(-self.score_gen))

    def _hinge_(self):
        """ hinge loss
        :return:
        """
        with tf.name_scope('hinge_loss'):
            self.loss_dis = tf.reduce_mean(
                tf.nn.relu(1.0 + self.score_gen)) + tf.reduce_mean(tf.nn.relu(1.0 - self.score_data))
            self.loss_gen = tf.reduce_mean(-self.score_gen)

    def _wasserstein_(self):
        """ wasserstein distance
        :return:
        """
        assert self.dis_penalty is not None, 'Discriminator penalty must be provided for wasserstein GAN'
        with tf.name_scope('wasserstein'):
            self.loss_gen = tf.reduce_mean(self.score_data) - tf.reduce_mean(self.score_gen)
            self.loss_dis = - self.loss_gen + self.dis_penalty

        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.summary.scalar('GANLoss/dis_penalty', self.dis_penalty)

        return self.loss_dis, self.loss_gen

    def _mmd_g_(self):
        """ maximum mean discrepancy with gaussian kernel
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)

        # mmd
        self.loss_gen = mixture_mmd_g(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=self.sigma,
            name='mmd_g', do_summary=self.do_summary)
        self.loss_dis = -self.loss_gen
        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty

    def _mmd_g_bound_(self):
        """ maximum mean discrepancy with gaussian kernel and bounds on dxy

        :return:
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)

        # mmd
        self.loss_gen = mmd_g(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='')
        mmd_b = mmd_g(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0, upper_bound=4, lower_bound=0.25,
            name='mmd_g_b', do_summary=self.do_summary, scope_prefix='')
        self.loss_dis = -mmd_b
        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty

    def _mmd_g_mix_(self, mix_threshold=1.0):
        """ maximum mean discrepancy with gaussian kernel and mixing score_gen and score_data
        if discriminator is too strong

        :param mix_threshold:
        :return:
        """
        # calculate pairwise distance
        pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
        dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)

        # mmd
        with tf.variable_scope('mmd_g_mix', reuse=tf.AUTO_REUSE):
            self.loss_gen = mixture_mmd_g(
                dist_gg, dist_gd, dist_dd, self.batch_size, sigma=self.sigma,
                name='mmd', do_summary=self.do_summary, scope_prefix='mmd_g_mix/')
            # mix data if self.loss_gen surpass loss_gen_threshold
            mix_indices, loss_average, mix_prob = get_mix_coin(
                self.loss_gen, mix_threshold, batch_size=self.batch_size, loss_average_name='gen_average')
            dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
            # mmd for mixed data
            loss_mix = mixture_mmd_g(
                dist_gg_mix, dist_gd_mix, dist_dd_mix, self.batch_size, sigma=self.sigma,
                name='mmd_mix', do_summary=self.do_summary, scope_prefix='mmd_g_mix/')
            self.loss_dis = -loss_mix

        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar('GANLoss/gen_average', loss_average)
                tf.summary.scalar('GANLoss/mix_prob', mix_prob)
                tf.summary.histogram('squared_dist/dxx', dist_gg)
                tf.summary.histogram('squared_dist/dyy', dist_dd)
                tf.summary.histogram('squared_dist/dxy', dist_gd)

    def _single_mmd_g_mix_(self, mix_threshold=0.2):
        """ maximum mean discrepancy with gaussian kernel and mixing score_gen and score_data
        if discriminator is too strong

        :param mix_threshold:
        :return:
        """
        # calculate pairwise distance
        pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
        dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)

        # mmd
        with tf.variable_scope('mmd_g_mix', reuse=tf.AUTO_REUSE):
            self.loss_gen = mmd_g(
                dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
                name='mmd', do_summary=self.do_summary, scope_prefix='mmd_g_mix/')
            # mix data if self.loss_gen surpass loss_gen_threshold
            mix_indices, loss_average, mix_prob = get_mix_coin(
                self.loss_gen, mix_threshold, batch_size=self.batch_size, loss_average_name='gen_average')
            dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
            # mmd for mixed data
            loss_mix = mmd_g(
                dist_gg_mix, dist_gd_mix, dist_dd_mix, self.batch_size, sigma=1.0,
                name='mmd_mix', do_summary=self.do_summary, scope_prefix='mmd_g_mix/')
            self.loss_dis = -loss_mix

        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.summary.scalar('GANLoss/gen_average', loss_average)
                tf.summary.scalar('GANLoss/mix_prob', mix_prob)
                tf.summary.histogram('squared_dist/dxx', dist_gg)
                tf.summary.histogram('squared_dist/dyy', dist_dd)
                tf.summary.histogram('squared_dist/dxy', dist_gd)

    def _mmd_t_(self):
        """ maximum mean discrepancy with t-distribution kernel
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
        # mmd
        self.loss_gen = mixture_mmd_t(
            dist_gg, dist_gd, dist_dd, self.batch_size, alpha=self.alpha, beta=self.beta,
            name='mmd_t', do_summary=self.do_summary)
        self.loss_dis = -self.loss_gen
        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty

    def _rand_g_(self):
        """ maximum mean discrepancy with gaussian kernel and random kernel scale
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)

        # mmd
        with tf.name_scope('rand_g'):
            omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
                if isinstance(self.omega_range, (list, tuple)) else self.omega_range
            loss_gr = rand_mmd_g_xy(
                dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
                max_iter=3, name='mmd_gr', do_summary=self.do_summary, scope_prefix='rand_g/')
            loss_gn = rand_mmd_g_xn(
                self.score_gen, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_gg, omega=omega,
                max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='rand_g/')
            loss_rn = rand_mmd_g_xn(
                self.score_data, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_dd, omega=omega,
                max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='rand_g/')
            # final loss
            self.loss_gen = loss_gr
            self.loss_dis = loss_rn - loss_gr

        # self.debug_register = [omega, loss_gr, loss_gn, loss_rn]
        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar('rand_g/omega', omega)
                tf.summary.scalar('GANLoss/gr', loss_gr)
                tf.summary.scalar('GANLoss/gn', loss_gn)
                tf.summary.scalar('GANLoss/rn', loss_rn)

    def _rand_g_bounded_(self):
        """ maximum mean discrepancy with gaussian kernel and random kernel scale, and upper bounds on dxy

        :return:
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)

        with tf.name_scope('rand_g'):
            omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
                if isinstance(self.omega_range, (list, tuple)) else self.omega_range
            loss_gr, loss_gr_b = rand_mmd_g_xy_bounded(
                dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
                max_iter=3, name='mmd', do_summary=self.do_summary, scope_prefix='rand_g/')
            # loss_gn = rand_mmd_g_xn(
            #     self.score_gen, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_gg, omega=omega,
            #     max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='rand_g/')
            # loss_rn = rand_mmd_g_xn(
            #     self.score_data, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_dd, omega=omega,
            #     max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='rand_g/')
            # final loss
            self.loss_gen = loss_gr
            self.loss_dis = - loss_gr_b

        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.summary.scalar('rand_g/omega', omega)
                tf.summary.scalar('GANLoss/gr', loss_gr)
                # tf.summary.scalar('GANLoss/gn', loss_gn)
                # tf.summary.scalar('GANLoss/rn', loss_rn)

    def _rand_g_mix_(self, mix_threshold=0.2):
        """ maximum mean discrepancy with gaussian kernel and random kernel scale
        and mixing score_gen and score_data if discriminator is too strong
        """
        # calculate pairwise distance
        pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
        dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)
        # mmd
        with tf.variable_scope('rand_g_mix', reuse=tf.AUTO_REUSE):
            omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
                if isinstance(self.omega_range, (list, tuple)) else self.omega_range
            loss_gr = rand_mmd_g_xy(
                dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
                max_iter=3, name='mmd_gr', do_summary=self.do_summary, scope_prefix='rand_g_mix/')
            loss_gn = rand_mmd_g_xn(
                self.score_gen, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_gg, omega=omega,
                max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='rand_g_mix/')
            loss_rn = rand_mmd_g_xn(
                self.score_data, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_dd, omega=omega,
                max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='rand_g_mix/')
            # mix data if self.loss_gen surpass loss_gen_threshold
            mix_indices, loss_average, mix_prob = get_mix_coin(
                loss_gr, mix_threshold, batch_size=self.batch_size, loss_average_name='gr_average')
            dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
            # mmd for mixed data
            loss_gr_mix = rand_mmd_g_xy(
                dist_gg_mix, dist_gd_mix, dist_dd_mix, self.batch_size, omega=omega,
                max_iter=3, name='mmd_gr_mix', do_summary=self.do_summary, scope_prefix='rand_g_mix/')
            # final loss
            self.loss_gen = loss_gr
            self.loss_dis = loss_rn - loss_gr_mix
            # self.debug_register = loss_rn

        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar('rand_g_mix/omega', omega)
                tf.summary.scalar('GANLoss/gr_average', loss_average)
                tf.summary.scalar('GANLoss/mix_prob', mix_prob)
                tf.summary.histogram('squared_dist/dxx', dist_gg)
                tf.summary.histogram('squared_dist/dyy', dist_dd)
                tf.summary.histogram('squared_dist/dxy', dist_gd)
                tf.summary.scalar('GANLoss/gr', loss_gr)
                tf.summary.scalar('GANLoss/gn', loss_gn)
                tf.summary.scalar('GANLoss/rn', loss_rn)
                tf.summary.scalar('GANLoss/gr_mix', loss_gr_mix)

    def _sym_rg_mix_(self, mix_threshold=0.2):
        """ symmetric version of rand_g_mix

        :param mix_threshold:
        :return:
        """
        # calculate pairwise distance
        pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
        dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)
        # mmd
        with tf.variable_scope('sym_rg_mix', reuse=tf.AUTO_REUSE):
            omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
                if isinstance(self.omega_range, (list, tuple)) else self.omega_range
            loss_gr = rand_mmd_g_xy(
                dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
                max_iter=3, name='mmd_gr', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
            loss_gn = rand_mmd_g_xn(
                self.score_gen, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_gg, omega=omega,
                max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
            loss_rn = rand_mmd_g_xn(
                self.score_data, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_dd, omega=omega,
                max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
            # mix data if self.loss_gen surpass loss_gen_threshold
            mix_indices, loss_average, mix_prob = get_mix_coin(
                loss_gr, mix_threshold, batch_size=self.batch_size, loss_average_name='gr_average')
            dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
            # mmd for mixed data
            loss_gr_mix = rand_mmd_g_xy(
                dist_gg_mix, dist_gd_mix, dist_dd_mix, self.batch_size, omega=omega,
                max_iter=3, name='mmd_gr_mix', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
            # final loss
            self.loss_gen = loss_gr + loss_gn
            self.loss_dis = loss_rn - loss_gr_mix - loss_gn

        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar('rand_g_mix/omega', omega)
                tf.summary.scalar('GANLoss/gr_average', loss_average)
                tf.summary.scalar('GANLoss/mix_prob', mix_prob)
                tf.summary.histogram('squared_dist/dxx', dist_gg)
                tf.summary.histogram('squared_dist/dyy', dist_dd)
                tf.summary.histogram('squared_dist/dxy', dist_gd)
                tf.summary.scalar('GANLoss/gr', loss_gr)
                tf.summary.scalar('GANLoss/gn', loss_gn)
                tf.summary.scalar('GANLoss/rn', loss_rn)
                tf.summary.scalar('GANLoss/gr_mix', loss_gr_mix)

    def _sym_rand_g_(self):
        """ Version 2 of symmetric rand_g. This function does not use label smoothing

        This function does not work.

        :return:
        """
        # calculate pairwise distance
        pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
        dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)
        # mmd
        with tf.variable_scope('sym_rg_mix', reuse=tf.AUTO_REUSE):
            omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
                if isinstance(self.omega_range, (list, tuple)) else self.omega_range
            loss_gr = rand_mmd_g_xy(
                dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
                max_iter=3, name='mmd_gr', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
            loss_gn = rand_mmd_g_xn(
                self.score_gen, self.ref_normal, self.batch_size, self.num_scores, y_mu=-0.5, dist_xx=dist_gg,
                omega=omega, max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
            loss_rn = rand_mmd_g_xn(
                self.score_data, self.ref_normal, self.batch_size, self.num_scores, y_mu=0.5, dist_xx=dist_dd,
                omega=omega, max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
            self.loss_gen = loss_gr
            self.loss_dis = 0.5*(loss_rn + loss_gn) - loss_gr

        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.summary.scalar('sym_rg_mix/omega', omega)
                tf.summary.histogram('squared_dist/dxx', dist_gg)
                tf.summary.histogram('squared_dist/dyy', dist_dd)
                tf.summary.histogram('squared_dist/dxy', dist_gd)
                tf.summary.scalar('GANLoss/gr', loss_gr)
                tf.summary.scalar('GANLoss/gn', loss_gn)
                tf.summary.scalar('GANLoss/rn', loss_rn)

    def _rand_g_instance_noise_(self, mix_threshold=0.2):
        """ This function tests instance noise

        :param mix_threshold:
        :return:
        """
        with tf.variable_scope('ins_noise'):
            sigma = tf.get_variable(
                'sigma', shape=[], dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
            stddev = tf.log(sigma + 1.0)  # to slow down sigma increase
            noise_gen = tf.random_normal(
                self.score_gen.get_shape().as_list(), mean=0.0, stddev=stddev,
                name='noise_gen', dtype=tf.float32)
            noise_x = tf.random_normal(
                self.score_data.get_shape().as_list(), mean=0.0, stddev=stddev,
                name='noise_x', dtype=tf.float32)
            self.score_gen = self.score_gen + noise_gen
            self.score_data = self.score_data + noise_x
            # use rand_g loss
            self._rand_g_()
            # update sigma
            loss_average = moving_average_copy(self.loss_gen, 'mmd_mean')
            tf.add_to_collection(
                tf.GraphKeys.UPDATE_OPS,
                tf.assign(
                    sigma,
                    tf.clip_by_value(
                        sigma + 0.001 * (loss_average - mix_threshold),
                        clip_value_min=0.0, clip_value_max=1.7183)))

        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap 
                tf.summary.scalar('GANLoss/gr_average', loss_average)
                tf.summary.scalar('GANLoss/sigma', sigma)

    def _repulsive_mmd_g_(self):
        """ repulsive loss

        :return:
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
        # self.loss_gen, self.loss_dis = mmd_g(
        #     dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.6,
        #     name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        self.loss_gen, self.loss_dis = mmd_g(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.summary.scalar('GANLoss/dis_penalty', self.dis_penalty)
        if self.dis_scale is not None:
            self.loss_dis = (self.loss_dis - 1.0) * self.dis_scale
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.summary.scalar('GANLoss/dis_scale', self.dis_scale)

    def _repulsive_mmd_g_bounded_(self):
        """ rmb loss

        :return:
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
        self.loss_gen, self.loss_dis = mmd_g_bounded(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0, lower_bound=0.25, upper_bound=4.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.summary.scalar('GANLoss/dis_penalty', self.dis_penalty)
        if self.dis_scale is not None:
            self.loss_dis = self.loss_dis * self.dis_scale
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.summary.scalar('GANLoss/dis_scale', self.dis_scale)

    def _test_(self):
        self.loss_dis = 0.0
        self.loss_gen = 0.0

    def __call__(self, score_gen, score_data, loss_type='logistic', **kwargs):
        """  This function calls one of the loss functions.

        :param score_gen:
        :param score_data:
        :param loss_type:
        :param kwargs:
        :return:
        """
        # IO and hyperparameters
        self.score_gen = score_gen
        self.score_data = score_data
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        if 'd' in kwargs:
            self.num_scores = kwargs['d']
        if 'dis_penalty' in kwargs:
            self.dis_penalty = kwargs['dis_penalty']
        if 'dis_scale' in kwargs:
            self.dis_scale = kwargs['dis_scale']
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        if 'beta' in kwargs:
            self.beta = kwargs['beta']
        if 'omega' in kwargs:
            self.omega_range = kwargs['omega']
        if 'ref_normal' in kwargs:
            self.ref_normal = kwargs['ref_normal']
        if 'rep_weights' in kwargs:
            self.repulsive_weights = kwargs['rep_weights']
        # check inputs
        if loss_type in {'fixed_g', 'mmd_g', 'fixed_t', 'mmd_t', 'mmd_g_mix', 'fixed_g_mix',
                         'rand_g', 'rand_g_mix', 'sym_rg_mix', 'instance_noise', 'ins_noise',
                         'sym_rg', 'rgb', 'rep', 'rep_gp', 'rmb', 'rmb_gp'}:
            assert self.batch_size is not None, 'GANLoss: batch_size must be provided'
            if loss_type in {'rand_g', 'rand_g_mix', 'sym_rg_mix', 'sym_rg'}:
                assert self.num_scores is not None, 'GANLoss: d must be provided'
        if loss_type in {'rep_gp', 'rmb_gp', 'wasserstein'}:
            assert self.dis_penalty is not None, 'Discriminator penalty must be provided.'
        if loss_type in {'rep_ds', 'rmb_ds'}:
            assert self.dis_scale is not None, 'Discriminator loss scale must be provided.'

        # loss
        if loss_type in {'logistic', ''}:
            self._logistic_()
        elif loss_type == 'hinge':
            self._hinge_()
        elif loss_type == 'wasserstein':
            self._wasserstein_()
        elif loss_type in {'fixed_g', 'mmd_g'}:
            self._mmd_g_()
        elif loss_type in {'mgb'}:
            self._mmd_g_bound_()
        elif loss_type in {'fixed_t', 'mmd_t'}:
            self._mmd_t_()
        elif loss_type in {'mmd_g_mix', 'fixed_g_mix'}:
            if 'mix_threshold' in kwargs:
                self._mmd_g_mix_(kwargs['mix_threshold'])
            else:
                self._mmd_g_mix_()
        elif loss_type in {'sgm'}:  # single mmd-g mix
            if 'mix_threshold' in kwargs:
                self._single_mmd_g_mix_(kwargs['mix_threshold'])
            else:
                self._single_mmd_g_mix_()
        elif loss_type == 'rand_g':
            self._rand_g_()
        elif loss_type == 'rgb':
            self._rand_g_bounded_()
        elif loss_type == 'rand_g_mix':
            if 'mix_threshold' in kwargs:
                self._rand_g_mix_(kwargs['mix_threshold'])
            else:
                self._rand_g_mix_()
        elif loss_type == 'sym_rg_mix':
            if 'mix_threshold' in kwargs:
                self._sym_rg_mix_(kwargs['mix_threshold'])
            else:
                self._sym_rg_mix_()
        elif loss_type in {'sym_rg', 'sym_rand_g'}:
            self._sym_rand_g_()
        elif loss_type in {'instance_noise', 'ins_noise'}:
            if 'mix_threshold' in kwargs:
                self._rand_g_instance_noise_(kwargs['mix_threshold'])
            else:
                self._rand_g_instance_noise_()
        elif loss_type in {'rep', 'rep_mmd_g', 'rep_gp', 'rep_ds'}:
            self._repulsive_mmd_g_()
        elif loss_type in {'rmb', 'rep_b', 'rep_mmd_b', 'rmb_gp', 'rmb_ds'}:
            self._repulsive_mmd_g_bounded_()
        elif loss_type == 'test':
            self._test_()
        else:
            raise NotImplementedError('Not implemented.')

        self._add_summary_()

        return self.loss_gen, self.loss_dis

    def apply(self, score_gen, score_data, loss_type='logistic', **kwargs):
        return self.__call__(score_gen, score_data, loss_type=loss_type, **kwargs)

    def get_register(self):
        """ This function returns the registered tensor

        :return:
        """
        # loss object always forgets self.debug_register after its value returned
        registered_info = self.debug_register
        self.debug_register = None
        return registered_info


def sqrt_sym_mat_np(mat, eps=None):
    """ This function calculates the square root of symmetric matrix

    :param mat:
    :param eps:
    :return:
    """
    if eps is None:
        eps = FLAGS.EPSI
    u, s, vh = np.linalg.svd(mat)
    si = np.where(s < eps, 0.0, np.sqrt(s))

    return np.matmul(np.matmul(u, np.diag(si)), vh)


def trace_sqrt_product_np(cov1, cov2):
    """ This function calculates trace(sqrt(cov1 * cov2))

    This code is inspired from:
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

    :param cov1:
    :param cov2:
    :return:
    """
    sqrt_cov1 = sqrt_sym_mat_np(cov1)
    cov_121 = np.matmul(np.matmul(sqrt_cov1, cov2), sqrt_cov1)

    return np.trace(sqrt_sym_mat_np(cov_121))


def sqrt_sym_mat_tf(mat, eps=None):
    """ This function calculates the square root of symmetric matrix

    :param mat:
    :param eps:
    :return:
    """
    if eps is None:
        eps = FLAGS.EPSI
    s, u, v = tf.svd(mat)
    si = tf.where(tf.less(s, eps), s, tf.sqrt(s))

    return tf.matmul(tf.matmul(u, tf.diag(si)), v, transpose_b=True)


def trace_sqrt_product_tf(cov1, cov2):
    """ This function calculates trace(sqrt(cov1 * cov2))

    This code is inspired from:
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

    :param cov1:
    :param cov2:
    :return:
    """
    sqrt_cov1 = sqrt_sym_mat_tf(cov1)
    cov_121 = tf.matmul(tf.matmul(sqrt_cov1, cov2), sqrt_cov1)

    return tf.trace(sqrt_sym_mat_tf(cov_121))


def jacobian(y, x, name='jacobian'):
    """ This function calculates the jacobian matrix: dy/dx and returns a list

    :param y: batch_size-by-d matrix
    :param x: batch_size-by-s tensor
    :param name:
    :return:
    """
    with tf.name_scope(name):
        batch_size, d = y.get_shape().as_list()
        if d == 1:
            return tf.reshape(tf.gradients(y, x)[0], [batch_size, -1])  # b-by-s
        else:
            return tf.transpose(
                tf.stack(
                    [tf.reshape(tf.gradients(y[:, i], x)[0], [batch_size, -1]) for i in range(d)], axis=0),  # d-b-s
                perm=(1, 0, 2))  # b-d-s tensor


def jacobian_squared_frobenius_norm(y, x, name='J_fnorm', do_summary=False):
    """ This function calculates the squared frobenious norm, e.g. sum of square of all elements in Jacobian matrix

    :param y: batch_size-by-d matrix
    :param x: batch_size-by-s tensor
    :param name:
    :param do_summary:
    :return:
    """
    with tf.name_scope(name):
        batch_size, d = y.get_shape().as_list()
        # sfn - squared frobenious norm
        if d == 1:
            jaco_sfn = tf.reduce_sum(tf.square(tf.reshape(tf.gradients(y, x)[0], [batch_size, -1])), axis=1)
        else:
            jaco_sfn = tf.reduce_sum(
                tf.stack(
                    [tf.reduce_sum(
                        tf.square(tf.reshape(tf.gradients(y[:, i], x)[0], [batch_size, -1])),  # b-vector
                        axis=1) for i in range(d)],
                    axis=0),  # d-by-b
                axis=0)  # b-vector

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.summary.histogram('Jaco_sfn', jaco_sfn)

        return jaco_sfn
