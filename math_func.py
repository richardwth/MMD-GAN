# default modules
import numpy as np
import tensorflow as tf

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
        # print(z_batch.get_shape().as_list())
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
    # return w / (tf.sqrt(tf.reduce_sum(tf.square(w))) + FLAGS.EPSI)

    # tf.norm is slightly faster than tf.sqrt(tf.reduce_sum(tf.square()))
    # it is important that axis=None; in this case, norm(w) = norm(vec(w))
    return w / (tf.norm(w, ord='euclidean', axis=None) + FLAGS.EPSI)


########################################################################
def power_iter(w, u=None, v=None, step=None):
    """ This function does one power iteration. It is primarily used as the body of tf.while_loop.

    :param w: a p-by-q matrix
    :param u: a 1-by-p vector; either u or v must be provided.
    :param v: a 1-by-q vector
    :param step:
    :return:
    """
    if (u is None) and (v is not None):
        # get 1-by-p vector u
        u = l2normalization(tf.matmul(v, w, transpose_b=True))
        # get 1-by-q vector v
        _update = l2normalization(tf.matmul(u, w))
        # calculate singular value
        # sigma = tf.norm(tf.matmul(w, _update, transpose_b=True), ord='euclidean', axis=None)
        sigma = tf.matmul(tf.matmul(u, w), _update, transpose_b=True)[0, 0]
    elif (u is not None) and (v is None):
        # get 1-by-q vector v
        v = l2normalization(tf.matmul(u, w))
        # get 1-by-p vector u
        _update = l2normalization(tf.matmul(v, w, transpose_b=True))
        # calculate singular value
        sigma = tf.norm(tf.matmul(_update, w), ord='euclidean', axis=None)
    else:
        raise AttributeError('Either u or v must be provided.')

    if step is None:
        return sigma, _update
    else:
        return sigma, _update, step + 1


########################################################################
def power_iter_conv(w, u, conv_def, step=None):
    """ This function does one power iteration for conv2d without dilation. It is primarily used as the body of
    tf.while_loop.

    :param w: a h-w-in-out tensor
    :param u: a 1-H-W-in or 1-in-H-W tensor
    :param step:
    :param conv_def: a dictionary with keys ['strides', 'padding', 'data_format']
    :return:
    """
    if conv_def['data_format'] in ['NCHW', 'channels_first']:
        strides = (1, 1, conv_def['strides'], conv_def['strides'])
    else:
        strides = (1, conv_def['strides'], conv_def['strides'], 1)

    def _conv_(input_u):
        return tf.nn.conv2d(
            input_u, w, strides=strides, padding=conv_def['padding'],
            data_format=conv_def['data_format'], name='conv')

    def _conv_t_(input_v):
        return tf.nn.conv2d_transpose(
            input_v, w, output_shape=u.shape, strides=strides, padding=conv_def['padding'],
            data_format=conv_def['data_format'], name='conv_t')

    # do conv on v
    v = l2normalization(_conv_(u))
    # get 1-by-p vector u
    u_update = l2normalization(_conv_t_(v))
    # calculate singular value
    sigma = tf.norm(_conv_(u_update), ord='euclidean', axis=None)

    if step is None:
        return sigma, u_update
    else:
        return sigma, u_update, step + 1


########################################################################
def power_iter_transpose_conv(w, u, conv_def, step=None):
    """ This function does one power iteration for transpose conv2d without dilation.
    It is primarily used as the body of tf.while_loop.

    :param w: a h-w-out-in tensor
    :param u: a 1-H-W-in or 1-in-H-W tensor
    :param step:
    :param conv_def: a dictionary with keys ['strides', 'padding', 'data_format', 'output_shape']
    :return:
    """
    if conv_def['data_format'] in ['NCHW', 'channels_first']:
        strides = (1, 1, conv_def['strides'], conv_def['strides'])
    else:
        strides = (1, conv_def['strides'], conv_def['strides'], 1)

    def _conv_(input_v):
        return tf.nn.conv2d(
            input_v, w, strides=strides, padding=conv_def['padding'],
            data_format=conv_def['data_format'], name='conv')

    def _conv_t_(input_u):
        return tf.nn.conv2d_transpose(
            input_u, w, output_shape=conv_def['output_shape'], strides=strides, padding=conv_def['padding'],
            data_format=conv_def['data_format'], name='conv_t')

    # do conv on v
    v = l2normalization(_conv_t_(u))
    # get 1-by-p vector u
    u_update = l2normalization(_conv_(v))
    # calculate singular value
    sigma = tf.norm(_conv_t_(u_update), ord='euclidean', axis=None)

    if step is None:
        return sigma, u_update
    else:
        return sigma, u_update, step + 1


########################################################################
def power_iter_atrous_conv(w, u, conv_def, step=None):
    """ This function does one power iteration for conv2d with dilation. It is primarily used as the body of
    tf.while_loop.

    :param w: a h-w-in-out tensor
    :param u: must be a 1-H-W-in tensor. The tf.nn.atrous_conv2d does not support NCHW format
    :param step:
    :param conv_def: a dictionary with keys ['dilation', 'padding']
    :return:
    """

    def _conv_(input_u):
        return tf.nn.atrous_conv2d(
            input_u, w, rate=conv_def['dilation'], padding=conv_def['padding'], name='conv')

    def _conv_t_(input_v):
        return tf.nn.atrous_conv2d_transpose(
            input_v, w, output_shape=u.shape, rate=conv_def['dilation'], padding=conv_def['padding'], name='conv_t')

    # do conv on v
    v = l2normalization(_conv_(u))
    # get 1-by-p vector u
    u_update = l2normalization(_conv_t_(v))

    # calculate singular value
    sigma = tf.norm(_conv_(u), ord='euclidean', axis=None)

    if step is None:
        return sigma, u_update
    else:
        return sigma, u_update, step + 1


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
def get_squared_dist(x, y=None, scale=None, z_score=False, mode='xxxyyy', name='squared_dist', do_summary=False):
    """ This function calculates the pairwise distance between x and x, x and y, y and y

    Warning: when x, y has mean far away from zero, the distance calculation is not accurate; use get_dist_ref instead

    :param x: batch_size-by-d matrix
    :param y: batch_size-by-d matrix
    :param scale: 1-by-d vector, the precision vector. dxy = x*scale*y
    :param z_score:
    :param mode: 'xxxyyy', 'xx', 'xy', 'xxxy'
    :param name:
    :param do_summary:
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
                tf.summary.histogram('dxx', dist_xx)

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
                    tf.summary.histogram('dxy', dist_xy)

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
                    tf.summary.histogram('dxy', dist_xy)
                    tf.summary.histogram('dyy', dist_yy)

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
                tf.summary.histogram('dxy', dist_xy)

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
def squared_dist_triplet(x, y, z, name='squared_dist', do_summary=False):
    """ This function calculates the pairwise distance between x and x, x and y, y and y, y and z, z and z in 'seq'
    mode, or any two pairs in 'all' mode

    :param x:
    :param y:
    :param z:
    :param name:
    :param do_summary:
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
            tf.summary.histogram('dxx', d_x_x)
            tf.summary.histogram('dyy', d_y_y)
            tf.summary.histogram('dzz', d_z_z)
            tf.summary.histogram('dxy', d_x_y)
            tf.summary.histogram('dyz', d_y_z)
            tf.summary.histogram('dxz', d_x_z)

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
    x = x - fx / gx

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
def mmd_t(dist_xx, dist_xy, dist_yy, batch_size, alpha=1.0, beta=2.0, var_target=None, name='mmd', do_summary=False):
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
            tf.summary.scalar('kxx', e_kxx)
            tf.summary.scalar('kyy', e_kyy)
            tf.summary.scalar('kxy', e_kxy)

        # return e_kxx, e_kxy, e_kyy
        if var_target is None:
            return mmd
        else:
            var = e_kxx + e_kyy + 2.0 * e_kxy
            loss_sigma = tf.square(var - var_target)
            if do_summary:
                tf.summary.scalar('loss_sigma', loss_sigma)

            return mmd, loss_sigma


#########################################################################
def mixture_mmd_t(
        dist_xx, dist_xy, dist_yy, batch_size, alpha=None, beta=2.0, var_targets=None, name='mmd', do_summary=False):
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
    :return:
    """
    num_alpha = len(alpha) if isinstance(alpha, list) else len(var_targets)
    with tf.name_scope(name):
        mmd = 0.0
        if var_targets is None:
            for i in range(num_alpha):
                mmd_i = mmd_t(
                    dist_xx, dist_xy, dist_yy, batch_size, alpha=alpha[i], beta=beta,
                    name='d{}'.format(i), do_summary=do_summary)
                mmd = mmd + mmd_i

            return mmd
        else:
            loss_alpha = 0.0
            for i in range(num_alpha):
                mmd_i, loss_i = mmd_t(
                    dist_xx, dist_xy, dist_yy, batch_size, alpha=alpha[i], beta=beta, var_target=var_targets[i],
                    name='d{}'.format(i), do_summary=do_summary)
                mmd = mmd + mmd_i
                loss_alpha = loss_alpha + loss_i

            return mmd, loss_alpha


#########################################################################
def witness_t(dist_zx, dist_zy, alpha=1.0, beta=2.0, name='witness', do_summary=False):
    """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on t-distribution kernel

    :param dist_zx:
    :param dist_zy:
    :param alpha:
    :param beta:
    :param name:
    :param do_summary:
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
            tf.summary.histogram('kzx', e_kx)
            tf.summary.histogram('kzy', e_ky)

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
def cramer(dist_xx, dist_xy, dist_yy, batch_size, name='mmd', epsi=1e-16, do_summary=False):
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
            tf.summary.scalar('kxx', e_kxx)
            tf.summary.scalar('kyy', e_kyy)
            tf.summary.scalar('kxy', e_kxy)

        # return e_kxx, e_kxy, e_kyy
        return e_kxx + e_kyy - 2.0 * e_kxy


#########################################################################
def mmd_g(dist_xx, dist_xy, dist_yy, batch_size, sigma=1.0, var_target=None, name='mmd', do_summary=False):
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
    :param name:
    :param do_summary:
    :return:
    """
    with tf.name_scope(name):
        k_xx = tf.exp(-dist_xx / (2.0 * sigma), name='k_xx')
        k_xy = tf.exp(-dist_xy / (2.0 * sigma), name='k_xy')
        k_yy = tf.exp(-dist_yy / (2.0 * sigma), name='k_yy')

        m = tf.constant(batch_size, tf.float32)
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)

        mmd = e_kxx + e_kyy - 2.0 * e_kxy

        if do_summary:
            tf.summary.scalar('kxx', e_kxx)
            tf.summary.scalar('kyy', e_kyy)
            tf.summary.scalar('kxy', e_kxy)

        # calculate perplexity for each x
        # k_x_sum = tf.reduce_sum(k_xy, axis=1)
        # entropy = tf.log(k_x_sum) + tf.reduce_sum(k_xy * dist_xy, axis=1) / k_x_sum / (2.0 * sigma)
        #
        # tf.summary.histogram('stat/entropy', entropy)
        # return e_kxx, e_kxy, e_kyy
        if var_target is None:
            return mmd
        else:
            var = e_kxx + e_kyy + 2.0 * e_kxy
            loss_sigma = tf.square(var - var_target)
            if do_summary:
                tf.summary.scalar('loss_sigma', loss_sigma)

            return mmd, loss_sigma


#########################################################################
def mixture_mmd_g(dist_xx, dist_xy, dist_yy, batch_size, sigma=None, var_targets=None, name='mmd', do_summary=False):
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
    :return:
    """
    num_sigma = len(sigma) if isinstance(sigma, list) else len(var_targets)
    with tf.name_scope(name):
        mmd = 0.0
        if var_targets is None:
            for i in range(num_sigma):
                mmd_i = mmd_g(
                    dist_xx, dist_xy, dist_yy, batch_size, sigma=sigma[i],
                    name='d{}'.format(i), do_summary=do_summary)
                mmd = mmd + mmd_i

            return mmd
        else:
            loss_sigma = 0.0
            for i in range(num_sigma):
                mmd_i, loss_i = mmd_g(
                    dist_xx, dist_xy, dist_yy, batch_size, sigma=sigma[i], var_target=var_targets[i],
                    name='d{}'.format(i), do_summary=do_summary)
                mmd = mmd + mmd_i
                loss_sigma = loss_sigma + loss_i

            return mmd, loss_sigma


#########################################################################
def witness_g(dist_zx, dist_zy, sigma=2.0, name='witness', do_summary=False):
    """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on Gaussian kernel

    :param dist_zx:
    :param dist_zy:
    :param sigma:
    :param name:
    :param do_summary:
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
            tf.summary.histogram('kzx', e_kx)
            tf.summary.histogram('kzy', e_ky)

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


def mmd_g_xn(batch_size, d, sigma, x, dist_xx=None, y_mu=0.0, y_var=1.0, name='mmd', do_summary=False):
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
            tf.summary.scalar('kxx', e_kxx)
            tf.summary.scalar('kyy', e_kyy)
            tf.summary.scalar('kxy', e_kxy)

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
def rand_mmd_g(dist_all, batch_size, omega=0.5, max_iter=0, name='mmd', do_summary=False):
    """ This function uses a global sigma to make e_k match the given omega which is sampled uniformly. The sigma is
    initialized with geometric mean of pairwise distances and updated with Newton's method.

    :param dist_all:
    :param batch_size:
    :param omega:
    :param max_iter:
    :param name:
    :param do_summary:
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
        beta = -tf.log(omega) / dist_mean  # beta = 1/2/sigma
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
            tf.summary.scalar('kxx', e_kxx)
            tf.summary.scalar('kyy', e_kyy)
            tf.summary.scalar('kxy_0', e_kxy_0)
            tf.summary.scalar('kxy_1', e_kxy_1)
            tf.summary.scalar('omega', omega)

        return e_kxx + e_kyy - e_kxy_0 - e_kxy_1


def rand_mmd_g_xy(
        dist_xx, dist_xy, dist_yy, batch_size=None, dist_yx=None, omega=0.5, max_iter=0, name='mmd', do_summary=False):
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
            beta = -tf.log(omega) / tf.reduce_mean(dist_xy)  # beta = 1/2/sigma
            # if max_iter is larger than one, do newton's update
            if max_iter > 0:
                beta, _ = tf.while_loop(
                    cond=lambda _1, i: i < max_iter,
                    body=lambda b, i: newton_root(b, f, df, step=i),
                    loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
        else:
            # initialize sigma as the geometric mean of dist_xy and dist_yx
            beta = -2.0 * tf.log(omega) / (tf.reduce_mean(dist_xy) + tf.reduce_mean(dist_yx))  # beta = 1/2/sigma
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
            tf.summary.scalar('kxx', e_kxx)
            tf.summary.scalar('kyy', e_kyy)
            tf.summary.scalar('kxy', e_kxy)
            tf.summary.scalar('omega', omega)
            # tf.summary.histogram('dxx', dist_xx)
            # tf.summary.histogram('dxy', dist_xy)
            # tf.summary.histogram('dyy', dist_yy)

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
                tf.summary.scalar('kyx', e_kyx)
            return e_kxx + e_kyy - e_kxy - e_kyx


def rand_mmd_g_xn(
        x, y_rho, batch_size, d, y_mu=0.0, dist_xx=None, omega=0.5, max_iter=0, name='mmd', do_summary=False):
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
            tf.summary.scalar('kxx', e_kxx)
            tf.summary.scalar('kyy', e_kyy)
            tf.summary.scalar('kxy', e_kxy)
            tf.summary.scalar('omega', omega)
            # tf.summary.histogram('dxx', dist_xx)
            # tf.summary.histogram('dist_xy', dist_xy)
            # tf.summary.histogram('dyy', dist_yy)

        return e_kxx + e_kyy - 2.0 * e_kxy
