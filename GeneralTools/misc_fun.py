""" This file contains mischievous functions


"""
# import tensorflow as tf
# import sys
# flags = tf.app.flags
#
# FLAGS = flags.FLAGS
#
# # Default folders
# flags.DEFINE_string(
#     'DEFAULT_IN',
#     'MMD-GAN/Data/',
#     """Default input folder.""")
## ...
#
# With 1.5.0-rc0 the Tensorflow maintainers have switched tf.app.flags to the flags module from abseil.
# Unfortunately, it is not 100% API compatible to the previous implementation.
# The following is added to avoid 'UnparsedFlagAccessError' and 'UnrecognizedFlagError'
# remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
# assert(remaining_args == [sys.argv[0]])


class SetFlag(object):
    def __init__(self):
        # machine config
        self.num_gpus = 1
        self.EPSI = 1e-10
        self.SILENT_MODE = False

        # library info
        self.TENSORFLOW_VERSION = '1.8.0'
        self.CUDA_VERSION = '9.0'
        self.CUDNN_VERSION = '7.0'
        self.DRIVER_VERSION = '396.54'

        # directory setup
        self.DEFAULT_IN = 'MMD-GAN/Data/'
        self.DEFAULT_OUT = 'MMD-GAN/Results/'
        self.DEFAULT_DOWNLOAD = 'MMD-GAN/Data/'
        self.INCEPTION_V1 = 'MMD-GAN/Addon/inception_v1/inceptionv1_for_inception_score.pb'
        self.INCEPTION_V3 = None
        
        # plotly account
        self.PLT_ACC = None
        self.PLT_KEY = None

        # model setup
        self.IMAGE_FORMAT = 'channels_first'
        self.IMAGE_FORMAT_ALIAS = 'NCHW'
        self.WEIGHT_INITIALIZER = 'default'  # 'default', 'sn_paper', 'pg_paper'
        self.SPECTRAL_NORM_MODE = 'default'  # 'default' = 'PICO', 'sn_paper'

    def print(self, info, force_print=False):
        if (not self.SILENT_MODE) or force_print:
            print(info)


FLAGS = SetFlag()
