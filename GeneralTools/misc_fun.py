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
#     '/home/richard/PycharmProjects/myNN/Data/',
#     """Default input folder.""")
# flags.DEFINE_string(
#     'DEFAULT_OUT',
#     '/home/richard/PycharmProjects/myNN/Results/',
#     """Default output folder to write summary and checkpoint files.""")
# flags.DEFINE_string(
#     'DEFAULT_DOWNLOAD',
#     '/home/richard/Downloads/Data/',
#     """Default download folder for large datasets.""")
#
# # machine specs
# flags.DEFINE_integer(
#     'num_gpus',
#     1,
#     """Number of GPUs to use.""")
# # minimum number in calculation
# flags.DEFINE_float(
#     'EPSI',
#     1e-10,
#     'Smallest positive number allowed in modeling')
#
# # plotly setup
# # plotly credential file can be found at "C:\Users\Richard Wang\.plotly\.credentials"
# flags.DEFINE_string(
#     'PLT_ACC',
#     'Richard_wth',
#     """User name for plotly platform."""
# )
# flags.DEFINE_string(
#     'PLT_KEY',
#     'cqBAQrgDsHm1blKmVVn8',
#     """api_key for plotly platform."""
# )
#
# # data format
# flags.DEFINE_string(
#     'IMAGE_FORMAT',
#     'channels_first',
#     """Image data format, could be channels_first or channels_last""")
#
# # With 1.5.0-rc0 the Tensorflow maintainers have switched tf.app.flags to the flags module from abseil.
# # Unfortunately, it is not 100% API compatible to the previous implementation.
# # The following is added to avoid 'UnparsedFlagAccessError' and 'UnrecognizedFlagError'
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
        self.INCEPTION_V1 = 'MMD-GAN/Code/inception_v1/inceptionv1_for_inception_score.pb'
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
