""" This module contains functions handling data IO.

Name list:
my_test()
my_readcsv() - read csv data to ndarray in memory
my_split() - split the ndarray
my_readcsv_tf() - read csv data to ndarray in memory using tensorflow
my_np2tfrecord() - write ndarray in memory to tfrecord
my_read_tfrecord() - create a tensor that can be used to read tfrecord file later
csv2tfrecord() - converts csv file to tfrecord file, a wrapper function 
                that calls both my_readcsv() and my_np2tfrecord()
image2csv() - read png files and convert to csv file
"""

# helper function
import os
import os.path
import sys
import time
# import warnings
# from PIL import Image  # PIL is not available on Spartan
# default modules
import numpy as np
import tensorflow as tf
from GeneralTools.misc_fun import FLAGS

DEFAULT_IN_FILE_DIR = FLAGS.DEFAULT_IN
DEFAULT_OUT_FILE_DIR = FLAGS.DEFAULT_OUT
DEFAULT_DOWNLOAD_FILE_DIR = FLAGS.DEFAULT_DOWNLOAD  # similar to DEFAULT_IN_FILE_DIR, but for large dataset


########################################################################
# define macro
# FloatList, Int64List and BytesList are three base feature types
def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])
        if isinstance(value, float) else tf.train.FloatList(value=value))


def _int64_feature(value):  # numpy int is not int!
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])
        if isinstance(value, int) else tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
        if isinstance(value, (str, bytes)) else tf.train.BytesList(value=value))
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


########################################################################
def my_np2tfrecord(filename, data, label=None):
    """ This function writes numpy ndarray raw_data to tfrecord file

    :param data: ndarray
    :param label: ndarray
    :param filename:
    :type filename: str
    :return:
    """
    # prepare
    filename = os.path.join(DEFAULT_IN_FILE_DIR, filename + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    num_examples = data.shape[0]

    # check data type
    if data.dtype == np.float32:
        feature_fun = _float_feature
    elif data.dtype == np.uint8:
        feature_fun = lambda x: _bytes_feature(x.tobytes())
    elif data.dtype == np.int32:
        data = data.astype(np.float32)
        feature_fun = _float_feature
    else:
        raise AttributeError('Supported data type: uint8, float32, int32; got {}'.format(data.type))

    if label is None:
        for i in range(num_examples):
            instance = tf.train.Example(features=tf.train.Features(feature={
                'x': feature_fun(data[i, :])
            }))
            writer.write(instance.SerializeToString())
            if (i + 1) % 5000 == 0:
                sys.stdout.write('\r %d instance finished.' % (i + 1))
                # sys.stdout.flush()
        writer.close()
    else:
        if label.shape[0] != num_examples:
            raise ValueError('Data size and label size do not match.')
        assert np.issubdtype(label.dtype, int), 'Supported data type: int; got {}'.format(data.type)
        for i in range(num_examples):
            instance = tf.train.Example(features=tf.train.Features(feature={
                'x': feature_fun(data[i, :]),
                'y': _int64_feature(int(label[i, :]))  # numpy int is not int
            }))
            writer.write(instance.SerializeToString())
            if (i + 1) % 5000 == 0:
                sys.stdout.write('\r %d instance finished.' % (i + 1))
                # sys.stdout.flush()
        writer.close()


#######################################################################
def binary_image_to_tfrecords(
        image_path, output_filename, num_images, image_size, num_labels=1, label_first=True,
        resize=None, crop=None, image_transpose=False, image_format_in_file='NCHW',
        target_image_format='NCHW', save_label=False):
        """ This function converts images stored in image_path.bin to target_path.tfrecords. It assumes RGB images.

        Note: label is not written to tfrecord

        For stl10 dataset, the following code is used:
            from GeneralTools.input_func import binary_image_to_tfrecords

            binary_image_to_tfrecords(
                'stl10/unlabeled_X', 'stl_NCHW/stl', 100000, [3, 96, 96],
                num_labels=0, resize=(48, 48), image_transpose=True,
                image_format_in_file='NCHW', target_image_format='NCHW')

        For cifar10 dataset, the following code is used:
            from GeneralTools.misc_fun import FLAGS
            FLAGS.DEFAULT_DOWNLOAD = '/media/richard/My Book/MyBackup/Data/'
            from GeneralTools.input_func import binary_image_to_tfrecords

            filename = ['cifar/cifar_{}'.format(i) for i in range(1, 6)]
            binary_image_to_tfrecords(
                filename, 'cifar_NCHW/cifar', 50000, [3, 32, 32], num_labels=1,
                image_format_in_file='NCHW', target_image_format='NCHW')


        :param image_path: e.g. stl10/unlabeled_X (in DEFAULT_DOWNLOAD_FILE_DIR)
        :param output_filename: e.g. stl10_NCHW/stl (in DEFAULT_IN_FILE_DIR)
        :param num_images: integer, e.g. 100000
        :param image_size: [num_channels, height, weight]
        :param num_labels: num of labels for each instance; if no label, set label_bytes = 0
        :param label_first: if data is stored as [label, feature] or [feature, label]
        :param resize: (new_height, new_weight)
        :param crop: e.g. (36-32, 44-32, 36+32, 44+32)
        :param image_transpose: in datasets like MNIST and stl10, the height and weight dimensions are reversed
        :param image_format_in_file:
        :param target_image_format:
        :param save_label: bool
        :return:
        """
        from PIL import Image

        def do_image_resize(im):
            """ This function resizes the input image

            :param im: an image in numpy array
            :return:
            """
            # convert ndarray to im object and do resize
            im = Image.fromarray(im, 'RGB')
            im = im.resize(resize, Image.LANCZOS)
            return np.array(im)

        def do_image_crop(np_image):
            """ This function crops the input image

            :param np_image:
            :return:
            """
            # convert ndarray to im object and do crop
            im = Image.fromarray(np_image, 'RGB')
            im = im.crop(crop)
            return np.array(im)

        # check inputs
        if isinstance(image_size, tuple):
            image_size = list(image_size)
        # read data into numpy
        num_features = image_size[0] * image_size[1] * image_size[2]
        start_time = time.time()
        data = bin2np(
            image_path, num_images, num_features=num_features, num_labels=num_labels, label_first=label_first,
            target_feature_type=tf.uint8, target_label_type=tf.int64)
        if num_labels > 0:
            images, labels = data
        else:
            save_label = False
            images = data
            labels = None

        # reshape and do transpose to NHWC
        if image_format_in_file in {'channels_first', 'NCHW'}:
            images = np.reshape(images, [num_images] + image_size)
            images = np.transpose(images, (0, 2, 3, 1))
        elif image_format_in_file in {'channels_last', 'NHWC'}:
            image_size = [image_size[1], image_size[2], image_size[0]]
            images = np.reshape(images, [num_images] + image_size)
        if image_transpose:  # in datasets like MNIST and stl10, the height and weight dimensions are reversed
            images = np.transpose(images, (0, 2, 1, 3))
        # show an example
        image0 = Image.fromarray(images[0], 'RGB')
        image0.show('A image')

        # resize and crop
        if resize is not None:
            images = np.array(list(map(lambda single_image: do_image_resize(single_image), images)))
            image0 = Image.fromarray(images[0], 'RGB')
            image0.show('Resize')
        if crop is not None:
            images = np.array(list(map(lambda single_image: do_image_crop(single_image), images)))
            image0 = Image.fromarray(images[0], 'RGB')
            image0.show('Crop')
        # transpose to NCHW if required
        if target_image_format in {'channels_first', 'NCHW'}:
            images = np.transpose(images, (0, 3, 1, 2))
        # flatten the data
        dataset = np.reshape(images, [num_images, -1])
        duration = time.time() - start_time
        print('Reading image file took {:.1f} seconds'.format(duration))

        # # save to tfrecord
        print('Images writing to tfrecord in process.')
        start_time = time.time()
        if save_label:
            my_np2tfrecord(output_filename, dataset, labels)
        else:
            my_np2tfrecord(output_filename, dataset)
        duration = time.time() - start_time
        print('Writing tfrecord file took {:.1f} seconds'.format(duration))


#######################################################################
def raw_image_to_tfrecords(
        image_folder, output_filename, resize=None, crop=None, save_image=False, save_folder=None,
        image_file_extension='png', num_images_per_tfrecord=20000, image_format=None):
        """ This function reads images in the image_folder and write it into tfrecords (20000 images per tfrecord).

        This function is based on the script from
        https://github.com/tensorflow/models/blob/master/research/real_nvp/lsun_formatting.py

        Unlike raw_image_to_csv, this function allows irregular image file names that cannot be simply indexed, e.g.
        'a908f9211863fa04f7d9b2b3212b3cec5d6d609a.webp'; it also allows irregular image size that may change for each
        image; however, in this case:
        1. either height or width will be downscaled to the required size, depending on which scaling factor is smaller
        2. only center crop can be used.

        For celebA dataset, the following code is used: (original image size 178*218, number 22511*9)
            from GeneralTools.input_func import raw_image_to_tfrecords
            image_folder = 'celebA/img_align_celeba_png'
            output_filename = 'celebA_NCHW/celebA'
            raw_image_to_tfrecords(
                image_folder, output_filename, resize=(72, 88), crop=(64, 64),
                num_images_per_tfrecord=22511, image_format='NCHW')

        For lsun dataset, the following code is used: (original image size 225*? or ?*225, number 3033042)
            from GeneralTools.input_func import raw_image_to_tfrecords
            image_folder = 'LSUN/tr'
            output_filename = 'lsun_NCHW/lsun'
            raw_image_to_tfrecords(
                image_folder, output_filename, resize=(64, 64), crop=(64, 64),
                image_file_extension='webp', num_images_per_tfrecord=49722, image_format='NCHW')

        :param image_folder: e.g. 'celebA/img_align_celeba_png' (in DEFAULT_DOWNLOAD_FILE_DIR)
        :param output_filename: e.g. stl10_NCHW/stl (in DEFAULT_IN_FILE_DIR)
        :param resize: (height, width)
        :param crop: (height, width), center crop is used
        :param save_image:
        :param save_folder:
        :param image_file_extension: 'webp', 'png', etc
        :param num_images_per_tfrecord:
        :param image_format: the image format in tfrecords, if None, check misc_fun.FLAGS.IMAGE_FORMAT
        :return:
        """
        from PIL import Image

        # get image format
        if image_format is None:
            image_format = FLAGS.IMAGE_FORMAT
        # prepare folders
        image_folder = os.path.join(DEFAULT_DOWNLOAD_FILE_DIR, image_folder)
        output_filename = os.path.join(DEFAULT_IN_FILE_DIR, output_filename)
        if save_image:
            if save_folder is None:
                save_folder = image_folder + '_copy'
            save_folder = os.path.join(DEFAULT_DOWNLOAD_FILE_DIR, save_folder)

        # get the list of image file names
        image_filename_list = os.listdir(image_folder)
        image_filename_list = [img_fn for img_fn in image_filename_list if img_fn.endswith('.' + image_file_extension)]
        num_images = len(image_filename_list)
        print('Number of images: {}'.format(num_images))
        # iteratively handle each image
        writer = None
        start_time = time.time()
        for image_index, image_filename in enumerate(image_filename_list):
            # configure the tfrecord file to write
            if image_index % num_images_per_tfrecord == 0:
                file_out = "{}_{:03d}.tfrecords".format(output_filename, image_index // num_images_per_tfrecord)
                # print("Writing on:", file_out)
                writer = tf.python_io.TFRecordWriter(file_out)

            # read image
            im = Image.open(os.path.join(image_folder, image_filename))
            # resize and crop the image
            if resize is not None:
                height, width = im.size
                factor = min(height / resize[0], width / resize[1])
                im = im.resize((int(height / factor), int(width / factor)), Image.LANCZOS)
            if crop is not None:
                height, width = im.size
                h_offset = int((height - crop[0]) / 2)
                w_offset = int((width - crop[1]) / 2)
                box = (h_offset, w_offset, h_offset + crop[0], w_offset + crop[1])
                im = im.crop(box)
            # save image if needed
            if save_image:
                image_save_file = os.path.join(save_folder, '%07d.png' % image_index)
                im.save(image_save_file)

            # if image not RGB format, convert to rbg
            if im.mode != 'RGB':
                im = im.convert('RGB')
            # if image format is channels_first, transpose im
            if image_format in {'channels_first', 'NCHW'}:
                im = np.array(im, dtype=np.uint8)
                im = np.transpose(im, axes=(2, 0, 1))

            # write to tfrecord
            instance = tf.train.Example(features=tf.train.Features(feature={
                'x': _bytes_feature(im.tobytes())  # both PIL.Image and np.ndarray has tobytes() method
            }))
            writer.write(instance.SerializeToString())

            if image_index % 5000 == 0:
                sys.stdout.write('\r {}/{} instances finished.'.format(image_index + 1, num_images))
            if image_index % num_images_per_tfrecord == (num_images_per_tfrecord - 1):
                writer.close()
        writer.close()
        duration = time.time() - start_time
        sys.stdout.write('\n All {} instances finished in {:.1f} seconds'.format(num_images, duration))


#######################################################################
def get_files_in_child_folders(parent_folder, file_extension='JPEG', do_sort=True):
    """ This function returns a list of sub-lists of filenames, each sub-list for a child folder and
    each file ending with file_extension.

    :param parent_folder: a folder that contains multiple child folders
    :param file_extension:
    :param do_sort: sort the list in alphabet order to make sure the order is unique in different file systems.
    :return:
        file_name_list - list of sub-lists
        file_count_list - list of numbers of files in each child folder
    """
    child_folder_list = os.listdir(parent_folder)
    if do_sort:
        child_folder_list.sort()
    file_name_list = []
    file_count_list = []
    for child_folder in child_folder_list:
        image_names = os.listdir(os.path.join(parent_folder, child_folder))
        image_names = [os.path.join(child_folder, img_fn) for img_fn in image_names
                       if img_fn.endswith('.' + file_extension)]
        if do_sort:
            image_names.sort()
        file_name_list.append(image_names)  # file_name_list: [['child_folder/image_name']]
        file_count_list.append(len(image_names))

    return file_name_list, file_count_list


#######################################################################
def raw_image_to_np(
        image_file, image_format='NCHW', resize=None, crop=None, dtype=np.uint8,
        save_image=False, save_folder=None, save_name=None):
        """ This function reads a single image to numpy array with resize and crop.
        Note that resize is done before crop and crop is done w.r.t. the image center

        :param image_file:
        :param image_format:
        :param resize:
        :param crop:
        :param dtype:
        :param save_image:
        :param save_folder:
        :param save_name:
        :return:
        """
        from PIL import Image

        # read image
        im = Image.open(image_file)
        # resize and crop the image
        if resize is not None:
            height, width = im.size
            factor = min(height / resize[0], width / resize[1])
            im = im.resize((int(height / factor), int(width / factor)), Image.LANCZOS)
        if crop is not None:
            height, width = im.size
            h_offset = int((height - crop[0]) / 2)
            w_offset = int((width - crop[1]) / 2)
            box = (h_offset, w_offset, h_offset + crop[0], w_offset + crop[1])
            im = im.crop(box)
        # save image if needed
        if save_image:
            image_save_file = os.path.join(save_folder, save_name)
            im.save(image_save_file)

        # if image not RGB format, convert to rbg
        if im.mode != 'RGB':
            im = im.convert('RGB')
        # output np array
        im = np.array(im, dtype=dtype)
        # if image format is channels_first, transpose im
        if image_format in {'channels_first', 'NCHW'}:
            im = np.transpose(im, axes=(2, 0, 1))

        return im


#######################################################################
def raw_image_labels_to_tfrecords(
        parent_folder, output_filename, resize=None, crop=None, save_image=False, save_folder=None,
        image_file_extension='JPEG', num_images_per_tfrecord=None, image_format=None):
    """ This function reads images in the image_folder and write it into tfrecords (20000 images per tfrecord).

    This function is based on the script from
    https://github.com/tensorflow/models/blob/master/research/real_nvp/lsun_formatting.py

    This function assumes:
    1. images are distributed in several folders;
    2. each folder has irregular names that cannot be simply indexed, e.g. 'n01443537';
    3. each folder contain images of one class
    This function allows:
    1. irregular image names that cannot be simply indexed, e.g. 'n01440764_18.JPEG';
    2. it also allows irregular image size that may change for each image; however, in this case:
        i. either height or width will be downscaled to the required size, depending on which scaling factor is smaller
        ii. only center crop can be used.

    For ImageNet dataset, the following code is used: (original image size variable, number 1281167 = 10168 * 126 - 1)

        # Download in terminal, took around 5 days
        wget -P "/media/richard/My Book/MyBackup/Data/ImageNet"
        "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar"

        # Extract .tar file into images (under 1000 folders), it took 4:04:59 to extract
        from Code.ImageNet.imagenet import extract_train
        tar_name = '/media/richard/My Book/MyBackup/Data/ImageNet/ILSVRC2012_img_train.tar'
        target_dir = '/media/richard/My Book/MyBackup/Data/ImageNet/train_jpeg'
        extract_train(tar_name, target_dir)

        # convert images to tfrecords, it took 5:45:09 to write tfrecord
        # to create 126 tfrecords with roughly equal size, set num_images_per_tfrecord=10168
        from GeneralTools.misc_fun import FLAGS
        FLAGS.DEFAULT_DOWNLOAD = '/media/richard/My Book/MyBackup/Data'
        FLAGS.DEFAULT_IN = '/home/richard/PycharmProjects/myNN/Data/imagenet_NCHW'
        from GeneralTools.input_func import raw_image_labels_to_tfrecords

        parent_folder = 'ImageNet/train_jpeg'
        output_filename = 'imagenet'
        raw_image_labels_to_tfrecords(
            parent_folder, output_filename, resize=(128, 128), crop=(128, 128),
            image_file_extension='JPEG', num_images_per_tfrecord=None, image_format='NCHW')

        There are 9 images with PIL corrupt EXIF data warning

    :param parent_folder: e.g. 'ImageNet/train_jpeg' (in DEFAULT_DOWNLOAD_FILE_DIR)
    :param output_filename: e.g. imagenet_NCHW/imagenet (in DEFAULT_IN_FILE_DIR)
    :param resize: (height, width)
    :param crop: (height, width), center crop is used
    :param save_image:
    :param save_folder:
    :param image_file_extension: 'webp', 'png', 'JPEG', etc
    :param num_images_per_tfrecord:
    :param image_format: the image format in tfrecords, if None, check misc_fun.FLAGS.IMAGE_FORMAT
    :return:
    """
    import itertools

    # get image format
    if image_format is None:
        image_format = FLAGS.IMAGE_FORMAT
    # prepare folders
    parent_folder = os.path.join(DEFAULT_DOWNLOAD_FILE_DIR, parent_folder)
    output_filename = os.path.join(DEFAULT_IN_FILE_DIR, output_filename)
    if save_image:
        if save_folder is None:
            save_folder = parent_folder + '_copy'
        save_folder = os.path.join(DEFAULT_DOWNLOAD_FILE_DIR, save_folder)

    # get the list of image file names
    # do sort to keep the order unique for different file systems
    image_name_list, image_count_list = get_files_in_child_folders(
        parent_folder, image_file_extension, do_sort=True)
    num_class = len(image_count_list)
    num_images = sum(image_count_list[:])  # sum up all elements in list
    print('Number of images: {}; number of classes: {}'.format(num_images, num_class))
    # flatten image names
    image_name_list_flat = list(itertools.chain.from_iterable(image_name_list))
    # get labels
    labels = np.zeros(num_images, int)
    index_list = np.cumsum(image_count_list)  # a list of num_class elements, [1300, 2600, ..., 1281167],
    for class_i in range(1, num_class):
        labels[index_list[class_i - 1]:index_list[class_i]] = class_i

    # save image names and class counts to txt
    np.savetxt(os.path.join(DEFAULT_IN_FILE_DIR, 'image_names.txt'), image_name_list_flat, delimiter=" ", fmt="%s")
    count = np.zeros(num_class, dtype=[('index', int), ('counts', int)])
    count['index'] = np.array(list(range(1000)))
    count['counts'] = image_count_list
    np.savetxt(os.path.join(DEFAULT_IN_FILE_DIR, 'image_class_counts.txt'), count, fmt='%.d %.d', delimiter=' ')

    # save data to tfrecords
    print("Writing start...")
    start_time = time.time()
    if num_images_per_tfrecord is None:  # save images of each class to a tfrecord file

        for class_i in range(num_class):
            file_out = "{}_{:03d}.tfrecords".format(output_filename, class_i)
            # print("Writing on:", file_out)
            writer = tf.python_io.TFRecordWriter(file_out)

            for image_filename in image_name_list[class_i]:
                image_file = os.path.join(parent_folder, image_filename)
                im = raw_image_to_np(
                    image_file, image_format, resize=resize, crop=crop, dtype=np.uint8,
                    save_image=save_image, save_folder=save_folder, save_name='%s.png' % image_filename)

                # write to tfrecord
                instance = tf.train.Example(features=tf.train.Features(feature={
                    'x': _bytes_feature(im.tobytes()),  # both PIL.Image and np.ndarray has tobytes() method
                    'y': _int64_feature(int(class_i))  # numpy int is not int
                }))
                writer.write(instance.SerializeToString())

            sys.stdout.write('\r {}/{} classes finished.'.format(class_i + 1, num_class))
            writer.close()

    else:  # save images to tfrecord files so that each contains num_images_per_tfrecord images
        # iteratively handle each image
        writer = None
        start_time = time.time()
        for image_index, image_filename in enumerate(image_name_list_flat):
            # configure the tfrecord file to write
            if image_index % num_images_per_tfrecord == 0:
                file_out = "{}_{:03d}.tfrecords".format(output_filename, image_index // num_images_per_tfrecord)
                # print("Writing on:", file_out)
                writer = tf.python_io.TFRecordWriter(file_out)

            # read image
            image_file = os.path.join(parent_folder, image_filename)
            im = raw_image_to_np(
                image_file, image_format, resize=resize, crop=crop, dtype=np.uint8,
                save_image=save_image, save_folder=save_folder, save_name='%07d.png' % image_index)

            # write to tfrecord
            instance = tf.train.Example(features=tf.train.Features(feature={
                'x': _bytes_feature(im.tobytes()),  # both PIL.Image and np.ndarray has tobytes() method
                'y': _int64_feature(int(labels[image_index]))  # numpy int is not int
            }))
            writer.write(instance.SerializeToString())

            if image_index % 5000 == 0:
                sys.stdout.write('\r {}/{} instances finished.'.format(image_index + 1, num_images))
            if image_index % num_images_per_tfrecord == (num_images_per_tfrecord - 1):
                writer.close()
        writer.close()

    duration = time.time() - start_time
    sys.stdout.write('\n All {} instances finished in {:.1f} seconds'.format(num_images, duration))


########################################################################
def my_read_tfrecord(filename, num_features, batch_size, num_epochs, num_threads=1, if_shuffle=False):
    """ This function creates a tensor that reads tfrecord file. 
    The actual reading will be run in other code

    Inputs:
    dataset - ndarray
    label - ndarray
    filename = 'USPS'
    """
    # check input
    if isinstance(filename, str):  # if string, add file location and .csv
        filename = [DEFAULT_IN_FILE_DIR + filename + '.tfrecords']
    else:  # if list, add file location and .csv to each element in list
        filename = [DEFAULT_IN_FILE_DIR + file + '.tfrecords' for file in filename]

    # build file queue
    min_queue_examples = 10000
    capacity = min_queue_examples + batch_size * (num_threads + 2)
    filename_queue = tf.train.string_input_producer(
        filename,
        num_epochs=num_epochs,
        shuffle=if_shuffle,
        capacity=capacity)
    reader = tf.TFRecordReader()
    _, value = reader.read(filename_queue)
    # decode examples
    instances = tf.parse_single_example(
        value,
        features={
            'x': tf.FixedLenFeature([num_features], tf.float32),
            'y': tf.FixedLenFeature([], tf.int64)
        })
    features, label = instances['x'], instances['y']
    # create batch
    if if_shuffle:
        x_batch, y_batch = tf.train.shuffle_batch(
            [features, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples)
    else:
        x_batch, y_batch = tf.train.batch(
            [features, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity)

    return x_batch, y_batch


########################################################################
def bin2np(
        filename, num_instances, num_features=None, num_labels=0, label_first=True, raw_data_type=tf.uint8,
        target_feature_type=tf.float32, target_label_type=tf.int32, num_threads=7):
    """ This function reads binary file into ndarray

    :param filename: local file name, e.g. 'cifar/cifar' or a list of such names
    :param num_instances: number of instances in the binary files in total
    :param num_features: num of features for each instance, e.g. 32 * 32 * 3
    :param num_labels: num of labels for each instance; if no label, set label_bytes = 0
    :param label_first: if data is stored as [label, feature] or [feature, label]
    :param raw_data_type: data type saved in the binary file, e.g. tf.uint8
    :param target_feature_type: feature type to output
    :param target_label_type: label type to output
    :param num_threads:
    :return:
    """
    # check file existence
    if isinstance(filename, str):
        filename_queue = [os.path.join(DEFAULT_DOWNLOAD_FILE_DIR, filename + '.bin')]
    elif isinstance(filename, (list, tuple)):
        filename_queue = [os.path.join(DEFAULT_DOWNLOAD_FILE_DIR, each_file + '.bin') for each_file in filename]
    else:
        raise AttributeError('Filename must be provided as str, list or tuple, not {}'.format(type(filename)))
    for each_file in filename_queue:
        assert os.path.isfile(each_file), 'File {} does not exist.'.format(each_file)

    # produce file name queue
    min_queue_examples = 1000
    capacity = min_queue_examples + num_instances * (num_threads + 2)
    filename_queue = tf.train.string_input_producer(filename_queue, num_epochs=None, shuffle=False)

    # decode one datum
    # set row length to label_bytes + data_bytes
    if raw_data_type == tf.uint8:
        record_bytes = num_labels + num_features
    elif raw_data_type in {tf.float32, tf.int32}:
        record_bytes = (num_labels + num_features) * 4
    else:
        raise AttributeError('raw-data-type must be uint8, float32, int32; {} is not supported.'.format(raw_data_type))
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(filename_queue)  # returns key, value
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record = tf.decode_raw(value, raw_data_type)

    # slice the record and do cast
    if num_labels > 0:
        if label_first:
            label = record[0:num_labels]
            features = record[num_labels:(num_labels + num_features)]
        else:
            label = record[0:num_features]
            features = record[num_features:(num_labels + num_features)]
        if target_feature_type is not raw_data_type:
            features = tf.cast(features, target_feature_type)
        if target_label_type is not raw_data_type:
            label = tf.cast(label, target_label_type)
        # set shape, this is because features and label do not have shape information,
        # which is required by following operations.
        features.set_shape([num_features])  # the batch operation will automatically add dimension
        label.set_shape([num_labels])
        # create batch
        data_batch = tf.train.batch(
            [features, label],
            batch_size=num_instances,
            num_threads=num_threads,
            capacity=capacity)
    else:
        features = record
        if target_feature_type is not raw_data_type:
            features = tf.cast(features, target_feature_type)
        # set shape, this is because features and label do not have shape information,
        # which is required by following operations.
        features.set_shape([num_features])  # the batch operation will automatically add dimension
        # create batch
        data_batch = tf.train.batch(
            [features],
            batch_size=num_instances,
            num_threads=num_threads,
            capacity=capacity)

    # get data into np
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        # run initialization
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # run session to read the raw_data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        data_batch = sess.run(data_batch)
        # close session
        coord.request_stop()
        coord.join(threads)

    return data_batch


########################################################################
class ReadTFRecords(object):
    def __init__(
            self, filename, num_features=None, num_labels=0, x_dtype=tf.string, y_dtype=tf.int64, batch_size=64,
            skip_count=0, file_repeat=1, num_epoch=None, file_folder=None,
            num_threads=8, buffer_size=10000, shuffle_file=False):
        """ This function creates a dataset object that reads data from files.

        :param filename: e.g., cifar
        :param num_features: e.g., 3*64*64
        :param num_labels:
        :param x_dtype: default tf.string, the dtype of features stored in tfrecord file
        :param y_dtype:
        :param num_epoch:
        :param buffer_size:
        :param batch_size:
        :param skip_count: if num_instance % batch_size != 0, we could skip some instances
        :param file_repeat: if num_instance % batch_size != 0, we could repeat the files for k times
        :param num_epoch:
        :param file_folder: if not specified, DEFAULT_IN_FILE_DIR is used.
        :param num_threads:
        :param buffer_size:
        :param shuffle_file: bool, whether to shuffle the filename list

        """
        if file_folder is None:
            file_folder = DEFAULT_IN_FILE_DIR
        # check inputs
        if isinstance(filename, str):  # if string, add file location and .tfrecords
            filename = [os.path.join(file_folder, filename + '.tfrecords')]
        else:  # if list, add file location and .tfrecords to each element in list
            filename = [os.path.join(file_folder, file + '.tfrecords') for file in filename]
        for file in filename:
            assert os.path.isfile(file), 'File {} does not exist.'.format(file)
        if file_repeat > 1:
            filename = filename * int(file_repeat)
        if shuffle_file:  # shuffle operates on the original list and returns None / does not return anything
            from random import shuffle
            shuffle(filename)

        # training information
        self.num_features = num_features
        self.num_labels = num_labels
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.batch_size = batch_size
        self.batch_shape = [self.batch_size, self.num_features]
        self.num_epoch = num_epoch
        self.skip_count = skip_count
        # read data,
        dataset = tf.data.TFRecordDataset(filename)  # setting num_parallel_reads=num_threads decreased the performance
        self.dataset = dataset.map(self.__parser__, num_parallel_calls=num_threads)
        self.iterator = None
        self.buffer_size = buffer_size
        self.scheduled = False
        self.num_threads = num_threads

    ###################################################################
    def __parser__(self, example_proto):
        """ This function parses a single datum

        :param example_proto:
        :return:
        """
        # configure feature and label length
        # It is crucial that for tf.string, the length is not specified, as the data is stored as a single string!
        x_config = tf.FixedLenFeature([], tf.string) \
            if self.x_dtype == tf.string else tf.FixedLenFeature([self.num_features], self.x_dtype)
        if self.num_labels == 0:
            proto_config = {'x': x_config}
        else:
            y_config = tf.FixedLenFeature([], tf.string) \
                if self.y_dtype == tf.string else tf.FixedLenFeature([self.num_labels], self.y_dtype)
            proto_config = {'x': x_config, 'y': y_config}

        # decode examples
        datum = tf.parse_single_example(example_proto, features=proto_config)
        if self.x_dtype == tf.string:  # if input is string / bytes, decode it to float32
            # first decode data to uint8, as data is stored in this way
            datum['x'] = tf.decode_raw(datum['x'], tf.uint8)
            # then cast data to tf.float32
            datum['x'] = tf.cast(datum['x'], tf.float32)
            # cannot use string_to_number as there is only one string for a whole sample
            # datum['x'] = tf.strings.to_number(datum['x'], tf.float32)  # this results in possibly a large number

        # return data
        if 'y' in datum:
            # y can be present in many ways:
            # 1. a single integer, which requires y to be int32 or int64 (e.g, used in tf.gather in cbn)
            # 2. num-class bool/integer/float variables. This form is more flexible as it allows multiple classes and
            # prior probabilities as targets
            # 3. float variables in regression problem.
            # but...
            # y is stored as int (for case 1), string (for other int cases), or float (for float cases)
            # in the case of tf.string and tf.int64, convert to to int32
            if self.y_dtype == tf.string:
                # avoid using string labels like 'cat', 'dog', use integers instead
                datum['y'] = tf.decode_raw(datum['y'], tf.uint8)
                datum['y'] = tf.cast(datum['y'], tf.int32)
            if self.y_dtype == tf.int64:
                datum['y'] = tf.cast(datum['y'], tf.int32)
            return datum['x'], datum['y']
        else:
            return datum['x']

    ###################################################################
    def shape2image(self, channels, height, width, resize=None):
        """ This function shapes the input instance to image tensor

        :param channels:
        :param height:
        :param width:
        :param resize: list of tuple
        :type resize: list, tuple
        :return:
        """

        def image_preprocessor(image):
            # scale image to [-1,1]
            image = tf.subtract(tf.divide(image, 127.5), 1)
            # reshape - note this is determined by how the data is stored in tfrecords, modify with caution
            image = tf.reshape(image, (channels, height, width)) \
                if FLAGS.IMAGE_FORMAT == 'channels_first' else tf.reshape(image, (height, width, channels))
            # resize
            if isinstance(resize, (list, tuple)):
                if FLAGS.IMAGE_FORMAT == 'channels_first':
                    image = tf.transpose(
                        tf.image.resize_images(  # resize only support HWC
                            tf.transpose(image, perm=(1, 2, 0)), resize, align_corners=True), perm=(2, 0, 1))
                else:
                    image = tf.image.resize_images(image, resize, align_corners=True)

            return image

        # do image pre-processing
        if self.num_labels == 0:
            self.dataset = self.dataset.map(
                lambda image_data: image_preprocessor(image_data),
                num_parallel_calls=self.num_threads)
        else:
            self.dataset = self.dataset.map(
                lambda image_data, label: (image_preprocessor(image_data), label),
                num_parallel_calls=self.num_threads)

        # write batch shape
        if isinstance(resize, (list, tuple)):
            height, width = resize
        self.batch_shape = [self.batch_size, height, width, channels] \
            if FLAGS.IMAGE_FORMAT == 'channels_last' else [self.batch_size, channels, height, width]

    ###################################################################
    def scheduler(
            self, batch_size=None, num_epoch=None, shuffle_data=True, buffer_size=None, skip_count=None,
            sample_same_class=False, sample_class=None):
        """ This function schedules the batching process

        :param batch_size:
        :param num_epoch:
        :param buffer_size:
        :param skip_count:
        :param sample_same_class: if the data must be sampled from the same class at one iteration
        :param sample_class: if provided, the data will be sampled from class of this label, otherwise,
            data of a random class aresampled.
        :param shuffle_data:
        :return:
        """
        if not self.scheduled:
            # update batch information
            if batch_size is not None:
                self.batch_size = batch_size
                self.batch_shape[0] = self.batch_size
            if num_epoch is not None:
                self.num_epoch = num_epoch
            if buffer_size is not None:
                self.buffer_size = buffer_size
            if skip_count is not None:
                self.skip_count = skip_count
            # skip instances
            if self.skip_count > 0:
                print('Number of {} instances skipped.'.format(self.skip_count))
                self.dataset = self.dataset.skip(self.skip_count)
            # shuffle
            if shuffle_data:
                self.dataset = self.dataset.shuffle(self.buffer_size)
            # set batching process
            if sample_same_class:
                if sample_class is None:
                    print('Caution: samples from the same class at each call.')
                    group_fun = tf.contrib.data.group_by_window(
                        key_func=lambda data_x, data_y: data_y,
                        reduce_func=lambda key, d: d.batch(self.batch_size),
                        window_size=self.batch_size)
                    self.dataset = self.dataset.apply(group_fun)
                else:
                    print('Caution: samples from class {}. This should not be used in training'.format(sample_class))
                    self.dataset = self.dataset.filter(lambda x, y: tf.equal(y[0], sample_class))
                    self.dataset = self.dataset.batch(self.batch_size)
            else:
                self.dataset = self.dataset.batch(self.batch_size)
            # self.dataset = self.dataset.padded_batch(batch_size)
            if self.num_epoch is None:
                self.dataset = self.dataset.repeat()
            else:
                FLAGS.print('Num_epoch set: {} epochs.'.format(num_epoch))
                self.dataset = self.dataset.repeat(self.num_epoch)

            self.iterator = self.dataset.make_one_shot_iterator()
            self.scheduled = True

    ###################################################################
    def next_batch(self, sample_same_class=False, sample_class=None, shuffle_data=True):
        """ This function generates next batch

        :param sample_same_class: if the data must be sampled from the same class at one iteration
        :param sample_class: if provided, the data will be sampled from class of this label, otherwise,
            data of a random class are sampled.
            Note that class_label should not be provided during training.
        :param shuffle_data:
        :return:
        """
        if self.num_labels == 0:
            if not self.scheduled:
                self.scheduler(shuffle_data=shuffle_data)
            x_batch = self.iterator.get_next()

            x_batch.set_shape(self.batch_shape)

            return {'x': x_batch}
        else:
            if sample_class is not None:
                assert isinstance(sample_class, (np.integer, int)), \
                    'class_label must be integer.'
                assert sample_class < self.num_labels, \
                    'class_label {} is larger than maximum value {} allowed.'.format(
                        sample_class, self.num_labels - 1)
                sample_same_class = True
            if not self.scheduled:
                self.scheduler(
                    shuffle_data=shuffle_data, sample_same_class=sample_same_class,
                    sample_class=sample_class)
            x_batch, y_batch = self.iterator.get_next()

            x_batch.set_shape(self.batch_shape)
            y_batch.set_shape([self.batch_size, self.num_labels])

            return {'x': x_batch, 'y': y_batch}


########################################################################
class SimData(object):
    def __init__(
            self, method, batch_size=64, x_dof=None, z_dof=None,
            probs=None, mu=None, std_or_cov=None,
            low=0.0, high=1.0):
        """ This class defines a distribution that can be sampled from.

        :param method:
        :param batch_size:
        :param x_dof:
        :param z_dof:
        :param probs:
        :param mu:
        :param std_or_cov:
        :param low:
        :param high:
        """
        self.batch_size = batch_size
        self.D = x_dof  # num_features for samples
        self.d = z_dof  # num_intrinsic_features for samples
        if (self.d is not None) and (self.D is not None) and (self.d != self.D):
            self.w = self.rand_w()
        else:
            self.w = None

        # prepare
        if isinstance(mu, list):
            mu = np.array(mu, dtype=np.float32)
        if isinstance(std_or_cov, list):
            std_or_cov = np.array(std_or_cov, dtype=np.float32)

        # declare the distribution
        self.dist = None
        if method in {'normal', 'gaussian'}:
            self.multivariate_normal(mu, std_or_cov)
        elif method in {'gaussian_mixture', 'gm'}:
            self.gaussian_mixture(probs, mu, std_or_cov)
        elif method in {'shell'}:
            self.distribution_shell()
        elif method in {'shell2'}:
            self.distribution_shell2()
        elif method in {'star'}:
            self.distribution_star()
        elif method in {'uniform', 'uni', 'u'}:
            self.uniform(low, high)
        else:
            raise NotImplementedError('{} distribution not implemented yet.'.format(method))

    def rand_w(self):
        """ This function creates a random projection matrix of size d-by-D

        :return:
        """
        w = tf.random_normal([self.d, self.D], mean=0.0, stddev=1.0)
        s, u, v = tf.svd(w)
        return tf.get_variable(
            'W', [self.d, self.D], dtype=tf.float32, initializer=tf.matmul(u, v, transpose_b=True), trainable=False)

    def multivariate_normal(self, mu, std_or_cov):
        """ This function declares a multivariate Gaussian distribution

        :param mu: must be a [d] vector
        :param std_or_cov: either a [d] vector for std or a [d,d] matrix for cov
        :return:
        """
        tfd = tf.contrib.distributions

        if len(std_or_cov.shape) == 1:
            self.dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=std_or_cov)
        elif len(std_or_cov.shape) == 2:
            self.dist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=std_or_cov)
        else:
            raise AttributeError('std_or_cov must be either a [d] vector or a [d,d] matrix.')

    def gaussian_mixture(self, probs, mu=None, std_or_cov=None):
        """ This function declares a Gaussian mixture distribution

        :param probs: a [C] vector that sums up to one
        :param mu: [C, d] matrix, each row referring to the mean of a Gaussian
        :param std_or_cov: either a [C, d] matrix, each row as a std, or a [C, d, d] tensor, each page as a cov
        :return:
        """
        tfd = tf.contrib.distributions

        if len(std_or_cov.shape) == 2:
            self.dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=probs),
                components_distribution=tfd.MultivariateNormalDiag(loc=mu, scale_diag=std_or_cov))
        elif len(std_or_cov.shape) == 3:
            self.dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=probs),
                components_distribution=tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=std_or_cov))
        else:
            raise AttributeError('std_or_cov must be either a [C, d] matrix or a [C, d,d] tensor.')

    def uniform(self, low=0.0, high=1.0):
        """ This function defines a multivariate uniform distribution

        :param low: the lower bound of the interval, can be a list
        :param high:
        :return:
        """
        tfd = tf.contrib.distributions

        self.dist = tfd.Uniform(low, high)

    def distribution_shell(self):
        """ This function defines a custom distribution called shell, consisting of 8 Gaussian distributions.

        :return:
        """
        c1 = 0.707106  # sqrt(2)/2
        c2 = [[0.03, 0.0], [0.0, 0.03]]
        c3 = [[0.04, 0.0395], [0.0395, 0.04]]
        c4 = [[0.04, -0.0395], [-0.0395, 0.04]]

        probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        mu = [[1.0, 0.0], [c1, c1], [0.0, 1.0], [-c1, c1], [-1.0, 0.0], [-c1, -c1], [0.0, -1.0], [c1, -c1]]
        cov = [c2, c3, c2, c4, c2, c3, c2, c4]

        mu = np.array(mu, dtype=np.float32) / 1.5  # so that all data are in [-1, 1]
        cov = np.array(cov, dtype=np.float32) / 2.25

        self.gaussian_mixture(probs, mu, cov)

    def distribution_shell2(self):
        """ This function defines a custom distribution called shell, consisting of 8 Gaussian distributions.
        dist_shell2 has different orientations for the Gaussian distributions from the dist_shell,

        :return:
        """
        c1 = 0.707106  # sqrt(2)/2
        c2 = [[0.03, 0.0], [0.0, 0.03]]
        c3 = [[0.04, 0.0], [0.0, 0.0005]]
        c4 = [[0.0005, 0.0], [0.0, 0.04]]

        probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        mu = [[c1, 0.0], [c1, c1], [0.0, c1], [-c1, c1], [-c1, 0.0], [-c1, -c1], [0.0, -c1], [c1, -c1]]
        cov = [c3, c2, c4, c2, c3, c2, c4, c2]

        mu = np.array(mu, dtype=np.float32) / 1.5  # so that all data are in [-1, 1]
        cov = np.array(cov, dtype=np.float32) / 2.25

        self.gaussian_mixture(probs, mu, cov)

    def distribution_star(self):
        """ This function defines a custom distribution called shell, consisting of 8 Gaussian distributions.

        :return:
        """
        c1 = 0.8
        c2 = c1 * np.tan(22.5/180.0*np.pi)
        c3 = [[0.001, 0.0], [0.0, 0.001]]

        probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        mu = [[c2, c1], [c1, c2], [c1, -c2], [c2, -c1], [-c2, -c1], [-c1, -c2], [-c1, c2], [-c2, c1]]
        cov = [c3, c3, c3, c3, c3, c3, c3, c3]

        mu = np.array(mu, dtype=np.float32)
        cov = np.array(cov, dtype=np.float32)

        self.gaussian_mixture(probs, mu, cov)

    def next_batch(self, batch_size=None):
        """ This function samples data from self.dist

        :return:
        """
        if batch_size is None:
            batch_size = self.batch_size
        z = self.dist.sample(batch_size)

        if self.w is not None:
            z = tf.matmul(z, self.w)

        return z

    def __call__(self, batch_size=None):
        return self.next_batch(batch_size)

    def prob(self, x):
        """ This function calculates the probability density value of self.dist at x

        :param x:
        :return:
        """
        return self.dist.prob(x)

    def log_prob(self, x):
        """ This function calculates the log-probability density value of self.dist at x

        :param x:
        :return:
        """
        return self.dist.log_prob(x)


def read_event_file(dataset_name, sub_folder, event_name, tags):
    """ This function reads tag data from event file.

    Currently only scalar data are supported.

    :param dataset_name:
    :param sub_folder:
    :param event_name:
    :param tags:
    :return:
    """
    path = os.path.join(DEFAULT_OUT_FILE_DIR, dataset_name + '_log', sub_folder, event_name)
    assert os.path.exists(path), 'Path does not exist.'

    if os.path.isfile(path):
        if isinstance(tags, str):
            data_list = []
            for e in tf.train.summary_iterator(path):
                for v in e.summary.value:
                    if v.tag == tags:
                        data_list.append(v.simple_value)
        elif isinstance(tags, list) or isinstance(tags, tuple):
            n = len(tags)
            data_list = [[] for _ in range(n)]
            for e in tf.train.summary_iterator(path):
                for v in e.summary.value:
                    for i in range(n):
                        if v.tag == tags[i]:
                            data_list[i].append(v.simple_value)
        else:
            raise AttributeError('tags should be str, list or tuple.')

        return data_list
    elif os.path.isdir(path):
        raise AttributeError('The given path is a directory.')
