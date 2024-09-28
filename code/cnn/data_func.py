import os
import shutil
import sys

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from PIL import Image

import data_settings as ds
import model_settings as ms


def model_begin_filename(filename=None, width=ds.WIDTH, height=ds.HEIGHT):
    name = filename
    if not name:
        name = f'{ms.OPTIMIZER_NAME.lower()}_{width}x{height}'
    return f'{ds.CWD}\\{ms.MODEL_DATA_DIR}\\{name}'


def model_end_filename(version, lr=ms.OPTIMIZER_LR, runs=ms.TRAIN_EPOCHS, epochs=ms.FIT_EPOCHS):
    # print(f'model_end_filename() _{lr}_{runs}x{epochs}_{version}')
    # return f'_{lr}_{runs}x{epochs}_{version}'
    return f'_{version}'


def model_filename(version, filename=None):
    mod_name = model_begin_filename(filename) + model_end_filename(version)
    return mod_name


def full_index_filename(dataset_name):
    name = f'{ds.CWD}\\{dataset_name}'
    if ds.INDEX_FILE_EXT:
        name += '.' + ds.INDEX_FILE_EXT
    return name


def train_index_filename(version, filename=None):
    name = f'{model_begin_filename(filename)}_{ms.MODEL_TRAIN_PREFIX}{model_end_filename(version)}'
    if ds.INDEX_FILE_EXT:
        name += '.' + ds.INDEX_FILE_EXT
    return name


def test_index_filename(version, filename=None):
    name = f'{model_begin_filename(filename)}_{ms.MODEL_TEST_PREFIX}{model_end_filename(version)}'
    if ds.INDEX_FILE_EXT:
        name += '.' + ds.INDEX_FILE_EXT
    return name


def load_index(filename):
    return pd.read_csv(filename)


def save_index(index_df, filename):
    index_df.to_csv(filename, index=False)


def save_all(version, model: tf.keras.Sequential, train_index: pd.DataFrame, test_index: pd.DataFrame,
             directory_dataset: bool = False) -> None:
    """
    Save the current model and state.

    :rtype: None
    :param version: The model version (integer or string).
    :param model: The keras.Sequential model.
    :param train_index: The pandas.DataFrame for the training set.
    :param test_index: The pandas.DataFrame for the evaluation set.
    :param directory_dataset:
    """
    model.save(model_filename(version))
    if not directory_dataset:
        save_index(train_index, train_index_filename(version))
        save_index(test_index, test_index_filename(version))


def load_all(version, model_data_filename=None, directory_dataset: bool = ds.SEPARATE_TRAIN_TEST):
    """
    Load a model and state.

    :rtype: tuple[keras.Sequential, pandas.DataFrame, pandas.DataFrame]
    :param version: The model version (integer or string).
    :param model_data_filename:
    :param directory_dataset:
    :return: A keras.Sequential, a training set pandas.DataFrame, an evaluation set pandas.DataFrame.
    """
    model_data_filename = model_filename(version, filename=model_data_filename)
    model = tf.keras.models.load_model(model_data_filename)
    if not directory_dataset:
        train_index = load_index(train_index_filename(version))
        test_index = load_index(test_index_filename(version))
        return model, train_index, test_index
    return model


def copy_index_files(src_version, dst_version, src_name=None):
    # Copy index files from src_version to dest_version
    s_train = train_index_filename(src_version, filename=src_name)
    d_train = train_index_filename(dst_version)
    shutil.copy(s_train, d_train)
    s_test = test_index_filename(src_version, filename=src_name)
    d_test = test_index_filename(dst_version)
    shutil.copy(s_test, d_test)


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Making', dir)


def get_index_and_dirs(
        src_index_file: str = ds.PREPARED_INDEX_FILE, src_images_folder: str = ds.MID_IMAGES_FOLDER,
        dst_images_folder: str = ds.PREPARED_IMAGES_FOLDER):
    src_dir = f'{ds.CWD}\\{src_images_folder}'
    dst_dir = f'{ds.CWD}\\{dst_images_folder}'
    ensure_dir(dst_dir)

    src_df = load_index(full_index_filename(src_index_file))
    return src_df, src_dir, dst_dir


def path_category(image_dir, label, use_class):
    return f'{image_dir}\\{ds.CATEGORY_NAMES[label]}' if use_class else image_dir


# Use source index csv file to copy a uniform number of images for each class
# into the destination folder, indexed by the dest_index_file.
# Set max_images to limit the number of images per class
def equalize_class_sample_size(
        src_index_file: str = ds.SRC_INDEX_FILE, dst_index_file: str = ds.PREPARED_INDEX_FILE,
        src_images_folder: str = ds.SRC_IMAGES_FOLDER, dst_images_folder: str = ds.MID_IMAGES_FOLDER,
        use_src_class: bool = ds.MASTER_USES_CLASS_FOLDERS, use_dst_class: bool = ds.USE_CLASS_FOLDERS,
        max_images: int = 0) -> int:
    src_df, src_dir, dst_dir = get_index_and_dirs(src_index_file, src_images_folder, dst_images_folder)

    # Group the source DataFrame by class labels (last column in src_df) into individual classes
    gdf = src_df.groupby(src_df.columns[1])

    # Find the smallest sample size of each class
    smallest = sys.maxsize
    for label, files in gdf:
        rows = files.shape[0]
        if rows < smallest:
            smallest = rows
    if smallest > max_images > 0:
        smallest = max_images

    dest_df = pd.DataFrame(columns=src_df.columns)

    for label, files in gdf:
        s_dir = path_category(src_dir, label, use_src_class)
        d_dir = path_category(dst_dir, label, use_dst_class)
        # Get min random rows for this class label
        samples = files.sample(n=smallest, replace=False, random_state=0)

        print(f'Class {label} has {samples.shape[0]} images')

        # Add samples to dest_df DataFrame
        dest_df = dest_df.append(samples)

        ensure_dir(dst_dir)

        for i in range(smallest):
            filename = samples.iloc[i, 0]

            try:
                # Copy samples files from image_src_folder to image_dest_folder
                shutil.copy(f'{s_dir}\\{filename}', d_dir)
            except FileNotFoundError:
                print('Missing File: ' + filename)  # << SHOULD REMOVE FROM INDEX
                continue

    save_index(dest_df, full_index_filename(dst_index_file))
    return smallest


# Get a size for an image to fit a desired size while maintaining aspect ratio.
def size_to_fit(image_size: tuple[int, int], desired_size: tuple[int, int]) -> tuple[int, int]:
    """
    Get a size for an image to fit a desired size while maintaining aspect ratio.

    :return: The largest width and height to fit the desired size.
    :rtype: tuple[int, int]
    :param image_size: The actual image width and height.
    :param desired_size: The target image width and height.
    """
    iw, ih = image_size
    dw, dh = desired_size
    h = dw / iw
    v = dh / ih
    ratio = h if (h < v) else v
    new_width = iw * ratio
    new_height = ih * ratio
    return round(new_width), round(new_height)


def copy_all_images(
        src_index_file: str = ds.PREPARED_INDEX_FILE, src_images_folder: str = ds.MID_IMAGES_FOLDER,
        dst_images_folder: str = ds.PREPARED_IMAGES_FOLDER, use_src_class: bool = ds.USE_CLASS_FOLDERS,
        use_dst_class: bool = ds.USE_CLASS_FOLDERS,
        resize: bool = False, desired_size: tuple[int, int] = ds.IMAGE_SIZE) -> None:
    """
    Copy the sample images with option to resize to a set width and height while maintaining aspect ratios.

    :rtype: None
    :param src_index_file: The name of the source dataset index file.
    :param src_images_folder: The relative directory of the source image samples.
    :param dst_images_folder: The relative directory to store the new image samples.
    :param use_src_class: Boolean indicates whether source image samples are stored in separate class directories.
    :param use_dst_class: Boolean indicates whether target image samples are stored in separate class directories.
    :param resize: Boolean indicates whether to resize sample images to desired_size.
    :param desired_size: Tuple of integer (width, height) for resizing.
    """
    src_df, src_dir, dst_dir = get_index_and_dirs(src_index_file, src_images_folder, dst_images_folder)

    # for filename in src_df['image_id']:
    for i in range(src_df.shape[0]):
        label = src_df['labels'].iloc[i]
        s_dir = path_category(src_dir, label, use_src_class)
        d_dir = path_category(dst_dir, label, use_dst_class)
        ensure_dir(d_dir)
        filename = src_df['image_id'].iloc[i]
        if resize:
            img = Image.open(f'{s_dir}\\{filename}')
            new_size = size_to_fit(img.size, desired_size)
            img2 = img.resize(new_size)
            new_im = Image.new("RGB", desired_size)
            new_im.paste(img2, ((desired_size[0] - new_size[0]) // 2,
                                (desired_size[1] - new_size[1]) // 2))
            new_im.save(f'{d_dir}\\{filename}')
        else:
            try:
                # Copy samples files from image_src_folder to image_dest_folder
                shutil.copy(f'{s_dir}\\{filename}', d_dir)
            except FileNotFoundError:
                print(f'Missing File: {s_dir}\\{filename}')  # << SHOULD REMOVE FROM INDEX
                continue


def split_index(index_df, frac=ds.FRACTION_FOR_TRAIN):
    train_index = pd.DataFrame(columns=index_df.columns)
    test_index = pd.DataFrame(columns=index_df.columns)
    # Group Dataframe by class labels (last column in src_df)
    gdf = index_df.groupby(index_df.columns[1])
    for name, grp in gdf:
        rows = grp.shape[0]
        br = int(rows * frac)
        grp = sklearn.utils.shuffle(grp)
        train_index = train_index.append(grp[:br], ignore_index=True)
        test_index = test_index.append(grp[br:], ignore_index=True)
    return train_index, test_index


def image_array(index_df, images_folder, start_index=0, images=0, use_class_folders=ds.USE_CLASS_FOLDERS):
    src_dir = f'{ds.CWD}\\{images_folder}'
    if images == 0 or images > index_df.shape[0] - start_index:
        images = index_df.shape[0] - start_index
    if images > ds.MAX_ARRAY_IMAGES:
        images = ds.MAX_ARRAY_IMAGES
    classes = index_df['labels'].max() - index_df['labels'].min() + 1

    s_dir = path_category(src_dir, index_df['labels'].iloc[0], use_class_folders)
    img = Image.open(s_dir + '\\' + index_df['image_id'].iloc[0])
    w, h = img.size

    d_type = 'float32' if ds.NORMALISE_IMAGES else 'int16'
    X = np.empty(shape=(images, h, w, 3), dtype=d_type)
    y = np.empty(images, dtype='int32')
    for ind in range(images):
        i = ind + start_index
        filename = index_df['image_id'].iloc[i]

        label = index_df['labels'].iloc[i]
        s_dir = path_category(src_dir, label, use_class_folders)

        img = Image.open(f'{s_dir}\\{filename}')
        im_arr = np.asarray(img)
        if ds.NORMALISE_IMAGES:
            X[ind] = im_arr.astype("float32") / 255
        else:
            X[ind] = im_arr
        y[ind] = label
    return X, tf.keras.utils.to_categorical(y, num_classes=classes, dtype="int32")
