import sys
import math

import pandas as pd
import sklearn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf

import data_settings as ds
import model_settings as ms

from data_func import path_category, image_array, load_all, save_all, train_index_filename, test_index_filename, \
    load_index, model_filename, copy_index_files
from util_func import start_timer, dataset_memory_use, show_timer, sci_form_bytes, step_time_str, years_secs_str


def history_list_avg(vals):
    mean = None
    if vals is not None and len(vals) > 0:
        mean = sum(vals) / len(vals)
    return mean


def hist_str_append(st, val_name, vals):
    mean = history_list_avg(vals)
    if mean is not None:
        if st != '':
            st += ' - '
        st += f'{val_name}: {mean:.6f}'
    return st


def history_string(history):
    st = ''
    if history is not None:
        for key in history.history.keys():
            st = hist_str_append(st, key, history.history[key])
    return st


def setup_conv_layers(model):
    if ms.FILTERS_1 > 0:
        model.add(layers.Conv2D(
            ms.FILTERS_1, kernel_size=ms.KERNEL_SIZE_1, strides=ms.KERNEL_STRIDES_1, activation=ms.CONV_ACTIVATION_1))
        model.add(layers.MaxPooling2D(pool_size=ms.POOL_SIZE_1, strides=ms.POOL_STRIDES_1))
    if ms.FILTERS_2 > 0:
        model.add(layers.Conv2D(
            ms.FILTERS_2, kernel_size=ms.KERNEL_SIZE_2, strides=ms.KERNEL_STRIDES_2, activation=ms.CONV_ACTIVATION_2))
        model.add(layers.MaxPooling2D(pool_size=ms.POOL_SIZE_2, strides=ms.POOL_STRIDES_2))
    if ms.FILTERS_3 > 0:
        model.add(layers.Conv2D(
            ms.FILTERS_3, kernel_size=ms.KERNEL_SIZE_3, strides=ms.KERNEL_STRIDES_3, activation=ms.CONV_ACTIVATION_3))
        model.add(layers.MaxPooling2D(pool_size=ms.POOL_SIZE_3, strides=ms.POOL_STRIDES_3))
    model.add(layers.Flatten())


def setup_dense_layers(model):
    if ms.DROPOUT_RATE_1 > 0:
        model.add(layers.Dropout(ms.DROPOUT_RATE_1))
    if ms.DENSE_OUT_1 > 0:
        model.add(layers.Dense(ms.DENSE_OUT_1, kernel_initializer=ms.get_dense_kernel_initializer(),
                               activation=ms.DENSE_ACTIVATION_1))
    if ms.DROPOUT_RATE_2 > 0:
        model.add(layers.Dropout(ms.DROPOUT_RATE_2))
    if ms.DENSE_OUT_2 > 0:
        model.add(layers.Dense(ms.DENSE_OUT_2, kernel_initializer=ms.get_dense_kernel_initializer(),
                               activation=ms.DENSE_ACTIVATION_2))
    if ms.DROPOUT_RATE_3 > 0:
        model.add(layers.Dropout(ms.DROPOUT_RATE_3))
    if ms.DENSE_OUT_3 > 0:
        model.add(layers.Dense(ms.DENSE_OUT_3, kernel_initializer=ms.get_dense_kernel_initializer(),
                               activation=ms.DENSE_ACTIVATION_3))


def compile_model(model):
    optimizer = ms.get_model_optimizer()
    # K.set_value(optimizer.lr, ms.OPTIMIZER_LR)
    model.compile(optimizer=optimizer, loss=ms.MODEL_LOSS, metrics=ms.MODEL_METRICS, run_eagerly=False)
    # print(f'Learning Rate: {ms.OPTIMIZER_LR}')
    lr = K.get_value(model.optimizer.lr)
    print(f'Learning Rate: {lr:.6f}')
    # lr = K.get_value(ms.MODEL_OPTIMIZER.lr)
    # print(f'Learning Rate: {lr:.6f}')

    print(model.summary())


def setup_model(width, height, channels):
    model = keras.Sequential()
    model.add(keras.Input(shape=(height, width, channels)))
    # inputs = keras.Input(shape=(height, width, channels))
    # a = inputs
    setup_conv_layers(model)
    if ms.INITIAL_DENSE_LAYERS:
        setup_dense_layers(model)
    model.add(layers.Dense(ms.OUT, activation=ms.OUT_ACTIVATION))

    # model = keras.Model(inputs=inputs, outputs=outputs)
    compile_model(model)

    return model


def set_conv_training(model, trainable=False):
    for layer in model.layers:
        if layer.name.startswith('conv2d') or layer.name.startswith('max_pooling2d'):
            layer.trainable = trainable


def has_hidden_layers(model):
    hidden = False
    c = 0
    for layer in model.layers:
        if layer.name.startswith('dense'):
            c += 1
            if c > 1:
                hidden = True
                break
    return hidden


# Only has effect if there are Dense layers other than the output layer.
def trainable_conv(src_version, dst_version, src_name, trainable=False):
    # Load model
    src_filename = model_filename(src_version, filename=src_name)
    model = tf.keras.models.load_model(src_filename)
    if has_hidden_layers(model):
        set_conv_training(model, trainable)
        compile_model(model)
        # Save model
        model.save(model_filename(dst_version))
        copy_index_files(src_version, dst_version, src_name)
    else:
        print('Cannot change conv training status without hidden Dense layers.')


def update_with_dense(src_version, dst_version, src_name):
    # Load model
    src_filename = model_filename(src_version, filename=src_name)
    model = tf.keras.models.load_model(src_filename)
    incorporate_dense_layers(model)
    # Save model
    model.save(model_filename(dst_version))
    copy_index_files(src_version, dst_version, src_name)


def incorporate_dense_layers(model):
    if not has_hidden_layers(model):
        # Remove last layer of model
        model.pop()

        # Setup new dense layers
        setup_dense_layers(model)
        model.add(layers.Dense(ms.OUT, activation=ms.OUT_ACTIVATION))

        set_conv_training(model, trainable=False)
    else:
        print('Hidden Dense layers already exist.')
    compile_model(model)


def validate_batch_size(batch_size):
    if batch_size == 0 or batch_size > ds.MAX_ARRAY_IMAGES:
        batch_size = ds.MAX_ARRAY_IMAGES
    return batch_size


def get_generator(datagen, batch_size, index_df, shuffle=True, images_folder=ds.MID_IMAGES_FOLDER,
                  use_class_folders=ds.USE_CLASS_FOLDERS):
    src_dir = f'{ds.CWD}\\{images_folder}'
    pd.set_option('mode.chained_assignment', None)
    prep_df = index_df.copy()

    images = prep_df.shape[0]

    # prep_df = pd.DataFrame(index_df.copy())

    if use_class_folders:
        for ind in range(images):
            filename = prep_df['image_id'].iloc[ind]

            label = prep_df['labels'].iloc[ind]
            s_dir = path_category(src_dir, label, use_class_folders)

            prep_df['image_id'].iloc[ind] = f'{s_dir}\\{filename}'

    classes = prep_df['labels'].max() - prep_df['labels'].min() + 1
    y = pd.DataFrame(tf.keras.utils.to_categorical(prep_df['labels'], num_classes=classes, dtype="int32"),
                     columns=ds.CATEGORY_NAMES)
    prep_df = prep_df.drop(['labels'], axis=1)
    prep_df = pd.concat([prep_df, y], axis=1)

    test_generator = datagen.flow_from_dataframe(
        prep_df,
        directory=src_dir,
        x_col='image_id',
        y_col=y.columns.values.astype(str),
        target_size=ds.IMAGE_SIZE,
        color_mode='rgb',  # one of "grayscale", "rgb", "rgba"
        class_mode='raw',  # 'categorical',
        batch_size=batch_size,
        shuffle=shuffle,
        seed=None,
        interpolation='nearest',
        # validate_filenames=False
    )
    return test_generator


def train_cnn(model, train_index):
    t_start = start_timer()
    batch_size = validate_batch_size(ms.TRAIN_BATCH_SIZE)
    datagen = ImageDataGenerator(
        rotation_range=ms.TRAIN_ROTATION_RANGE,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0.0,
        rescale=1. / 255
    )
    train_generator = get_generator(datagen, batch_size, train_index, True, images_folder=ds.MID_IMAGES_FOLDER,
                                    use_class_folders=ds.USE_CLASS_FOLDERS)

    print(f'TRAINING {train_index.shape[0]} images')
    print('Expected dataset batch size: ' + dataset_memory_use(batch_size, ds.WIDTH, ds.HEIGHT))
    t = t_start
    for e in range(ms.TRAIN_EPOCHS):
        model.fit(train_generator, batch_size=batch_size, epochs=ms.FIT_EPOCHS, verbose=1)
        t += show_timer(t, f'RUN {e + 1} of {ms.TRAIN_EPOCHS} took')
    show_timer(t_start, 'Training took')


def train_cnn2(model, train_index):
    t_start = start_timer()
    batch_size = ms.TRAIN_BATCH_SIZE
    if batch_size == 0 or batch_size > ds.MAX_ARRAY_IMAGES:
        batch_size = ds.MAX_ARRAY_IMAGES

    datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0.0,
        rescale=1. / 255
    )
    train_generator = datagen.flow_from_directory(
        f'{ds.CWD}/{ds.TRAIN_TEST_IMAGES_FOLDER}/train',
        target_size=ds.IMAGE_SIZE,
        color_mode='rgb',
        # classes=None,  # CATEGORY_NAMES,
        class_mode='categorical',
        batch_size=ms.TRAIN_BATCH_SIZE,
        shuffle=True,
        seed=None,
        # save_to_dir=None,
        # save_prefix='',
        # save_format='png',
        # follow_links=False,
        # subset=None,
        interpolation='nearest'
    )

    print(f'TRAINING {train_index.shape[0]} images')
    print('Expected dataset batch size: ' + dataset_memory_use(batch_size, ds.WIDTH, ds.HEIGHT))
    for e in range(ms.TRAIN_EPOCHS):
        model.fit(train_generator, batch_size=ms.TRAIN_BATCH_SIZE, epochs=ms.FIT_EPOCHS, verbose=1)
    show_timer(t_start, 'Training took')


def init_image_batch(epoch, total_epochs, batch_size, i, secs, start, steps, test_index, t_start, total_images,
                     history):
    X, y = image_array(test_index, ds.PREPARED_IMAGES_FOLDER, start_index=start, images=batch_size)

    st = history_string(history)
    if i == 0:
        if epoch < 2:
            x_size = sys.getsizeof(X)
            y_size = sys.getsizeof(y)
            print('Actual dataset batch size: ' + sci_form_bytes(x_size + y_size)
                  + ' {' + str(x_size) + ' + ' + str(y_size) + ' bytes}')
            secs = show_timer(t_start, f'Time to compile batch of {batch_size} images:')
        if total_epochs == 0:
            print(f'BATCH {i + 1:d}/{steps:d}                       ', end=' ')
        else:
            ts = ''
            if epoch > 1:
                epoch -= 1
                i = steps - 1
                secs, total, ts = step_time_str(t_start, secs, epoch, total_epochs, 1, 1)
            print(f'EPOCH {epoch:d}/{total_epochs:d} - BATCH {i + 1:d}/{steps:d}' + ts + '    ' + st, end='\n')
    else:
        secs, total, ts = step_time_str(t_start, secs, epoch, total_epochs, start, total_images)
        if total_epochs == 0:
            t_left = years_secs_str(total - secs)
            print(f'BATCH {i + 1:d}/{steps:d}: {t_left:s} remaining     ', end=' ')
        else:
            print(f'EPOCH {epoch:d}/{total_epochs:d} - BATCH {i:d}/{steps:d}' + ts + '    ' + st, end='\n')
    return secs, start + batch_size, X, y


def train_cnn1(model, train_index):
    t_start = start_timer()
    batch_size = ms.TRAIN_BATCH_SIZE
    if batch_size == 0 or batch_size > ds.MAX_ARRAY_IMAGES:
        batch_size = ds.MAX_ARRAY_IMAGES
    steps = math.ceil(train_index.shape[0] / batch_size)
    print(f'TRAINING {train_index.shape[0]} images')
    print('Expected dataset batch size: ' + dataset_memory_use(batch_size, ds.WIDTH, ds.HEIGHT))
    start = secs = 0
    history = None
    for e in range(ms.TRAIN_EPOCHS):
        train_index = sklearn.utils.shuffle(train_index)
        for i in range(steps):
            secs, start, train_X, train_y = \
                init_image_batch(e + 1, ms.TRAIN_EPOCHS, batch_size, i, secs, start, steps, train_index, t_start,
                                 train_index.shape[0], history)
            # history = model.fit(train_X, train_y, batch_size=TRAIN_BATCH_SIZE, epochs=FIT_EPOCHS, verbose=0)
            history = model.fit(train_X, train_y, batch_size=ms.TRAIN_BATCH_SIZE, epochs=ms.FIT_EPOCHS, verbose=0,
                                workers=1, use_multiprocessing=True)
        start = 0
    # _, total, ts = step_time_str(t_start, secs, steps, steps)
    _, total, ts = step_time_str(t_start, secs, 1, 1, 1, 1)
    # print('BATCH', steps, 'of', steps, ts)
    show_timer(t_start, 'Training took')


def evaluate_cnn(model, test_index):
    t_start = start_timer()
    batch_size = validate_batch_size(ms.TEST_BATCH_SIZE)
    datagen = ImageDataGenerator(
        rotation_range=0,  # 20,
        width_shift_range=0,  # 0.2,
        height_shift_range=0,  # 0.2,
        horizontal_flip=False,  # True,
        fill_mode='constant',
        cval=0.0,
        rescale=1. / 255
    )
    test_generator = get_generator(datagen, batch_size, test_index, False, images_folder=ds.MID_IMAGES_FOLDER,
                                   use_class_folders=ds.USE_CLASS_FOLDERS)
    # test_generator = get_generator(datagen, batch_size, test_index, False, images_folder=PREPARED_IMAGES_FOLDER,
    #                                use_class_folders=USE_CLASS_FOLDERS)

    print(f'EVALUATING {test_index.shape[0]} images')
    print('Expected dataset batch size: ' + dataset_memory_use(batch_size, ds.WIDTH, ds.HEIGHT))
    loss, acc = model.evaluate(test_generator, verbose=1)
    show_timer(t_start, 'Evaluation took')
    return loss, acc


def evaluate_cnn1(model, test_index):
    t_start = start_timer()
    samples = test_index.shape[0]
    batch_size = ms.TEST_BATCH_SIZE
    if batch_size == 0 or batch_size > ds.MAX_ARRAY_IMAGES:
        batch_size = ds.MAX_ARRAY_IMAGES
    steps = math.ceil(samples / batch_size)
    print('EVALUATING ' + str(test_index.shape[0]) + ' images')
    print('Expected dataset batch size: ' + dataset_memory_use(batch_size, ds.WIDTH, ds.HEIGHT))
    start = secs = 0
    test_loss = test_acc = 0
    history = None
    for i in range(steps):
        secs, start, test_X, test_y = \
            init_image_batch(0, 0, batch_size, i, secs, start, steps, test_index, t_start, samples, history)
        # print()
        loss, acc = model.evaluate(test_X, test_y, verbose=2)
        test_loss += loss * test_X.shape[0]
        test_acc += acc * test_X.shape[0]
    # _, total, ts = step_time_str(t_start, secs, steps, steps)
    _, total, ts = step_time_str(t_start, secs, 1, 1, 1, 1)
    # print('BATCH', steps, 'of', steps, ts)
    show_timer(t_start, 'Evaluation took')
    return test_loss / samples, test_acc / samples


def train_with_included_layers(src_version, dst_version, eval_train=True, train=False):
    model, train_index, test_index = load_all(src_version, None, ds.SEPARATE_TRAIN_TEST)
    incorporate_dense_layers(model)
    results = train_eval(dst_version, model, train_index, test_index, eval_train, train)
    return model, train_index, test_index, results


def train_eval(dst_version, model, train_index, test_index, eval_train=True, train=False):
    if train:
        train_cnn(model, train_index)
        save_all(dst_version, model, train_index, test_index)
    train_loss, train_acc = 0.0, 0.0
    if eval_train:
        train_loss, train_acc = evaluate_cnn(model, train_index)
    test_loss, test_acc = evaluate_cnn(model, test_index)
    results = train_loss, train_acc, test_loss, test_acc
    display_loss_acc(results)
    return results


def load_train_eval(
        src_version, dst_version, eval_train=True, train=False,
        model_data_filename=None, directory_dataset: bool = ds.SEPARATE_TRAIN_TEST):
    model, train_index, test_index = load_all(src_version, model_data_filename, directory_dataset)
    compile_model(model)
    results = train_eval(dst_version, model, train_index, test_index, eval_train, train)

    return model, train_index, test_index, results


def setup_train_test(dst_version, train_index, test_index):
    model = setup_model(ds.WIDTH, ds.HEIGHT, ds.IMAGE_CHANNELS)
    results = train_eval(dst_version, model, train_index, test_index, train=True)
    return model, train_index, test_index, results


def setup_train_with_existing(from_model_name, from_version, dst_version):
    train_index = load_index(train_index_filename(from_version, filename=from_model_name))
    test_index = load_index(test_index_filename(from_version, filename=from_model_name))
    return setup_train_test(dst_version, train_index, test_index)


def display_loss_acc(results):
    train_loss, train_acc, test_loss, test_acc = results
    loss_diff = acc_diff = None
    print()
    if train_loss != 0 or train_acc != 0:
        loss_diff = test_loss - train_loss
        acc_diff = test_acc - train_acc
        print(f'Train loss: {train_loss:.4f}    Train acc: {train_acc:.4f}')
    if test_loss != 0 or test_acc != 0:
        print(f' Test loss: {test_loss:.4f}     Test acc: {test_acc:.4f}')
    if loss_diff is not None:
        print(f'      diff  {loss_diff:.4f}               {acc_diff:.4f}')
    print()


def train_dataset_batches(images_folder, version, eval_train=True):
    val_split = 1 - ds.FRACTION_FOR_TRAIN
    train_ds = tf.keras.utils.image_dataset_from_directory(
        images_folder + '\\train',
        labels='inferred',
        label_mode='categorical',
        validation_split=val_split,
        subset="training",
        class_names=None,
        batch_size=ms.TRAIN_BATCH_SIZE,
        image_size=(ds.HEIGHT, ds.WIDTH),
        shuffle=True)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        images_folder + '\\train',
        labels='inferred',
        label_mode='categorical',
        validation_split=val_split,
        subset="training",
        class_names=None,
        batch_size=ms.TEST_BATCH_SIZE,
        image_size=(ds.HEIGHT, ds.WIDTH))
    test_ds = tf.keras.utils.image_dataset_from_directory(
        images_folder + '\\test',
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        batch_size=ms.TEST_BATCH_SIZE,
        image_size=(ds.HEIGHT, ds.WIDTH))
    model = setup_model(ds.WIDTH, ds.HEIGHT, ds.IMAGE_CHANNELS)
    t_start = start_timer()
    print('Training')
    _ = model.fit(train_ds, validation_data=val_ds, batch_size=ms.TRAIN_BATCH_SIZE, epochs=ms.TRAIN_EPOCHS, verbose=1)
    show_timer(t_start, 'Training took')
    t_start = start_timer()
    print('Evaluating')
    train_loss, train_acc = 0.0, 0.0
    if eval_train:
        train_loss, train_acc = model.evaluate(train_ds, verbose=1)
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    show_timer(t_start, 'Evaluation took')
    display_loss_acc((train_loss, train_acc, test_loss, test_acc))
    model.save(model_filename(version, filename=None))
