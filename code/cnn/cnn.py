import sys

from tensorflow.keras.backend import clear_session

import data_settings as ds
import model_settings as ms
from util_func import show_timer, start_timer
from data_func import equalize_class_sample_size, copy_all_images, split_index, train_index_filename, \
    test_index_filename, full_index_filename, load_index, model_begin_filename
from cnn_func import setup_train_with_existing, setup_train_test, load_train_eval, \
    train_with_included_layers, trainable_conv, update_with_dense


def reduce_and_equalise_dataset():
    t_start = start_timer(show_time=True)
    class_size = equalize_class_sample_size()
    message = 'Equalising took'
    show_timer(t_start, message)
    print(class_size, "images for each class")


def make_samples_uniform_size():
    t_start = start_timer(show_time=True)
    copy_all_images(resize=True, desired_size=ds.IMAGE_SIZE)
    message = 'Resizing took'
    show_timer(t_start, message)


def change_dir_structure(src_class_dirs=False):
    t_start = start_timer(show_time=True)
    copy_all_images(use_src_class=src_class_dirs, use_dst_class=~src_class_dirs)
    message = 'Changing directory structure took'
    show_timer(t_start, message)


def change_dir_train_test(use_version, src_folder=ds.PREPARED_IMAGES_FOLDER, dst_folder=ds.PREPARED_IMAGES_FOLDER,
                          src_class_dirs=ds.USE_CLASS_FOLDERS, dst_class_dirs=ds.USE_CLASS_FOLDERS):
    t_start = start_timer(show_time=True)
    train_index_name = train_index_filename(use_version)
    test_index_name = test_index_filename(use_version)
    copy_all_images(train_index_name, src_folder, dst_folder + '\\train',
                    use_src_class=src_class_dirs, use_dst_class=dst_class_dirs)
    copy_all_images(test_index_name, src_folder, dst_folder + '\\test',
                    use_src_class=src_class_dirs, use_dst_class=dst_class_dirs)
    message = 'Changing directory structure took'
    show_timer(t_start, message)


def train_with_existing_set(src_name, src_version, dst_version):
    setup_train_with_existing(src_name, src_version, dst_version)


def initial_training(version):
    index_df = load_index(full_index_filename(ds.PREPARED_INDEX_FILE))
    train_index, test_index = split_index(index_df, ds.FRACTION_FOR_TRAIN)
    setup_train_test(version, train_index, test_index)


def extra_training(src_version, dst_version):
    load_train_eval(src_version, dst_version, eval_train=True, train=True)


def extra_dense_training(src_version, dst_version):
    train_with_included_layers(src_version, dst_version, eval_train=True, train=True)


def evaluate_model(version, eval_train):
    load_train_eval(version, version, eval_train=eval_train, train=False)


def other_steps(src_version, dst_version, lrs):
    global OPTIMIZER_LR
    for lr in lrs:
        OPTIMIZER_LR = lr
        print(f'Learning Rate at {OPTIMIZER_LR}')
        extra_training(src_version, dst_version)
        src_version = dst_version


# 'conv', src_version, dst_version, src_name
def conv_learning_from_existing_named(src_version, dst_version, src_name, lrs=ms.DEFAULT_STEPPED_LEARNING_RATE):
    ms.INITIAL_DENSE_LAYERS = False
    ms.OPTIMIZER_LR = lrs[0]
    print(f'Learning Rate at {ms.OPTIMIZER_LR}')
    train_with_existing_set(src_name, src_version, dst_version)
    other_steps(dst_version, dst_version, lrs[1:])


# 'conv', src_version, dst_version
def conv_learning_from_existing_set(src_version, dst_version, lrs=ms.DEFAULT_STEPPED_LEARNING_RATE):
    ms.INITIAL_DENSE_LAYERS = False
    ms.OPTIMIZER_LR = lrs[0]
    print(f'Learning Rate at {ms.OPTIMIZER_LR}')
    extra_training(src_version, dst_version)
    other_steps(dst_version, dst_version, lrs[1:])


# 'conv', src_version
def conv_learning(src_version, lrs=ms.DEFAULT_STEPPED_LEARNING_RATE):
    ms.INITIAL_DENSE_LAYERS = False
    ms.OPTIMIZER_LR = lrs[0]
    print(f'Learning Rate at {ms.OPTIMIZER_LR}')
    initial_training(src_version)
    other_steps(src_version, src_version, lrs[1:])


# 'dense', src_version[, [dst_version]
def dense_learning(src_version, dst_version, lrs=ms.DEFAULT_STEPPED_LEARNING_RATE):
    ms.OPTIMIZER_LR = lrs[0]
    print(f'Learning Rate at {ms.OPTIMIZER_LR}')
    extra_dense_training(src_version, dst_version)
    other_steps(dst_version, dst_version, lrs[1:])


# 'more', src_version[, [dst_version]
def more_learning(src_version, dst_version, lrs=ms.DEFAULT_STEPPED_LEARNING_RATE):
    other_steps(src_version, dst_version, lrs)


def get_paras(args, ensure_dest=True):
    p1 = p2 = p3 = None
    if len(args) > 2:
        p1 = args[2]
        if len(args) > 3:
            p2 = args[3]
            if len(args) > 4:
                p3 = args[4]
        elif ensure_dest:
            p2 = p1
    return p1, p2, p3


#  model_function(['', 'prepare']) calls reduce_and_equalise_dataset()
#  model_function(['', 'resize']) calls make_samples_uniform_size()
#  model_function(['', 'train', model_version]) calls initial_training(model_version)
#  model_function(['', 'train', src_version, dst_version])
#      calls train_with_existing_set(src_name=model_begin_filename(filename=src_name), src_version, dst_version)
#  model_function(['', 'retrain', src_version, dst_version]) calls extra_training(src_version, dst_version)
#  model_function(['', 'evaluate', model_version, 'True']) calls evaluate_model(model_version, eval_train)
#  model_function(['', 'add_dense', src_version, dst_version, src_name])
#  model_function(['', 'trainable_conv', src_version, dst_version, flag, src_name])
def model_function(args: list) -> None:
    """
    Perform some model operation.\n
    Can be executed in the terminal
        py cnn.py <function> [<src_version>] [<dst_version>] [<...>]\n
    or called from another function
        model_function(['', '<function>'])\n
        model_function(['', '<function>', <src_version>])\n
        model_function(['', '<function>', <src_version>, <dst_version>])\n
        model_function(['', '<function>', <src_version>, <dst_version>, <...>])\n
    The src_version and dst_version arguments can be integer or string values.\n
    function:
        prepare - Copy a uniform number of images for each class to be used for models.\n

        resize - Resize images to IMAGE_SIZE := (WIDTH, HEIGHT).\n

        train <src_version> - Train model and save as src_version.\n
        train <src_version>, <dst_version> - Train model using training set from src_version
        and save as dst_version.\n
        train <src_version>, <dst_version>, <src_model_name> - Train model using training set from src_version
        of a different model and save as dst_version.\n

        retrain <src_version>, <dst_version> - Retrain model src_version and save as dst_version.\n

        evaluate <src_version> - Evaluate test set on model src_version.\n
        evaluate <src_version> <any_value> - Evaluate test and training sets on model src_version.\n

        add_dense <src_version>[, <dst_version>, [<src_name>] - Add dense layers to model src_version\n

        trainable_conv <src_version> [<dst_version> [<flag> [<src_name>]]] - Changes the model to either
        train conv layers or not. Only has effect if there are Dense layers other than the output layer.\n

        conv <src_version> - Iteratively train convolution layers of model and save as src_version.\n
        conv <src_version>, <dst_version> - Iteratively train convolution layers of model
        using training set from src_version and save as dst_version.\n
        conv <src_version>, <dst_version>, <src_model_name> - Iteratively train convolution layers
        of model using training set from src_version of a different model and save as dst_version.\n

        dense <src_version>[, <dst_version>] - Iteratively retrain dense layers of model src_version
        with added dense layers and save as dst_version.\n

        more <src_version>[, <dst_version>] - Iteratively retrain all layers of model src_version
        and save as dst_version.\n


    :rtype: None
    :param args: A list of '', 'function', [src_version], [dst_version], [...]
    """
    if len(args) > 1:  # py cnn_model train 0
        if args[1] == 'prepare':
            print('Copy a uniform number of images for each class to be used for models')
            reduce_and_equalise_dataset()
        elif args[1] == 'resize':
            print(f'Resize images to ({ds.WIDTH:d}, {ds.HEIGHT:d})')
            make_samples_uniform_size()
        elif args[1] == 'train':
            src_version, dst_version, src_name = get_paras(args, False)
            if src_version is not None:
                if dst_version is not None:
                    # src_name = model_begin_filename(filename=src_name)
                    print(f'Train model using training set from {src_version} and save as {dst_version}')
                    train_with_existing_set(src_name, src_version, dst_version)
                else:
                    print(f'Train model and save as given version {src_version}')
                    initial_training(src_version)
        elif args[1] == 'retrain':
            src_version, dst_version, _ = get_paras(args)
            if src_version is not None and dst_version is not None:
                print(f'Retrain model version {src_version} and save as {dst_version}')
                extra_training(src_version, dst_version)
        elif args[1] == 'evaluate':
            model_version, eval_train, _ = get_paras(args, False)
            if eval_train:
                eval_train = True
                st = 'and training sets'
            else:
                eval_train = False
                st = ''
            print(f'Evaluate test {st} on model version {model_version}')
            evaluate_model(model_version, eval_train)
        elif args[1] == 'add_dense':
            src_version, dst_version, src_name = get_paras(args)
            if src_version is not None:
                print(f'Add dense layers to model version {src_version}')
                update_with_dense(src_version, dst_version, src_name)
                # extra_dense_training(src_version, dst_version)
        elif args[1] == 'trainable_conv':
            src_version, dst_version, fg = get_paras(args)
            if src_version is not None:
                src_name = args[4] if len(args) > 4 else None
                if fg is None or str(fg).lower().startswith('t'):
                    flag = True
                    print(f'Allow conv layers to be trained: Updates model version {src_version}')
                else:
                    flag = False
                    print(f'Stop conv layers being trained: Updates model version {src_version}')
                trainable_conv(src_version, dst_version, src_name, flag)
        elif args[1] == 'conv':
            src_version, dst_version, src_name = get_paras(args, False)
            if src_version is not None:
                if dst_version is not None:
                    if src_name:
                        print(f'Train model (convolution w/ variable learning rate) using training set'
                              f'from {src_name} version {src_version} and save as {dst_version}')
                        conv_learning_from_existing_named(src_version, dst_version, src_name)
                    else:
                        print(f'Train model (convolution w/ variable learning rate) using training set'
                              f'from {src_version} and save as {dst_version}')
                        conv_learning_from_existing_set(src_version, dst_version)
                else:
                    print(f'Train model (convolution w/ variable learning rate) and save'
                          f'as given version {src_version}')
                    conv_learning(src_version)
        elif args[1] == 'dense':
            src_version, dst_version, src_name = get_paras(args)
            if src_version is not None and dst_version is not None:
                print(f'Retrain model (dense w/ variable learning rate) version {src_version} and save'
                      f'as {dst_version}')
                dense_learning(src_version, dst_version)
        elif args[1] == 'more':
            src_version, dst_version, src_name = get_paras(args)
            if src_version is not None and dst_version is not None:
                print(f'Retrain model (conv + dense w/ variable learning rate) version {src_version} and'
                      f'save as {dst_version}')
                more_learning(src_version, dst_version)
        else:
            print('Unrecognized command')
        clear_session()

    else:
        print('No command')


model_function(sys.argv)

# model_function(['', 'prepare'])
# change_dir_structure(src_class_dirs=False)
# model_function(['', 'resize'])
# model_function(['', 'train', 0])
# model_function(['', 'train', 9, 0])
# model_function(['', 'train', 0, '0.1ad0', 'adamax_256x256'])
# model_function(['', 'evaluate', 1])
# model_function(['', 'evaluate', 0, 'train'])
# model_function(['', 'add_dense', '0.01c3', '0.01cX'])
# extra_dense_training('0.01c3', '0.01cX')

# change_dir_train_test(0, dst_folder='bydataset')
# train_dataset_batches(PREPARED_IMAGES_FOLDER, 9)


# Set Learning Rate to 0.1, rotation to 40, TRAIN_EPOCHS=2, FIT_EPOCHS=5
# model_function(['', 'train', 0, '0.1_2x5_o0', 'adamax_256x256'])
# model_function(['', 'retrain', '0.1_2x5_o0', '0.1_2x5_o1'])
# model_function(['', 'trainable_conv', '0.1_2x5_o1', False]) # Should be unable to change setting
# model_function(['', 'retrain', '0.1_2x5_o1', '0.1_2x5_o2'])
# model_function(['', 'retrain', '0.1_2x5_o3', '0.1_2x5_o4'])
# Retrain adadelta_280x210_0.1_2x5_o4 with 0.01 learning rate
# model_function(['', 'retrain', '0.1_2x5_o4', '0.01_2x5_o5'])
# model_function(['', 'retrain', '0.01_2x5_o5', '0.01_2x5_o6'])
# model_function(['', 'retrain', '0.01_2x5_o6', '0.01_2x5_o7'])
# model_function(['', 'retrain', '0.01_2x5_o7', '0.01_2x5_o8'])
# model_function(['', 'retrain', '0.01_2x5_o8', '0.01_2x5_o9'])
# Add hidden layers and retrain  adadelta_280x210_0.1_2x5_o9A dense layers only with 0.01 learning rate:
# model_function(['', 'add_dense', '0.01_2x5_o9', '0.01_2x5_o9A'])
# model_function(['', 'evaluate', '0.01_2x5_o9A', True])
# model_function(['', 'retrain', '0.01_2x5_o9A', '0.01_2x5_oA'])
# model_function(['', 'retrain', '0.01_2x5_oA', '0.01_2x5_oB'])
# model_function(['', 'retrain', '0.01_2x5_oB', '0.01_2x5_oC'])
# model_function(['', 'retrain', '0.01_2x5_oC', '0.01_2x5_oD'])
# model_function(['', 'retrain', '0.01_2x5_oD', '0.01_2x5_oE'])
