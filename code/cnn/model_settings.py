import tensorflow as tf
from data_settings import PREPARED_IMAGES_FOLDER

TRAIN_BATCH_SIZE = 10  # The size of a batch of images to compile for each training pass
TEST_BATCH_SIZE = 10  # The size of a batch of images to compile for each evaluation pass

TRAIN_EPOCHS = 2  # epochs parameter for fitting
FIT_EPOCHS = 5  # epochs parameter for fitting

TRAIN_ROTATION_RANGE = 40

DEFAULT_STEPPED_LEARNING_RATE = [0.2, 0.1, 0.01, 0.001]

# Model layers
FILTERS_1 = 64  # For 1st Conv2d layer
CONV_ACTIVATION_1 = 'relu'
KERNEL_HEIGHT_1, KERNEL_WIDTH_1 = KERNEL_SIZE_1 = (3, 3)  # (2, 2)  # For 1st Conv2d layer
KERNEL_STRIDES_1 = (1, 1)  # For 1st Conv2d layer
POOL_HEIGHT_1, POOL_WIDTH_1 = POOL_SIZE_1 = (2, 2)  # For 1st Pooling layer
STRIDE_HEIGHT_1, STRIDE_WIDTH_1 = POOL_STRIDES_1 = (2, 2)  # For 1st Pooling layer

FILTERS_2 = 128  # For 2nd Conv2d layer
CONV_ACTIVATION_2 = 'relu'
KERNEL_HEIGHT_2, KERNEL_WIDTH_2 = KERNEL_SIZE_2 = (3, 3)  # (2, 2)  # For 2nd Conv2d layer
KERNEL_STRIDES_2 = (1, 1)  # For 2nd Conv2d layer
POOL_HEIGHT_2, POOL_WIDTH_2 = POOL_SIZE_2 = (2, 2)  # For 2nd Pooling layer
STRIDE_HEIGHT_2, STRIDE_WIDTH_2 = POOL_STRIDES_2 = (2, 2)  # For 2nd Pooling layer

FILTERS_3 = 256  # 128  # For 3rd Conv2d layer
CONV_ACTIVATION_3 = 'relu'
KERNEL_HEIGHT_3, KERNEL_WIDTH_3 = KERNEL_SIZE_3 = (3, 3)  # (2, 2)  # For 3rd Conv2d layer
KERNEL_STRIDES_3 = (1, 1)  # For 3rd Conv2d layer
POOL_HEIGHT_3, POOL_WIDTH_3 = POOL_SIZE_3 = (2, 2)  # For 3rd Pooling layer
STRIDE_HEIGHT_3, STRIDE_WIDTH_3 = POOL_STRIDES_3 = (2, 2)  # For 3rd Pooling layer

INITIAL_DENSE_LAYERS = False  # Whether to set up dense layers with conv layers

DROPOUT_RATE_1 = 0.2  # Input dropout rate before 1st Dense Layer
DENSE_OUT_1 = 396  # 202752 / 512 # 496  # Outputs for the 1st Dense Layer
DENSE_ACTIVATION_1 = 'relu'

DROPOUT_RATE_2 = 0.1  # Input dropout rate before 2nd Dense Layer
DENSE_OUT_2 = 792  # 202752 / 256 # 992  # 62  # 40  # 00  # Outputs for the 2nd Dense Layer
DENSE_ACTIVATION_2 = 'relu'

DROPOUT_RATE_3 = 0.1  # Input dropout rate before 3rd Dense Layer
DENSE_OUT_3 = 198  # 202752 / 1024 # 248  # 0  # Outputs for the 3rd Dense Layer
DENSE_ACTIVATION_3 = 'relu'

OUT = 5  # The 5 classes
OUT_ACTIVATION = 'softmax'

# Model Compilation
OPTIMIZER_LR = 0.01  # 0.001
# OPTIMIZER_NAME = 'SGD'  # The name of the optimizer parameter for compiling model
OPTIMIZER_NAME = 'Adadelta'  # The name of the optimizer parameter for compiling model

# MODEL_OPTIMIZER = OPTIMIZER_NAME  # 'Ftrl'  # 'Adam'  # 'Adamax'  # The optimizer parameter for compiling model
MODEL_LOSS = 'categorical_crossentropy'  # loss parameter for compiling model
MODEL_METRICS = ['accuracy']  # metrics parameter for compiling model

# Model Execution
MODEL_DATA_DIR = 'cnn_model'  # Folder to store trained models
MODEL_TRAIN_PREFIX = 'train_ind'  # Base of training data index filename
MODEL_TEST_PREFIX = 'test_ind'  # Base of testing data index filename


# kernel_initializer parameter for Dense layers
def get_dense_kernel_initializer():
    return tf.keras.initializers.GlorotNormal(seed=42)


# The optimizer parameter for compiling model
def get_model_optimizer(name=OPTIMIZER_NAME, lr=OPTIMIZER_LR):
    if name.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr,  # 0.01
            momentum=0.0,
            nesterov=False,
            name=name
        )
    else:
        optimizer = tf.keras.optimizers.Adadelta(
            learning_rate=lr,  # 0.001
            rho=0.95,
            epsilon=1e-07,
            name=name
        )
    return optimizer
