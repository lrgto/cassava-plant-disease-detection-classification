import os

# Original Dataset:
CWD = os.getcwd()
SRC_INDEX_FILE = 'merged_train'  # The original (complete) data index file.
SRC_IMAGES_FOLDER = 'train_images'  # The folder (relative to the cwd) containing the original (complete) set of images.

# Data Preparation:
PREPARED_INDEX_FILE = 'dataset'  # The data index file for the new dataset with equal class sample size.
MID_IMAGES_FOLDER = 'dataset_images'  # The relative folder containing the images for the uniform sample size.
MASTER_USES_CLASS_FOLDERS = False  # Whether the master dataset stores images in descendent class folders.
USE_CLASS_FOLDERS = True  # Whether image samples are stored in descendent class folders
SEPARATE_TRAIN_TEST = False
INDEX_FILE_EXT = 'csv'

# labels  category        Name
# ------  --------        ----
# 0       CBB             Cassava Bacterial Blight
# 1       CBSD            Cassava Brown Streak Disease
# 2       CGM             Cassava Green Mottle
# 3       CMD             Cassava Mosaic Disease
# 4       Healthy         No disease
CATEGORY_NAMES = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']  # The directory names for each class

WIDTH, HEIGHT = IMAGE_SIZE = (280, 210)  # (256, 256)  # (400, 300)  # Every image must be this size
IMAGE_CHANNELS = 3
SAMPLE_COLOUR_MODE = 'rgb',  # one of "grayscale", "rgb", "rgba"
PREPARED_IMAGES_FOLDER = f'{WIDTH}x{HEIGHT}'  # Folder containing uniform sized images for dataset
TRAIN_TEST_IMAGES_FOLDER = 't' + MID_IMAGES_FOLDER

FRACTION_FOR_TRAIN = 0.85

MAX_ARRAY_IMAGES = 1500  # Too large will exhaust memory
NORMALISE_IMAGES = True  # Normalise image pixels when loading into dataset
