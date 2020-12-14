import os

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)
TRAIN_PATH = "val2017"
RESNET_PATH = "data/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"
TRAINING_SIZE = 10
train_ids = next(os.walk(TRAIN_PATH))[2][:TRAINING_SIZE]
SEED = 42
TEST_SIZE = 2
BATCH_SIZE = 3
