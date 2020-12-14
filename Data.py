import os
import sys

from skimage.color import rgb2gray, gray2rgb, rgb2lab
from tensorflow.python.framework import ops
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.ops import get_default_graph
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet_v2 import preprocess_input

from Utils import train_ids, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH, SEED, TEST_SIZE, BATCH_SIZE, RESNET_PATH


def training_data():
    print("Getting train images ...")
    sys.stdout.flush()
    x_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    nr = 0

    for id in train_ids:
        path = os.path.join(TRAIN_PATH, id)
        img = imread(path)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
        x_train[nr] = img
        nr += 1
    x_train = x_train.astype('float32') / 255.
    x_train, x_test = train_test_split(x_train, test_size=TEST_SIZE, random_state=SEED)
    return x_train, x_test


datagen = ImageDataGenerator(shear_range=0.2,
                             zoom_range=0.2,
                             rotation_range=20,
                             horizontal_flip=True)

inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights(RESNET_PATH)
inception.graph = get_default_graph()

def create_inception_embedding(grayscaled_rgb):
    def resize_gray(x):
        return resize(x,(299,299,3),mode='constant')
    grayscaled_rgb_resized = np.array([resize_gray(x) for x in grayscaled_rgb])
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    embed = inception.predict(grayscaled_rgb_resized)
    return embed

def image_a_b_gen(dataset):
    for batch in datagen.flow(dataset, batch_size=BATCH_SIZE):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        x_batch = lab_batch[:, :, :, 0]
        x_batch = x_batch.reshape(x_batch.shape + (1,))
        y_batch = lab_batch[:, :, :, 1:] / 128
        yield [x_batch, embed], y_batch


def prepare_test_data(dataset):
    color_me = gray2rgb(rgb2gray(dataset))
    color_me_embed = create_inception_embedding(color_me)
    color_me = rgb2lab(color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape + (1,))
    return color_me, color_me_embed



