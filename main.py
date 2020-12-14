
from skimage.color import gray2rgb, rgb2gray, rgb2lab, lab2rgb
from skimage.io import imsave
from tensorflow.python import enable_eager_execution
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from Data import training_data, create_inception_embedding, image_a_b_gen, prepare_test_data
from Network import Colorize
from Utils import train_ids, BATCH_SIZE, TEST_SIZE
import tensorflow as tf
import numpy as np

x_train, x_test = training_data()

learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
filepath = "Art_Colorization_Model.h5"
checkpoint = ModelCheckpoint(filepath,
                             save_best_only=True,
                             monitor='loss',
                             mode='min')
model_callbacks = [learning_rate_reduction,checkpoint]
model = Colorize()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
model.summary()

model.fit_generator(image_a_b_gen(x_train),
                    epochs=5,
                    verbose=1,
                    steps_per_epoch=5, callbacks=model_callbacks)

model.save(filepath)
model.save_weights("Art_Colorization_Weights.h5")

x_test_input, x_test_input_embedded = prepare_test_data(x_test)

output = model.predict([x_test_input, x_test_input_embedded])
output = output * 128
acc = 0
for i in range(len(x_test)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = x_test_input[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    gray = np.zeros((256, 256, 3))
    gray[:, :, 0] = x_test_input[i][:, :, 0]
    final_output = lab2rgb(cur)
    outp = np.concatenate((lab2rgb(gray), final_output, x_test[i]), axis=0)
    imsave("output\\img_" + str(i) + ".jpg", outp)
    acc += np.sum((final_output - x_test)**2)

acc /= len(x_test)
print("Accuracy: ", acc)



