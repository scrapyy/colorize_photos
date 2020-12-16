from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, RepeatVector, Reshape, concatenate, UpSampling2D


def Colorize():
    embed_input = Input(shape=(1000,))

    # Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)

    # Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

    # Decoder
    decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    return Model(inputs=[encoder_input, embed_input], outputs=decoder_output)