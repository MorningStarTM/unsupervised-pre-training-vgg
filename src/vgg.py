import tensorflow as tf
import keras.layers as L
from keras.models import Model


def vgg_encoder(inputs:tuple):
    inputs = L.Input(shape=inputs)

    c = L.Conv2D(16, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(inputs)
    c = L.Conv2D(16, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c)
    c = L.Conv2D(16, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c)
    c = L.MaxPooling2D((2, 2), strides=(2, 2))(c)


    c1 = L.Conv2D(32, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c)
    c1 = L.Conv2D(32, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c1)
    c1 = L.Conv2D(32, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c1)
    c1 = L.MaxPooling2D((2, 2), strides=(2, 2))(c1)

    c2 = L.Conv2D(64, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c1)
    c2 = L.Conv2D(64, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c2)
    c2 = L.Conv2D(64, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c2)
    c2 = L.MaxPooling2D((2,2), strides=(2,2))(c2)

    c3 = L.Conv2D(128, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c2)
    c3 = L.Conv2D(128, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c3)
    c3 = L.Conv2D(128, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c3)
    c3 = L.MaxPooling2D((2,2), strides=(2,2))(c3)

    c4 = L.Conv2D(256, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c3)
    c4 = L.Conv2D(256, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c4)
    c4 = L.Conv2D(256, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(c4)
    c4 = L.MaxPooling2D((2,2), strides=(2,2))(c4)

    encoder_model = Model(inputs, c4, name='encoder')
    return encoder_model



def vgg_decoder(inputs:tuple):
    decoder_in = L.Input(shape=(8,8,256))

    d = L.Conv2DTranspose(256, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(decoder_in)
    d = L.Conv2DTranspose(256, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d)
    d = L.Conv2DTranspose(256, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d)
    d = L.UpSampling2D((2,2))(d)

    d1 = L.Conv2DTranspose(128, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d)
    d1 = L.Conv2DTranspose(128, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d1)
    d1 = L.Conv2DTranspose(128, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d1)
    d1 = L.UpSampling2D((2,2))(d1)

    d2 = L.Conv2DTranspose(64, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d1)
    d2 = L.Conv2DTranspose(64, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d2)
    d2 = L.Conv2DTranspose(64, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d2)
    d2 = L.UpSampling2D((2,2))(d2)

    d3 = L.Conv2DTranspose(32, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d2)
    d3 = L.Conv2DTranspose(32, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d3)
    d3 = L.Conv2DTranspose(32, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d3)
    d3 = L.UpSampling2D((2,2))(d3)

    d4 = L.Conv2DTranspose(16, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d3)
    d4 = L.Conv2DTranspose(16, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d4)
    d4 = L.Conv2DTranspose(16, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d4)
    d4 = L.UpSampling2D((2,2))(d4)

    outputs = L.Conv2DTranspose(3, (6,6), activation='sigmoid', padding='same', strides=(1,1), kernel_initializer='he_normal')(d4)

    decoder_model = Model(decoder_in, outputs, name='decoder')

