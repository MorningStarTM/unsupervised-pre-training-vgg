import tensorflow as tf
from keras import layers as L
from keras.models import Model
from vgg import vgg_decoder, vgg_encoder


class Autoencoder:
    def __init__(self, inputs:tuple):
        self.encoder = vgg_encoder(inputs)
        self.deocder = vgg_decoder(self.encoder.output)

    def build_autoencoder(self):
        autoencoder_inputs = self.encoder.input
        autoencoder_outputs = self.deocder(self.encoder(autoencoder_inputs))

        autoencoder_model = Model(autoencoder_inputs, autoencoder_outputs, name='autoencoder')
        return autoencoder_model