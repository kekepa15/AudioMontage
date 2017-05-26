import tensorflow as tf
import numpy as np


from Utils import FLAGS



class Encoder(object):
    def __init__(self, name):
        self.n = FLAGS.hidden_n # hidden number of first dense ouput layer
        self.name = name

    def Encode(self, image, reuse=False):
                
        with tf.variable_scope(self.name, reuse=reuse):
            
