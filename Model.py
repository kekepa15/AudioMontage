"""
BEGAN for audio montage project
"""
import tensorflow as tf
import numpy as np

import Decoder
import Encoder
import utils


with tf.variable_scope("Generator") #create instance of decoder under the name of generator
    G = Decoder
    G.build()