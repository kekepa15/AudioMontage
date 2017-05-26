import tensorflow as tf
import numpy as np
import librosa

from PIL import Image
from glob import glob
from os import walk, mkdir



#Get image information  
image = Image.open("./sample_image.jpg")
image_w, image_h = image.size
image_c = 3


#Get spectrogram information  
offset = 0
duration = 3
sampling_rate = 16000
fft_size = 1024

y,sr = librosa.load("./sample_audio.wav", offset=offset, duration=duration, sr=sampling_rate) # load audio
D = librosa.stft(y, n_fft=fft_size, hop_length=int(fft_size/2), win_length=fft_size, window='hann') # make spectrogram
spectrogram_h = D.shape[0] # height of spectrogram
spectrogram_w = D.shape[1] # width of spectrogram
spectrogram_c = 1 # channel of spectrogram


#Define constant
flags = tf.app.flags
FLAGS = flags.FLAGS

#training parameters
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100000, 'Maximum epochs to iterate.')
flags.DEFINE_integer('bn', 16, "Batch number")

#data parameters
flags.DEFINE_integer('spec_h', spectrogram_h, "Height of spectrogram" )
flags.DEFINE_integer('spec_w', spectrogram_w, "Width of spectrogram" )
flags.DEFINE_integer('spec_c', spectrogram_c, "Channel of spectrogram" )
flags.DEFINE_integer('img_h', image_h, "Height of image" )
flags.DEFINE_integer('img_w', image_w, "Width of image" )
flags.DEFINE_integer('img_c', image_c, "Channel of image" )


#layer parameters
flags.DEFINE_integer('hidden_n', 64, "Hidden convolution number")
flags.DEFINE_integer('output_channel', 3, "Output channel number")
# gen_conv_infos = {
#         "conv_layer_number": layer_depth,
#         "filter":[
#             [3, 3, 8*8*n, 64],
#             [3, 3, 64, 64*2],
#             [3, 3, 64*2, 64*4],
#             [3, 3, 64*4, 64*8],
#         ],
#         "stride" : [[1, 2, 2, 1] for _ in range(layer_depth)],
#                  }   


#---------------------------------------------------------------------------#

#Functions

def generate_z(size=64):
    return tf.random_uniform(shape=(1,size), minval=-1, maxval=1, dtype=tf.float32)


def upsample(images, size):
    """    
    images : image having shape with [batch, height, width, channels], 
    size : output_size with [new_height, new_width]
    """
    return tf.image.resize_nearest_neighbor(images=images, size=size)  

