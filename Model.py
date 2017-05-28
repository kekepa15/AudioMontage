"""
BEGAN for audio montage project
"""
from Utils import generate_z
from Decoder import Decoder
from Encoder import Encoder

import tensorflow as tf
import numpy as np



#___________________________Layer info_____________________________________


Encoder_infos = {
			        "outdim":[n,n,2*n,2*n,2*n,3*n,3*n, 3*n, 3*n] \
			        "kernel":[ \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
			       			 ], \
			        "stride":[ \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        		 ], \
			    } 


Decoder_infos = {
			        "outdim":[n,n,n,n,n,n,3] \
			        "kernel":[ \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
			       			 ], \
			        "stride":[ \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        		 ], \
			       } 

Generator_infos = {
			        "outdim":[n,n,n,n,n,n,3] \
			        "kernel":[ \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
					            [3, 3], \
			       			 ], \
			        "stride":[ \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        			[1, 1], \
			        		 ], \
			       } 





#---------------------------------------------------------------------------



#____________________________________Model composition________________________________________

Encoder = Encoder("Encoder", Encoder_infos)
Decoder = Decoder("Decoder", Decoder_infos)
Generator = Decoder("Generator", Generator_infos)


#Generator
image = Generator.decode(generate_z())


#Discriminator (Auto-Encoder)
embedding_vector = Encoder.encode(image)
reconstructed_image = Decoder.decode(embedding_vector)

#---------------------------------------------------------------------------------------







