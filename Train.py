"""
BEGAN for audio montage project
"""
from Utils import generate_z, FLAGS, get_loss
from Decoder import Decoder
from Encoder import Encoder
from Loader import Image_Loader, save_image

import tensorflow as tf
import numpy as np
import os

def main(_):

	"""
	Run main function
	"""

	#___________________________________________Layer info_____________________________________________________
	n = FLAGS.hidden_n

	Encoder_infos = {
						"outdim":[n,n,2*n,2*n,2*n,3*n,3*n, 3*n, 3*n], \
						"kernel":[ \
									[3, 3], \
									[3, 3], \
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
									[2, 2], \
									[1, 1], \
									[1, 1], \
									[2, 2], \
									[1, 1], \
									[1, 1], \
								], \
					} 


	Decoder_infos = {
						"outdim":[n,n,n,n,n,n,3], \
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
						"outdim":[n,n,n,n,n,n,3], \
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


	"""
	Prepare Image Loader
	"""
	root = "./CelebA/images"
	batch_size = FLAGS.bn
	scale_size = [32,32]
	data_format = "NHWC"
	loader = Image_Loader(root, batch_size, scale_size, data_format, file_type="jpg")



	"""
	Make Saving Directories
	"""
	os.makedirs("./Check_Point", exist_ok=True)
	os.makedirs("./logs", exist_ok=True) # make logs directories to save summaries
	os.makedirs("./Generated_Images", exist_ok=True)
	os.makedirs("./Decoded_Generated_Images", exist_ok=True)




	#----------------------------------------------------------------------------------------------------







	#____________________________________Model composition________________________________________


	k = tf.Variable(0, name = "k_t", trainable = False, dtype = tf.float32) #init value of k_t = 0

	E = Encoder("Encoder", Encoder_infos)
	D = Decoder("Decoder", Decoder_infos)
	G = Decoder("Generator", Generator_infos)

	#Generator
	z_G = tf.placeholder(tf.float32, shape=(FLAGS.bn, 1, FLAGS.hidden_n), name = "z_G") # Place holder for  z_G
	z_D = tf.placeholder(tf.float32, shape=(FLAGS.bn, 1, FLAGS.hidden_n), name = "z_D") # Place holder for  z_D

	generated_image = G.decode(z_G)
	generated_image_for_disc = G.decode(z_D, reuse = True)

	#Discriminator (Auto-Encoder)
	image = tf.placeholder(tf.float32, shape=(FLAGS.bn, scale_size[0], scale_size[1], 3), name = "Real_Image") #get embedding vector z_G

	embedding_vector_real = E.encode(image)
	reconstructed_image_real = D.decode(embedding_vector_real)

	embedding_vector_fake = E.encode(generated_image_for_disc, reuse=True)
	reconstructed_image_fake = D.decode(embedding_vector_fake, reuse=True)


	#-----------------------------------------------------------------------------------------------







	#____________________________________________Loss_______________________________________________


	"""
	Define Loss
	"""
	real_image_loss = get_loss(image, reconstructed_image_real)
	generator_loss_for_disc = get_loss(generated_image_for_disc, reconstructed_image_fake)
	discriminator_loss = real_image_loss - tf.multiply(k, generator_loss_for_disc)
	generator_loss = get_loss(generated_image, reconstructed_image_fake)
	global_measure = real_image_loss + tf.abs(tf.multiply(FLAGS.gamma,real_image_loss) - generator_loss)

	
	#-----------------------------------------------------------------------------------------------







	#_____________________________________________Train_______________________________________________

	discriminator_parameters = []
	generator_parameters = []

	for v in tf.trainable_variables():
		if 'Encoder' in v.name:
			discriminator_parameters.append(v)
			print("Discriminator parameter : ", v.name)
		elif 'Decoder' in v.name:
			discriminator_parameters.append(v)
			print("Discriminator parameter : ", v.name)			
		elif 'Generator' in v.name:
			generator_parameters.append(v)
			print("Generator parameter : ", v.name)
		else:
			print("None of Generator and Discriminator parameter : ", v.name)

	optimizer_D = tf.train.AdamOptimizer(FLAGS.lr,beta1=FLAGS.B1,beta2=FLAGS.B2).minimize(discriminator_loss,var_list=discriminator_parameters)
	optimizer_G = tf.train.AdamOptimizer(FLAGS.lr,beta1=FLAGS.B1,beta2=FLAGS.B2).minimize(generator_loss,var_list=generator_parameters)

	init = tf.global_variables_initializer()	



	with tf.Session() as sess:

		sess.run(init) # Initialize Variables


		coord = tf.train.Coordinator() # Set Coordinator to Manage Queue Runners
		threads = tf.train.start_queue_runners(sess, coord=coord) # Set Threads
		

#_______________________________Restore____________________________________

		saver = tf.train.Saver(max_to_keep=1000)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir="./Check_Point")

		
		try :	
			if ckpt and ckpt.model_checkpoint_path:
				print("check point path : ", ckpt.model_checkpoint_path)
				saver.restore(sess, ckpt.model_checkpoint_path)	
				print('Restored!')
		except AttributeError:
				print("No checkpoint")	

#---------------------------------------------------------------------------
		for t in range(FLAGS.iteration): # Mini-Batch Iteration Loop

			if coord.should_stop():
				break

		#___________________Get DATA________________________		

			real_image = loader.get_image_from_loader(sess) # Get Image Mini-Batch from Queue
			z_g = sess.run(generate_z())
			z_d = sess.run(generate_z())

		#---------------------------------------------------

			_, l_D = sess.run([optimizer_D, discriminator_loss], feed_dict={image : real_image, z_D : z_d})
			_, l_G = sess.run([optimizer_G, generator_loss], feed_dict={z_D : z_d, z_G : z_g})
			l_Global = sess.run(global_measure, feed_dict={image : real_image, z_D : z_d, z_G : z_g})
			
			tf.summary.scalar('Discriminator loss', l_D)
			tf.summary.scalar('Generator loss', l_G)
			tf.summary.scalar('Global_Measure', l_Global)

			print("Step : {}".format(t), "Global measure of convergence : ", l_Global, "  Generator Loss : ", l_G, "  Discriminator Loss : ", l_D) 

			k = k + FLAGS.lamb*(FLAGS.gamma*real_image_loss - generator_loss)

	#       ____________________________Save____________________________________


			merged_summary = tf.summary.merge_all() # merege summaries
			summary = sess.run(merged_summary)

			writer = tf.summary.FileWriter('./logs', sess.graph) # add the graph to the file './logs'
			
			

			if t % 200 == 0:
				writer.add_summary(summary, t)

				Generated_images = sess.run(generated_image, feed_dict={z_G : z_g})
				Decoded_Generated_images = sess.run(reconstructed_image_fake, feed_dict={z_D : z_d, z_G : z_g})
				
				save_image(Generated_images, '{}/{}.png'.format("./Generated_Images", t))
				save_image(Decoded_Generated_images, '{}/{}.png'.format("./Decoded_Generated_Images", t))



			if t % 500 == 0:
				saver.save(sess, "./Check_Point/model.ckpt", global_step = t)


	#       --------------------------------------------------------------------


		coord.request_stop()
		coord.join(threads)


			



#-----------------------------------Train Finish---------------------------------



if __name__ == "__main__" :
	tf.app.run()


