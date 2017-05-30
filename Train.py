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
	n = FLAGS.hidden_n
	"""
	Run main function
	"""



	#___________________________________________Layer info_____________________________________________________


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
									[1, 1], \
									[1, 1], \
									[1, 1], \
									[1, 1], \
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

	E = Encoder("Encoder", Encoder_infos)
	D = Decoder("Decoder", Decoder_infos)
	G = Decoder("Generator", Generator_infos)

	#Generator
	z_G = tf.placeholder(tf.float32, shape=(FLAGS.bn, 1, FLAGS.hidden_n, 1), name = "z_G") #get embedding vector z_G
	z_D = tf.placeholder(tf.float32, shape=(FLAGS.bn, 1, FLAGS.hidden_n, 1), name = "z_D") #get embedding vector z_G

	generated_image = G.decode(z_G)
	generated_image_for_disc = G.decode(z_D, reuse = True)

	#Discriminator (Auto-Encoder)
	image = tf.placeholder(tf.float32, shape=(FLAGS.bn, scale_size[0], scale_size[1], 1), name = "Real_Image") #get embedding vector z_G

	embedding_vector_real = E.encode(image)
	reconstructed_image_real = D.decode(embedding_vector_real)

	embedding_vector_fake = E.encode(generated_image_for_disc)
	reconstructed_image_fake = D.decode(embedding_vector_fake)


	#-----------------------------------------------------------------------------------------------







	#____________________________________________Loss_______________________________________________


	"""
	Define Loss
	"""
	real_image_loss = get_loss(real_image, reconstructed_image_real)
	generator_loss_for_disc = get_loss(generated_image_for_disc, reconstructed_image_fake)
	discriminator_loss = real_image_loss - k*generator_loss_for_disc
	generator_loss = get_loss(generated_image, reconstructed_image_fake)
	global_measure = real_image_loss + tf.abs(FLAGS.gamma*real_image_loss - generator_loss)


	#-----------------------------------------------------------------------------------------------







	#_____________________________________________Train_______________________________________________

	discriminator_parameters = []
	generator_parameters = []

	for v in tf.trainable_variables():
		if 'Encoder' or 'Decoder' in v.name:
			discriminator_parameters.append(v)
			print("Discriminator parameter : ", v.name)
		if 'Generator' in v.name:
			generator_parameters.append(v)
			print("Generator parameter : ", v.name)


	optimizer_D = tf.train.AdamOptimizer(FLAGS.lr,beta1=FLAGS.B1,beta2=FLAGS.B2).minimize(discriminator_loss,var_list=discriminator_parameters)
	optimizer_G = tf.train.AdamOptimizer(FLAGS.lr,beta1=FLAGS.B1,beta2=FLAGS.B2).minimize(generator_loss,var_list=generator_parameters)

	init = tf.global_variables_initializer()	



	with tf.session() as sess:
		sess.run(init)
		saver = tf.train.Saver(max_to_keep=1000)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess, coord=coord)
		
		real_image = loader.get_image_from_loader(sess) #get image mini-batch
		

		k = 0 # init value of k_t

		for t in range(FLAGS.iteration):
			if coord.should_stop():
				break

			z_G = generate_z()
			z_D = generate_z()	

			_, l_d = sess.run([optimizer_D, discriminator_loss], feed_dict={image : real_image, z_D : z_D})
			_, l_g = sess.run([optimizer_G, generator_loss], feed_dict={z_G : z_G, z_D : z_D})
			l_global = sess.run(global_measure)
			
			tf.summary.scalar('Generator loss', l_g)
			tf.summary.scalar('Discriminator loss', l_d)
			tf.summary.scalar('Global_Measure', l_global)

			print("Global measure of convergence : ", l_global, "  Generator Loss : ", l_g, "  Discriminator Loss : ", l_d) 

			k = k + FLAGS.lamb*(FLAGS.gamma*real_image_loss - generator_loss)

	#       ____________________________Save____________________________________


			summary = tf.summary.merge_all() # merege summaries    
			writer = tf.summary.FileWriter('./logs', sess.graph) # add the graph to the file './logs'
			
			

			if t % 200 == 0:
				writer.add_summary(summary_run, step)

				Generated_images = self.sess.run(generated_image)
				Decoded_Generated_images = self.sess.run(reconstructed_image_fake)
				
				save_image(Generated_images, '{}/{}.png'.format("./Generated_Images", t))
				save_image(Decoded_Generated_images, '{}/{}.png'.format("./Decoded_Generated_Images", t))



			if t % 500 == 0:
				saver.save(self.sess, "./Check_Point/model.ckpt", global_step = step)


	#       --------------------------------------------------------------------


		coord.request_stop()
		coord.join(threads)


			



#-----------------------------------Train Finish---------------------------------



if __name__ == "__main__" :
    tf.app.run()





