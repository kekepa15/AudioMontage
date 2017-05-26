import tensorflow as tf

from Utils import FLAGS   

def main(_):
    print(FLAGS.epochs)    

if __name__ == "__main__" :
    tf.app.run()
