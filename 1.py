import tensorflow as tf
import tflearn

hello = tf.constant('Hello, Tensorflow')
sess = tf.Session()
print(sess.run(hello))

