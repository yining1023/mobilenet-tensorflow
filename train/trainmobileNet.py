# setup path
import sys
sys.path.append('../models/research/slim')

import tensorflow as tf
from nets.mobilenet import mobilenet_v2
from datasets import imagenet

tf.reset_default_graph()

# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

# Note: arg_scope is optional for inference.
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
  logits, endpoints = mobilenet_v2.mobilenet(images)
  
# Restore using exponential moving average since it produces (1.5-2%) higher 
# accuracy
ema = tf.train.ExponentialMovingAverage(0.999)
vars = ema.variables_to_restore()

saver = tf.train.Saver(vars)

with tf.Session() as sess:
  saver.restore(sess,  'mobilenet_v2_1.0_224.ckpt')
  x = endpoints['Predictions'].eval(feed_dict={file_input: 'cat.jpg'})
  print('x', x)
  y = endpoints['Predictions'].eval(feed_dict={file_input: 'panda.jpg'})
  print('y', y)
label_map = imagenet.create_readable_names_for_imagenet_labels()  
print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())
label_map2 = imagenet.create_readable_names_for_imagenet_labels()  
print("Top 1 prediction2: ", y.argmax(),label_map[y.argmax()], y.max())
