import tensorflow as tf
tf.config.list_physical_devices()


with tf.device('/GPU'):
  a = tf.random.normal(shape=(2,), dtype=tf.float32)
  b = tf.nn.relu(a)

print(a)
print(b)
