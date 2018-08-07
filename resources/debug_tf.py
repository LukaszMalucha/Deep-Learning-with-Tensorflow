import tensorflow as tf
print(tf.__version__)



def some_method(data):
  a = data[:,0:2]
  c = data[:,1]
  s = (a + c)
  return tf.sqrt(tf.matmul(s, tf.transpose(s)))

with tf.Session() as sess:
  fake_data = tf.constant([
      [5.0, 3.0, 7.1],
      [2.3, 4.1, 4.8],
      [2.8, 4.2, 5.6],
      [2.9, 8.3, 7.3]
    ])
  print sess.run(some_method(fake_data))




def some_method(data):
  a = data[:,0:2]
  print a.get_shape()
  c = data[:,1]
  print c.get_shape()
  s = (a + c)
  return tf.sqrt(tf.matmul(s, tf.transpose(s)))

with tf.Session() as sess:
  fake_data = tf.constant([
      [5.0, 3.0, 7.1],
      [2.3, 4.1, 4.8],
      [2.8, 4.2, 5.6],
      [2.9, 8.3, 7.3]
    ])
  print sess.run(some_method(fake_data))


def some_method(data):
  a = data[:,0:2]
  print a.get_shape()
  c = data[:,1:3]
  print c.get_shape()
  s = (a + c)
  return tf.sqrt(tf.matmul(s, tf.transpose(s)))

with tf.Session() as sess:
  fake_data = tf.constant([
      [5.0, 3.0, 7.1],
      [2.3, 4.1, 4.8],
      [2.8, 4.2, 5.6],
      [2.9, 8.3, 7.3]
    ])
  print sess.run(some_method(fake_data))



import tensorflow as tf

x = tf.constant([[3, 2],
                 [4, 5],
                 [6, 7]])
print "x.shape", x.shape
expanded = tf.expand_dims(x, 1)
print "expanded.shape", expanded.shape
sliced = tf.slice(x, [0, 1], [2, 1])
print "sliced.shape", sliced.shape

with tf.Session() as sess:
  print "expanded: ", expanded.eval()
  print "sliced: ", sliced.eval()
  
  
  
###   Vector vs scalar

