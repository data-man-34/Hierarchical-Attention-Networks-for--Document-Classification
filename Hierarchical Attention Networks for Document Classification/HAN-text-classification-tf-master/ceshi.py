import collections
import numpy as np
import tensorflow as tf
# s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
#
# d = collections.defaultdict(list)
# for k, v in s:
#     d[k].append(v)
# print (d)
# print (type(d))
# list(d.items())

# L = [('b',2),('a',1),('c',3),('d',4)]
# print (sorted(L, key=lambda x:-x[1]))

# a = 3
# b= [0] * 2
# print (b)
# c =[[]]
# a = [np.array([1,2,3]),np.array([4,5,6])]
# print (a[0].shape[0])
# x= ((0,3),(0,0))
# a= np.pad (a,x, 'constant')
# print (a.shape)
# print (type(a))

a =np.array([ [[1,2,13],
     [3,4,14]],
     [[5,6,15],
     [7,8,16]],
     [[9,10,17],
     [11,12,18]]])
# b = [2,2,2]
# print (a.shape)
# print (np.multiply(a, b))
# print (len(a))
#
# a = [[1,2],
#      [3,4]]
# b= [2,3]
# c = np.multiply(a, b)
# print (c)

# x = np.array([[[1],[4]],[[7],[10]]],dtype = 'float32')
# y = np.array([[1,2],
#               [3,4]])
# a = tf.reduce_sum(x, axis=1, keep_dims=True)
# b = tf.nn.softmax(x, dim = 1)
# with tf.Session() as sess:
#      print (sess.run(b))


# a = 2
# # b = np.sum(a)
# # print (b)
# print (pow(a,2))

# for i in range(0, 200000, 64):
#     print (i)
a = tf.reduce_max(a,reduction_indices=2)
with tf.Session() as sess:
      print (sess.run(a))

# a = tf.truncated_normal([10])
# print (type(a))
