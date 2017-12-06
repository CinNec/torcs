import numpy as np
import random
import tensorflow as tf

from Normalize import Normalize
import pickle

Ndata = Normalize()
#
with tf.Graph().as_default() as accbrk_graph:
  saver1 = tf.train.import_meta_graph("./model_accbrk/model_accbrk.meta")
sess1 = tf.Session(graph=accbrk_graph)
saver1.restore(sess1,'./model_accbrk/model_accbrk')

with tf.Graph().as_default() as steer_graph:
  saver2 = tf.train.import_meta_graph("./model_steer/model_steer.meta")
sess2 = tf.Session(graph=steer_graph)
saver2.restore(sess2,'./model_steer/model_steer')

nn_input = np.array([0,	1,	4.024009678671506274e-01,	5.505602450838025658e-02,	1,	2.209881655715679322e-02,	1.243855721393034852e-02,	1.328388059701492672e-02,	1.469303482587064683e-02,	1.716248756218905630e-02,	2.231218905472636890e-02,	2.136940298507462588e-01,	1.706293532338308550e-01,	1.393412935323383173e-01,	1.146248756218905540e-01,	9.614975124378109805e-02,	1.544768429554125855e-01,	4.367189537834055835e-01,	5.277345734187532250e-02,	4.464304204231009376e-01,	4.461021254342385500e-01])


i=0
while(i <= 20):
    nn_input[i] = (nn_input[i] - Ndata.minarray[i])/(Ndata.maxarray[i]-Ndata.minarray[i])
    i += 1

nn_input.shape = (1, 1, nn_input.shape[0])



#saver = tf.train.import_meta_graph("./model_accbrk/model_accbrk.meta")
#
##with tf.Session() as sess:
##    saver.restore(sess, tf.train.latest_checkpoint("./"))
##    output = sess.run("prediction", feed_dict={"x:0": tf.cast(input, tf.float32)})
##    print("output:", output)
###        result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 3.3})
#
#with tf.Session() as sess:
#    saver.restore(sess,'./model_accbrk/model_accbrk')
#    output = sess.run("prediction:0", feed_dict={"x:0": inputs})
##    output = sess.run("prediction", feed_dict={"x:0": tf.cast(input, tf.float32)})
#    print("output:", output)
##    print (sess.run(tf.get_default_graph().get_tensor_by_name('x:0')))
#    # [ 0.43350926  1.02784836]
##    print(tf.global_variables()) # print tensor variables
##    # [<tf.Variable 'w1:0' shape=(2,) dtype=float32_ref>,
##    #  <tf.Variable 'w2:0' shape=(5,) dtype=float32_ref>]
###    for op in tf.get_default_graph().get_operations():
###        print (str(op.name)) # print all the operation nodes' name
##
##    g = tf.get_default_graph()
##
##    x = g.get_tensor_by_name("x:0")
#    
#    
##with tf.Session(graph=tf.Graph()) as sess:
##    tf.saved_model.loader.load(sess, [tag_constants.TRAINING], './SavedModel/')
#    print(tf.global_variables())


#with tf.Session() as sess:
#    saver1.restore(sess,'./model_accbrk/model_accbrk')
accbrk = sess1.run("accbrk:0", feed_dict={"x_accbrk:0": nn_input})

#with tf.Session() as sess2:
#    saver2.restore(sess2,'./model_steer/model_steer')
steer = sess2.run("steer:0", feed_dict={"x_steer:0": nn_input})

#%%
print("accbrk:", accbrk)
accbrk = np.round(accbrk)
print("acc:", accbrk[0, 0])
print("brk:", accbrk[0, 1])
print("steer:", steer)
#%%