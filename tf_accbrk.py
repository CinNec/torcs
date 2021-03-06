import numpy as np
import random
import tensorflow as tf

from Normalize_clean import Normalize
import pickle

from tensorflow.python.ops import rnn, rnn_cell


#%%

def generate_sequence(data, time_steps, pos):
    pos += 1
    sequence = data[pos - time_steps : pos]
    return sequence

def generate_batch(data, batch_size, time_steps, position):
    batch = []
    start = position
    while position < start + batch_size:
        sequence = generate_sequence(data, time_steps, position)
        batch.append(sequence)
        position += 1
        
    batch = np.array(batch)
    x = batch[:, :, :INPUT_SIZE]
    y = batch[:, time_steps - 1,  OUTPUT]
    return x, y, position


#%%

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([RNN_HIDDEN, OUTPUT_SIZE])),
             'biases':tf.Variable(tf.random_normal([OUTPUT_SIZE]))}
    
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, INPUT_SIZE])
    x = tf.split(x, TIME_STEPS, 0)

    lstm_cell = rnn_cell.LSTMCell(RNN_HIDDEN,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    sigmoid_prediction = tf.nn.sigmoid(prediction, name="accbrk")
    error = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)         
    correct = tf.equal(tf.round(tf.nn.sigmoid(prediction)), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            epoch_error = 0
            accura = 0
            iterations_per_epoch = 0
            POSITION = TIME_STEPS - 1
            while POSITION - 1 + BATCH_SIZE < len(data):
                epoch_x, epoch_y, POSITION = generate_batch(data, BATCH_SIZE, TIME_STEPS, position=POSITION)
                _, cost, accu, pred, cor = sess.run([optimizer, error, accuracy, sigmoid_prediction, correct], feed_dict={x: epoch_x, y: epoch_y})
                epoch_error += cost
                accura += accu
                iterations_per_epoch += 1

            if epoch % 10 == 0: 
                print('Epoch', epoch, 'completed out of',EPOCHS,'error:',epoch_error / iterations_per_epoch, "accuracy:", accura / iterations_per_epoch)
                accu = sess.run([accuracy], feed_dict={x: X, y: Y})
                print("Training accuracy:", accu)
                
            if epoch % 10001 == 0:
                saver.save(sess, './model_accbrk/model_accbrk')
                accu = sess.run([accuracy], feed_dict={x: X, y: Y})
                print("Training accuracy:", accu)
                with open('./model_accbrk/stats.txt', "w+") as file:
                    print("Input size:", INPUT_SIZE, '\nEpoch:', epoch, "\nBatch size:", BATCH_SIZE, "\nLayer size:", RNN_HIDDEN, "\nLearning rate:", LEARNING_RATE, "\naccuracy:", accu, "\n", file=file)
                    
            if epoch % 12001 == 0:
                saver.save(sess, './model_accbrk/model_accbrk')
                accu = sess.run([accuracy], feed_dict={x: X, y: Y})
                print("Training accuracy:", accu)
                with open('./model_accbrk/stats.txt', "w+") as file:
                    print("Input size:", INPUT_SIZE, '\nEpoch:', epoch, "\nBatch size:", BATCH_SIZE, "\nLayer size:", RNN_HIDDEN, "\nLearning rate:", LEARNING_RATE, "\naccuracy:", accu, "\n", file=file)
            
            if epoch % 14001 == 0:
                saver.save(sess, './model_accbrk/model_accbrk')
                accu = sess.run([accuracy], feed_dict={x: X, y: Y})
                print("Training accuracy:", accu)
                with open('./model_accbrk/stats.txt', "w+") as file:
                    print("Input size:", INPUT_SIZE, '\nEpoch:', epoch, "\nBatch size:", BATCH_SIZE, "\nLayer size:", RNN_HIDDEN, "\nLearning rate:", LEARNING_RATE, "\naccuracy:", accu, "\n", file=file)
            
            if epoch % 16001 == 0:
                saver.save(sess, './model_accbrk/model_accbrk')
                accu = sess.run([accuracy], feed_dict={x: X, y: Y})
                print("Training accuracy:", accu)
                with open('./model_accbrk/stats.txt', "w+") as file:
                    print("Input size:", INPUT_SIZE, '\nEpoch:', epoch, "\nBatch size:", BATCH_SIZE, "\nLayer size:", RNN_HIDDEN, "\nLearning rate:", LEARNING_RATE, "\naccuracy:", accu, "\n", file=file)
            
        saver.save(sess, './model_accbrk/model_accbrk')
        
        accu = sess.run([accuracy], feed_dict={x: X, y: Y})
        print("Training accuracy:", accu)

#%%

# Initialize model parameters

INPUT_SIZE    = 22
OUTPUT_SIZE   = 2
OUTPUT = [22, 23]
RNN_HIDDEN    = 192
#RNN_HIDDEN    = [50, 50]
LEARNING_RATE = 0.002

EPOCHS = 16000
BATCH_SIZE = 2064
TIME_STEPS = 1
POSITION = TIME_STEPS - 1 # Should be 1 less than timesteps

x = tf.placeholder(tf.float32, (None, None, INPUT_SIZE), name="x_accbrk")  # (batch, time, in)
y = tf.placeholder(tf.float32, (None, OUTPUT_SIZE), name="y") # (batch, time, out)
#%%

# Initialize data and train and test sets
Ndata = Normalize()
data = Ndata.data

X = data[:, :22]
X.shape = (X.shape[0], 1, X.shape[1])
Y = data[:, [22, 23]]

#%%
train_neural_network(x)