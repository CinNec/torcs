import numpy as np
import random
import tensorflow as tf

from Normalize import Normalize
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
    y = batch[:, time_steps - 1,  23]
    y.shape = (len(y), 1)
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
    identity_prediction = tf.identity(prediction, name="steer")
    error = tf.reduce_mean(tf.losses.mean_squared_error(predictions=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
    accuracy = tf.reduce_mean(tf.abs(prediction - y))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            np.random.shuffle(data)
            epoch_error = 0
            accura = 0
            iterations_per_epoch = 0
            POSITION = TIME_STEPS - 1
            while POSITION - 1 + BATCH_SIZE < len(data):
                epoch_x, epoch_y, POSITION = generate_batch(data, BATCH_SIZE, TIME_STEPS, position=POSITION)
                _, cost, accu = sess.run([optimizer, error, accuracy], feed_dict={x: epoch_x, y: epoch_y})
                epoch_error += cost
                accura += accu
                iterations_per_epoch += 1

            
            if epoch % 10 == 0: 
                print('Epoch', epoch, 'completed out of',EPOCHS,'error:',epoch_error / iterations_per_epoch, "accuracy:", accura / iterations_per_epoch)
                
            if epoch % 500 == 0:
                saver.save(sess, './model_steer/model_steer')
                accu = sess.run([accuracy], feed_dict={x: X, y: Y})
                print("Training accuracy:", accu)
                with open('./model_steer/stats.txt', "w+") as file:
                    print("Input size:", INPUT_SIZE, '\nEpoch:', epoch, "\nBatch size:", BATCH_SIZE, "\nLayer size:", RNN_HIDDEN, "\nLearning rate:", LEARNING_RATE, "\naccuracy:", accu, "\n", file=file)
        
        saver.save(sess, './model_steer/model_steer')
        
        accu = sess.run([accuracy], feed_dict={x: X, y: Y})
        print("Training accuracy:", accu)
#                correct = tf.equal(tf.round(prediction), y)
#                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#                print('prediction:', prediction.eval({x:x_small, y:y_small}))
#                print('tf.round(prediction):', tf.round(prediction).eval({x:x_small, y:y_small}))
#                print('correct:',  correct.eval({x:x_small, y:y_small}))
#                print('y:', y.eval({x:x_small, y:y_small}))
#                print('Accuracy on test set:', accuracy.eval({x:x_small, y:y_small}))


#%%

# Initialize model parameters

INPUT_SIZE    = 21
OUTPUT_SIZE   = 1 
RNN_HIDDEN    = 128
#RNN_HIDDEN    = [50, 50]
LEARNING_RATE = 0.001

EPOCHS = 10000
BATCH_SIZE = 2048
TIME_STEPS = 1
POSITION = TIME_STEPS - 1 # Should be 1 less than timesteps

x = tf.placeholder(tf.float32, (None, None, INPUT_SIZE), name="x_steer")  # (batch, time, in)
y = tf.placeholder(tf.float32, (None, OUTPUT_SIZE)) # (batch, time, out)
#%%

# Initialize data and train and test sets
Ndata = Normalize()
data = Ndata.data

X = data[:, :21]
X.shape = (X.shape[0], 1, X.shape[1])
Y = data[:, 23]
Y.shape = (Y.shape[0], 1)

#%%
train_neural_network(x)

#%