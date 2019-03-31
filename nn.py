import tensorflow as tf
import numpy as np
import pandas as pd

learning_rate = 0.5
epochs = 10
batch_size = 31

column_names=['y','x']
#test_names=['yt','xt']
dataset=pd.read_csv('train.csv',names=column_names)
#print(dataset)
xd=dataset.x.tolist()
yd=dataset.y.tolist()
xd=np.array(xd)
yd=np.array([0,1])
yd=yd.reshape(1,2)
xd=xd.reshape(1,31)
print(xd.shape)
print(xd,yd)

x = tf.placeholder(tf.float32, [1, 31])
y = tf.placeholder(tf.float32, [1, 2])

#weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([31, 20], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([20]), name='b1')
#weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([20, 2], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([2]), name='b2')

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = 31
   for epoch in range(epochs):
        avg_cost = 0
        #for (q,w) in zip(xd,yd):
            #batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
        _, c = sess.run([optimiser, cross_entropy], 
        feed_dict={x: xd, y: yd})
        avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    #print(sess.run(accuracy, feed_dict={x: xd, y: yd}))

