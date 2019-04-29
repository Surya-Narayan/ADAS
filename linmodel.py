import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
column_names=['y','x']
#test_names=['yt','xt']
dataset=pd.read_csv('train.csv',names=column_names)
#print(dataset)
x=dataset.x.tolist()
y=dataset.y.tolist()
n=len(x)
X=tf.placeholder(dtype="float32",name='X')
Y=tf.placeholder(dtype="float32",name='Y')
W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b") 
learning_rate = 0.001
training_epochs = 300


y_pred = tf.add(tf.multiply(X, W), b) 
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n) 
total_error = tf.reduce_sum(tf.square(Y - tf.reduce_mean(Y)))
unexplained_error = tf.reduce_sum(tf.square(Y - y_pred))
R_squared = (1 - tf.div(unexplained_error, total_error))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
init = tf.global_variables_initializer() 

# Starting the Tensorflow Session 
with tf.Session() as sess: 
	sess.run(init)  
	for epoch in range(training_epochs): 
		# Feeding each data point into the optimizer using Feed Dictionary 
		for (xt, yt) in zip(x, y): 
			sess.run(optimizer, feed_dict = {X : xt, Y : yt}) 
		# Displaying the result after every epochs 
		if (epoch + 1) % 1 == 0: 
			# Calculating the cost after every epoch 
			c = sess.run(cost, feed_dict = {X : x, Y : y}) 
			R_sq=sess.run(R_squared, feed_dict = {X : x, Y : y}) 
			print("Epoch", (epoch + 1), ": total_cost =", c,"R-SQuared=",R_sq, "W =", sess.run(W), "b =", sess.run(b)) 
	 
	training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
	weight = sess.run(W) 
	bias = sess.run(b) 

test_names=['y_t','x_t']
tdataset=pd.read_csv('test.csv',names=test_names)
x_t=tdataset.x_t.tolist()
x_t=np.array(x_t)
y_t=tdataset.y_t.tolist()

plt.scatter(x,y)
# plt.scatter(x_t,y_t)
print("Y=",weight,"*X+(",bias,")")
x=np.array(x)
y1=weight*x+bias
plt.plot(x,y1)
xpredtmp=x[n-1]+1
plt.show()

# tdataset=pd.read_csv('test.csv',names=test_names)
# xt=dataset.x.tolist()
# xt=np.array(xt)
# yt=dataset.y.tolist()
# yt=np.array(yt)
#print(xt,"\n",yt)
