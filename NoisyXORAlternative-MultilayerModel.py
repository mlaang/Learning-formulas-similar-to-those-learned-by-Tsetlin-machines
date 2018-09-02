# This computes a multilayer version of formulas similar to those computed by Tsetlin machines for a
# noisy XOR problem. With the current parameters it achieves an average test accuracy around 90%.
# This is much worse than  Tsetlin machines.
#
# The dataset this is intended for is that supplied with Granmo's demonstration
# implementation of Tsetlin machines, which can be found here: https://github.com/cair/TsetlinMachine

import numpy as np
import tensorflow as tf
from DNFF import Relaxed_DNFF

# Model parameters
number_of_clauses = 10
number_of_hidden_variables = 5
final_epsilon = 0.01

# Parameters of the pattern recognition problem
number_of_features = 12
number_of_classes = 2

# Training configuration
starting_epsilon = 0.25
epochs = 300#2000
batch_size = 32
starting_learning_rate = 0.005	
final_learning_rate = 0.001
starting_discretization_importance = 0.0
final_discretization_importance = 0.2

# Loading of training and test data
training_data = np.loadtxt("NoisyXORTrainingData.txt").astype(dtype=np.int32)
test_data = np.loadtxt("NoisyXORTestData.txt").astype(dtype=np.int32)

X_training = training_data[:,0:12] # Input features
y_training = np.transpose([training_data[:,12], 1-training_data[:,12]]) # Target value

X_test = test_data[:,0:12] # Input features
y_test = np.transpose([test_data[:,12], 1-test_data[:,12]]) # Target value-

x = tf.placeholder(tf.float32, [None, number_of_features])
y = tf.placeholder(tf.float32, [None, number_of_classes])
epsilon = tf.placeholder(tf.float32, [])
discretization_importance = tf.placeholder(tf.float32, [])

first_layer  = Relaxed_DNFF(number_of_hidden_variables, number_of_clauses, number_of_features)
second_layer = Relaxed_DNFF(number_of_classes, number_of_clauses, number_of_hidden_variables)

def model_fn(x):
    first_layer_output = first_layer.model_function(x, epsilon)
    return second_layer.model_function(first_layer_output, epsilon/2)
	
def discretized_model_fn(x):
    first_layer_output = first_layer.discretized_formula().model_function(x, epsilon)
    return second_layer.discretized_formula().model_function(first_layer_output, epsilon)

learning_rate = tf.placeholder(tf.float32, [])

discretization_error = first_layer.non_discreteness() + second_layer.non_discreteness()
underlying_cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_fn(x), labels=y))
cost = underlying_cost + discretization_error*discretization_importance

def accuracy(x, y, model_fn):
    correct_pred = tf.equal(tf.argmax(model_fn(x), 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.MomentumOptimizer(learning_rate,0.90).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

current_epsilon = starting_epsilon
current_discretization_importance = starting_discretization_importance
current_learning_rate = starting_learning_rate
for i in range(epochs):
    if(i % 50 == 0):
        print("Epoch " + str(i) + ": Training set accuracy: "  + str(sess.run(accuracy(x, y, model_fn), feed_dict={x:X_training, y:y_training, epsilon:final_epsilon})) + ".")
    perm = np.arange(X_training.shape[0])
    np.random.shuffle(perm)
    X_training = X_training[perm]
    y_training = y_training[perm]
    index_in_epoch = 0
    while(index_in_epoch + batch_size <= X_training.shape[0]):
        start = index_in_epoch
        index_in_epoch += batch_size
        end = index_in_epoch
        sess.run(optimizer, feed_dict={x:X_training[start:end], y:y_training[start:end], epsilon:current_epsilon, discretization_importance:current_discretization_importance, learning_rate:current_learning_rate})
    current_epsilon -= (starting_epsilon-final_epsilon)/epochs
    current_discretization_importance -= (starting_discretization_importance - final_discretization_importance)/epochs
    current_learning_rate -= (starting_learning_rate - final_learning_rate)/epochs

print("Undiscretized test accuracy: " + str(sess.run(accuracy(x, y, model_fn), feed_dict={x:X_test, y:y_test, epsilon:final_epsilon})) + ".")
print("Discretized test accuracy: " + str(sess.run(accuracy(x, y, discretized_model_fn), feed_dict={x:X_test, y:y_test, epsilon:final_epsilon})) + ".")