# This computes formulas similar to those computed by Tsetlin machines for a noisy XOR problem.
# With the current parameters it achieves around 70% test accuracy,
# but parameters are not carefully tuned and during experimentation
# some gave accuracies of 82%. This is much worse than Tsetlin machines
# and worse than the neural network baseline in the Tsetlin machine paper, but
# the lower of these is better than an SVM on this problem.
#
# The dataset this is intended for is that supplied with Granmo's demonstration
# implementation of Tsetlin machines, which can be found here: https://github.com/cair/TsetlinMachine

import numpy as np
import tensorflow as tf

# Model parameters
number_of_clauses = 20
final_epsilon = 0.01

# Parameters of the pattern recognition problem
number_of_features = 12
number_of_classes = 2

# Training configuration
epochs = 2000
starting_epsilon = 0.25
batch_size = 32
starting_learning_rate = 0.005
final_learning_rate = 0.001
starting_discretization_importance = 0.0
final_discretization_importance = 0.1

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

parameters = {	
    'W': tf.Variable(tf.random_uniform([number_of_classes, number_of_clauses, number_of_features], dtype=tf.float32)),
    'V': tf.Variable(tf.random_uniform([number_of_classes, number_of_clauses, number_of_features], dtype=tf.float32))
}

def discretize(params):
    dict = {
        'W': tf.round(params['W']),
        'V': tf.round(params['V'])
    }
    return dict

learning_rate = tf.placeholder(tf.float32, [])

# A DNF formula is an expression of the form \sum_i \prod_i x_i^{u_i} (1-x_i)^{v_i} where for all i u_i,v_i \in {0,1} 
# and x_i \in {0, 1} and the sum is counted modulo 2. In a Tsetlin machine such a formula is modified as follows:
# instead of counting modulo 2 we count the sums of the products directly, and instead of having only positive clauses
# we have both positive and negative clauses, so that we get an expression like
#
# I_{x>0}[\sum_i \prod_j x_j^{a_ij} (1-x_j)^{b_ij} - \sum_i \prod_x^{A_ij} (1-x_i)^{B_ij}].
#
# In order to obtain something which can be optimized using derivatives we replace x_i \in {0,1} with x_i+\epsilon
# and allow u,v,U,V to take any values instead of only 0 and 1, instead using a non-discreteness penalty to bring the
# weight close to {0, 1} before the final hard discretization.
#
# We thus compute
#
# sigmoid(\sum_{i=1}^n (-1)^i \prod_j (x_j + \epsilon)^{W_kij} (1-x_j + \epsilon)^((1 - W_kij)U_kij)) for k={1,2}
#
# where n is the number of clauses. To avoid making this needlessly computation intensive we use logarithms:
#
# sigmoid(\sum_{i=1}^n (-1)^i exp(\sum_j W_kij log(x_j + \epsilon) + \sum_j (1 - W_kij)U_kij log(1- x_j + \epsilon))).

def model_fn(x, params):
    x_transformed          = tf.log(tf.add(x, epsilon))
    x_transformed_negative = tf.log(tf.add(1-x, epsilon))
    negative_clause_sums   = tf.einsum('ijk,lk->lij', tf.multiply((1-tf.abs(params['W'])),params['V']), x_transformed_negative)
    positive_clause_sums   = tf.einsum('ijk,lk->lij', tf.abs(params['W']), x_transformed)
    clause_outputs         = tf.exp(tf.add(negative_clause_sums, positive_clause_sums))
    return tf.sigmoid(tf.einsum('j,lij->li', tf.constant(np.tile([-1.0,1.0],number_of_clauses//2).tolist()), clause_outputs))

discretization_error = tf.reduce_sum(tf.abs(parameters['W'])+tf.abs(tf.add(parameters['W'],-1))-tf.abs(tf.add(parameters['W'],-0.5)) + tf.abs(parameters['V'])+tf.abs(tf.add(parameters['V'],-1))-tf.abs(tf.add(parameters['V'],-0.5)))
underlying_cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_fn(x, parameters), labels=y))
cost = underlying_cost + discretization_error*discretization_importance

def accuracy(x, y, params):
    correct_pred = tf.equal(tf.argmax(model_fn(x, params), 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

current_epsilon = starting_epsilon
current_discretization_importance = starting_discretization_importance
current_learning_rate = starting_learning_rate
for i in range(epochs):
    if(i % 50 == 0):
        print("Epoch " + str(i) + ": Training set accuracy: "  + str(sess.run(accuracy(x, y, parameters), feed_dict={x:X_training, y:y_training, epsilon:final_epsilon})) + ".")
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

print("Undiscretized test accuracy: " + str(sess.run(accuracy(x, y, parameters), feed_dict={x:X_test, y:y_test, epsilon:final_epsilon})) + ".")
print("Discretized test accuracy: " + str(sess.run(accuracy(x, y, discretize(parameters)), feed_dict={x:X_test, y:y_test, epsilon:final_epsilon})) + ".")