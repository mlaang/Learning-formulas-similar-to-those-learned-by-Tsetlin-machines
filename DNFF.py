import numpy as np
import tensorflow as tf

def non_discreteness(X):
    return tf.reduce_sum(tf.abs(X) + tf.abs(X-1) - tf.abs(X-0.5))
	
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
# sigmoid(\sum_{i=1}^n (-1)^i \prod_j (x_j + \epsilon)^{W_kij U_kij} (1-x_j + \epsilon)^((1 - W_kij)V_kij)) for k={1,2}
#
# where n is the number of clauses. To avoid making this needlessly computation intensive we use logarithms:
#
# sigmoid(\sum_{i=1}^n (-1)^i exp(\sum_j W_kij U_kij log(x_j + \epsilon) + \sum_j (1 - W_kij)V_kij log(1- x_j + \epsilon))).

class Relaxed_DNFF:
    def __init__(self, number_of_classes, number_of_clauses, number_of_features, W = None, V = None, U = None, trainable = False):
        self.number_of_classes  = number_of_classes
        self.number_of_clauses  = number_of_clauses
        self.number_of_features = number_of_features
        if(W == None):
            self.W = tf.Variable(tf.random_uniform([number_of_classes, number_of_clauses, number_of_features], dtype=tf.float32))
            self.V = tf.Variable(tf.random_uniform([number_of_classes, number_of_clauses, number_of_features], dtype=tf.float32))
            self.U = tf.Variable(tf.random_uniform([number_of_classes, number_of_clauses, number_of_features], dtype=tf.float32))
        elif(trainable == False):
            self.W = W
            self.V = V
            self.U = U
        else:
            print("Error: Unimplemented!")

    def model_function(self, x, epsilon):
        x_transformed          = tf.log(x + epsilon)
        x_transformed_negative = tf.log(1 - x + epsilon)
        negative_clause_sums   = tf.einsum('ijk,lk->lij', tf.multiply((1-tf.abs(self.W)),tf.abs(self.V)), x_transformed_negative)
        positive_clause_sums   = tf.einsum('ijk,lk->lij', tf.multiply(tf.abs(self.W),tf.abs(self.U)), x_transformed)
        clause_outputs         = tf.exp(tf.add(negative_clause_sums, positive_clause_sums))
        return tf.sigmoid(tf.einsum('j,lij->li', tf.constant(np.tile([-1.0,1.0],self.number_of_clauses//2).tolist()), clause_outputs))

    def non_discreteness(self):
        return non_discreteness(self.W) + 0.5*non_discreteness(self.U) + 0.5*non_discreteness(self.U)

    def discretized_formula(self):
        return Relaxed_DNFF(
            self.number_of_classes,
            self.number_of_clauses,
            self.number_of_features,
			tf.round(self.W),
			tf.round(self.V),
			tf.round(self.U)
        )