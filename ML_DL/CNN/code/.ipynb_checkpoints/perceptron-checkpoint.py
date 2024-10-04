import numpy as np

def linear_classifier(X, W, b):
    """
    Function to compute a linear classifier Wx + b.

    Parameters:
    - X: array-like, shape = [n_features,]. Features for one sample.
    - W: array-like, shape = [n_features,]. Weight vector.
    - b: scalar. Bias term for the classifier.

    Returns:
    - y_pred: the result of the linear classifier.
    """
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    
    return y_pred


def perceptron(X, W, b):
    """
    Function to compute the output of a perceptron unit for binary classification with labels -1 and 1,
    using the linear_classifier function to compute Wx + b and then applying a sign function.

    Parameters:
    - X: array-like, shape = [n_samples, n_features]. Feature matrix where each row represents a sample.
    - W: array-like, shape = [n_features,]. Weight vector for the perceptron.
    - b: scalar. Bias term for the perceptron.

    Returns:
    - y_pred: array-like, shape = [n_samples,]. Predicted class label (-1 or 1) for each input sample.
    """
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    return y_pred

def perceptron_grad(X, y, W, b):
    """
    Compute gradients for weights and bias for a perceptron model based on the
    perceptron loss function L = Theta(y), where Theta is the Heaviside step function.
    
    Parameters:
    - X: Input features (numpy array of shape (n_samples, n_features)).
    - y: True labels (numpy array of shape (n_samples,)), expected to be -1 or 1.
    - W: Current weights (numpy array of shape (n_features,)).
    - b: Current bias (scalar).
    
    Returns:
    - gradients_W: Gradients of the loss with respect to the weights.
    - gradients_b: Gradients of the loss with respect to the bias.
    """
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    return gradients_W, gradients_b

def train_perceptron(W, b, X, y, compute_gradients_func, learning_rate=0.01, iterations=1000):
    """
    Optimizes W and b by running gradient descent over iterations.
    
    Parameters:
    - W: Initial weights, numpy array of shape (n_features,).
    - b: Initial bias, scalar or numpy array if multiple outputs.
    - X: Input data, numpy array of shape (n_samples, n_features).
    - y: True labels, numpy array of shape (n_samples,).
    - compute_gradients_func: Function to compute gradients of W and b.
    - learning_rate: Learning rate for the update step.
    - iterations: Number of iterations to run gradient descent.
    
    Returns:
    - W, b: Optimized weights and bias.
    """
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    return W, b