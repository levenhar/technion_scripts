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
    y_pred = np.dot(X, W) + b
    ### YOUR CODE HERE
    return y_pred.reshape(-1)


def perceptron(X, W, b):
    """
    Function to compute the output of a perceptron (single unit).

    Parameters:
    - X: array-like, shape = [n_features,]. Features for one sample.
    - W: array-like, shape = [n_features,]. Weight vector for the perceptron.
    - b: scalar. Bias term for the perceptron.

    Returns:
    - y_pred: int. Predicted class label (-1 or 1) for the input sample.
    """
    # Compute the linear classifier
    linear_class = linear_classifier(X, W, b)

    ### YOUR CODE HERE
    # linear_class[linear_class >= 0] = 1
    # linear_class[linear_class < 0] = -1

    linear_class = linear_class / np.abs(linear_class)
    ### YOUR CODE HERE
    
    return linear_class

def perceptron_grad(X, y, W, b):
    """
    Compute gradients for weights and bias for a perceptron model based on the
    perceptron loss function L = Theta(yhat - y), where Theta is the Heaviside step function.
    
    Parameters:
    - X: Input features (numpy array of shape (n_samples, n_features)).
    - y: True labels (numpy array of shape (n_samples,)), expected to be -1 or 1.
    - W: Current weights (numpy array of shape (n_features,)).
    - b: Current bias (scalar).
    
    Returns:
    - gradients_W: Gradients of the loss with respect to the weights.
    - gradients_b: Gradients of the loss with respect to the bias.
    """
    y_pred = (np.dot(X, W) + b)
    distances = - y * y_pred 
    
    # Initialize derivative to zero
    dw = W
    db = 0

    for i in range(len(distances)):
        if distances[i] > 0:
            dw += -y[i]*X[i]
            db += -y[i]
        else:
            dw += 0
            db += 0

    dw /= len(distances)
    db /= len(distances)
    
    ### YOUR CODE HERE
    gradients_W = dw
    # print(gradients_W)
    gradients_b = db 
    
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
    for i in range(iterations):
        ### YOUR CODE HERE
        grad_W, grad_b = compute_gradients_func(X, y, W, b)
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b
    
        ### YOUR CODE HERE

    print("Optimization Complete.")
    return W, b