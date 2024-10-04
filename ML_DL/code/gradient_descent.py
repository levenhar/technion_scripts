import numpy as np
from hinge_loss_derivative import *

def gradient_descent(X, t, learning_rate=0.01,  iterations=1000):
    """
    Perform gradient descent to minimize the hinge loss, with a simple progress bar.
    
    Parameters:
    X: numpy array of shape (n_samples, n_features), input features.
    y: numpy array of shape (n_samples,), true labels, expected to be -1 or 1.
    learning_rate: float, the step size at each iteration.
    iterations: int, the number of iterations to run gradient descent.
    
    Returns:
    w: Optimized weights, numpy array of shape (n_features,).
    """
    w = np.zeros(X.shape[1])  # Initialize weights to zeros

    ## YOUR CODE GOES HERE
    dw = np.inf * np.ones(X.shape[1])
    TrashHold = 1**-20
    c = 0
    # while np.linalg.norm(dw) > TrashHold and c < iterations:
    for _ in range(iterations):
        dw = hinge_loss_derivative(X, t, w)
        w -= learning_rate * dw
        c+=1
        if np.linalg.norm(dw) < 0.001:
            break
    
    ## YOUR CODE ENDS HERE
            
    print("Training Complete")  # Signal that training is complete
    return w, c