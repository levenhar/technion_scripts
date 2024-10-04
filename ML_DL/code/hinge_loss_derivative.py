import numpy as np

def hinge_loss_derivative(X, t, w):
    """
    Compute the derivative of the hinge loss function with respect to the weights (w).
    
    This derivative indicates how a change in weights would impact the hinge loss value, 
    guiding the gradient descent optimization in adjusting weights to minimize the loss.
    
    Parameters:
    X: numpy array of shape (n_samples, n_features), input features.
    y: numpy array of shape (n_samples,), true labels, expected to be -1 or 1.
    w: numpy array of shape (n_features,), current weights of the model.
    
    Returns:
    grad: The gradient of the hinge loss with respect to the weights, numpy array of shape (n_features,).
    """
    # Compute the distance from the margin for each sample
    distances = 1 - t * np.dot(X, w)
    
    # Initialize derivative to zero
    dw = np.zeros(len(w)) 

    for i in range(len(distances)):
        if distances[i] > 0:
            dw += -t[i]*X[i]
        else:
            dw += 0

    dw /= len(distances)




    ## YOUR CODE GOES HERE
    # check = t.reshape(1,-1)@y
    #
    # if check < 1:
    #     dw = (t.reshape(1,-1)@ X).reshape(-1)
    
    ## YOUR CODE ENDS HERE
    
    return dw
