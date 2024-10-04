import numpy as np

def linear_least_squares(X, y, use_bias=True):
    """
    Solve for theta (θ) using the normal equation method for linear least squares.

    Inputs:
    X: 2D numpy array (each row represents a sample, and each column represents a parameter).
    y: 1D numpy array of target values.
    use_bias: Boolean, if True adds a bias term to the model, otherwise no bias term is used.

    Returns:
    theta: 1D numpy array of coefficients (θ).
    """

    ### YOUR CODE GOES HERE
    
    if use_bias == True:
        X = np.column_stack((X,np.ones((X.shape[0],1))))
        
    theta =  np.linalg.inv(X.T@X)@(X.T@y)
        
    ### YOUR CODE ENDS HERE

    return theta
