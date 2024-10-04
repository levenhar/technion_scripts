import numpy as np

def mse_loss(y_pred, y_true):
    """Computes the mean squared error loss."""
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    return mse_loss

def relu(x):
    """Applies the ReLU activation function."""
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    return relu_activation

def relu_derivative(x):
    """Computes the derivative of the ReLU activation function."""
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    return relu_grad

def sigmoid(z):
    """Compute the sigmoid of z."""    
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    return sigmoid_activation

def binary_cross_entropy_loss(y_true, y_pred):
    '''
    Compute the binary cross entropy loss function
    Note: this is implemented since we did not see it in class
    '''
    epsilon = 1e-9  # To prevent log(0)

    loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return loss

def forward_pass(X, W1, b1, W2, b2, W3, b3, W4, b4):
    """
    Performs the forward pass through a 3-hidden-layer neural network.
    
    Parameters:
    - X: Input data, numpy array of shape (n_samples, n_features).
    - W1, W2, W3, W4: Weight matrices for each layer.
    - b1, b2, b3, b4: Bias vectors for each layer.
    
    Returns:
    - A tuple containing the activations and network output
    """

    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    
    return h1, h2, h3, output

def backward_pass(X, y_true, 
                  W1, W2, W3, W4, b1, b2, b3, b4, h1, h2, h3, 
                  output, loss_type = 'mse'):
    """
    Performs the backward pass through a 3-hidden-layer neural network.
    
    Parameters:
    - X, y_true: Input data and true labels.
    - W1, W2, W3, W4, b1, b2, b3, b4: Current weights and biases.
    - h1, h2, h3: Activations from each of the hidden layers.
    - output: The final output of the network.
    
    Returns:
    - Gradients for each weight and bias in the network.
    """
    n_samples = X.shape[0]
    
    # Gradient of loss w.r.t. output
    if loss_type == 'mse':
        ### YOUR CODE HERE
    
        ### YOUR CODE HERE
    if loss_type == 'bce': # note: this is implemented since we did not see it in class
        dLoss_dOutput = (output - y_true) / n_samples

    # Output layer gradients (follow this example to complete the rest)
    dOutput_dW4 = h3.T
    dW4 = np.dot(dOutput_dW4, dLoss_dOutput)
    db4 = np.sum(dLoss_dOutput, axis=0)
    
    # Backpropagate through ReLU and layer 3
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    
    # Layer 2
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    
    # Layer 1
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    
    return dW1, db1, dW2, db2, dW3, db3, dW4, db4

def train_network(X, y_true, 
                  W1, b1, W2, b2, W3, b3, W4, b4, 
                  loss_type = 'mse', 
                  learning_rate=0.01, 
                  iterations=1000, 
                  X_test = [], y_test = []):
    """
    Trains a neural network with three hidden layers using gradient descent.
    
    Parameters:
    - X: Input data, a numpy array of shape (n_samples, n_features).
    - y: True labels, a numpy array of shape (n_samples,) or (n_samples, n_outputs).
    - W1, W2, W3, W4: Weight matrices for each layer of the network.
    - b1, b2, b3, b4: Bias vectors for each layer of the network.
    - learning_rate: The learning rate for gradient descent.
    - iterations: The number of iterations to run the training loop.
    - Optional: X_test and y_test to compute test loss
    
    Returns:
    - The updated weight matrices and bias vectors after training.
    """
    train_loss_lst = list()
    test_loss_lst = list()
    
    for i in range(iterations):
        h1, h2, h3, output = forward_pass(X, W1, b1, W2, b2, W3, b3, W4, b4)

        if loss_type == 'mse':
            train_loss = mse_loss(output, y_true)
            train_loss_lst.append(train_loss)
            
            if np.size(X_test) > 1:
                _, _, _, output_test = forward_pass(X_test, W1, b1, W2, b2, W3, b3, W4, b4)
                test_loss = mse_loss(y_test, output_test)
                test_loss_lst.append(test_loss)
        
        elif loss_type == 'bce':
            train_loss = binary_cross_entropy_loss(y_true, output)
            train_loss_lst.append(train_loss)
            
            if np.size(X_test) > 1:
                _, _, _, output_test = forward_pass(X_test, W1, b1, W2, b2, W3, b3, W4, b4)
                test_loss = binary_cross_entropy_loss(y_test, output_test)
                test_loss_lst.append(test_loss)
        
        dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backward_pass(X, y_true, W1, W2, W3, W4, b1, b2, b3, b4, h1, h2, h3, output)
        
        # Update weights and biases using the gradients computed during the backward pass.
        ### YOUR CODE HERE
        
        ### YOUR CODE HERE
        
        # Optional: Print the loss at regular intervals to monitor training progress.
        if i % 100 == 0:
            print(f"Iteration {i}: Train Loss = {train_loss:.4f}")
            if np.size(X_test) > 1:
                print(f"Iteration {i}: Test Loss = {test_loss:.4f}")
    
    # Return the updated parameters after training.
    return W1, b1, W2, b2, W3, b3, W4, b4, np.array(train_loss_lst), np.array(test_loss_lst)
