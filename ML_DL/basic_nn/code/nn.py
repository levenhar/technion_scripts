import numpy as np

def mse_loss(y_pred, y_true):
    """Computes the mean squared error loss."""
    ### YOUR CODE HERE
    mse_loss = np.mean((y_pred - y_true)**2)
    ### YOUR CODE HERE
    return mse_loss

def relu(x):
    """Applies the ReLU activation function."""
    ### YOUR CODE HERE
    relu_activation = np.copy(x)
    relu_activation[relu_activation < 0] = 0
    ### YOUR CODE HERE
    return relu_activation

def relu_derivative(x):
    """Computes the derivative of the ReLU activation function."""
    ### YOUR CODE HERE
    relu_grad = np.zeros_like(x)
    relu_grad[x>0] = 1
    ### YOUR CODE HERE
    return relu_grad

def sigmoid(z):
    """Compute the sigmoid of z."""    
    ### YOUR CODE HERE
    sigmoid_activation = 1 / (1 + np.exp(-z))
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
    h0 = np.copy(X)
    U1 = np.dot(h0, W1) + b1
    h1 = relu(U1)

    U2 = np.dot(h1, W2) + b2
    h2 = relu(U2)

    U3 = np.dot(h2, W3) + b3
    h3 = relu(U3)

    U4 = np.dot(h3, W4) + b4
    output = sigmoid(U4)
    
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
        dLoss_dOutput = 2 * (output - y_true) * (output*(1- output)) / n_samples
        ### YOUR CODE HERE
    # note: this is implemented since we did not see it in class
    if loss_type == 'bce':
        dLoss_dOutput = (output - y_true) / n_samples

    # Output layer gradients (follow this example to complete the rest)
    dOutput_dW4 = h3.T
    dW4 = np.dot(dOutput_dW4, dLoss_dOutput)
    db4 = np.sum(dLoss_dOutput, axis=0, keepdims=True)

    # Backpropagate through ReLU and layer 3
    ### YOUR CODE HERE
    dh3 = np.dot(dLoss_dOutput, W4.T)
    dU3 = dh3 * relu_derivative(np.dot(h2, W3) + b3)
    dW3 = np.dot(h2.T, dU3)
    db3 = np.sum(dU3, axis=0, keepdims=True)


    ### YOUR CODE HERE

    # Layer 2
    ### YOUR CODE HERE

    dh2 = np.dot(dU3, W3.T)
    dU2 = dh2 * relu_derivative(np.dot(h1, W2) + b2)
    dW2 = np.dot(h1.T, dU2)
    db2 = np.sum(dU2, axis=0, keepdims=True)



    ### YOUR CODE HERE
    
    # Layer 1
    ### YOUR CODE HERE

    dh1 = np.dot(dU2, W2.T)
    dU1 = dh1 * relu_derivative(np.dot(X, W1) + b1)
    dW1 = np.dot(X.T, dU1)
    db1 = np.sum(dh1, axis=0, keepdims=True)

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

    W1s = []
    W2s = []
    W3s = []
    W4s = []
    b1s = []
    b2s = []
    b3s = []
    b4s = []
    
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
        
        dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backward_pass(X, y_true, W1, W2, W3, W4, b1, b2, b3, b4, h1, h2, h3, output, loss_type = loss_type)
        
        # Update weights and biases using the gradients computed during the backward pass.
        ### YOUR CODE HERE
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
        W3 -= learning_rate * dW3
        W4 -= learning_rate * dW4

        b1 -= learning_rate * db1.reshape(-1)
        b2 -= learning_rate * db2.reshape(-1)
        b3 -= learning_rate * db3.reshape(-1)
        b4 -= learning_rate * db4.reshape(-1)
        ### YOUR CODE HERE

        W1s.append(W1)
        W2s.append(W2)
        W3s.append(W3)
        W4s.append(W4)
        b1s.append(b1)
        b2s.append(b2)
        b3s.append(b3)
        b4s.append(b4)
        
        # Optional: Print the loss at regular intervals to monitor training progress.
        # if i % 100 == 0:
        #     print(f"Iteration {i}: Train Loss = {train_loss:.4f}")
        #     if np.size(X_test) > 1:
        #         print(f"Iteration {i}: Test Loss = {test_loss:.4f}")
    if len(test_loss_lst) > 1:
        inx = np.argsort(test_loss_lst)[0]
    else:
        inx = np.argsort(train_loss_lst)[0]
    # print(f"the number of iteration is {inx}")
    # Return the updated parameters after training.
    return W1s[inx], b1s[inx], W2s[inx], b2s[inx], W3s[inx], b3s[inx], W4s[inx], b4s[inx], np.array(train_loss_lst), np.array(test_loss_lst)
