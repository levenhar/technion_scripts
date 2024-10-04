# Create the utils.py file with the necessary functions to load and prepare CIFAR-10 data

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def load_cifar10_batch(batch_filename):
    '''
    Load a single batch of the CIFAR-10 dataset
    '''
    with open(batch_filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_training_data(data_dir):
    '''
    Load all training data from CIFAR-10 dataset
    '''
    # Initialize variables to hold training data and labels
    X_train = np.empty((0, 32, 32, 3), dtype='float')
    y_train = np.empty((0), dtype='int')
    
    # Load each batch and accumulate
    for i in range(1, 6):
        batch_filename = os.path.join(data_dir, 'data_batch_{}'.format(i))
        X_batch, y_batch = load_cifar10_batch(batch_filename)
        X_train = np.vstack((X_train, X_batch))
        y_train = np.append(y_train, y_batch)
    
    # Normalize pixel values to 0-1
    X_train = X_train / 255.0
    
    return X_train, y_train

def load_testing_data(data_dir):
    """
    Load testing data from the CIFAR-10 dataset.

    Parameters:
    data_dir: Path to the directory containing the CIFAR-10 dataset files.

    Returns:
    X_test: 4D numpy array of test images (number of images, height, width, channels).
    y_test: 1D numpy array of test labels.
    """
    test_batch_filename = os.path.join(data_dir, 'test_batch')
    with open(test_batch_filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X_test = datadict[b'data']
        y_test = datadict[b'labels']
        X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y_test = np.array(y_test)

        # Normalize pixel values to 0-1
        X_test = X_test / 255.0
        
        return X_test, y_test

def show_binary_class_images(X, T, class_index, label_names, num_examples=5):
    """
    Display images with binary class labels for a specific class,
    rescaling images back to [0, 1] for proper display.
    
    Parameters:
    X: numpy array of images, normalized to [-1, 1].
    T: numpy array of binary labels for all classes (one-vs-all setup).
    class_index: int, index of the class to display images for.
    label_names: list of str, names of all classes.
    num_examples: int, number of positive and negative examples to display.
    """
    class_name = label_names[class_index]

    # Get the binary labels for the chosen class
    binary_labels = T[:, class_index]

    # Find indices for positive and negative examples
    positive_indices = np.where(binary_labels == 1)[0][:num_examples]
    negative_indices = np.where(binary_labels == -1)[0][:num_examples]

    # Combine indices to show a mix of positive and negative examples
    selected_indices = np.concatenate([positive_indices, negative_indices])

    # Plotting
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(selected_indices):
        plt.subplot(2, num_examples, i + 1)
        img = X[idx].astype('float32')
        img = (img + 1) / 2  # Rescale images from [-1, 1] back to [0, 1]
        plt.imshow(img)
        label = class_name if binary_labels[idx] == 1 else f"Not {class_name}"
        plt.title(label)
        plt.axis('off')
    plt.show()

