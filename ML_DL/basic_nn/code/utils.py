import numpy as np
import matplotlib.pyplot as plt
import os

def generate_circle_data(radius, center, n_samples):
    # Generate angles
    angles = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
    # Generate radii with a little noise
    radii = np.random.normal(loc=radius, scale=0.1, size=n_samples)
    # Polar to cartesian conversion
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return x, y


def compute_accuracy(y_true, y_pred):
    """
    Compute the accuracy of predictions.

    Parameters:
    - y_true: True labels (numpy array of shape (n_samples,)).
    - y_pred: Predicted labels (numpy array of shape (n_samples,)).

    """
    correct_predictions = np.count_nonzero(y_true.flatten() == y_pred)
    accuracy = correct_predictions / len(y_true)
    print("Your model's accuracy is " + str(accuracy))
    return accuracy

def load_images_and_labels(directory, label):
    '''
    Function to read images and assign labels for the wildfires notebook
    directory: images directory
    
    '''
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'): 
            filepath = os.path.join(directory, filename)
            image = plt.imread(filepath)
            images.append((image, label))
    return images