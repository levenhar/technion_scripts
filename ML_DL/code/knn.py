import numpy as np
from scipy.stats import mode



def knn_norm(vec, p):
    """
    Compute the norm of order p of a vector

    Parameters:
    vec: Numpy array; the input vector (x_1, x_2, ..., x_n)
    p: the norm order.

    Returns:
    the norm of order p.
    """
    return np.sum(np.abs(vec) ** p) ** (1/p)

def compute_distances(test_img, train_imgs):
    """
    Compute Euclidean distances between the test image and the training images

    Parameters:
    test_img: A flattened image from the CIFAR-10 test set
    train_imgs: A flattened list of training images from the CIFAR-10 set
    
    Returns:
    A list of distances between test_img and train_imgs
    """

    
    distance_list = list()

    for train_img in train_imgs:
        diff = train_img - test_img
        distance = knn_norm(diff, 2)
    
        distance_list.append(distance)
    
    return distance_list


def predict_class(distance_list, k, labels):
    """
    Predict image class based on k neighbors. 

    Parameters:
    distance_list: Numpy array; the distances between the image and the training images
    k: the number of neighbors to consider
    labels: the class labels

    Returns:
    the predicted class index in CIFAR-10 (1 through 10).
    """
    k_closest_indices = np.argsort(distance_list)[:k]
    k_closest_classes = labels[k_closest_indices]
    predicted_class_index = np.argmax(np.bincount(k_closest_classes))

    return predicted_class_index

def predict_class_vec(distance_list, k, labels):
    """
    Predict image class based on k neighbors. 

    Parameters:
    distance_list: Numpy array; the distances between the image and the training images
    k: the number of neighbors to consider
    labels: the class labels

    Returns:
    the predicted class index in CIFAR-10 (1 through 10).
    """
    k_closest_indices = np.argsort(distance_list, axis = 0)[:,:k]
    k_closest_classes = labels[k_closest_indices]
    # predicted_class_index = np.argmax(np.bincount(k_closest_classes, axis =0), axis =0)
    predicted_class_index = mode(k_closest_classes, axis = 1,keepdims = False).mode
    return predicted_class_index

def knn_norm_vecotr(vec, p):
    return np.sum(np.abs(vec) ** p, axis =0) ** (1/p)
    
def vectorized_compute_distances(test_img, train_imgs,p, num_images):
    """
    Compute the Euclidean distances between a test image (or multiple test images) 
    and a set of training images using vectorized operations for efficiency.

    Parameters:
    test_img: numpy array of shape (num_test_samples, num_features), test images.
    train_imgs: numpy array of shape (num_train_samples, num_features), training images.

    Returns:
    dists: numpy array of shape (num_test_samples, num_train_samples), Euclidean distances.
    """
    # Ensure test_img is a 2D array (for a single image or multiple test images)
    
    ### YOUR CODE GOES HERE

    X_train_flat1 = train_imgs.T.reshape(3072, 1, train_imgs.shape[0])
    
    diff = X_train_flat1 - test_img.T.reshape(3072,num_images,1)
    
    distances = knn_norm_vecotr(diff, p)

    ### YOUR CODE ENDS HERE
    
    return distances
