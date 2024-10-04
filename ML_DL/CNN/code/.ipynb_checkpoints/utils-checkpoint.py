import numpy as np
import matplotlib.pyplot as plt
import torch 

def calculate_precision(true_positives, false_positives):
    """
    Calculate precision for each class.
    
    Parameters:
    - true_positives (list): Number of true positives for each class.
    - false_positives (list): Number of false positives for each class.
    
    Returns:
    - precision (list): Precision value for each class.
    """
    precisions = []
    ###
    for c in range(len(true_positives)):
        precision =  round((true_positives[c])/(true_positives[c] + false_positives[c]),3)
        precisions.append(precision)
    ### YOUR CODE GOES HERE
    ###
    return precisions

def calculate_recall(true_positives, false_negatives):
    """
    Calculate recall for each class.
    
    Parameters:
    - true_positives (list): Number of true positives for each class.
    - false_negatives (list): Number of false negatives for each class.
    
    Returns:
    - recall (list): Recall value for each class.
    """
    recalls = []
    for c in range(len(true_positives)):
        recall =  round((true_positives[c])/(true_positives[c] + false_negatives[c]),3)
        recalls.append(recall)
    ###
    ### YOUR CODE GOES HERE
    ###
    return recalls

def calculate_f1_score(precisions, recalls):
    """
    Calculate F1 score for each class, and then compute the macro-average F1 score.
    
    Parameters:
    - precisions (list): precisions for each class.
    - recalls (list): recalls for each class.
    
    Returns:
    - f1_scores (list): F1 score for each class.
    - macro_average_f1 (float): Macro-average F1 score across all classes.
    """
    f1_scores = []
    for c in range(len(precisions)):
        f1_score =  round(2*(precisions[c]*recalls[c])/(precisions[c] + recalls[c]),3)
        f1_scores.append(f1_score)
    ###
    ### YOUR CODE GOES HERE
    ###
    
    macro_average_f1 = np.mean(f1_scores)
    return f1_scores, macro_average_f1

###
###
### NO NEED TO CHANGE THE BELOW FUNCTIONS
###
###

def calculate_accuracy(model, dataloader):
    """
    Computes the accuracy of the model given a test data loader
    
    Parameters:
    - model: Pytorch model
    - dataloader: pyrtorch test set data loader
    
    Returns:
    - model accuracy (double)
    """
    correct = 0
    total = 0
    with torch.no_grad():  # No need to calculate gradients
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

def draw_imgs_labels(imgs, labels, predicted_labels, classes):
    '''
    Draw images with labels
    '''
    # imgs are normalized tensors, unnormalize them
    imgs = imgs / 2 + 0.5
    npimgs = imgs.numpy()
    
    # Number of images
    num_images = imgs.size(0)
    
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axes = [axes]  # Make it iterable
    
    for i, ax in enumerate(axes):
        ax.imshow(np.transpose(npimgs[i], (1, 2, 0)))
        ax.set_title(f"True: {classes[labels[i]]}\nPred: {classes[predicted_labels[i]]}")
        ax.axis('off')
    
    plt.show()

def plot_training_validation_loss(epoch_losses, epoch_val_losses):
    """
    Plots the training and validation loss curves.

    Parameters:
    - epoch_losses (list): A list of average training losses per epoch.
    - epoch_val_losses (list): A list of average validation losses per epoch.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.plot(epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()