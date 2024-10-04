import torch
import torch.nn as nn
from utils import *

def train_network(model, trainloader, validloader, optimizer, epochs=10):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize lists to store average loss per epoch for both training and validation
    epoch_losses = []
    epoch_val_losses = []

    print('Training model...')
    
    for epoch in range(epochs):
        running_loss = 0.0
        val_running_loss = 0.0
        total_batches = len(trainloader)  # For training progress update
        total_val_batches = len(validloader)  # For validation progress update

        
        # Training Phase
        for i, (inputs, labels) in enumerate(trainloader):

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Print progress for each batch in the training phase
            print(f'\rEpoch {epoch+1}/{epochs} - Training Batch {i+1}/{total_batches}', end='')

        avg_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_loss)

        # Validation Phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(validloader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                # Print progress for each batch in the validation phase
                print(f'\rEpoch {epoch+1}/{epochs} - Validation Batch {j+1}/{total_val_batches}', end='')

        avg_val_loss = val_running_loss / len(validloader)
        epoch_val_losses.append(avg_val_loss)

        model.train()  # Set model back to train mode

        # Print epoch summary for both training and validation
        print(f'\nEpoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.3f}, Validation Loss: {avg_val_loss:.3f}')

    print('Finished Training')
    return epoch_losses, epoch_val_losses  # Return the vectors of loss values