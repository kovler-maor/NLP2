import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

class NER_FF_NN(nn.Module):
    """
    A simple feedforward neural network for Named Entity Recognition (NER) with training and testing capabilities.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(NER_FF_NN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        self.hidden_layers.append(nn.Linear(input_size, hidden_size[0]))
        for k in range(len(hidden_size) - 1):
            self.hidden_layers.append(nn.Linear(hidden_size[k], hidden_size[k+1]))
        self.output = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output(x)
        return x

    def train_model(self, train_loader, criterion, optimizer, epochs=5):
        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            total_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Clear gradients
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

    def test_model(self, test_loader):
        self.eval()  # Set the model to evaluation mode
        all_predictions = []
        all_labels = []
        with torch.no_grad():  # No need to compute gradients
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.numpy())  # Store predictions
                all_labels.extend(labels.numpy())  # Store true labels

        # Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        print(f'F1 Score: {f1:.4f}')

        # Generate and print the confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print('Confusion Matrix:\n', cm)

        # Calculate accuracy from the confusion matrix
        accuracy = np.trace(cm) / np.sum(cm)
        print(f'Accuracy: {accuracy * 100:.2f}%')

    def run(self, train_loader, test_loader, lr=0.001, epochs=5):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        print("Starting training...")
        self.train_model(train_loader, criterion, optimizer, epochs)
        print("\nStarting testing...")
        self.test_model(test_loader)
