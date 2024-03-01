import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np



class NER_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(NER_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

    def train_model(self, train_loader, criterion, optimizer, epochs):
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

    def test_model(self, test_loader):
        self.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                outputs = outputs.view(-1, outputs.shape[-1])
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        cm = confusion_matrix(all_labels, all_predictions)
        accuracy = np.trace(cm) / np.sum(cm)
        print(f'\nTest Results:\nAccuracy: {accuracy * 100:.2f}%\nF1 Score: {f1:.4f}')
        print(f'Confusion Matrix:\n{cm}')

    def run(self, train_loader, test_loader, input_dim, hidden_dim, output_dim, num_layers, dropout, lr, epochs):
        # Reinitialize the model with the new hyperparameters
        self.__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Train the model
        print("Starting training...")
        self.train_model(train_loader, criterion, optimizer, epochs)

        # Test the model
        print("\nStarting testing...")
        self.test_model(test_loader)
