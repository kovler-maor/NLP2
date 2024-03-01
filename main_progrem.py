import os
import pickle

import torch
from gensim import downloader
from torch import nn
from torch.utils.data import DataLoader

from NERDataSet import NERDataSet
from NERDataSetForNN import NERDataSetForNN
from FF import NER_FF_NN
from LSTM import NER_LSTM
from SVM import SVMModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    window_size = int(input("Enter the window size for the NER data set: "))
    run_svm_model_flag = input("Do you want to run the SVM model? (y/n): ")
    run_ner_ff_nn_model_flag = input("Do you want to run the NER_FF_NN model? (y/n): ")
    run_lstm_model_flag = input("Do you want to run the LSTM model? (y/n): ")


    train_data_path = 'data/train.tagged'
    dev_data_path = 'data/dev.tagged'

    # Load GloVe model
    glove_model_path = 'glove_model.pkl'
    if os.path.exists(glove_model_path):
        with open(glove_model_path, 'rb') as file:
            glove_model = pickle.load(file)
    else:
        glove_model = downloader.load('glove-twitter-200')
        with open(glove_model_path, 'wb') as file:
            pickle.dump(glove_model, file)

    # Initialize NER data set
    ner_data = NERDataSet(train_data_path, dev_data_path, glove_model, window_size= window_size)

    # If we are running any of the neural network models, we need to prepare the data for them
    if run_ner_ff_nn_model_flag == 'y' or run_lstm_model_flag == 'y':
        ner_dataset_nn = NERDataSetForNN(ner_data.train_vectors, ner_data.train_labels)
        ner_dataset_nn_test = NERDataSetForNN(ner_data.dev_vectors, ner_data.dev_labels)
        train_loader = DataLoader(ner_dataset_nn, batch_size=32, shuffle=True)
        test_loader = DataLoader(ner_dataset_nn_test, batch_size=32, shuffle=True)

    if run_svm_model_flag == 'y':
        # SVM Model Training and Evaluation
        print("Training SVM Model...")
        svm_model = SVMModel(ner_data.train_vectors, ner_data.train_labels, ner_data.dev_vectors, ner_data.dev_labels,
                             pca_components=100, use_pca=True)
        svm_model.train()
        print("Evaluating SVM Model...")
        svm_model.evaluate()

    if run_ner_ff_nn_model_flag == 'y':
        # NER_FF_NN Model Training and Evaluation
        print("Training and evaluating NER_FF_NN Model...")

        # Create the model with specified input, hidden, and output sizes
        ff_model = NER_FF_NN(input_size=100, hidden_size=[128, 64], output_size=2)  # Example dimensions, adjust as needed
        ff_model.to(device)
        # Run the model with specified learning rate and epochs
        ff_model.run(train_loader=train_loader, test_loader=test_loader, lr=0.001, epochs=5)

    if run_lstm_model_flag == 'y':
        # LSTM Model Training and Evaluation
        print("Training and evaluating LSTM model ...")
        lstm_model = NER_LSTM(ner_dataset_nn.vectors.shape[1], 100, 2, 1)
        lstm_model.to(device)
        lstm_model.run(
            train_loader,
            test_loader,
            ner_dataset_nn.vectors.shape[1],
            100,
            2,
            1,
            0.5,
            0.001,
            5
        )



if __name__ == '__main__':
    main()

