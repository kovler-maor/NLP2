import os
import pickle

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


class NERDataSet:
    def __init__(self, train_data_path, dev_data_path,  glove_model, window_size=0):
        self.train_sentences, self.train_tags = self.parse_tagged_file(train_data_path)
        self.dev_sentences, self.dev_tags = self.parse_tagged_file(dev_data_path)
        self.glove_model = glove_model
        self.window_size = window_size
        self.vector_size = glove_model.vector_size


    def parse_tagged_file(self, file_path):
        sentences = [] # List of sentences
        tags = [] # List of corresponding tags
        current_sentence = [] # Buffer for the current sentence
        current_tags = [] # Buffer for the current tags

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() == "": # Sentence boundary
                    sentences.append(current_sentence) 
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
                else:
                    parts = line.strip().split('\t')
                    word = parts[0]
                    tag = parts[1] if len(parts) == 2 else 'O'
                    if tag != 'O':
                        tag = 'ENTITY'
                    current_sentence.append(word)
                    current_tags.append(tag)

        return sentences, tags
    

    def word2vector(self):
        """
        Convert the words to vectors using the glove model
        """
        train_vectors = []
        dev_vectors = []

        for sentence in self.train_sentences:
            sentence_vectors = []
            for word in sentence:
                sentence_vectors.append(self.get_vector(word, self.glove_model))
            train_vectors.append(sentence_vectors)

        for sentence in self.dev_sentences:
            sentence_vectors = []
            for word in sentence:
                sentence_vectors.append(self.get_vector(word, self.glove_model))
            dev_vectors.append(sentence_vectors)

        self.train_vectors = train_vectors 
        self.dev_vectors = dev_vectors


    def get_vector(self, word, model):
        """
        Get the vector of a word from the glove model
        """
        try:
            # Try to retrieve the word's vector from the model
            return model[word]
        except KeyError:
            # Return a zero vector if the word is not in the vocabulary
            return np.zeros(model.vector_size)


    def vector2window(self):
        """
        Convert the vectors to bigger vector that represent the window of the word.
        for each sentence, we will pad the sentence beginning and end with vectors of zeros and then create a window of size 2 around each word.
        we create the window by taking the vectors of the words in the window and concatenating them.
        """
        for i in range(len(self.train_vectors)):
            sentence = self.train_vectors[i]
            padded_sentence = np.pad(sentence, ((self.window_size, self.window_size), (0, 0)), 'constant') # pad the sentence with zeros
            window_vectors = [np.concatenate(padded_sentence[i - self.window_size: i + self.window_size + 1]) for i in range(self.window_size, len(padded_sentence) - self.window_size)] # create the window vectors
            self.train_vectors[i] = window_vectors # replace the sentence with the window vectors

        for i in range(len(self.dev_vectors)):
            sentence = self.dev_vectors[i]
            padded_sentence = np.pad(sentence, ((self.window_size, self.window_size), (0, 0)), 'constant') # pad the sentence with zeros
            window_vectors = [np.concatenate(padded_sentence[i - self.window_size: i + self.window_size + 1]) for i in range(self.window_size, len(padded_sentence) - self.window_size)] # create the window vectors
            self.dev_vectors[i] = window_vectors # replace the sentence with the window vectors
        
        self.vector_size = self.vector_size * (2 * self.window_size + 1) # update the vector size


    def preper2train(self):
        """
        change the shape of the train_vectors to be 1d array and the same for the tags
        """
        # flatten the train and dev vectors (list of np.arrays)
        flattened_train_vectors = []
        for sublist in self.train_vectors:
            for item in sublist:
                flattened_train_vectors.append(item)
        self.train_vectors = flattened_train_vectors

        flattened_dev_vectors = []
        for sublist in self.dev_vectors:
            for item in sublist:
                flattened_dev_vectors.append(item)
        self.dev_vectors = flattened_dev_vectors

        # flatten the tags for train and dev (list of lists)
        flattened_train_tags = []
        for sublist in self.train_tags:
            for item in sublist:
                flattened_train_tags.append(item)
        self.train_tags = flattened_train_tags

        flattened_dev_tags = []
        for sublist in self.dev_tags:
            for item in sublist:
                flattened_dev_tags.append(item)
        self.dev_tags = flattened_dev_tags


    def print_results(self, y_pred):
            """
            Print the results of a model
            """
            # print the classification report of the model
            y_dev = self.dev_tags
            print(classification_report(y_dev, y_pred))

            # calculate and print the final f1 score
            print(f"\nThe final f1 score of the SVM model is: {f1_score(y_dev, y_pred, average='weighted')}\n")

            # print the confusion matrix of the model with the labels and tp\fp\tn\fn for each row and column
            print("confusion matrix:")
            print(['ENTITY', 'O'])
            print(confusion_matrix(y_dev, y_pred))
            print("\n\n")

            
class svm_model():
    def __init__(self, ner_data, pca_components=50, use_pca=True):
        self.ner_data = ner_data
        self.svm_model = make_pipeline(StandardScaler(), SVC(kernel='poly', degree=3, gamma='scale', coef0=1))
        self.use_pca = use_pca
        if use_pca:
            self.pca = PCA(n_components=pca_components)
        

    def train_svm_model(self):
        """
        Train a svm model using the glove window vectors
        """
        if self.use_pca:
            X_train = self.pca.fit_transform(self.ner_data.train_vectors)
        else:
            X_train = self.ner_data.train_vectors
        
        y_train = self.ner_data.train_tags

        self.svm_model.fit(X_train, y_train)


    def evaluate_svm_model(self):
        """
        Evaluate the svm model
        """
        # evaluate the svm model here
        if self.use_pca:
            X_dev = self.pca.transform(self.ner_data.dev_vectors)
        else:
            X_dev = self.ner_data.dev_vectors
        y_dev = self.ner_data.dev_tags

        # pridict and return
        y_pred = self.svm_model.predict(X_dev)
        return y_pred
        

class nn_model(nn.Module):
    
    def __init__(self, ner_data, hidden_dim=100, num_epochs=10, learning_rate=0.1, batch_size=16):
        super(nn_model, self).__init__()
        # data attributes
        self.ner_data = ner_data
        self.num_classes = len(set(ner_data.train_tags))
        self.vec_dim = ner_data.vector_size #! the size of the window vector but might be somting else
        # layer model attributes
        self.first_layer = nn.Linear(self.vec_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, self.num_classes)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        # model parameters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size


    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids) # (batch_size, vec_dim) -> (batch_size, hidden_dim)
        x = self.activation(x) # (batch_size, hidden_dim)
        x = self.second_layer(x) # (batch_size, hidden_dim) -> (batch_size, num_classes)
        if labels is None:
            return x, None
        loss = self.loss(x, labels) # (batch_size, num_classes) -> (batch_size)
        return x, loss


    def train_nn_model(self, model, optimizer):
        """
        Train a nn model using the glove window vectors
        """
        # check if cuda is available and set the device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using GPU for nn model training')
        else:
            device = torch.device('cpu')
            print('Using CPU for nn model training')

        # create the data loaders
        data_loaders = {"train": DataLoader(ner_data.train_vectors, batch_size=self.batch_size, shuffle=True),
                        "dev": DataLoader(ner_data.dev_vectors, batch_size=self.batch_size, shuffle=False)}
        model.to(device)

        best_acc = 0.0

        #! start debbuging here
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            for phase in ['train', 'dev']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                labels, preds = [], []

                for batch in data_loaders[phase]:
                    batch_size = 0
                    for k, v in batch.item():
                        batch[k] = v.to(device)
                        batch_size = v.shape[0]

                    optimizer.zero_grad()
                    if phase == 'train':
                        outputs, loss = model(**batch)
                        loss.backward()
                        optimizer.step()
                    else:
                        with torch.no_grad():
                            outputs, loss = model(**batch)
                    pred = outputs.argmax(dim=-1).clone().detach().cpu()
                    labels += batch['labels'].cpu().view(-1).tolist()
                    preds += pred.view(-1).tolist()
                    running_loss += loss.item() * batch_size

                epoch_loss = running_loss / len(data_sets[phase])
                epoch_acc = accuracy_score(labels, preds)

                epoch_acc = round(epoch_acc, 5)

                if phase.title() == "dev":
                    print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
                else:
                    print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')
                if phase == 'dev' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    with open('nn_model.pkl', 'wb') as f:
                        torch.save(model, f)
            print()

        print(f'Best Validation Accuracy: {best_acc:4f}')
        #! start debbuging here



    def evaluate_nn_model(self):
        """
        Evaluate the nn model
        """
        if self.use_pca:
            X_dev = self.pca.transform(self.ner_data.dev_vectors)
        else:
            X_dev = self.ner_data.dev_vectors

        y_dev = self.ner_data.dev_tags
        
        # pridict and return
        y_pred = self.nn_model.predict(X_dev) #! ?????
        return y_pred
        




if __name__ == "__main__":
    #* START of the data preperation part
    # load the glove model
    glove_path = 'glove_model.pkl'
    if os.path.exists(glove_path):
        with open(glove_path, 'rb') as f:
            glove_model = pickle.load(f)
    
    # create the NERDataSet object
    train_data_path = 'data/train.tagged'
    dev_data_path = 'data/dev.tagged'
    window_size = 0
    ner_data = NERDataSet(train_data_path, dev_data_path, glove_model, window_size)

    # print the first sentence and tags
    print("Train data: ")
    print(f"First sentence: {ner_data.train_sentences[0]}")
    print(f"First tags: {ner_data.train_tags[0]}\n")
    
    print("Dev data: ")
    print(f"First sentence: {ner_data.dev_sentences[0]}")
    print(f"First tags: {ner_data.dev_tags[0]}\n")

    # create features
    ner_data.word2vector()
    print(f"The start of the vec of the word '{ner_data.train_sentences[0][3]}' \nfrom the Train vectors: {ner_data.train_vectors[0][3][:5]}\n")
    print(f"The start of the vec of the word '{ner_data.dev_sentences[0][3]}' \nfrom the Dev vectors: {ner_data.dev_vectors[0][3][:5]}\n")
    
    # create window
    print("lenght of the first train sentance before padding: ", len(ner_data.train_vectors[0]))
    print(f"Vector size: {ner_data.vector_size}")
    print(f"Window size: {ner_data.window_size}\n")
    print("creating windows...\n")
    ner_data.vector2window()
    print("lenght of the first train sentance after padding: ", len(ner_data.train_vectors[0]))
    print(f"Window Vector size: {ner_data.vector_size}\n")
    ner_data.preper2train()
    print(f"Number of Train vectors (words):  {len(ner_data.train_vectors)}")
    print(f"Number of Train tags:  {len(ner_data.train_tags)}\n")
    print(f"Number of Dev vectors (words):  {len(ner_data.dev_vectors)}")
    print(f"Number of Dev tags:  {len(ner_data.dev_tags)}\n\n")
    #* END of the data preperation part

    run_svm_model = False
    if run_svm_model:
        #* START of the SVM model training and evaluation part
        #? choose the number of pca components 
        pca_components = ner_data.vector_size // 5 # 1/3 of the vector size
        # train a svm model using the glove window vectors
        print("Init the SVM model...\n")
        svm_model = svm_model(ner_data, pca_components, use_pca=True)
        print("Training the SVM model...\n")
        svm_model.train_svm_model()
        print("Evaluating the SVM model...\n")
        y_pred_svm = svm_model.evaluate_svm_model()
        print("printing the results for SVM model...\n")
        ner_data.print_results(y_pred_svm)
        print("SVM model trained and evaluated successfully!\n\n")
        #* END of the SVM model training and evaluation part

    run_nn_model = True
    if run_nn_model:
        # #* START of the NN model training and evaluation part
        #? choose the nn parameters
        hidden_dim = 100
        num_epochs=5
        learning_rate=0.1
        batch_size=16
        # train a nn model using the glove window vectors
        print("Init the NN model...\n")
        nn_model = nn_model(ner_data, hidden_dim, num_epochs, learning_rate, batch_size)
        print("Training the NN model...\n")
        optimizer = Adam(params=nn_model.parameters())
        nn_model.train_nn_model(nn_model, optimizer)
        print("Evaluating the NN model...\n")
        y_pred_nn = nn_model.evaluate_nn_model()
        print("printing the results for SVM model...\n")
        ner_data.print_results(y_pred_nn)
        print("NN model trained and evaluated successfully!\n\n")
        # #* END of the nn model training and evaluation part

    run_lstm_model = False
    if run_lstm_model:
        #* START of the LSTM model training and evaluation part
        # train a lstm model using the glove window vectors
        print("Init the LSTM model...\n")
        lstm_model = lstm_model(ner_data)
        print("Training the LSTM model...\n")
        lstm_model.train_lstm_model()
        print("Evaluating the LSTM model...\n")
        lstm_model.evaluate_lstm_model()
        print("LSTM model trained and evaluated successfully!\n\n")
        #* END of the LSTM model training and evaluation part




    
