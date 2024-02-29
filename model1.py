# Description: This file contains the code to train a SVM model to predict if a word in a sentence
# is a named entity or not.

# Part 1:
# Load and Preprocess the Data:
# We need to parse the uploaded files (train.tagged, dev.tagged, and test.untagged)
# to extract words and their corresponding tags for the training and development datasets.
# For the test dataset, we'll only extract words since it's untagged.
#--------------------------------------------------------------


# Part 1.1: Load and Preview the Data:
train_file_path = 'data/train.tagged'
dev_file_path = 'data/dev.tagged'
# Function to load and preview a file
def load_and_preview(file_path, num_lines=10):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [next(file) for _ in range(num_lines)]
    return lines

train_preview = load_and_preview(train_file_path)

# Part 1.2:
# Parse the train.tagged, dev.tagged, and test.untagged files to extract sentences and their corresponding word tags.
# Prepare the data for feature extraction by aligning it with the GloVe embeddings.
# Modified function to handle lines that might not strictly follow the "word\ttag" format
def parse_tagged_file(file_path, is_tagged=True):
    sentences = []
    tags = []
    current_sentence = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == "":  # Sentence boundary
                if current_sentence:  # If the current sentence is not empty
                    sentences.append(current_sentence)
                    if is_tagged:
                        tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
            else:
                parts = line.strip().split('\t')
                if len(parts) == 2:  # Correctly formatted line
                    word, tag = parts
                    current_tags.append(tag)
                else:  # Handle untagged or improperly formatted lines
                    word = parts[0]
                    if is_tagged:
                        current_tags.append('O')  # Assume 'O' if tag is missing
                current_sentence.append(word)

    # Add the last sentence if the file does not end with an empty line
    if current_sentence:
        sentences.append(current_sentence)
        if is_tagged:
            tags.append(current_tags)

    return sentences, tags if is_tagged else None


# Re-parse the train and development datasets with the modified function
train_sentences, train_tags = parse_tagged_file(train_file_path)
dev_sentences, dev_tags = parse_tagged_file(dev_file_path)

# Parse the test dataset, which is untagged
test_file_path = 'data/test.untagged'
test_sentences, _ = parse_tagged_file(test_file_path, is_tagged=False)

# Preview the parsed data if needed
# print(train_sentences[:2], train_tags[:2], dev_sentences[:2], dev_tags[:2], test_sentences[:2])

# End of Part 1
#--------------------------------------------------------------


# Part 2:
# Prepare Word Embeddings:
# We'll download the GloVe pre-trained embeddings (100d version)
# and create a lookup table to map words to their vector representations.
# For words not found in GloVe, we'll use a zero vector or a special "unknown" vector.
#--------------------------------------------------------------

# Part 2.1: Download GloVe Embeddings:
# Download the GloVe embeddings

import gensim
from gensim import downloader
import numpy as np
GLOVE_PATH = 'glove-twitter-200'
glove_model = gensim.downloader.load(GLOVE_PATH)

# Part 2.2: Create a Word Embedding Lookup Table:
def get_vector(word, model):
    try:
        return model[word]
    except KeyError:
        # Return a zero vector if the word is not in the vocabulary
        return np.zeros(model.vector_size)
# Example usage if needed
# vector = get_vector("university", glove_model)
# print(vector)

# End of Part 2
#--------------------------------------------------------------

# Part 3:
# Feature Extraction:
# For each word in the dataset,
# create a feature vector that includes its GloVe embedding and the embeddings of its neighbors.
# This step is crucial for capturing the context around each word,
# which can significantly improve the performance of your NER system
#--------------------------------------------------------------
def create_features(sentences, glove_model, window_size=2):
    features = []
    labels = []  # Assuming you're working with labeled data for training
    for sentence in sentences:
        for index, word in enumerate(sentence):
            # Context words
            context_words = sentence[max(index - window_size, 0): index] + \
                            sentence[index + 1: index + 1 + window_size]
            context_vectors = [get_vector(w, glove_model) for w in context_words]
            # Word vector
            word_vector = get_vector(word, glove_model)
            # Combine vectors (simple concatenation here; could be more sophisticated)
            feature = np.concatenate([v for v in context_vectors] + [word_vector])
            features.append(feature)
            # Add label for this word if available
            # labels.append(...)
    return features, labels
# End of Part 3
#--------------------------------------------------------------

# Part 4:
# Model Training:
# Vector Representation: Transform your sentences into sequences of vectors,
# incorporating the context as decided (e.g., two words before and after the target word).

# Flatten for SVM: Since SVMs (and most Scikit-learn models) do not directly accept sequences as input,
# you'll need to flatten your context-inclusive word vectors into a single feature vector per word or tag prediction.

# Training: Train your SVM model on the flattened feature vectors.
#--------------------------------------------------------------

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Example feature preparation
X_train, y_train = create_features(train_sentences, glove_model)

# Scaling features and training SVM
svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
svm_model.fit(X_train, y_train)
# End of Part 4
#--------------------------------------------------------------

# Part 5:
# Model Evaluation:
# Evaluate the trained model on the development dataset.
#--------------------------------------------------------------
X_dev, y_dev = create_features(dev_sentences, glove_model)
y_pred = svm_model.predict(X_dev)

# Evaluation
from sklearn.metrics import classification_report
print(classification_report(y_dev, y_pred))
# End of Part 5
#--------------------------------------------------------------