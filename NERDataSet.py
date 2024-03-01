import numpy as np


class NERDataSet:
    """
    A class to prepare and handle data for NER tasks.
    """
    def __init__(self, train_data_path, dev_data_path, glove_model, window_size=0):
        self.glove_model = glove_model
        self.window_size = window_size
        self.vector_size = glove_model.vector_size
        self.train_sentences, self.train_tags = self._parse_tagged_file(train_data_path)
        self.dev_sentences, self.dev_tags = self._parse_tagged_file(dev_data_path)
        self.train_vectors, self.train_labels = self._prepare_data(self.train_sentences, self.train_tags)
        self.dev_vectors, self.dev_labels = self._prepare_data(self.dev_sentences, self.dev_tags)

    def _parse_tagged_file(self, data_path):
        sentences = []  # List of sentences
        tags = []  # List of corresponding tags
        current_sentence = []  # Buffer for the current sentence
        current_tags = []  # Buffer for the current tags

        with open(data_path, 'r', encoding='utf-8') as file:
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

    def _prepare_data(self, sentences, tags):
        """
        Prepares the data by converting words to vectors and applying windowing.
        """
        vectors, labels = [], []
        for sentence, tag_sequence in zip(sentences, tags):
            sentence_vectors = [self._word_to_vector(word) for word in sentence]
            window_vectors = self._apply_windowing(sentence_vectors)
            vectors.extend(window_vectors)
            labels.extend([1 if tag == 'ENTITY' else 0 for tag in tag_sequence])
        return np.array(vectors), np.array(labels)

    def _word_to_vector(self, word):
        """
        Converts a word to its vector representation using the GloVe model.
        If the word is not in the model, returns a zero vector.
        """
        try:
            # Attempt to retrieve the word's vector from the model
            return self.glove_model[word]
        except KeyError:
            # Return a zero vector if the word is not in the model
            return np.zeros(self.vector_size)

    def _apply_windowing(self, sentence_vectors):
        """
        Applies windowing to sentence vectors.
        """
        padded_sentence = np.pad(sentence_vectors, ((self.window_size, self.window_size), (0, 0)), 'constant')
        window_vectors = [np.concatenate(padded_sentence[i:i + 2 * self.window_size + 1]) for i in range(len(sentence_vectors))]
        return window_vectors