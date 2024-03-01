from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVMModel:
    """
    A class for training and evaluating an SVM model for NER tasks.
    """
    def __init__(self, train_vectors, train_labels, dev_vectors, dev_labels, pca_components=50, use_pca=True):
        self.train_vectors = train_vectors
        self.train_labels = train_labels
        self.dev_vectors = dev_vectors
        self.dev_labels = dev_labels
        self.use_pca = use_pca
        self.model = make_pipeline(StandardScaler(), SVC(kernel='poly', degree=3, gamma='scale', coef0=1))
        if use_pca:
            self.pca = PCA(n_components=pca_components)
            self.train_vectors = self.pca.fit_transform(self.train_vectors)
            self.dev_vectors = self.pca.transform(self.dev_vectors)

    def train(self):
        """
        Trains the SVM model.
        """
        self.model.fit(self.train_vectors, self.train_labels)

    def evaluate(self):
        """
        Evaluates the SVM model and prints the results.
        """
        y_pred = self.model.predict(self.dev_vectors)
        print(classification_report(self.dev_labels, y_pred))
        print(f"The final f1 score is: {f1_score(self.dev_labels, y_pred, average='weighted')}")
        print("Confusion Matrix:")
        print(confusion_matrix(self.dev_labels, y_pred))
