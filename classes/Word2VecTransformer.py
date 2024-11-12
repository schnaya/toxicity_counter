import os
import re
import joblib
import nltk
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import spacy
import torch
from tqdm import tqdm
from nltk.corpus import stopwords

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,device, vector_size=10000,trainings=False, window=2, min_count=2, epochs=50,dm=1, model_path=None, train_df=None):
        self.device=device
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model_path = model_path
        self.dm=dm
        self.model = None
        self.df=None
        if trainings:
            if train_df is not None and 'content' in train_df.columns:
                self.df = train_df
                self.fit(self.df['content'].values)
            else:
                raise RuntimeError("Training Dataframe is not provided")

        else:
            if self.model is None and model_path:
                self.load_model()

    def fit(self, X, y=None):
        #tokenized_data = [self.preprocess_text(row).split() for row in X]

        sentences = [row.split() for row in X]

        if self.model is None:
            self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window,
                                  min_count=self.min_count, workers=10, epochs=self.epochs, sg=self.dm)
            self.save_model()
        else:
            self.model.build_vocab(sentences, update=True)
            self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.epochs)
            self.save_model()
        return self
    
    def transform(self, X):
        if self.model is None:
            raise RuntimeError("Word2Vec model has not been trained or loaded.")
        vectors = []
        for row in X:
            tokens = (row).split()
            word_vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if word_vectors:
                sentence_vector = np.mean(word_vectors, axis=0)
            else:
                sentence_vector = np.zeros(self.model.vector_size)
            vectors.append(sentence_vector)

        return np.array(vectors)
    def transform_to_vectors(self, X):
        if self.model is None:
            raise RuntimeError("Word2Vec model has not been trained or loaded.")
        vectors = [self.model.infer_vector(self.preprocess_text(row).split()) for row in X]
        return vectors
    def save_model(self):
        if not self.model_path:
            raise ValueError("Model path is not provided. Cannot save the model.")
        dir_name = os.path.dirname(self.model_path)
        os.makedirs(dir_name, exist_ok=True)  # Create directory if it doesn't exist

        try:
            joblib.dump(self.model, self.model_path)
            print(f"Model saved to {self.model_path}")
        except PermissionError:
            raise RuntimeError(f"Permission denied: Unable to save the model to {self.model_path}.")
        except Exception as e:
            raise RuntimeError(f"Failed to save the model: {str(e)}")

    def load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError("Model path is not provided or file does not exist.")
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {str(e)}")