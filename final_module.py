import warnings

import nltk
import numpy as np
import spacy
import torch
import pandas as pd
from nltk.corpus import stopwords
import os

from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, root_mean_squared_error, mean_absolute_error, \
    log_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
warnings.simplefilter(action='ignore', category=FutureWarning)
from classes.Word2VecTransformer import Word2VecTransformer
from classes.LogisticRegressionTorch import LogisticRegressionTorchTransformer
from toxic_coms_task.classes.ToxiicityFinderTransformer import ToxicityTransformer
from toxic_coms_task.classes.TranslationTransformer import TranslationTransformer
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

class ToxicityPipeline:
    def __init__(self,trainings=[False,False,True],testing=False, word2vec_model_path=None, logreg_model_path=None,catboost_model_path=None,  device =torch.device('cuda'),train_df_path=None,vector_size=10000,batch_size=32, epochs=100,word2vec_and_logreg_df_path=None,train_on_valid=False,config_path=None):
        self.train_on_valid = train_on_valid
        self.word2vec_and_logreg_training_df = pd.DataFrame()
        self.catboost_train_df = pd.DataFrame()
        if (trainings[0] or trainings[1] or testing) and word2vec_model_path:
            self.word2vec_and_logreg_training_df = pd.read_csv(word2vec_and_logreg_df_path, on_bad_lines='skip')
        if (trainings[2] or testing) and train_df_path:
            self.catboost_train_df = pd.read_csv(train_df_path)
        self.word2vec_transformer = Word2VecTransformer( trainings=trainings[0],model_path=word2vec_model_path,device=device,vector_size=vector_size,train_df = self.word2vec_and_logreg_training_df)
        self.logreg_transformer = LogisticRegressionTorchTransformer(trainings=trainings[1],testing=testing,model_path=logreg_model_path,device=device,input_size=vector_size, train_df = self.word2vec_and_logreg_training_df,word2vec=self.word2vec_transformer, batch_size=batch_size, num_epochs= epochs,config_path=config_path)
        self.config_path =self.logreg_transformer.config_path
        self.translation_transformer=TranslationTransformer(config_path=self.config_path)
        self.toxicity_transformer = ToxicityTransformer(trainings=trainings[2],testing=True,model_path=catboost_model_path,device=device,train_df=self.catboost_train_df,vectorizer=self.word2vec_transformer, batch_size=batch_size)

    def predict(self, texts):
        vectors_tensor = self.word2vec_transformer.transform(texts)
        predictions = self.logreg_transformer.predict(vectors_tensor)
        translated = self.translation_transformer.transform(texts, predictions)
        vectors_translated = self.word2vec_transformer.transform(translated)
        results = self.toxicity_transformer.transform(vectors_translated)
        return results

    def fit(self, X_train, langs, y_train):
        if self.word2vec_transformer:
            self.word2vec_transformer.fit(X_train)
        if self.logreg_transformer:
            self.logreg_transformer.valid_fit(X_train, langs,config_path="C:\\ongoing\\too_lot_of\\toxic_coms_task\\config.ini")
        if self.toxicity_transformer:
            vectors_tensor = self.word2vec_transformer.transform(X_train)
            predictions = self.logreg_transformer.predict(vectors_tensor)
            translated = self.translation_transformer.transform(X_train, predictions)
            self.toxicity_transformer.fit(pd.DataFrame({'comment_text': translated, 'toxic': y_train}))


    def validate(self, path_to_validation_data):
        validation_df = pd.read_csv(path_to_validation_data)

        assert 'comment_text' in validation_df.columns
        assert 'lang' in validation_df.columns
        assert 'toxic' in validation_df.columns

        texts = validation_df['comment_text'].values
        langs = validation_df['lang'].values
        true_labels = validation_df['toxic'].values

        predictions = self.predict(texts)

        if isinstance(predictions[0], (list, tuple, np.ndarray)):
            predictions = [pred[0] for pred in predictions]

        rmse = root_mean_squared_error(true_labels, predictions)
        mae = mean_absolute_error(true_labels, predictions)
        logloss = log_loss(true_labels, predictions)
        f1 = f1_score(true_labels, np.round(predictions), average='weighted')  # Rounding for F1

        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if self.train_on_valid:
            self.fit(texts,langs, true_labels)

    def test(self, path_to_test_data):
        test_df = pd.read_csv(path_to_test_data)
        texts = test_df['comment_text'].values
        predictions = self.predict(texts)
        submission_df = pd.DataFrame({'toxic': predictions})
        submission_df.to_csv('data/submission.csv', index=True)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = ToxicityPipeline(
        trainings=[False, False, False],
        word2vec_model_path='models/word2vec/word2vec.model',
        logreg_model_path='models/lang_detect/logistic_regression_torch.pth',
        catboost_model_path = 'models/toxicity/toxicity_catboost_model.joblib',
        vector_size=5000,

        device=device,
        testing= False,
        batch_size=32,
        epochs=100,
        word2vec_and_logreg_df_path='data/combined.csv',
        train_df_path="data/train.csv",
        config_path = "config.ini",
        train_on_valid=False,


    )
    #pipeline.validate('data/validation-processed-seqlen128.csv')
    pipeline.test('data/test-processed-seqlen128.csv')