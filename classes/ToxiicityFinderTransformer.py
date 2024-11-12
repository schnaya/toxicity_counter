import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor, Pool
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def convert_to_floats(vector_str):
    return np.array([float(i) for i in vector_str.strip('[]').split()])

class ToxicityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, device, vector_size=10000, trainings=True,testing = False, window=2, min_count=2, epochs=50, dm=1,
                 model_path=None, train_df=None, vectorizer=None,batch_size =32):
        self.device = device
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model_path = model_path
        self.dm = dm
        self.model = None
        self.df = pd.DataFrame()
        self.vectorizer = vectorizer
        self.testing = testing
        if not trainings:
            self.load_model()
        if trainings or testing:
            if train_df is not None and not train_df.empty:

                if trainings:
                    self.fit(train_df)
    def fit(self, train_df):
        self.df = train_df[:2000].copy()
        if 'vectorized' in self.df.columns:
            self.df.loc[:, 'vectorized'] = self.df['vectorized'].apply(convert_to_floats)
        else:
            vectors = self.vectorizer.transform(self.df['comment_text'].values)
            self.df.loc[:, 'vectorized'] = [row for row in vectors]
            self.df.to_csv('data/train_cut_vectorized.csv', index=False)

        X = np.array(self.df['vectorized'].tolist())
        y = self.df['toxic'].values

        assert all(isinstance(x, (int, float)) for row in X for x in row), "Все элементы X должны быть числами"

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model is None:
            self.model = CatBoostRegressor(iterations=1000, depth=10, learning_rate=0.1, loss_function='RMSE', verbose=100)

        train_pool = Pool(X_train, y_train)
        self.model.fit(train_pool)

        if self.testing:
            self.evaluate(X_test, y_test)

        self.save_model()
        return self

    def transform(self, X):
        predictions = self.model.predict(X)
        return predictions

    def save_model(self):
        if not self.model_path:
            raise ValueError("Model path is not provided. Cannot save the model.")
        dir_name = os.path.dirname(self.model_path)
        os.makedirs(dir_name, exist_ok=True)  # Create directory if it doesn't exist

        try:
            dump(self.model, self.model_path)
            print(f"Model saved to {self.model_path}")
        except PermissionError:
            raise RuntimeError(f"Permission denied: Unable to save the model to {self.model_path}.")
        except Exception as e:
            raise RuntimeError(f"Failed to save the model: {str(e)}")

    def load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError("Model path is not provided or file does not exist.")
        try:
            self.model = load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {str(e)}")

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        print(f"Mean Squared Error: {mse}")
        return mse

