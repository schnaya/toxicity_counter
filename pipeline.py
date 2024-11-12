import re
import nltk
import pandas as pd
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Для сохранения модели
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Загрузка данных
train = pd.read_csv('data/train.csv')
validation = pd.read_csv('data/validation.csv')
test = pd.read_csv('data/test.csv')

# Загрузка необходимых библиотек
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Функция предобработки текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

# Класс для Doc2Vec с прогрессом
class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=2, min_count=1, epochs=40):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, X, y=None):

        tagged_data = [TaggedDocument(words=row.split(), tags=[str(i)]) for i, row in enumerate(X)]

        # Обучение модели Doc2Vec с прогрессом
        self.model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=4, epochs=self.epochs)
        self.model.build_vocab(tagged_data)
        for epoch in tqdm(range(self.epochs), desc="Training Doc2Vec"):
            self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=1)
        return self

    def transform(self, X):
        return pd.DataFrame([self.model.infer_vector(row.split()) for row in X])

# Создание Pipeline с Doc2Vec
pipeline = Pipeline([
    ('doc2vec', Doc2VecTransformer()),  # Преобразование текстов с помощью Doc2Vec
    ('catboost', CatBoostRegressor(iterations=1000, depth=10, learning_rate=0.1,
                                   loss_function='RMSE', devices='0', verbose=100))
])

# Определение целевой переменной
X = train['cleaned_comment_text'].fillna("  ")  # Используем исходный текст для предобработки внутри трансформера
y = train['toxic']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение Pipeline с прогрессом
print("Fitting pipeline...")
pipeline.fit(X_train, y_train)

# Кросс-валидация с прогрессом
print("Cross-validation...")
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
mean_mse = -scores.mean()
print(f'Mean Squared Error from Cross-Validation: {mean_mse}')

# Предсказание
y_pred = pipeline.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error with CatBoost: {mse}')
print(f'R² Score: {r2}')

# Сохранение модели
joblib.dump(pipeline, 'catboost_pipeline_with_doc2vec.pkl')