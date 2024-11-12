import json
from configparser import ConfigParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class LogisticRegressionTorchTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_size, device,testing = False,trainings=True,train_df=None,word2vec=None, model_path=None, lr=0.01,batch_size=32, num_epochs=100,langs=7,config_path=None):
        self.input_size = input_size
        self.model_path = model_path 
        self.device = device
        self.word2vec=word2vec
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.lang_to_id = None
        self.criterion = nn.CrossEntropyLoss()
        self.model=LogisticRegressionTorch(input_size,langs).to(self.device)
        self.config_path=config_path
        if self.config_path:
            config = ConfigParser()
            config.read(config_path)
            self.lang_to_id = {lang: int(id) for lang, id in config.items("languages")}
        if not trainings:
            if model_path:
                self.load_model()
        if trainings or testing:
            self.train_df = train_df
            num_classes,train_loader,test_loader=self.prepare_train_df()
            if trainings:
                self.model = LogisticRegressionTorch(input_size, num_classes).to(self.device)
                self.fit(train_loader)
            if testing:
                self.score(test_loader)


    def save_config_to_ini(self,lang_to_id, config_path="config.ini"):
        config = ConfigParser()
        config.add_section("languages")
        for lang, id in lang_to_id.items():
            config.set("languages", lang, str(id))

        with open(config_path, "w") as f:
            config.write(f)
        self.config_path=config_path
    def prepare_train_df(self):
        if self.lang_to_id is None:
            unique_langs = self.train_df['lang'].unique()
            self.lang_to_id = {lang: idx for idx, lang in enumerate(unique_langs)}
            if self.config_path == None :
                self.save_config_to_ini(self.lang_to_id)
        self.train_df['lang_id'] = self.train_df['lang'].map(self.lang_to_id)
        X = self.word2vec.transform(self.train_df['content'])
        y = self.train_df['lang_id']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train.values, dtype=torch.long, device=self.device)
        y_test = torch.tensor(y_test.values, dtype=torch.long, device=self.device)

    # Создание TensorDataset и DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        return len(self.lang_to_id), train_loader, test_loader

    def fit(self, train_loader=None,config_path=None):
        losses = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0  
            num_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()  
                num_batches += 1 

            average_loss = total_loss / num_batches
            losses.append(average_loss)
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{self.num_epochs}], Loss: {average_loss:.4f}')
        
        if self.model_path:
            self.save_model()

    def predict(self, X):
        # Transform (predict) for scikit-learn compatibility
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X).float().to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def valid_fit(self, X_train, y_train, config_path=None):
        self.train_df=pd.DataFrame({'content':X_train,'lang':y_train})
        num_classes, train_loader, test_loader = self.prepare_train_df()
        self.fit(train_loader)
        self.score(test_loader)
    def score(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                test_outputs = self.model(batch_X)
                _, predicted = torch.max(test_outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy
    def save_model(self):
        if not self.model_path:
            raise ValueError("Model path is not provided. Cannot save the model.")
        try:
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Model saved to {self.model_path}")
            joblib.dump(self.model,'models/lang_detect/logistic_regression_torch.joblib')
        except Exception as e:
            raise RuntimeError(f"Failed to save the model: {str(e)}")

    def load_model(self):
        if not self.model_path:
            raise ValueError("Model path is not provided or the file does not exist.")
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {str(e)}")
