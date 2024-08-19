
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow


df = pd.read_csv('data.csv')

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

features = x_train.columns
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = pd.DataFrame(x_train, columns=features)
x_test = pd.DataFrame(x_test, columns=features)

x_train.drop('Unnamed: 32', axis=1, inplace=True)
x_test.drop('Unnamed: 32', axis=1, inplace=True)

x_train.drop('id', axis=1, inplace=True)
x_test.drop('id', axis=1, inplace=True)

params = {
    'max_depth': 5,
    'random_state': 42
}

model = DecisionTreeClassifier(**params)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision_score = precision_score(y_test, y_pred, pos_label='M')
recall_score = recall_score(y_test, y_pred, pos_label='M')
f1_score = f1_score(y_test, y_pred, pos_label='M')
confusion_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred, output_dict=True)
print(class_report)


mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Breast Cancer")

mlflow.log_params(params)
mlflow.log_metrics(
    {
        "accuracy": class_report['accuracy'],
        "recall_class_0": class_report['B']['recall'],
        "recall_class_1": class_report['M']['recall'],
        "f1_score": class_report['macro avg']['f1-score']
    }
)
mlflow.sklearn.log_model(model, "DecicisonTreeClassifier")