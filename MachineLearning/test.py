import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from DecisionStumpClassifier import DecisionStumpClassifier 

auto_mpg = pd.read_parquet('./data/auto-mpg.parquet', engine='pyarrow')
autos = pd.read_parquet('./data/autos.parquet', engine='pyarrow')
hungarian_heart_disease = pd.read_parquet('./data/hungarian-heart-disease.parquet', engine='pyarrow')

datasets = {
    'auto-mpg': auto_mpg,
    # 'autos': autos
    'hungarian-heart-disease': hungarian_heart_disease
}

def evaluate_model(classifier, X, y):
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        scores.append(balanced_accuracy_score(y_test, y_pred))
    return np.mean(scores)

results = pd.DataFrame(columns=['DS', 'DT (max_depth=1)', 'DT'])

for name, dataset in datasets.items():
    X = dataset.drop(columns=['class'])
    y = dataset['class']
    results.loc[name] = [
        evaluate_model(DecisionStumpClassifier(), X, y),
        evaluate_model(DecisionTreeClassifier(max_depth=1), X, y),
        evaluate_model(DecisionTreeClassifier(), X, y)
    ]

print(results)