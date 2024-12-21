# Generating models using algorithms which will take whole dataset
# Supported algorthms: random_forest, svm, decision_trees, naive_bayes, knn, lightgbm, logistic_regression, xgboost
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

import numpy as np

from imblearn.over_sampling import SMOTE
import json
import pickle
import os

data = []
feature = []
dataset_path = 'Dataset & Code/dataset/conversation-collection-senti-all.json'
temp_path = input("Enter JSON dataset path for Training (Default: Dataset & Code/dataset/conversation-collection-senti-all.json) > ")

if os.path.exists(temp_path):
    dataset_path = temp_path
elif temp_path == '':
    print('[SYS] Path not provided. Using default path.')
else:
    print("[SYS] File not found. Using default path.")

with open('Dataset & Code/dataset/5.1 featured/features.json', 'r') as f:
    feature = json.load(f)

with open(dataset_path, 'r') as f:
    data = json.load(f)

available_algo = [
    ['random_forest', 'Random Forest'],
    ['svm', 'SVM'],
    ['decision_trees', 'Decision Trees'],
    ['naive_bayes', 'Naive Bayes'],
    ['knn', 'K-Nearest Neighbors'],
    ['lightgbm', 'LightGBM'],
    ['logistic_regression', 'Logistic Regression'],
    ['xgboost', 'XGBoost']
]

print("Choose a algorithm: ")
for i in range(len(available_algo)):
    print(f"{i+1}. {available_algo[i][1]}")

algorithm_no = int(input("Algorithm > "))
while algorithm_no not in range(1, len(available_algo)+1):
    print("[SYS] Invalid algorithm number. Please try again.")
    algorithm_no = input("Algorithm > ")

model_algo = available_algo[algorithm_no-1][0]

# Create directory if it does not exist
model_path = f'Dataset & Code/models/{model_algo}'
os.makedirs(f'{model_path}', exist_ok=True)

def gen_model(data, model_label, feature_label, algorithm):
    X = []
    Y = []
    for i in data:
        temp = []

        for j in feature[feature_label]:
            temp.append(i[model_label][j])
        
        X.append(temp)

        if i['result'] == 'AI':
            Y.append(1)
        else:
            Y.append(0)

    # Split data
    X_train = X
    y_train = Y
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = None

    if algorithm == 'random_forest':
        # Initialize and train random_forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif algorithm == 'svm':
        # Initialize and train svm
        model = SVC(kernel='poly', C=1)

    elif algorithm == 'decision_trees':
        # Initialize and train decision_trees
        model = DecisionTreeClassifier(random_state=42)

    elif algorithm == 'naive_bayes':
        # Initialize and train Naive Bayes
        model = MultinomialNB()

    elif algorithm == 'knn':
        # Initialize and train KNN
        model = KNeighborsClassifier(n_neighbors=5)
    
    elif algorithm == 'lightgbm':
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Initialize and train LightGBM
        model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', boosting_type='gbdt', num_leaves=31, learning_rate=0.05, feature_fraction=0.9, verbose=-1)
    
    elif algorithm == 'logistic_regression':
        # Initialize and train Logistic Regression
        model = LogisticRegression()
    
    elif algorithm == 'xgboost':
        # Initialize and train XGBoost
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Initialize and train XGBoost
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)

    

    model.fit(X_train, y_train)

    return model

def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)

print(f"[SYS] Generating models for {available_algo[algorithm_no-1][1]}...")
print("-- Model 1 Created --")
model_m1 = gen_model(data,'senti-trans', 'senti-trans-feature', model_algo)
save_model(model_m1, f'{model_path}/model_m1.pkl')

print("\n-- Model 2 Created --")
model_m2 = gen_model(data,'senti-trans2', 'senti-trans-feature2', model_algo)
save_model(model_m2, f'{model_path}/model_m2.pkl')

print("\n-- Model 3 Created --")
model_m3 = gen_model(data,'senti-trans3', 'senti-trans-feature3', model_algo)
save_model(model_m3, f'{model_path}/model_m3.pkl')

print("\n-- Model 4 Created --")
model_m4 = gen_model(data,'senti-trans4', 'senti-trans-feature4', model_algo)
save_model(model_m4, f'{model_path}/model_m4.pkl')

print("\n-- Model 5 Created --")
model_m5 = gen_model(data,'senti-trans5', 'senti-trans-feature5', model_algo)
save_model(model_m5, f'{model_path}/model_m5.pkl')

print("\n-- Model 6 Created --")
model_m6 = gen_model(data,'senti-emotion', 'trans-emo-feature', model_algo)
save_model(model_m6, f'{model_path}/model_m6.pkl')

print("\n-- Model 7 Created --")
model_m7 = gen_model(data,'senti-emotion2', 'trans-emo-feature2', model_algo)
save_model(model_m7, f'{model_path}/model_m7.pkl')

print("\n-- Model 8 Created --")
model_m8 = gen_model(data,'senti-emotion3', 'trans-emo-feature3', model_algo)
save_model(model_m8, f'{model_path}/model_m8.pkl')

print("\n-- Model 9 Creaeted --")
model_m9 = gen_model(data,'senti-emotion4', 'trans-emo-feature4', model_algo)
save_model(model_m9, f'{model_path}/model_m9.pkl')

print("\n-- Model 10 Created --")
model_m10 = gen_model(data,'senti-emotion5', 'trans-emo-feature5', model_algo)
save_model(model_m10, f'{model_path}/model_m10.pkl')

print(f"[SYS] Models created and saved in {model_path}.")