# Checking XGBoost performance by splitting training and testing data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import json
import os
import xgboost as xgb
import numpy as np

dataset_path = 'Dataset & Code/dataset/conversation-collection-senti-all.json'
temp_path = input("Enter JSON dataset path (Default: Dataset & Code/dataset/conversation-collection-senti-all.json) > ")
if os.path.exists(temp_path):
    dataset_path = temp_path
elif temp_path == '':
    print('[SYS] Path not provided. Using default path.')
else:
    print("[SYS] File not found. Using default path.")

data = []
feature = []
with open('Dataset & Code/dataset/5.1 featured/features.json', 'r') as f:
    feature = json.load(f)

with open(dataset_path, 'r') as f:
    data = json.load(f)

def xgboost_model(data, model_label, feature_label):
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
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Initialize and train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.05,
        'max_depth': 6,
        'verbosity': 0  # Suppress warnings
    }
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=100, evals=evallist, verbose_eval=False)

    # Make predictions
    y_pred = bst.predict(dtest)
    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return bst, accuracy, precision, recall, f1

print("\n-- XGBoost Model 1 --")
model_xgb1, accuracy_xgb1, precision_xgb1, recall_xgb1, f1_xgb1 = xgboost_model(data,'senti-trans', 'senti-trans-feature')
print(f"Accuracy: {accuracy_xgb1}, Precision: {precision_xgb1}, Recall: {recall_xgb1}, F1: {f1_xgb1}")

print("\n-- XGBoost Model 2 --")
model_xgb2, accuracy_xgb2, precision_xgb2, recall_xgb2, f1_xgb2 = xgboost_model(data,'senti-trans2', 'senti-trans-feature2')
print(f"Accuracy: {accuracy_xgb2}, Precision: {precision_xgb2}, Recall: {recall_xgb2}, F1: {f1_xgb2}")

print("\n-- XGBoost Model 3 --")
model_xgb3, accuracy_xgb3, precision_xgb3, recall_xgb3, f1_xgb3 = xgboost_model(data,'senti-trans3', 'senti-trans-feature3')
print(f"Accuracy: {accuracy_xgb3}, Precision: {precision_xgb3}, Recall: {recall_xgb3}, F1: {f1_xgb3}")

print("\n-- XGBoost Model 4 --")
model_xgb4, accuracy_xgb4, precision_xgb4, recall_xgb4, f1_xgb4 = xgboost_model(data,'senti-trans4', 'senti-trans-feature4')
print(f"Accuracy: {accuracy_xgb4}, Precision: {precision_xgb4}, Recall: {recall_xgb4}, F1: {f1_xgb4}")

print("\n-- XGBoost Model 5 --")
model_xgb5, accuracy_xgb5, precision_xgb5, recall_xgb5, f1_xgb5 = xgboost_model(data,'senti-trans5', 'senti-trans-feature5')
print(f"Accuracy: {accuracy_xgb5}, Precision: {precision_xgb5}, Recall: {recall_xgb5}, F1: {f1_xgb5}")

print("\n-- XGBoost Model 6 --")
model_xgb6, accuracy_xgb6, precision_xgb6, recall_xgb6, f1_xgb6 = xgboost_model(data,'senti-emotion', 'trans-emo-feature')
print(f"Accuracy: {accuracy_xgb6}, Precision: {precision_xgb6}, Recall: {recall_xgb6}, F1: {f1_xgb6}")

print("\n-- XGBoost Model 7 --")
model_xgb7, accuracy_xgb7, precision_xgb7, recall_xgb7, f1_xgb7 = xgboost_model(data,'senti-emotion2', 'trans-emo-feature2')
print(f"Accuracy: {accuracy_xgb7}, Precision: {precision_xgb7}, Recall: {recall_xgb7}, F1: {f1_xgb7}")

print("\n-- XGBoost Model 8 --")
model_xgb8, accuracy_xgb8, precision_xgb8, recall_xgb8, f1_xgb8 = xgboost_model(data,'senti-emotion3', 'trans-emo-feature3')
print(f"Accuracy: {accuracy_xgb8}, Precision: {precision_xgb8}, Recall: {recall_xgb8}, F1: {f1_xgb8}")

print("\n-- XGBoost Model 9 --")
model_xgb9, accuracy_xgb9, precision_xgb9, recall_xgb9, f1_xgb9 = xgboost_model(data,'senti-emotion4', 'trans-emo-feature4')
print(f"Accuracy: {accuracy_xgb9}, Precision: {precision_xgb9}, Recall: {recall_xgb9}, F1: {f1_xgb9}")

print("\n-- XGBoost Model 10 --")
model_xgb10, accuracy_xgb10, precision_xgb10, recall_xgb10, f1_xgb10 = xgboost_model(data,'senti-emotion5', 'trans-emo-feature5')
print(f"Accuracy: {accuracy_xgb10}, Precision: {precision_xgb10}, Recall: {recall_xgb10}, F1: {f1_xgb10}")