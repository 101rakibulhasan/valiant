# Checking LightGBM performance by splitting training and testing data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import json
import os
import lightgbm as lgb
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

def lightgbm_model(data, model_label, feature_label):
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

    # Initialize and train LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1  # Suppress warnings
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval)

    # Make predictions
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return gbm, accuracy, precision, recall, f1

print("\n-- LightGBM Model 1 --")
model_lgb1, accuracy_lgb1, precision_lgb1, recall_lgb1, f1_lgb1 = lightgbm_model(data,'senti-trans', 'senti-trans-feature')
print(f"Accuracy: {accuracy_lgb1}, Precision: {precision_lgb1}, Recall: {recall_lgb1}, F1: {f1_lgb1}")

print("\n-- LightGBM Model 2 --")
model_lgb2, accuracy_lgb2, precision_lgb2, recall_lgb2, f1_lgb2 = lightgbm_model(data,'senti-trans2', 'senti-trans-feature2')
print(f"Accuracy: {accuracy_lgb2}, Precision: {precision_lgb2}, Recall: {recall_lgb2}, F1: {f1_lgb2}")

print("\n-- LightGBM Model 3 --")
model_lgb3, accuracy_lgb3, precision_lgb3, recall_lgb3, f1_lgb3 = lightgbm_model(data,'senti-trans3', 'senti-trans-feature3')
print(f"Accuracy: {accuracy_lgb3}, Precision: {precision_lgb3}, Recall: {recall_lgb3}, F1: {f1_lgb3}")

print("\n-- LightGBM Model 4 --")
model_lgb4, accuracy_lgb4, precision_lgb4, recall_lgb4, f1_lgb4 = lightgbm_model(data,'senti-trans4', 'senti-trans-feature4')
print(f"Accuracy: {accuracy_lgb4}, Precision: {precision_lgb4}, Recall: {recall_lgb4}, F1: {f1_lgb4}")

print("\n-- LightGBM Model 5 --")
model_lgb5, accuracy_lgb5, precision_lgb5, recall_lgb5, f1_lgb5 = lightgbm_model(data,'senti-trans5', 'senti-trans-feature5')
print(f"Accuracy: {accuracy_lgb5}, Precision: {precision_lgb5}, Recall: {recall_lgb5}, F1: {f1_lgb5}")

print("\n-- LightGBM Model 6 --")
model_lgb6, accuracy_lgb6, precision_lgb6, recall_lgb6, f1_lgb6 = lightgbm_model(data,'senti-emotion', 'trans-emo-feature')
print(f"Accuracy: {accuracy_lgb6}, Precision: {precision_lgb6}, Recall: {recall_lgb6}, F1: {f1_lgb6}")

print("\n-- LightGBM Model 7 --")
model_lgb7, accuracy_lgb7, precision_lgb7, recall_lgb7, f1_lgb7 = lightgbm_model(data,'senti-emotion2', 'trans-emo-feature2')
print(f"Accuracy: {accuracy_lgb7}, Precision: {precision_lgb7}, Recall: {recall_lgb7}, F1: {f1_lgb7}")

print("\n-- LightGBM Model 8 --")
model_lgb8, accuracy_lgb8, precision_lgb8, recall_lgb8, f1_lgb8 = lightgbm_model(data,'senti-emotion3', 'trans-emo-feature3')
print(f"Accuracy: {accuracy_lgb8}, Precision: {precision_lgb8}, Recall: {recall_lgb8}, F1: {f1_lgb8}")

print("\n-- LightGBM Model 9 --")
model_lgb9, accuracy_lgb9, precision_lgb9, recall_lgb9, f1_lgb9 = lightgbm_model(data,'senti-emotion4', 'trans-emo-feature4')
print(f"Accuracy: {accuracy_lgb9}, Precision: {precision_lgb9}, Recall: {recall_lgb9}, F1: {f1_lgb9}")

print("\n-- LightGBM Model 10 --")
model_lgb10, accuracy_lgb10, precision_lgb10, recall_lgb10, f1_lgb10 = lightgbm_model(data,'senti-emotion5', 'trans-emo-feature5')
print(f"Accuracy: {accuracy_lgb10}, Precision: {precision_lgb10}, Recall: {recall_lgb10}, F1: {f1_lgb10}")