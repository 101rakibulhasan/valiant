# Checking Logistic Regression performance by splitting training and testing data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import json
import os
from sklearn.linear_model import LogisticRegression

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

def logistic_regression(data, model_label, feature_label):
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

    # Initialize and train Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions
    y_pred = lr_model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return lr_model, accuracy, precision, recall, f1

print("\n-- Logistic Regression Model 1 --")
model_lr1, accuracy_lr1, precision_lr1, recall_lr1, f1_lr1 = logistic_regression(data,'senti-trans', 'senti-trans-feature')
print(f"Accuracy: {accuracy_lr1}, Precision: {precision_lr1}, Recall: {recall_lr1}, F1: {f1_lr1}")

print("\n-- Logistic Regression Model 2 --")
model_lr2, accuracy_lr2, precision_lr2, recall_lr2, f1_lr2 = logistic_regression(data,'senti-trans2', 'senti-trans-feature2')
print(f"Accuracy: {accuracy_lr2}, Precision: {precision_lr2}, Recall: {recall_lr2}, F1: {f1_lr2}")

print("\n-- Logistic Regression Model 3 --")
model_lr3, accuracy_lr3, precision_lr3, recall_lr3, f1_lr3 = logistic_regression(data,'senti-trans3', 'senti-trans-feature3')
print(f"Accuracy: {accuracy_lr3}, Precision: {precision_lr3}, Recall: {recall_lr3}, F1: {f1_lr3}")

print("\n-- Logistic Regression Model 4 --")
model_lr4, accuracy_lr4, precision_lr4, recall_lr4, f1_lr4 = logistic_regression(data,'senti-trans4', 'senti-trans-feature4')
print(f"Accuracy: {accuracy_lr4}, Precision: {precision_lr4}, Recall: {recall_lr4}, F1: {f1_lr4}")

print("\n-- Logistic Regression Model 5 --")
model_lr5, accuracy_lr5, precision_lr5, recall_lr5, f1_lr5 = logistic_regression(data,'senti-trans5', 'senti-trans-feature5')
print(f"Accuracy: {accuracy_lr5}, Precision: {precision_lr5}, Recall: {recall_lr5}, F1: {f1_lr5}")

print("\n-- Logistic Regression Model 6 --")
model_lr6, accuracy_lr6, precision_lr6, recall_lr6, f1_lr6 = logistic_regression(data,'senti-emotion', 'trans-emo-feature')
print(f"Accuracy: {accuracy_lr6}, Precision: {precision_lr6}, Recall: {recall_lr6}, F1: {f1_lr6}")

print("\n-- Logistic Regression Model 7 --")
model_lr7, accuracy_lr7, precision_lr7, recall_lr7, f1_lr7 = logistic_regression(data,'senti-emotion2', 'trans-emo-feature2')
print(f"Accuracy: {accuracy_lr7}, Precision: {precision_lr7}, Recall: {recall_lr7}, F1: {f1_lr7}")

print("\n-- Logistic Regression Model 8 --")
model_lr8, accuracy_lr8, precision_lr8, recall_lr8, f1_lr8 = logistic_regression(data,'senti-emotion3', 'trans-emo-feature3')
print(f"Accuracy: {accuracy_lr8}, Precision: {precision_lr8}, Recall: {recall_lr8}, F1: {f1_lr8}")

print("\n-- Logistic Regression Model 9 --")
model_lr9, accuracy_lr9, precision_lr9, recall_lr9, f1_lr9 = logistic_regression(data,'senti-emotion4', 'trans-emo-feature4')
print(f"Accuracy: {accuracy_lr9}, Precision: {precision_lr9}, Recall: {recall_lr9}, F1: {f1_lr9}")

print("\n-- Logistic Regression Model 10 --")
model_lr10, accuracy_lr10, precision_lr10, recall_lr10, f1_lr10 = logistic_regression(data,'senti-emotion5', 'trans-emo-feature5')
print(f"Accuracy: {accuracy_lr10}, Precision: {precision_lr10}, Recall: {recall_lr10}, F1: {f1_lr10}")