# Checking KNN performance by splitting training and testing data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import json
import os
from sklearn.neighbors import KNeighborsClassifier

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

def knn(data, model_label, feature_label):
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

    # Initialize and train KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Make predictions
    y_pred = knn_model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return knn_model, accuracy, precision, recall, f1

print("\n-- KNN Model 1 --")
model_knn1, accuracy_knn1, precision_knn1, recall_knn1, f1_knn1 = knn(data,'senti-trans', 'senti-trans-feature')
print(f"Accuracy: {accuracy_knn1}, Precision: {precision_knn1}, Recall: {recall_knn1}, F1: {f1_knn1}")

print("\n-- KNN Model 2 --")
model_knn2, accuracy_knn2, precision_knn2, recall_knn2, f1_knn2 = knn(data,'senti-trans2', 'senti-trans-feature2')
print(f"Accuracy: {accuracy_knn2}, Precision: {precision_knn2}, Recall: {recall_knn2}, F1: {f1_knn2}")

print("\n-- KNN Model 3 --")
model_knn3, accuracy_knn3, precision_knn3, recall_knn3, f1_knn3 = knn(data,'senti-trans3', 'senti-trans-feature3')
print(f"Accuracy: {accuracy_knn3}, Precision: {precision_knn3}, Recall: {recall_knn3}, F1: {f1_knn3}")

print("\n-- KNN Model 4 --")
model_knn4, accuracy_knn4, precision_knn4, recall_knn4, f1_knn4 = knn(data,'senti-trans4', 'senti-trans-feature4')
print(f"Accuracy: {accuracy_knn4}, Precision: {precision_knn4}, Recall: {recall_knn4}, F1: {f1_knn4}")

print("\n-- KNN Model 5 --")
model_knn5, accuracy_knn5, precision_knn5, recall_knn5, f1_knn5 = knn(data,'senti-trans5', 'senti-trans-feature5')
print(f"Accuracy: {accuracy_knn5}, Precision: {precision_knn5}, Recall: {recall_knn5}, F1: {f1_knn5}")

print("\n-- KNN Model 6 --")
model_knn6, accuracy_knn6, precision_knn6, recall_knn6, f1_knn6 = knn(data,'senti-emotion', 'trans-emo-feature')
print(f"Accuracy: {accuracy_knn6}, Precision: {precision_knn6}, Recall: {recall_knn6}, F1: {f1_knn6}")

print("\n-- KNN Model 7 --")
model_knn7, accuracy_knn7, precision_knn7, recall_knn7, f1_knn7 = knn(data,'senti-emotion2', 'trans-emo-feature2')
print(f"Accuracy: {accuracy_knn7}, Precision: {precision_knn7}, Recall: {recall_knn7}, F1: {f1_knn7}")

print("\n-- KNN Model 8 --")
model_knn8, accuracy_knn8, precision_knn8, recall_knn8, f1_knn8 = knn(data,'senti-emotion3', 'trans-emo-feature3')
print(f"Accuracy: {accuracy_knn8}, Precision: {precision_knn8}, Recall: {recall_knn8}, F1: {f1_knn8}")

print("\n-- KNN Model 9 --")
model_knn9, accuracy_knn9, precision_knn9, recall_knn9, f1_knn9 = knn(data,'senti-emotion4', 'trans-emo-feature4')
print(f"Accuracy: {accuracy_knn9}, Precision: {precision_knn9}, Recall: {recall_knn9}, F1: {f1_knn9}")

print("\n-- KNN Model 10 --")
model_knn10, accuracy_knn10, precision_knn10, recall_knn10, f1_knn10 = knn(data,'senti-emotion5', 'trans-emo-feature5')
print(f"Accuracy: {accuracy_knn10}, Precision: {precision_knn10}, Recall: {recall_knn10}, F1: {f1_knn10}")