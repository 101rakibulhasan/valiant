from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import json
import os

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

def decision_trees(data, model_label, feature_label):
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

    # Initialize and train decision_trees
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    # Make predictions
    y_pred = dt_model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return dt_model, accuracy, precision, recall, f1

print("-- Model 1 --")
model_m1, accuracy_m1, precision_m1, recall_m1, f1_m1 = decision_trees(data,'senti-trans', 'senti-trans-feature')
print(f"Accuracy: {accuracy_m1}, Precision: {precision_m1}, Recall: {recall_m1}, F1: {f1_m1}")

print("\n-- Model 2 --")
model_m2, accuracy_m2, precision_m2, recall_m2, f1_m2 = decision_trees(data,'senti-trans2', 'senti-trans-feature2')
print(f"Accuracy: {accuracy_m2}, Precision: {precision_m2}, Recall: {recall_m2}, F1: {f1_m2}")

print("\n-- Model 3 --")
model_m3, accuracy_m3, precision_m3, recall_m3, f1_m3 = decision_trees(data,'senti-trans3', 'senti-trans-feature3')
print(f"Accuracy: {accuracy_m3}, Precision: {precision_m3}, Recall: {recall_m3}, F1: {f1_m3}")

print("\n-- Model 4 --")
model_m4, accuracy_m4, precision_m4, recall_m4, f1_m4 = decision_trees(data,'senti-trans4', 'senti-trans-feature4')
print(f"Accuracy: {accuracy_m4}, Precision: {precision_m4}, Recall: {recall_m4}, F1: {f1_m4}")

print("\n-- Model 5 --")
model_m5, accuracy_m5, precision_m5, recall_m5, f1_m5 = decision_trees(data,'senti-trans5', 'senti-trans-feature5')
print(f"Accuracy: {accuracy_m5}, Precision: {precision_m5}, Recall: {recall_m5}, F1: {f1_m5}")

print("\n-- Model 6 --")
model_m6, accuracy_m6, precision_m6, recall_m6, f1_m6 = decision_trees(data,'senti-emotion', 'trans-emo-feature')
print(f"Accuracy: {accuracy_m6}, Precision: {precision_m6}, Recall: {recall_m6}, F1: {f1_m6}")

print("\n-- Model 7 --")
model_m7, accuracy_m7, precision_m7, recall_m7, f1_m7 = decision_trees(data,'senti-emotion2', 'trans-emo-feature2')
print(f"Accuracy: {accuracy_m7}, Precision: {precision_m7}, Recall: {recall_m7}, F1: {f1_m7}")

print("\n-- Model 8 --")
model_m8, accuracy_m8, precision_m8, recall_m8, f1_m8 = decision_trees(data,'senti-emotion3', 'trans-emo-feature3')
print(f"Accuracy: {accuracy_m8}, Precision: {precision_m8}, Recall: {recall_m8}, F1: {f1_m8}")

print("\n-- Model 9 --")
model_m9, accuracy_m9, precision_m9, recall_m9, f1_m9 = decision_trees(data,'senti-emotion4', 'trans-emo-feature4')
print(f"Accuracy: {accuracy_m9}, Precision: {precision_m9}, Recall: {recall_m9}, F1: {f1_m9}")

print("\n-- Model 10 --")
model_m10, accuracy_m10, precision_m10, recall_m10, f1_m10 = decision_trees(data,'senti-emotion5', 'trans-emo-feature5')
print(f"Accuracy: {accuracy_m10}, Precision: {precision_m10}, Recall: {recall_m10}, F1: {f1_m10}")