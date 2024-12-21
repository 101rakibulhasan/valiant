# Checking Naive Bayes performance by splitting training and testing data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import json
import os
from sklearn.naive_bayes import MultinomialNB

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

def naive_bayes(data, model_label, feature_label):
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

    # Initialize and train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = nb_model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return nb_model, accuracy, precision, recall, f1

print("\n-- Naive Bayes Model 1 --")
model_nb1, accuracy_nb1, precision_nb1, recall_nb1, f1_nb1 = naive_bayes(data,'senti-trans', 'senti-trans-feature')
print(f"Accuracy: {accuracy_nb1}, Precision: {precision_nb1}, Recall: {recall_nb1}, F1: {f1_nb1}")

print("\n-- Naive Bayes Model 2 --")
model_nb2, accuracy_nb2, precision_nb2, recall_nb2, f1_nb2 = naive_bayes(data,'senti-trans2', 'senti-trans-feature2')
print(f"Accuracy: {accuracy_nb2}, Precision: {precision_nb2}, Recall: {recall_nb2}, F1: {f1_nb2}")

print("\n-- Naive Bayes Model 3 --")
model_nb3, accuracy_nb3, precision_nb3, recall_nb3, f1_nb3 = naive_bayes(data,'senti-trans3', 'senti-trans-feature3')
print(f"Accuracy: {accuracy_nb3}, Precision: {precision_nb3}, Recall: {recall_nb3}, F1: {f1_nb3}")

print("\n-- Naive Bayes Model 4 --")
model_nb4, accuracy_nb4, precision_nb4, recall_nb4, f1_nb4 = naive_bayes(data,'senti-trans4', 'senti-trans-feature4')
print(f"Accuracy: {accuracy_nb4}, Precision: {precision_nb4}, Recall: {recall_nb4}, F1: {f1_nb4}")

print("\n-- Naive Bayes Model 5 --")
model_nb5, accuracy_nb5, precision_nb5, recall_nb5, f1_nb5 = naive_bayes(data,'senti-trans5', 'senti-trans-feature5')
print(f"Accuracy: {accuracy_nb5}, Precision: {precision_nb5}, Recall: {recall_nb5}, F1: {f1_nb5}")

print("\n-- Naive Bayes Model 6 --")
model_nb6, accuracy_nb6, precision_nb6, recall_nb6, f1_nb6 = naive_bayes(data,'senti-emotion', 'trans-emo-feature')
print(f"Accuracy: {accuracy_nb6}, Precision: {precision_nb6}, Recall: {recall_nb6}, F1: {f1_nb6}")

print("\n-- Naive Bayes Model 7 --")
model_nb7, accuracy_nb7, precision_nb7, recall_nb7, f1_nb7 = naive_bayes(data,'senti-emotion2', 'trans-emo-feature2')
print(f"Accuracy: {accuracy_nb7}, Precision: {precision_nb7}, Recall: {recall_nb7}, F1: {f1_nb7}")

print("\n-- Naive Bayes Model 8 --")
model_nb8, accuracy_nb8, precision_nb8, recall_nb8, f1_nb8 = naive_bayes(data,'senti-emotion3', 'trans-emo-feature3')
print(f"Accuracy: {accuracy_nb8}, Precision: {precision_nb8}, Recall: {recall_nb8}, F1: {f1_nb8}")

print("\n-- Naive Bayes Model 9 --")
model_nb9, accuracy_nb9, precision_nb9, recall_nb9, f1_nb9 = naive_bayes(data,'senti-emotion4', 'trans-emo-feature4')
print(f"Accuracy: {accuracy_nb9}, Precision: {precision_nb9}, Recall: {recall_nb9}, F1: {f1_nb9}")

print("\n-- Naive Bayes Model 10 --")
model_nb10, accuracy_nb10, precision_nb10, recall_nb10, f1_nb10 = naive_bayes(data,'senti-emotion5', 'trans-emo-feature5')
print(f"Accuracy: {accuracy_nb10}, Precision: {precision_nb10}, Recall: {recall_nb10}, F1: {f1_nb10}")