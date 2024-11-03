from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import pickle
import os

def svm(data, model_label, feature_label):
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
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize and train SVM
    svm_model = SVC(kernel='linear', C=1)
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return svm_model, accuracy, precision, recall, f1

def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)

data = []
feature = []
with open('Dataset & Code/dataset/featured/features.json', 'r') as f:
    feature = json.load(f)

with open('Dataset & Code/dataset/conversation-collection-senti.json', 'r') as f:
    data = json.load(f)

# Create directory if it does not exist
os.makedirs('Dataset & Code/models/svm', exist_ok=True)

print("-- Model 1 --")
model_m1, accuracy_m1, precision_m1, recall_m1, f1_m1 = svm(data,'senti-trans', 'senti-trans-feature')
save_model(model_m1, 'Dataset & Code/models/svm/model_m1.pkl')
print(f"Accuracy: {accuracy_m1}, Precision: {precision_m1}, Recall: {recall_m1}, F1: {f1_m1}")

print("\n-- Model 2 --")
model_m2, accuracy_m2, precision_m2, recall_m2, f1_m2 = svm(data,'senti-trans2', 'senti-trans-feature2')
save_model(model_m2, 'Dataset & Code/models/svm/model_m2.pkl')
print(f"Accuracy: {accuracy_m2}, Precision: {precision_m2}, Recall: {recall_m2}, F1: {f1_m2}")

print("\n-- Model 3 --")
model_m3, accuracy_m3, precision_m3, recall_m3, f1_m3 = svm(data,'senti-trans3', 'senti-trans-feature3')
save_model(model_m3, 'Dataset & Code/models/svm/model_m3.pkl')
print(f"Accuracy: {accuracy_m3}, Precision: {precision_m3}, Recall: {recall_m3}, F1: {f1_m3}")

print("\n-- Model 4 --")
model_m4, accuracy_m4, precision_m4, recall_m4, f1_m4 = svm(data,'senti-trans4', 'senti-trans-feature4')
save_model(model_m4, 'Dataset & Code/models/svm/model_m4.pkl')
print(f"Accuracy: {accuracy_m4}, Precision: {precision_m4}, Recall: {recall_m4}, F1: {f1_m4}")

print("\n-- Model 5 --")
model_m5, accuracy_m5, precision_m5, recall_m5, f1_m5 = svm(data,'senti-trans5', 'senti-trans-feature5')
save_model(model_m5, 'Dataset & Code/models/svm/model_m5.pkl')
print(f"Accuracy: {accuracy_m5}, Precision: {precision_m5}, Recall: {recall_m5}, F1: {f1_m5}")

print("\n-- Model 6 --")
model_m6, accuracy_m6, precision_m6, recall_m6, f1_m6 = svm(data,'senti-emotion', 'trans-emo-feature')
save_model(model_m6, 'Dataset & Code/models/svm/model_m6.pkl')
print(f"Accuracy: {accuracy_m6}, Precision: {precision_m6}, Recall: {recall_m6}, F1: {f1_m6}")

print("\n-- Model 7 --")
model_m7, accuracy_m7, precision_m7, recall_m7, f1_m7 = svm(data,'senti-emotion2', 'trans-emo-feature2')
save_model(model_m7, 'Dataset & Code/models/svm/model_m7.pkl')
print(f"Accuracy: {accuracy_m7}, Precision: {precision_m7}, Recall: {recall_m7}, F1: {f1_m7}")

print("\n-- Model 8 --")
model_m8, accuracy_m8, precision_m8, recall_m8, f1_m8 = svm(data,'senti-emotion3', 'trans-emo-feature3')
save_model(model_m8, 'Dataset & Code/models/svm/model_m8.pkl')
print(f"Accuracy: {accuracy_m8}, Precision: {precision_m8}, Recall: {recall_m8}, F1: {f1_m8}")

print("\n-- Model 9 --")
model_m9, accuracy_m9, precision_m9, recall_m9, f1_m9 = svm(data,'senti-emotion4', 'trans-emo-feature4')
save_model(model_m9, 'Dataset & Code/models/svm/model_m9.pkl')
print(f"Accuracy: {accuracy_m9}, Precision: {precision_m9}, Recall: {recall_m9}, F1: {f1_m9}")

print("\n-- Model 10 --")
model_m10, accuracy_m10, precision_m10, recall_m10, f1_m10 = svm(data,'senti-emotion5', 'trans-emo-feature5')
save_model(model_m10, 'Dataset & Code/models/svm/model_m10.pkl')
print(f"Accuracy: {accuracy_m10}, Precision: {precision_m10}, Recall: {recall_m10}, F1: {f1_m10}")