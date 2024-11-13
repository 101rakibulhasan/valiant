from collections import Counter
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json

def jsontodict(jsonfile):
    with open(jsonfile, 'r') as file:
        data = json.load(file)
    return data

feature = []
with open('Dataset & Code/dataset/5. featured/features.json', 'r') as f:
    feature = json.load(f)

def get_input(data, feature_label, model_label):
    X = []
    Y = []
    for i in data:
        temp = []

        global feature
        for j in feature[feature_label]:
            temp.append(i[model_label][j])
        
        X.append(temp)

        if i['result'] == 'AI':
            Y.append(1)
        else:
            Y.append(0)

    return X, Y

def ensemble_predict_classification(m1, m2, m3, X1, X2, X3):
    with open(m1, "rb") as file:
        m1_model = pickle.load(file)

    with open(m2, "rb") as file:
        m2_model = pickle.load(file)

    with open(m3, "rb") as file:
        m3_model = pickle.load(file)
        
    m1_preds = m1_model.predict(X1)
    m2_preds = m2_model.predict(X2)
    m3_preds = m3_model.predict(X3)

    ensemble_preds = []
    for i in range(len(m1_preds)):
        prediction = Counter([m1_preds[i], m2_preds[i], m3_preds[i]]).most_common(1)[0][0]
        ensemble_preds.append(prediction)
    
    return ensemble_preds

def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", conf_matrix)