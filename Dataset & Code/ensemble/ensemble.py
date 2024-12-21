from collections import Counter
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
from sklearn.ensemble import VotingClassifier

def jsontodict(jsonfile):
    with open(jsonfile, 'r') as file:
        data = json.load(file)
    return data

feature = []
with open('Dataset & Code/dataset/5.1 featured/features.json', 'r') as f:
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

def ensemble_predict_classification(path, best_models, features_label):
    y_predictions = []

    index = 0
    actual_y = []
    for model_name in best_models:
        with open(model_name, "rb") as file:
            model = pickle.load(file)
            X, Y = get_input(jsontodict(path), features_label[index][0], features_label[index][1])
            actual_y.append(Y)
            predictions = model.predict(X)
            y_predictions.append(predictions)
            index += 1

    ensemble_preds = []
    for i in range(len(y_predictions[0])):
        prediction = Counter([y_predictions[j][i] for j in range(len(y_predictions))]).most_common(1)[0][0]
        ensemble_preds.append(prediction)
    
    return actual_y, ensemble_preds

def save_ensemble_model(models_name, filename):
    estimators = []
    for model_name in models_name:
        with open(model_name, "rb") as file:
            model = pickle.load(file)
            estimators.append(model)
    
    voting_clf = VotingClassifier(estimators=estimators, voting='hard')
    with open(filename, "wb") as file:
        pickle.dump(voting_clf, file)
    
    print("[SYS] Ensemble model saved to", filename)

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

    return accuracy, f1, precision, recall, conf_matrix