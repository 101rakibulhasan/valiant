import json
import pickle

def get_model_result(pickle_file, X_test):
    with open(pickle_file, 'rb') as file:
        loaded_model = pickle.load(file)

        # Use the loaded model to make predictions
        y_pred = loaded_model.predict(X_test)
        return y_pred
    
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

    return X

data_path = 'Dataset & Code/example/chat-human.json'
data = jsontodict(data_path)

X_senti = get_input(data, "senti-trans-feature", "senti-trans")
X_senti2 = get_input(data, "senti-trans-feature2", "senti-trans2")
X_senti3 = get_input(data, "senti-trans-feature3", "senti-trans3")
X_senti4 = get_input(data, "senti-trans-feature4", "senti-trans4")
X_senti5 = get_input(data, "senti-trans-feature5", "senti-trans5")
X_emo= get_input(data, "trans-emo-feature", "senti-emotion")
X_emo2 = get_input(data, "trans-emo-feature2", "senti-emotion2")
X_emo3 = get_input(data, "trans-emo-feature3", "senti-emotion3")
X_emo4 = get_input(data, "trans-emo-feature4", "senti-emotion4")
X_emo5 = get_input(data, "trans-emo-feature5", "senti-emotion5")

def model_says(model, X_test1):
    temp = get_model_result(model, X_test1)
    if temp[0] == 1:
        return "AI"
    else:
        return "Human"
    
# 1 - AI
# 0 - Human
print("-- SVM --")
print(f"The model1 says: {model_says('Dataset & Code/models/svm/model_m1.pkl', X_senti)}")
print(f"The model2 says: {model_says('Dataset & Code/models/svm/model_m2.pkl', X_senti2)}")
print(f"The model3 says: {model_says('Dataset & Code/models/svm/model_m3.pkl', X_senti3)}")
print(f"The model4 says: {model_says('Dataset & Code/models/svm/model_m4.pkl', X_senti4)}")
print(f"The model5 says: {model_says('Dataset & Code/models/svm/model_m5.pkl', X_senti5)}")
print(f"The model6 says: {model_says('Dataset & Code/models/svm/model_m6.pkl', X_emo)}")
print(f"The model7 says: {model_says('Dataset & Code/models/svm/model_m7.pkl', X_emo2)}")
print(f"The model8 says: {model_says('Dataset & Code/models/svm/model_m8.pkl', X_emo3)}")
print(f"The model9 says: {model_says('Dataset & Code/models/svm/model_m9.pkl', X_emo4)}")
print(f"The model10 says: {model_says('Dataset & Code/models/svm/model_m10.pkl', X_emo5)}")

print("\n-- Decision Tree --")
print(f"The model1 says: {model_says('Dataset & Code/models/decision_trees/model_m1.pkl', X_senti)}")
print(f"The model2 says: {model_says('Dataset & Code/models/decision_trees/model_m2.pkl', X_senti2)}")
print(f"The model3 says: {model_says('Dataset & Code/models/decision_trees/model_m3.pkl', X_senti3)}")
print(f"The model4 says: {model_says('Dataset & Code/models/decision_trees/model_m4.pkl', X_senti4)}")
print(f"The model5 says: {model_says('Dataset & Code/models/decision_trees/model_m5.pkl', X_senti5)}")
print(f"The model6 says: {model_says('Dataset & Code/models/decision_trees/model_m6.pkl', X_emo)}")
print(f"The model7 says: {model_says('Dataset & Code/models/decision_trees/model_m7.pkl', X_emo2)}")
print(f"The model8 says: {model_says('Dataset & Code/models/decision_trees/model_m8.pkl', X_emo3)}")
print(f"The model9 says: {model_says('Dataset & Code/models/decision_trees/model_m9.pkl', X_emo4)}")
print(f"The model10 says: {model_says('Dataset & Code/models/decision_trees/model_m10.pkl', X_emo5)}")

print("\n-- Random Forest --")
print(f"The model1 says: {model_says('Dataset & Code/models/random_forest/model_m1.pkl', X_senti)}")
print(f"The model2 says: {model_says('Dataset & Code/models/random_forest/model_m2.pkl', X_senti2)}")
print(f"The model3 says: {model_says('Dataset & Code/models/random_forest/model_m3.pkl', X_senti3)}")
print(f"The model4 says: {model_says('Dataset & Code/models/random_forest/model_m4.pkl', X_senti4)}")
print(f"The model5 says: {model_says('Dataset & Code/models/random_forest/model_m5.pkl', X_senti5)}")
print(f"The model6 says: {model_says('Dataset & Code/models/random_forest/model_m6.pkl', X_emo)}")
print(f"The model7 says: {model_says('Dataset & Code/models/random_forest/model_m7.pkl', X_emo2)}")
print(f"The model8 says: {model_says('Dataset & Code/models/random_forest/model_m8.pkl', X_emo3)}")
print(f"The model9 says: {model_says('Dataset & Code/models/random_forest/model_m9.pkl', X_emo4)}")
print(f"The model10 says: {model_says('Dataset & Code/models/random_forest/model_m10.pkl', X_emo5)}")

print("\n-- Ensembled and SMOTE on Random Forest --")
with open('Dataset & Code/models/random_forest/best_rf_all_ensamble+smote_models.pkl', 'rb') as file:
    rf_all_loaded_model = pickle.load(file)

predictions = []
feature_set1 = rf_all_loaded_model.estimators[0][1].predict(X_senti)
feature_set2 = rf_all_loaded_model.estimators[1][1].predict(X_senti2)
feature_set3 = rf_all_loaded_model.estimators[2][1].predict(X_senti3)
feature_set4 = rf_all_loaded_model.estimators[3][1].predict(X_senti4)
feature_set5 = rf_all_loaded_model.estimators[4][1].predict(X_senti5)
feature_set6 = rf_all_loaded_model.estimators[5][1].predict(X_emo)
feature_set7 = rf_all_loaded_model.estimators[6][1].predict(X_emo2)
feature_set8 = rf_all_loaded_model.estimators[7][1].predict(X_emo3)
feature_set9 = rf_all_loaded_model.estimators[8][1].predict(X_emo4)
feature_set10 = rf_all_loaded_model.estimators[9][1].predict(X_emo5)

predictions.append(feature_set1)
predictions.append(feature_set2)
predictions.append(feature_set3)
predictions.append(feature_set4)
predictions.append(feature_set5)
predictions.append(feature_set6)
predictions.append(feature_set7)
predictions.append(feature_set8)
predictions.append(feature_set9)
predictions.append(feature_set10)

from collections import Counter
predictions = zip(*predictions)
y_pred = [Counter(sample).most_common(1)[0][0] for sample in predictions]

# Evaluate performance
most_common_prediction = Counter(y_pred).most_common(1)[0][0]
ans = ""
if most_common_prediction == 1:
    ans = "AI"
else:
    ans = "Human"
print(f"The model says: {ans}")