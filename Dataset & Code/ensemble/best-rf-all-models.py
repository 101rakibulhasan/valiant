## All models of Random Forest

import ensemble
import pickle
from collections import Counter

X_senti, Y_senti = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature", "senti-trans")
X_senti2, Y_senti2 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature2", "senti-trans2")
X_senti3, Y_senti3 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature3", "senti-trans3")
X_senti4, Y_senti4 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature4", "senti-trans4")
X_senti5, Y_senti5 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature5", "senti-trans5")
X_emo, Y_emo = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature", "senti-emotion")
X_emo2, Y_emo2 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature2", "senti-emotion2")
X_emo3, Y_emo3 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature3", "senti-emotion3")
X_emo4, Y_emo4 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature4", "senti-emotion4")
X_emo5, Y_emo5 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature5", "senti-emotion5")
# All Y value here are the same, so we can use any of them

with open("Dataset & Code/models/random_forest/model_m1.pkl", "rb") as file:
    m1_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/model_m2.pkl", "rb") as file:
    m2_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/model_m3.pkl", "rb") as file:
    m3_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/model_m4.pkl", "rb") as file:
    m4_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/model_m5.pkl", "rb") as file:
    m5_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/model_m6.pkl", "rb") as file:
    m6_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/model_m7.pkl", "rb") as file:
    m7_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/model_m8.pkl", "rb") as file:
    m8_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/model_m9.pkl", "rb") as file:
    m9_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/model_m10.pkl", "rb") as file:
    m10_model = pickle.load(file)
    
m1_preds = m1_model.predict(X_senti)
m2_preds = m2_model.predict(X_senti2)
m3_preds = m3_model.predict(X_senti3)
m4_preds = m4_model.predict(X_senti4)
m5_preds = m5_model.predict(X_senti5)
m6_preds = m6_model.predict(X_emo)
m7_preds = m7_model.predict(X_emo2)
m8_preds = m8_model.predict(X_emo3)
m9_preds = m9_model.predict(X_emo4)
m10_preds = m10_model.predict(X_emo5)

y_pred = []
for i in range(len(m1_preds)):
    prediction = Counter([m1_preds[i], m2_preds[i], m3_preds[i], m4_preds[i], m5_preds[i], m6_preds[i], m7_preds[i], m8_preds[i], m9_preds[i], m10_preds[i]]).most_common(1)[0][0]
    y_pred.append(prediction)

ensemble.evaluate_classification(Y_senti, y_pred)

# Output:
"""
Accuracy: 0.9488636363636364
F1 Score: 0.9488223320732772
Precision: 0.9503173986267651
Recall: 0.9488636363636364
Confusion Matrix:
 [[86  2]
 [ 7 81]]
"""
# After SMOTE
"""
Accuracy: 0.9602272727272727
F1 Score: 0.9601951471681044
Precision: 0.9617178390983288
Recall: 0.9602272727272727
Confusion Matrix:
 [[87  1]
 [ 6 82]]
"""