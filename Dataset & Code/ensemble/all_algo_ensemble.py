import pickle
import numpy as np
import ensemble
from sklearn.ensemble import VotingClassifier

new_estimetors = []

# Load the saved VotingClassifier
with open("Dataset & Code/models/random_forest/best_dt_all_ensamble+smote_models.pkl", "rb") as file:
    dt_model = pickle.load(file)
    new_estimetors.append(dt_model)

with open("Dataset & Code/models/random_forest/best_svm_all_ensamble+smote_models.pkl", "rb") as file:
    svm_model = pickle.load(file)
    new_estimetors.append(svm_model)

with open("Dataset & Code/models/random_forest/best_rf_all_ensamble+smote_models.pkl", "rb") as file:
    rf_model = pickle.load(file)
    new_estimetors.append(rf_model)



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

# Prepare the combined feature sets (as required by the sub-models)
X_combined = [X_senti, X_senti2, X_senti3, X_senti4, X_senti5,
              X_emo, X_emo2, X_emo3, X_emo4, X_emo5]

# Each sub-model should process its specific features
predictions = []
for estimator_name, model in dt_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions.append(model.predict(feature_set))

for estimator_name, model in svm_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions.append(model.predict(feature_set))

for estimator_name, model in rf_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions.append(model.predict(feature_set))

# Custom voting mechanism (majority vote)
from collections import Counter
predictions = zip(*predictions)
y_pred = [Counter(sample).most_common(1)[0][0] for sample in predictions]

# Evaluate performance
ensemble.evaluate_classification(Y_senti, y_pred)

voting_clf = VotingClassifier(estimators=new_estimetors, voting='soft')

with open("Dataset & Code/models/random_forest/best_all_ensamble+smote_models.pkl", "wb") as file:
    pickle.dump(voting_clf, file)