# ROC Curve of 3 Ensembled Model Curve

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import sys
sys.path.append ('./Dataset & Code/ensemble')
import ensemble

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

X_combined = [X_senti, X_senti2, X_senti3, X_senti4, X_senti5, X_emo, X_emo2, X_emo3, X_emo4, X_emo5]

Y_combined = Y_senti    # All Y values are the same, so we can use any of them

with open("Dataset & Code/models/random_forest/best_rf_all_ensamble+smote_models.pkl", 'rb') as file:
    loaded_model_rf = pickle.load(file)

with open("Dataset & Code/models/svm/best_svm_all_ensamble+smote_models.pkl", 'rb') as file:
    loaded_model_svm = pickle.load(file)

with open("Dataset & Code/models/decision_trees/best_dt_all_ensamble+smote_models.pkl", 'rb') as file:
    loaded_model_dt = pickle.load(file)

predictions_rf = []
for estimator_name, model in loaded_model_rf.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_rf.append(model.predict(feature_set))

predictions_svm = []
for estimator_name, model in loaded_model_svm.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_svm.append(model.predict(feature_set))

predictions_dt = []
for estimator_name, model in loaded_model_dt.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_dt.append(model.predict(feature_set))

from collections import Counter
predictions = zip(*predictions_rf)
y_pred_rf = [Counter(sample).most_common(1)[0][0] for sample in predictions]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(Y_combined, y_pred_rf)

predictions = zip(*predictions_svm)
y_pred_svm = [Counter(sample).most_common(1)[0][0] for sample in predictions]
fpr_svm, tpr_svm, thresholds_svm = roc_curve(Y_combined, y_pred_svm)

predictions = zip(*predictions_dt)
y_pred_dt = [Counter(sample).most_common(1)[0][0] for sample in predictions]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(Y_combined, y_pred_dt)


# Calculate AUC
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'RF ROC curve (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'SVM ROC curve (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_dt, tpr_dt, color='yellow', lw=2, label=f'DT ROC curve (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
