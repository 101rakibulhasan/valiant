# ROC Curve of 3 Ensembled Model Curve

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import sys
sys.path.append ('./Dataset & Code/ensemble')
import ensemble
import numpy as np

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

# Check unique values in Y_combined
unique_classes = np.unique(Y_combined)
print("Unique values in Y_combined:", unique_classes)

# Ensure Y_combined contains both classes
if len(unique_classes) < 2:
    raise ValueError("Y_combined contains only one class. Ensure the dataset contains both classes.")

with open("Dataset & Code/models/decision_trees/best_dt_all_ensamble+smote_models.pkl", "rb") as file:
    dt_model = pickle.load(file)

with open("Dataset & Code/models/svm/best_svm_all_ensamble+smote_models.pkl", "rb") as file:
    svm_model = pickle.load(file)

with open("Dataset & Code/models/random_forest/best_rf_all_ensamble+smote_models.pkl", "rb") as file:
    rf_model = pickle.load(file)

with open("Dataset & Code/models/knn/best_knn_all_ensamble+smote_models.pkl", "rb") as file:
    knn_model = pickle.load(file)

with open("Dataset & Code/models/lightgbm/best_lightgbm_all_ensamble+smote_models.pkl", "rb") as file:
    lightgbm_model = pickle.load(file)

with open("Dataset & Code/models/logistic_regression/best_logistic_regression_all_ensamble+smote_models.pkl", "rb") as file:
    logistic_regression_model = pickle.load(file)

with open("Dataset & Code/models/naive_bayes/best_naive_bayes_all_ensamble+smote_models.pkl", "rb") as file:
    naive_bayes_model = pickle.load(file)

with open("Dataset & Code/models/xgboost/best_xgboost_all_ensamble+smote_models.pkl", "rb") as file:
    xgboost_model = pickle.load(file)

predictions_rf = []
predictions_svm = []
predictions_dt = []
predictions_knn = []
predictions_lightgbm = []
predictions_logistic_regression = []
predictions_naive_bayes = []
predictions_xgboost = []

for estimator_name, model in rf_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_rf.append(model.predict(feature_set))

for estimator_name, model in svm_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_svm.append(model.predict(feature_set))

for estimator_name, model in dt_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_dt.append(model.predict(feature_set))

for estimator_name, model in knn_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_knn.append(model.predict(feature_set))

for estimator_name, model in lightgbm_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_lightgbm.append(model.predict(feature_set))

for estimator_name, model in logistic_regression_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_logistic_regression.append(model.predict(feature_set))

for estimator_name, model in naive_bayes_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_naive_bayes.append(model.predict(feature_set))

for estimator_name, model in xgboost_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_xgboost.append(model.predict(feature_set))


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

predictions = zip(*predictions_knn)
y_pred_knn = [Counter(sample).most_common(1)[0][0] for sample in predictions]
fpr_knn, tpr_knn, thresholds_knn = roc_curve(Y_combined, y_pred_knn)

predictions = zip(*predictions_lightgbm)
y_pred_lightgbm = [Counter(sample).most_common(1)[0][0] for sample in predictions]
fpr_lightgbm, tpr_lightgbm, thresholds_lightgbm = roc_curve(Y_combined, y_pred_lightgbm)

predictions = zip(*predictions_logistic_regression)
y_pred_logistic_regression = [Counter(sample).most_common(1)[0][0] for sample in predictions]
fpr_logistic_regression, tpr_logistic_regression, thresholds_logistic_regression = roc_curve(Y_combined, y_pred_logistic_regression)

predictions = zip(*predictions_naive_bayes)
y_pred_naive_bayes = [Counter(sample).most_common(1)[0][0] for sample in predictions]
fpr_naive_bayes, tpr_naive_bayes, thresholds_naive_bayes = roc_curve(Y_combined, y_pred_naive_bayes)

predictions = zip(*predictions_xgboost)
y_pred_xgboost = [Counter(sample).most_common(1)[0][0] for sample in predictions]
fpr_xgboost, tpr_xgboost, thresholds_xgboost = roc_curve(Y_combined, y_pred_xgboost)


# Calculate AUC
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_dt = auc(fpr_dt, tpr_dt)
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_lightgbm = auc(fpr_lightgbm, tpr_lightgbm)
roc_auc_logistic_regression = auc(fpr_logistic_regression, tpr_logistic_regression)
roc_auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes)
roc_auc_xgboost = auc(fpr_xgboost, tpr_xgboost)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'RF ROC curve (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'SVM ROC curve (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_dt, tpr_dt, color='yellow', lw=2, label=f'DT ROC curve (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_knn, tpr_knn, color='green', lw=2, label=f'KNN ROC curve (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_lightgbm, tpr_lightgbm, color='purple', lw=2, label=f'LightGBM ROC curve (AUC = {roc_auc_lightgbm:.2f})')
plt.plot(fpr_logistic_regression, tpr_logistic_regression, color='orange', lw=2, label=f'Logistic Regression ROC curve (AUC = {roc_auc_logistic_regression:.2f})')
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='cyan', lw=2, label=f'Naive Bayes ROC curve (AUC = {roc_auc_naive_bayes:.2f})')
plt.plot(fpr_xgboost, tpr_xgboost, color='magenta', lw=2, label=f'XGBoost ROC curve (AUC = {roc_auc_xgboost:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of Ensembled Models')
plt.legend(loc="lower right")
plt.show()
