import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append ('./Dataset & Code/ensemble')
import ensemble
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# Load the saved VotingClassifier
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
predictions_rf = []
predictions_svm = []
predictions_dt = []
predictions_knn = []
predictions_lightgbm = []
predictions_logistic_regression = []
predictions_naive_bayes = []
predictions_xgboost = []

for estimator_name, model in dt_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_dt.append(model.predict(feature_set))

for estimator_name, model in svm_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_svm.append(model.predict(feature_set))

for estimator_name, model in rf_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_rf.append(model.predict(feature_set))

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

# Custom voting mechanism (majority vote)
from collections import Counter
predictions_rf = zip(*predictions_rf)
y_pred_rf = [Counter(sample).most_common(1)[0][0] for sample in predictions_rf]

predictions_svm = zip(*predictions_svm)
y_pred_svm = [Counter(sample).most_common(1)[0][0] for sample in predictions_svm]

predictions_dt = zip(*predictions_dt)
y_pred_dt = [Counter(sample).most_common(1)[0][0] for sample in predictions_dt]

predictions_knn = zip(*predictions_knn)
y_pred_knn = [Counter(sample).most_common(1)[0][0] for sample in predictions_knn]

predictions_lightgbm = zip(*predictions_lightgbm)
y_pred_lightgbm = [Counter(sample).most_common(1)[0][0] for sample in predictions_lightgbm]

predictions_logistic_regression = zip(*predictions_logistic_regression)
y_pred_logistic_regression = [Counter(sample).most_common(1)[0][0] for sample in predictions_logistic_regression]

predictions_naive_bayes = zip(*predictions_naive_bayes)
y_pred_naive_bayes = [Counter(sample).most_common(1)[0][0] for sample in predictions_naive_bayes]

predictions_xgboost = zip(*predictions_xgboost)
y_pred_xgboost = [Counter(sample).most_common(1)[0][0] for sample in predictions_xgboost]

rf_accuracy, rf_f1, rf_precision, rf_recall, rf_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_rf)
dt_accuracy, dt_f1, dt_precision, dt_recall, dt_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_dt)
svm_accuracy, svm_f1, svm_precision, svm_recall, svm_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_svm)
knn_accuracy, knn_f1, knn_precision, knn_recall, knn_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_knn)
lightgbm_accuracy, lightgbm_f1, lightgbm_precision, lightgbm_recall, lightgbm_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_lightgbm)
logistic_regression_accuracy, logistic_regression_f1, logistic_regression_precision, logistic_regression_recall, logistic_regression_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_logistic_regression)
naive_bayes_accuracy, naive_bayes_f1, naive_bayes_precision, naive_bayes_recall, naive_bayes_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_naive_bayes)
xgboost_accuracy, xgboost_f1, xgboost_precision, xgboost_recall, xgboost_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_xgboost)

models = ['SVM', 'Random Forest', 'Decision Tree', 'KNN', 'LightGBM', 'Logistic Regression', 'Naive Bayes', 'XGBoost']
accuracy = [svm_accuracy, rf_accuracy, dt_accuracy, knn_accuracy, lightgbm_accuracy, logistic_regression_accuracy, naive_bayes_accuracy, xgboost_accuracy]
precision = [svm_precision, rf_precision, dt_precision, knn_precision, lightgbm_precision, logistic_regression_precision, naive_bayes_precision, xgboost_precision]
recall = [svm_recall, rf_recall, dt_recall, knn_recall, lightgbm_recall, logistic_regression_recall, naive_bayes_recall, xgboost_recall]

print('Accuracy:', accuracy)

x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width, accuracy, width, label='Accuracy')
ax.bar(x, precision, width, label='Precision')
ax.bar(x + width, recall, width, label='Recall')

ax.set_ylabel('Scores')
ax.set_ylim(0.7, 1)
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(svm_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(dt_conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
plt.title('Decision Trees Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(lightgbm_conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
plt.title('LightGBM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(logistic_regression_conf_matrix, annot=True, fmt='d', cmap='Greys', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(naive_bayes_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(xgboost_conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['True Negative', 'True Positive'])
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()