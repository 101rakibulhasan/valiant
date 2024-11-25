import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append ('./Dataset & Code/ensemble')
import ensemble
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# Load all models into a list
with open('Dataset & Code/models/random_forest/best_rf_all_ensamble+smote_models.pkl', 'rb') as file:
    rf_model = pickle.load(file)
with open('Dataset & Code/models/decision_trees/best_dt_all_ensamble+smote_models.pkl', 'rb') as file:
    dt_model = pickle.load(file)
with open('Dataset & Code/models/svm/best_svm_all_ensamble+smote_models.pkl', 'rb') as file:
    svm_model = pickle.load(file)

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
for estimator_name, model in dt_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_dt.append(model.predict(feature_set))

for estimator_name, model in svm_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_svm.append(model.predict(feature_set))

for estimator_name, model in rf_model.estimators:
    feature_set = X_combined[int(estimator_name.split("_m")[1]) - 1]  # Match features to models
    predictions_rf.append(model.predict(feature_set))

# Custom voting mechanism (majority vote)
from collections import Counter
predictions_rf = zip(*predictions_rf)
y_pred_rf = [Counter(sample).most_common(1)[0][0] for sample in predictions_rf]

predictions_svm = zip(*predictions_svm)
y_pred_svm = [Counter(sample).most_common(1)[0][0] for sample in predictions_svm]

predictions_dt = zip(*predictions_dt)
y_pred_dt = [Counter(sample).most_common(1)[0][0] for sample in predictions_dt]

rf_accuracy, rf_f1, rf_precision, rf_recall, rf_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_rf)
dt_accuracy, dt_f1, dt_precision, dt_recall, dt_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_dt)
svm_accuracy, svm_f1, svm_precision, svm_recall, svm_conf_matrix = ensemble.evaluate_classification(Y_senti, y_pred_svm)

models = ['SVM', 'Random Forest', 'Decision Tree']
accuracy = [svm_accuracy, rf_accuracy, dt_accuracy]
precision = [svm_precision, rf_precision, dt_precision]
recall = [svm_recall, rf_recall, dt_recall]

print('Accuracy:', accuracy)

x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width, np.array(accuracy) - 0.7, width, label='Accuracy', bottom=0.7)
ax.bar(x, np.array(precision) - 0.7, width, label='Precision', bottom=0.7)
ax.bar(x + width, np.array(recall) - 0.7, width, label='Recall', bottom=0.7)


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