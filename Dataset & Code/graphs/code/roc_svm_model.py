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
# All Y value here are the same, so we can use any of them

estimators = []
with open("Dataset & Code/models/svm/model_m1.pkl", "rb") as file:
    m1_model = pickle.load(file)
    estimators.append((f'model_m1', m1_model))

with open("Dataset & Code/models/svm/model_m2.pkl", "rb") as file:
    m2_model = pickle.load(file)
    estimators.append((f'model_m2', m2_model))

with open("Dataset & Code/models/svm/model_m3.pkl", "rb") as file:
    m3_model = pickle.load(file)
    estimators.append((f'model_m3', m3_model))

with open("Dataset & Code/models/svm/model_m4.pkl", "rb") as file:
    m4_model = pickle.load(file)
    estimators.append((f'model_m4', m4_model))

with open("Dataset & Code/models/svm/model_m5.pkl", "rb") as file:
    m5_model = pickle.load(file)
    estimators.append((f'model_m5', m5_model))

with open("Dataset & Code/models/svm/model_m6.pkl", "rb") as file:
    m6_model = pickle.load(file)
    estimators.append((f'model_m6', m6_model))

with open("Dataset & Code/models/svm/model_m7.pkl", "rb") as file:
    m7_model = pickle.load(file)
    estimators.append((f'model_m7', m7_model))

with open("Dataset & Code/models/svm/model_m8.pkl", "rb") as file:
    m8_model = pickle.load(file)
    estimators.append((f'model_m8', m8_model))

with open("Dataset & Code/models/svm/model_m9.pkl", "rb") as file:
    m9_model = pickle.load(file)
    estimators.append((f'model_m9', m9_model))

with open("Dataset & Code/models/svm/model_m10.pkl", "rb") as file:
    m10_model = pickle.load(file)
    estimators.append((f'model_m10', m10_model))
    
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

Y_combined = Y_senti    # All Y values are the same, so we can use any of them

fpr_m1, tpr_m1, thresholds_m1 = roc_curve(Y_combined, m1_preds)
fpr_m2, tpr_m2, thresholds_m2 = roc_curve(Y_combined, m2_preds)
fpr_m3, tpr_m3, thresholds_m3 = roc_curve(Y_combined, m3_preds)
fpr_m4, tpr_m4, thresholds_m4 = roc_curve(Y_combined, m4_preds)
fpr_m5, tpr_m5, thresholds_m5 = roc_curve(Y_combined, m5_preds)
fpr_m6, tpr_m6, thresholds_m6 = roc_curve(Y_combined, m6_preds)
fpr_m7, tpr_m7, thresholds_m7 = roc_curve(Y_combined, m7_preds)
fpr_m8, tpr_m8, thresholds_m8 = roc_curve(Y_combined, m8_preds)
fpr_m9, tpr_m9, thresholds_m9 = roc_curve(Y_combined, m9_preds)
fpr_m10, tpr_m10, thresholds_m10 = roc_curve(Y_combined, m10_preds)


# Calculate AUC
# roc_auc_rf = auc(fpr_rf, tpr_rf)
# roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_m1 = auc(fpr_m1, tpr_m1)
roc_auc_m2 = auc(fpr_m2, tpr_m2)
roc_auc_m3 = auc(fpr_m3, tpr_m3)
roc_auc_m4 = auc(fpr_m4, tpr_m4)
roc_auc_m5 = auc(fpr_m5, tpr_m5)
roc_auc_m6 = auc(fpr_m6, tpr_m6)
roc_auc_m7 = auc(fpr_m7, tpr_m7)
roc_auc_m8 = auc(fpr_m8, tpr_m8)
roc_auc_m9 = auc(fpr_m9, tpr_m9)
roc_auc_m10 = auc(fpr_m10, tpr_m10)

# Plot ROC curve
plt.figure(figsize=(8, 6))
# plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'RF ROC curve (AUC = {roc_auc_rf:.2f})')
# plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'SVM ROC curve (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_m1, tpr_m1, lw=2, label=f'M1 ROC curve (AUC = {roc_auc_m1:.2f})')
plt.plot(fpr_m2, tpr_m1, lw=2, label=f'M2 ROC curve (AUC = {roc_auc_m2:.2f})')
plt.plot(fpr_m3, tpr_m1, lw=2, label=f'M3 ROC curve (AUC = {roc_auc_m3:.2f})')
plt.plot(fpr_m4, tpr_m1, lw=2, label=f'M4 ROC curve (AUC = {roc_auc_m4:.2f})')
plt.plot(fpr_m5, tpr_m1, lw=2, label=f'M5 ROC curve (AUC = {roc_auc_m5:.2f})')
plt.plot(fpr_m6, tpr_m1, lw=2, label=f'M6 ROC curve (AUC = {roc_auc_m6:.2f})')
plt.plot(fpr_m7, tpr_m1, lw=2, label=f'M7 ROC curve (AUC = {roc_auc_m7:.2f})')
plt.plot(fpr_m8, tpr_m1, lw=2, label=f'M8 ROC curve (AUC = {roc_auc_m8:.2f})')
plt.plot(fpr_m9, tpr_m1, lw=2, label=f'M9 ROC curve (AUC = {roc_auc_m9:.2f})')
plt.plot(fpr_m10, tpr_m1, lw=2, label=f'M10 ROC curve (AUC = {roc_auc_m10:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc="lower right")
plt.show()