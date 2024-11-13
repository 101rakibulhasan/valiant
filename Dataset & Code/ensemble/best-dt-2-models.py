## Best models for each classifier
# rf model 3 - 0.93
# rf model 5 - 0.92

import ensemble
import ensemble
import pickle
from collections import Counter

X_test1, Y_test1 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature3", "senti-trans3")
X_test2, Y_test2 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature5", "senti-trans5")

with open("Dataset & Code/models/decision_trees/model_m3.pkl", "rb") as file:
    m3_model = pickle.load(file)

with open("Dataset & Code/models/decision_trees/model_m5.pkl", "rb") as file:
    m5_model = pickle.load(file)

m3_preds = m3_model.predict(X_test1)
m5_preds = m5_model.predict(X_test2)

y_pred = []
for i in range(len(m3_preds)):
    prediction = Counter([m3_preds[i], m5_preds[i]]).most_common(1)[0][0]
    y_pred.append(prediction)

ensemble.evaluate_classification(Y_test1, y_pred)

# Output:
# After SMOTE:
"""
Accuracy: 0.9772727272727273
F1 Score: 0.977269792070257
Precision: 0.9775193798449613
Recall: 0.9772727272727273
Confusion Matrix:
 [[85  3]
 [ 1 87]]
"""