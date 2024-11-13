## Best models for each classifier
# rf model 3 - 0.93
# rf model 5 - 0.92
# rf model 6 - 0.926

import ensemble

X_test1, Y_test1 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature3", "senti-trans3")
X_test2, Y_test2 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature5", "senti-trans5")
X_test3, Y_test3 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature5", "senti-emotion5")

y_pred = ensemble.ensemble_predict_classification(
    "Dataset & Code/models/decision_trees/model_m3.pkl",
    "Dataset & Code/models/decision_trees/model_m5.pkl",
    "Dataset & Code/models/decision_trees/model_m6.pkl", X_test1, X_test2, X_test3)

ensemble.evaluate_classification(Y_test1, y_pred)

# Output:
# After SMOTE:
"""
Accuracy: 0.9829545454545454
F1 Score: 0.982953995157385
Precision: 0.9830169185070387
Recall: 0.9829545454545454
Confusion Matrix:
 [[87  1]
 [ 2 86]]
"""