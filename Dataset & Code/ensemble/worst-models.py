## Worst models for each classifier
# svm model 8 - 0.755
# dt model 1 - 0.75
# rf model 1 - 0.784

import ensemble

X_test1, Y_test1 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature3", "senti-emotion3")
X_test2, Y_test2 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature", "senti-trans")
X_test3, Y_test3 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature", "senti-trans")

y_pred = ensemble.ensemble_predict_classification(
    "Dataset & Code/models/svm/model_m8.pkl",
    "Dataset & Code/models/decision_trees/model_m4.pkl",
    "Dataset & Code/models/random_forest/model_m1.pkl", X_test1, X_test2, X_test3)

ensemble.evaluate_classification(Y_test1, y_pred)

# Output:
"""
Accuracy: 0.7613636363636364
F1 Score: 0.7575757575757575
Precision: 0.7787878787878788
Recall: 0.7613636363636364
Confusion Matrix:
 [[56 32]
 [10 78]]
"""