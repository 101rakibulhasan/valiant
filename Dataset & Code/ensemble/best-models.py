## Best models for each classifier
# svm model 9 - 0.81
# dt model 3 - 0.926
# rf model 3 - 0.9318

import ensemble

test_path = 'Dataset & Code/dataset/conversation-collection-senti-test.json'

best_models = [
    "Dataset & Code/models/svm/model_m5.pkl",
    "Dataset & Code/models/decision_trees/model_m10.pkl",
    "Dataset & Code/models/random_forest/model_m10.pkl",
    "Dataset & Code/models/knn/model_m5.pkl",
    "Dataset & Code/models/lightgbm/model_m10.pkl",
    "Dataset & Code/models/logistic_regression/model_m10.pkl",
    "Dataset & Code/models/naive_bayes/model_m3.pkl",
    "Dataset & Code/models/xgboost/model_m7.pkl"
]

features_label = [
    ["senti-trans-feature5", "senti-trans5",],
    ["trans-emo-feature5", "senti-emotion5",],
    ["trans-emo-feature5", "senti-emotion5",],
    ["senti-trans-feature5", "senti-trans5",],
    ["trans-emo-feature5", "senti-emotion5",],
    ["trans-emo-feature5", "senti-emotion5",],
    ["senti-trans-feature3", "senti-trans3",],
    ["trans-emo-feature2", "senti-emotion2"]
]

actual_y, y_pred = ensemble.ensemble_predict_classification(test_path, best_models, features_label)

ensemble.evaluate_classification(actual_y[0], y_pred)

# Output:

# After SMOTE:
"""
Accuracy: 0.9166666666666666
F1 Score: 0.9164578111946533
Precision: 0.9208754208754207
Recall: 0.9166666666666666
Confusion Matrix:
 [[26  4]
 [ 1 29]]
"""