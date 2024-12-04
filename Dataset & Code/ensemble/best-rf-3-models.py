## Best models for each classifier
# rf model 3 - 0.93
# rf model 5 - 0.92
# rf model 6 - 0.926

import ensemble

X_test1, Y_test1 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature3", "senti-trans3")
X_test2, Y_test2 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature5", "senti-trans5")
X_test3, Y_test3 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature5", "senti-emotion5")

y_pred = ensemble.ensemble_predict_classification(
    "Dataset & Code/models/random_forest/model_m3.pkl",
    "Dataset & Code/models/random_forest/model_m5.pkl",
    "Dataset & Code/models/random_forest/model_m7.pkl", X_test1, X_test2, X_test3)

ensemble.evaluate_classification(Y_test1, y_pred)

# Output:
# After SMOTE:
"""
Accuracy: 0.9602272727272727
F1 Score: 0.9601642577682931
Precision: 0.9631578947368421
Recall: 0.9602272727272727
Confusion Matrix:
 [[88  0]
 [ 7 81]]
"""