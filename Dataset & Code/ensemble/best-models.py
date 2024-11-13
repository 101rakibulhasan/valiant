## Best models for each classifier
# svm model 9 - 0.81
# dt model 3 - 0.926
# rf model 3 - 0.9318

import ensemble

X_test1, Y_test1 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature4", "senti-emotion4")
X_test2, Y_test2 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature3", "senti-trans3")
X_test3, Y_test3 = ensemble.get_input(ensemble.jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature3", "senti-trans3")

y_pred = ensemble.ensemble_predict_classification(
    "Dataset & Code/models/svm/model_m9.pkl",
    "Dataset & Code/models/decision_trees/model_m3.pkl",
    "Dataset & Code/models/random_forest/model_m3.pkl", X_test1, X_test2, X_test3)

ensemble.evaluate_classification(Y_test1, y_pred)

# Output:
"""
Accuracy: 0.9431818181818182
F1 Score: 0.9431157078215903
Precision: 0.9452516865594187
Recall: 0.9431818181818182
Confusion Matrix:
 [[86  2]
 [ 8 80]]
"""