## Best models for each classifier
# lightgbm model 3 - 0.883
# lightgbm model 5 - 0.883
# lightgbm model 9 - 0.817

import ensemble

test_path = 'Dataset & Code/dataset/conversation-collection-senti-test.json'

mdl = 'svm'

feature_labels = [
    ["senti-trans-feature3", "senti-trans3"],
    ["senti-trans-feature5", "senti-trans5"],
    ["trans-emo-feature4", "senti-emotion4"]
]

best_models = [
    f"Dataset & Code/models/{mdl}/model_m3.pkl",
    f"Dataset & Code/models/{mdl}/model_m5.pkl",
    f"Dataset & Code/models/{mdl}/model_m9.pkl"
]

actual_y, y_pred = ensemble.ensemble_predict_classification(test_path, best_models, feature_labels)

ensemble.evaluate_classification(actual_y[0], y_pred)

ensemble.save_ensemble_model(best_models, f"Dataset & Code/models/{mdl}/best-{mdl}-3-models.pkl")