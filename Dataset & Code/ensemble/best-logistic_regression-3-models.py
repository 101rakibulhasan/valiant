## Best models for each classifier
# lightgbm model 6 - 0.883
# lightgbm model 7 - 0.883
# lightgbm model 10 - 0.9

import ensemble

test_path = 'Dataset & Code/dataset/conversation-collection-senti-test.json'

mdl = 'logistic_regression'

feature_labels = [
    ["trans-emo-feature", "senti-emotion"],
    ["trans-emo-feature2", "senti-emotion2"],
    ["trans-emo-feature5", "senti-emotion5"]
]

best_models = [
    f"Dataset & Code/models/{mdl}/model_m6.pkl",
    f"Dataset & Code/models/{mdl}/model_m7.pkl",
    f"Dataset & Code/models/{mdl}/model_m10.pkl"
]

actual_y, y_pred = ensemble.ensemble_predict_classification(test_path, best_models, feature_labels)

ensemble.evaluate_classification(actual_y[0], y_pred)

ensemble.save_ensemble_model(best_models, f"Dataset & Code/models/{mdl}/best-{mdl}-3-models.pkl")