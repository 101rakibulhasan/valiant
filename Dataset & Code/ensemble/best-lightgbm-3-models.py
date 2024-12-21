## Best models for each classifier
# lightgbm model 3 - 0.9
# lightgbm model 7 - 0.883
# lightgbm model 10 - 0.883

import ensemble

test_path = 'Dataset & Code/dataset/conversation-collection-senti-test.json'

mdl = 'lightgbm'

feature_labels = [
    ["senti-trans-feature3", "senti-trans3"],
    ["trans-emo-feature2", "senti-emotion2"],
    ["trans-emo-feature5", "senti-emotion5"]
]

best_models = [
    f"Dataset & Code/models/{mdl}/model_m3.pkl",
    f"Dataset & Code/models/{mdl}/model_m7.pkl",
    f"Dataset & Code/models/{mdl}/model_m10.pkl"
]

actual_y, y_pred = ensemble.ensemble_predict_classification(test_path, best_models, feature_labels)

ensemble.evaluate_classification(actual_y[0], y_pred)

ensemble.save_ensemble_model(best_models, f"Dataset & Code/models/{mdl}/best-knn-3-models.pkl")