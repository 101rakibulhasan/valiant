## Best models for each classifier
# knn model 3 - 0.833
# knn model 5 - 0.85
# knn model 7 - 0.85

import ensemble

test_path = 'Dataset & Code/dataset/conversation-collection-senti-test.json'

mdl = 'knn'

feature_labels = [
    ["senti-trans-feature5", "senti-trans5"],
    ["trans-emo-feature2", "senti-emotion2"]
]

best_models = [
    f"Dataset & Code/models/{mdl}/model_m5.pkl",
    f"Dataset & Code/models/{mdl}/model_m7.pkl"
]

actual_y, y_pred = ensemble.ensemble_predict_classification(test_path, best_models, feature_labels)

ensemble.evaluate_classification(actual_y[0], y_pred)

ensemble.save_ensemble_model(best_models, f"Dataset & Code/models/{mdl}/best-{mdl}-2-models.pkl")