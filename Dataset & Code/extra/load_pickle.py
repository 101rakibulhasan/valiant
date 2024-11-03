import pickle

def get_model_result(pickle_file, X_test):
    with open(pickle_file, 'rb') as file:
        loaded_model = pickle.load(file)

        # Use the loaded model to make predictions
        y_pred = loaded_model.predict(X_test)
        return y_pred

# AI - 1, Human - 0
print("-- SVM Model 1 Test --")
print(get_model_result('Dataset & Code/models/svm/model_m1.pkl', [[5,2,2,0,0,0,0]]))

print("\n-- Random Forest Model 1 Test --")
print(get_model_result('Dataset & Code/models/random_forest/model_m1.pkl', [[5,2,2,0,0,0,0]]))

print("\n-- Decision Trees Model 1 Test --")
print(get_model_result('Dataset & Code/models/decision_trees/model_m1.pkl', [[5,2,2,0,0,0,0]]))
