import json
import load_pickle

def jsontodict(jsonfile):
    with open(jsonfile, 'r') as file:
        data = json.load(file)
    return data

feature = []
with open('Dataset & Code/dataset/5.1 featured/features.json', 'r') as f:
    feature = json.load(f)

def get_input(data, feature_label, model_label):
    X = []
    Y = []
    for i in data:
        temp = []

        global feature
        for j in feature[feature_label]:
            temp.append(i[model_label][j])
        
        X.append(temp)

        if i['result'] == 'AI':
            Y.append(1)
        else:
            Y.append(0)

    return X, Y

X_senti, Y_senti = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature", "senti-trans")
X_senti2, Y_senti2 = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature2", "senti-trans2")
X_senti3, Y_senti3 = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature3", "senti-trans3")
X_senti4, Y_senti4 = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature4", "senti-trans4")
X_senti5, Y_senti5 = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "senti-trans-feature5", "senti-trans5")
X_emo, Y_emo = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature", "senti-emotion")
X_emo2, Y_emo2 = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature2", "senti-emotion2")
X_emo3, Y_emo3 = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature3", "senti-emotion3")
X_emo4, Y_emo4 = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature4", "senti-emotion4")
X_emo5, Y_emo5 = get_input(jsontodict('Dataset & Code/dataset/conversation-collection-senti-test.json'), "trans-emo-feature5", "senti-emotion5")

def accuracy_model(model, X_test1, Y_test1):
    temp = load_pickle.get_model_result(model, X_test1)
    count = 0
    for i in range(len(temp)):
        if temp[i] == Y_test1[i]:
            count += 1
    
    return count / len(temp)

print("-- SVM --")
print(f"The accuracy of SVM model 1 is {accuracy_model('Dataset & Code/models/svm/model_m1.pkl', X_senti, Y_senti)}")
print(f"The accuracy of SVM model 2 is {accuracy_model('Dataset & Code/models/svm/model_m2.pkl', X_senti2, Y_senti2)}")
print(f"The accuracy of SVM model 3 is {accuracy_model('Dataset & Code/models/svm/model_m3.pkl', X_senti3, Y_senti3)}")
print(f"The accuracy of SVM model 4 is {accuracy_model('Dataset & Code/models/svm/model_m4.pkl', X_senti4, Y_senti4)}")
print(f"The accuracy of SVM model 5 is {accuracy_model('Dataset & Code/models/svm/model_m5.pkl', X_senti5, Y_senti5)}")
print(f"The accuracy of SVM model 6 is {accuracy_model('Dataset & Code/models/svm/model_m6.pkl', X_emo, Y_emo)}")
print(f"The accuracy of SVM model 7 is {accuracy_model('Dataset & Code/models/svm/model_m7.pkl', X_emo2, Y_emo2)}")
print(f"The accuracy of SVM model 8 is {accuracy_model('Dataset & Code/models/svm/model_m8.pkl', X_emo3, Y_emo3)}")
print(f"The accuracy of SVM model 9 is {accuracy_model('Dataset & Code/models/svm/model_m9.pkl', X_emo4, Y_emo4)}")
print(f"The accuracy of SVM model 10 is {accuracy_model('Dataset & Code/models/svm/model_m10.pkl', X_emo5, Y_emo5)}")

print("\n-- Decision Trees --")
print(f"The accuracy of Decision Tree model 1 is {accuracy_model('Dataset & Code/models/decision_trees/model_m1.pkl', X_senti, Y_senti)}")
print(f"The accuracy of Decision Tree model 2 is {accuracy_model('Dataset & Code/models/decision_trees/model_m2.pkl', X_senti2, Y_senti2)}")
print(f"The accuracy of Decision Tree model 3 is {accuracy_model('Dataset & Code/models/decision_trees/model_m3.pkl', X_senti3, Y_senti3)}")
print(f"The accuracy of Decision Tree model 4 is {accuracy_model('Dataset & Code/models/decision_trees/model_m4.pkl', X_senti4, Y_senti4)}")
print(f"The accuracy of Decision Tree model 5 is {accuracy_model('Dataset & Code/models/decision_trees/model_m5.pkl', X_senti5, Y_senti5)}")
print(f"The accuracy of Decision Tree model 6 is {accuracy_model('Dataset & Code/models/decision_trees/model_m6.pkl', X_emo, Y_emo)}")
print(f"The accuracy of Decision Tree model 7 is {accuracy_model('Dataset & Code/models/decision_trees/model_m7.pkl', X_emo2, Y_emo2)}")
print(f"The accuracy of Decision Tree model 8 is {accuracy_model('Dataset & Code/models/decision_trees/model_m8.pkl', X_emo3, Y_emo3)}")
print(f"The accuracy of Decision Tree model 9 is {accuracy_model('Dataset & Code/models/decision_trees/model_m9.pkl', X_emo4, Y_emo4)}")
print(f"The accuracy of Decision Tree model 10 is {accuracy_model('Dataset & Code/models/decision_trees/model_m10.pkl', X_emo5, Y_emo5)}")

print("\n-- Random Forest --")
print(f"The accuracy of Random Forest model 1 is {accuracy_model('Dataset & Code/models/random_forest/model_m1.pkl', X_senti, Y_senti)}")
print(f"The accuracy of Random Forest model 2 is {accuracy_model('Dataset & Code/models/random_forest/model_m2.pkl', X_senti2, Y_senti2)}")
print(f"The accuracy of Random Forest model 3 is {accuracy_model('Dataset & Code/models/random_forest/model_m3.pkl', X_senti3, Y_senti3)}")
print(f"The accuracy of Random Forest model 4 is {accuracy_model('Dataset & Code/models/random_forest/model_m4.pkl', X_senti4, Y_senti4)}")
print(f"The accuracy of Random Forest model 5 is {accuracy_model('Dataset & Code/models/random_forest/model_m5.pkl', X_senti5, Y_senti5)}")
print(f"The accuracy of Random Forest model 6 is {accuracy_model('Dataset & Code/models/random_forest/model_m6.pkl', X_emo, Y_emo)}")
print(f"The accuracy of Random Forest model 7 is {accuracy_model('Dataset & Code/models/random_forest/model_m7.pkl', X_emo2, Y_emo2)}")
print(f"The accuracy of Random Forest model 8 is {accuracy_model('Dataset & Code/models/random_forest/model_m8.pkl', X_emo3, Y_emo3)}")
print(f"The accuracy of Random Forest model 9 is {accuracy_model('Dataset & Code/models/random_forest/model_m9.pkl', X_emo4, Y_emo4)}")
print(f"The accuracy of Random Forest model 10 is {accuracy_model('Dataset & Code/models/random_forest/model_m10.pkl', X_emo5, Y_emo5)}")


