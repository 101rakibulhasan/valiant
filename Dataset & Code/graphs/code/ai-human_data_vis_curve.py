import json
import os 
import numpy as np
import matplotlib.pyplot as plt

with open('Dataset & Code/dataset/5.1 featured/ffus-ai-human-collection-senti.json', 'r') as f:
    data_aihuman = json.load(f)

feature_with_val_aihuman = {}
matching_key = {
    "senti-trans-feature" : "senti-trans",
    "senti-trans-feature2" : "senti-trans2",
    "senti-trans-feature3" : "senti-trans3",
    "senti-trans-feature4" : "senti-trans4",
    "senti-trans-feature5" : "senti-trans5",
    "trans-emo-feature" : "senti-emotion",
    "trans-emo-feature2" : "senti-emotion2",
    "trans-emo-feature3" : "senti-emotion3",
    "trans-emo-feature4" : "senti-emotion4",
    "trans-emo-feature5" : "senti-emotion5"

}
file_path = "Dataset & Code/dataset/5.1 featured/features.json"
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, "r") as f:
        sent_features = json.load(f)
        for sent_feature in sent_features:
            feature_with_val_aihuman[sent_feature] = {}
            for i in sent_features[sent_feature]:
                feature_with_val_aihuman[sent_feature][i] = 0

for i in data_aihuman: # []
    for f in feature_with_val_aihuman: # {""}
        for k in i[matching_key[f]]:
            if k in feature_with_val_aihuman[f]:
                feature_with_val_aihuman[f][k] += i[matching_key[f]][k]

print(feature_with_val_aihuman)

model_no = 1
for i in feature_with_val_aihuman:
    features = list(feature_with_val_aihuman[i].keys())
    dataset_1_values = list(feature_with_val_aihuman[i].values())

    top_features = features
    top_dataset_1_values = dataset_1_values

    # Create a line plot to compare feature values without points
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(top_features, top_dataset_1_values, linestyle='-', color='blue', label=f'AI Human Training Dataset Model {model_no}')

    # Add title and legend
    ax.set_title('AI-Human Training Dataset Model ' + str(model_no) + ' Features Frequency')
    ax.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Hide x and y axis lines and labels
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

    model_no += 1