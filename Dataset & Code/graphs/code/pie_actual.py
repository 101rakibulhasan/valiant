import os
import json

import matplotlib.pyplot as plt

data = []
data_train = json.load(open('Dataset & Code/dataset/conversation-collection-senti-all.json', 'r'))
data_test = json.load(open('Dataset & Code/dataset/conversation-collection-senti-test.json', 'r'))

classes_train = {'AI': 0, 'Human': 0}

for i in data_train:
    classes_train[i['result']] += 1

classes_test = {'AI': 0, 'Human': 0}

for i in data_test:
    classes_test[i['result']] += 1

print(classes_train)
print(classes_test)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].pie(classes_train.values(), labels=classes_train.keys(), autopct='%1.1f%%')
ax[0].set_title('Training Data')

ax[1].pie(classes_test.values(), labels=classes_test.keys(), autopct='%1.1f%%')
ax[1].set_title('Testing Data')

plt.show()

