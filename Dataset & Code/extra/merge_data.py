import json
total = []

with open('Dataset & Code/dataset/featured/fus-ai-ai-collection-senti.json', 'r') as f:
    data = json.load(f)
    total.extend(data)

with open('Dataset & Code/dataset/featured/fus-ai-human-collection-senti.json', 'r') as f:
    data = json.load(f)
    total.extend(data)

with open('Dataset & Code/dataset/conversation-collection-senti.json', 'w') as f:
    json.dump(total, f, indent=4)

features = []
with open('Dataset & Code/dataset/featured/features.json', 'r') as f:
    features = json.load(f)

for i in total:
    for j in features['senti-trans-feature']:
        flag = 0
        for k in i['senti-trans']:
            if j == k:
                flag = 1
                break
            else:
                print(i['id'])
                break
         
        if flag == 1:
            break
            
print('Done')