import json
total = []

with open('Dataset & Code/dataset/5.1 featured/ffus-ai-ai-collection-senti-test.json', 'r') as f:
    data = json.load(f)
    total.extend(data)

with open('Dataset & Code/dataset/5.1 featured/ffus-ai-human-collection-senti-test.json', 'r') as f:
    data = json.load(f)
    total.extend(data)

with open('Dataset & Code/dataset/conversation-collection-senti-test.json', 'w') as f:
    json.dump(total, f, indent=4)
            
print('Done')