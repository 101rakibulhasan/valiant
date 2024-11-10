import json

faiai = []
faihuman = []
with open('Dataset & Code/dataset/1. raw/glm4-ai-ai-collection.json') as f:
    fgaiai = json.load(f)
with open('Dataset & Code/dataset/1. raw/internlm2-ai-ai-collection.json') as f:
    fiaiai = json.load(f)
with open('Dataset & Code/dataset/1. raw/yi-ai-ai-collection.json') as f:
    fyaiai = json.load(f)
with open('Dataset & Code/dataset/1. raw/ai-human-collection.json') as f:
    faihuman = json.load(f)

print("Ignoring messages with less than 8 messages...")
fgaiai = list(filter(lambda i: len(i['messages']) >= 8, fgaiai))

fiaiai = list(filter(lambda i: len(i['messages']) >= 8, fiaiai))

fyaiai = list(filter(lambda i: len(i['messages']) >= 8, fyaiai))

faihuman = list(filter(lambda i: len(i['messages']) >= 8, faihuman))

with open('Dataset & Code/dataset/2. filtered/f-glm4-ai-ai-collection.json', 'w') as f:  # Open in write mode
    json.dump(fgaiai, f, indent=4)

with open('Dataset & Code/dataset/2. filtered/f-internlm2-ai-ai-collection.json', 'w') as f:  # Open in write mode
    json.dump(fiaiai, f, indent=4)

with open('Dataset & Code/dataset/2. filtered/f-yi-ai-ai-collection.json', 'w') as f:  # Open in write mode
    json.dump(fyaiai, f, indent=4)

with open('Dataset & Code/dataset/2. filtered/f-ai-human-collection.json', 'w') as f:  # Open in write mode
    json.dump(faihuman, f, indent=4)

print("GLM4 AI-AI Length:", len(fgaiai))
print("InternLM2 AI-AI Length:", len(fiaiai))
print("Yi AI-AI Length:", len(fyaiai))
print("Total AI-AI Length:", len(fgaiai) + len(fiaiai) + len(fyaiai))
print("Total AI-Human Length:", len(faihuman))