import json

fgaiai = []
fiaiai = []
fyaiai = []
faiai = []
faihuman = []
with open('Dataset & Code/dataset/2. filtered/f-glm4-ai-ai-collection.json') as f:
    fgaiai = json.load(f)
with open('Dataset & Code/dataset/2. filtered/f-internlm2-ai-ai-collection.json') as f:
    fiaiai = json.load(f)
with open('Dataset & Code/dataset/2. filtered/f-yi-ai-ai-collection.json') as f:
    fyaiai = json.load(f)
with open('Dataset & Code/dataset/2. filtered/f-ai-human-collection.json') as f:
    faihuman = json.load(f)

print("GLM4 AI-AI Length:", len(fgaiai))
print("Internlm2 AI-AI Length:", len(fiaiai))
print("Yi AI-AI Length:", len(fyaiai))
print("AI-Human Length:", len(faihuman))

print("Processing...")
ai_len = len(fgaiai) + len(fiaiai) + len(fyaiai)
ai_human = len(faihuman)

min_len = min(ai_len, ai_human)
if min_len % 3 == 1:
    faiai = faiai + fgaiai[:(min_len//3)+1]
    faiai = faiai + fiaiai[:(min_len//3)]
    faiai = faiai + fyaiai[:(min_len//3)]

elif min_len % 3 == 2:
    faiai = faiai + fgaiai[:(min_len//3)+1]
    faiai = faiai + fiaiai[:(min_len//3)+1]
    faiai = faiai + fyaiai[:(min_len//3)]

else:
    faiai = faiai + fgaiai[:(min_len//3)]
    faiai = faiai + fiaiai[:(min_len//3)]
    faiai = faiai + fyaiai[:(min_len//3)]

faihuman = faihuman[-min_len:]

print("Filtered AI-AI Length:", len(faiai))
print("Filtered AI-Human Length:", len(faihuman))

with open('Dataset & Code/dataset/3. usable/fu-ai-ai-collection.json', 'w') as f:  # Open in write mode
    json.dump(faiai, f, indent=4)

with open('Dataset & Code/dataset/3. usable/fu-ai-human-collection.json', 'w') as f:  # Open in write mode
    json.dump(faihuman, f, indent=4)
