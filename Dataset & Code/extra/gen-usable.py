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

choice1 = input("Do you want to balance the dataset by trimming? (Y / (anything else)): ").upper()

if choice1 == "Y":
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

else:
    faiai = fgaiai + fiaiai + fyaiai
    faihuman = faihuman

print("Filtered AI-AI Length:", len(faiai))
print("Filtered AI-Human Length:", len(faihuman))

if choice1 == "Y":
    destination = "3.1 usable"
    # Filtered AI-AI Length: 87
    # Filtered AI-Human Length: 87
else:
    destination = "3.2 usable"
    # Filtered AI-AI Length: 139
    # Filtered AI-Human Length: 87

with open(f'Dataset & Code/dataset/{destination}/fu-ai-ai-collection.json', 'w') as f:  # Open in write mode
    json.dump(faiai, f, indent=4)

with open(f'Dataset & Code/dataset/{destination}/fu-ai-human-collection.json', 'w') as f:  # Open in write mode
    json.dump(faihuman, f, indent=4)

