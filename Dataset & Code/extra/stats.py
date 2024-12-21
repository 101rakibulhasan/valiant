import json

print("-- TRAIN STATS--")
faiai = []
faihuman = []
with open('Dataset & Code/dataset/1. raw/glm4-ai-ai-collection.json') as f:
    aiai = json.load(f)
    print(f"GLM4 Raw AI-AI Length: {len(aiai)}")

with open('Dataset & Code/dataset/1. raw/internlm2-ai-ai-collection.json') as f:
    aiai = json.load(f)
    print(f"Internlm2 Raw AI-AI Length: {len(aiai)}")

with open('Dataset & Code/dataset/1. raw/yi-ai-ai-collection.json') as f:
    aiai = json.load(f)
    print(f"Yi Raw AI-AI Length: {len(aiai)}")

with open('Dataset & Code/dataset/1. raw/ai-human-collection.json') as f:
    aihuman = json.load(f)
    print(f"Raw AI-Human Length: {len(aihuman)}")


faiai = []
faihuman = []
with open('Dataset & Code/dataset/2. filtered/f-glm4-ai-ai-collection.json') as f:
    faiai = json.load(f)
    print(f"Filtered GLM4 AI-AI Length: {len(faiai)}")

with open('Dataset & Code/dataset/2. filtered/f-internlm2-ai-ai-collection.json') as f:
    faiai = json.load(f)
    print(f"Filtered Internlm2 AI-AI Length: {len(faiai)}")

with open('Dataset & Code/dataset/2. filtered/f-yi-ai-ai-collection.json') as f:
    faiai = json.load(f)
    print(f"Filtered Yi AI-AI Length: {len(faiai)}")

with open('Dataset & Code/dataset/2. filtered/f-ai-human-collection.json') as f:
    faihuman = json.load(f)
    print(f"Filtered AI-Human Length: {len(faihuman)}")


uaiai = []
uaihuman = []
with open('Dataset & Code/dataset/3.1 usable/fu-ai-ai-collection.json') as f:
    uaiai = json.load(f)
    print(f"3.1 Usable AI-AI Length (No SMOTE): {len(uaiai)}")

with open('Dataset & Code/dataset/3.1 usable/fu-ai-human-collection.json') as f:
    uaihuman = json.load(f)
    print(f"3.1 Usable AI-Human Length (No SMOTE): {len(uaihuman)}")

with open('Dataset & Code/dataset/3.2 usable/fu-ai-ai-collection.json') as f:
    uaiai = json.load(f)
    print(f"3.2 Usable AI-AI Length (SMOTE): {len(uaiai)}")

with open('Dataset & Code/dataset/3.2 usable/fu-ai-human-collection.json') as f:
    uaihuman = json.load(f)
    print(f"3.2 Usable AI-Human Length (SMOTE): {len(uaihuman)}")

with open('Dataset & Code/dataset/4.1 sentiment/fus-ai-ai-collection-senti.json') as f:
    uaiai = json.load(f)
    print(f"4.1 Sentiment AI-AI Length (No SMOTE): {len(uaiai)}")

with open('Dataset & Code/dataset/4.1 sentiment/fus-ai-human-collection-senti.json') as f:
    uaihuman = json.load(f)
    print(f"4.1 Sentiment AI-Human Length (No SMOTE): {len(uaihuman)}")

with open('Dataset & Code/dataset/4.2 sentiment/fus-ai-ai-collection-senti.json') as f:
    uaiai = json.load(f)
    print(f"4.2 Sentiment AI-AI Length (SMOTE): {len(uaiai)}")

with open('Dataset & Code/dataset/4.2 sentiment/fus-ai-human-collection-senti.json') as f:
    uaihuman = json.load(f)
    print(f"4.2 Sentiment AI-Human Length (SMOTE): {len(uaihuman)}")

with open('Dataset & Code/dataset/5.1 featured/ffus-ai-ai-collection-senti.json') as f:
    uaiai = json.load(f)
    print(f"5.1 Featured AI-AI Length (No SMOTE): {len(uaiai)}")

with open('Dataset & Code/dataset/5.1 featured/ffus-ai-human-collection-senti.json') as f:
    uaihuman = json.load(f)
    print(f"5.1 Featured AI-Human Length (No SMOTE): {len(uaihuman)}")
with open('Dataset & Code/dataset/5.2 featured/ffus-ai-ai-collection-senti.json') as f:
    uaiai = json.load(f)
    print(f"5.2 Featured AI-AI Length (SMOTE): {len(uaiai)}")

with open('Dataset & Code/dataset/5.2 featured/ffus-ai-human-collection-senti.json') as f:
    uaihuman = json.load(f)
    print(f"5.2 Featured AI-Human Length (SMOTE): {len(uaihuman)}")
with open('Dataset & Code/dataset/conversation-collection-senti.json') as f:
    uaiai = json.load(f)
    print(f"Total conversation length (No SMOTE): {len(uaiai)}")

with open('Dataset & Code/dataset/conversation-collection-senti-all.json') as f:
    uaihuman = json.load(f)
    print(f"Total conversation length ALL (SMOTE): {len(uaihuman)}")

print("\n\n-- TEST STATS--")
with open('Dataset & Code/dataset/1. raw/ai-ai-collection-test.json') as f:
    aihuman = json.load(f)
    print(f"Test Raw AI-AI Length: {len(aihuman)}")
with open('Dataset & Code/dataset/1. raw/ai-human-collection-test.json') as f:
    aihuman = json.load(f)
    print(f"Raw AI-Human Length: {len(aihuman)}")
with open('Dataset & Code/dataset/5.1 featured/ffus-ai-ai-collection-senti-test.json') as f:
    aiai = json.load(f)
    print(f"Featured AI-AI Length: {len(aiai)}")
with open('Dataset & Code/dataset/5.1 featured/ffus-ai-human-collection-senti-test.json') as f:
    aihuman = json.load(f)
    print(f"Featured AI-Human Length: {len(aihuman)}")

with open('Dataset & Code/dataset/conversation-collection-senti-test.json') as f:
    uaihuman = json.load(f)
    print(f"Total conversation test length: {len(uaihuman)}")
