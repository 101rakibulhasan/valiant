import json

faiai = []
faihuman = []
with open('Dataset & Code/dataset/raw/ai-ai-collection-test.json') as f:
    aiai = json.load(f)
    print(f"Raw AI-AI Length: {len(aiai)}")

with open('Dataset & Code/dataset/raw/ai-human-collection-test.json') as f:
    aihuman = json.load(f)
    print(f"Raw AI-Human Length: {len(aihuman)}")


faiai = []
faihuman = []
with open('Dataset & Code/dataset/filtered/f-ai-ai-collection.json') as f:
    faiai = json.load(f)
    print(f"Filtered AI-AI Length: {len(faiai)}")

with open('Dataset & Code/dataset/filtered/f-ai-human-collection.json') as f:
    faihuman = json.load(f)
    print(f"Filtered AI-Human Length: {len(faihuman)}")


uaiai = []
uaihuman = []
with open('Dataset & Code/dataset/usable/u-ai-ai-collection-test.json') as f:
    uaiai = json.load(f)
    print(f"Usable AI-AI Length: {len(uaiai)}")

with open('Dataset & Code/dataset/usable/u-ai-human-collection-test.json') as f:
    uaihuman = json.load(f)
    print(f"Usable AI-Human Length: {len(uaihuman)}")