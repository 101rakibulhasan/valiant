import json

trim_end_no = int(input("Enter the number of trim from first: "))

faiai = []
faihuman = []
with open('Dataset & Code/dataset/filtered/f-ai-ai-collection.json') as f:
    faiai = json.load(f)
    faiai = faiai[-trim_end_no:] 

    with open('Dataset & Code/dataset/usable/u-ai-ai-collection.json', 'w') as f:  # Open in write mode
        json.dump(faiai, f, indent=4)

with open('Dataset & Code/dataset/filtered/f-ai-human-collection.json') as f:
    faihuman = json.load(f)
    faihuman = faihuman[-trim_end_no:]

    with open('Dataset & Code/dataset/usable/u-ai-human-collection.json', 'w') as f:  # Open in write mode
        json.dump(faihuman, f, indent=4)