import json
import os 

sent_message = []
trans_senti_feature = []
trans_senti_feature2 = []
trans_senti_feature3 = []
trans_senti_feature4 = []
trans_senti_feature5 = []
trans_emo_feature = []
trans_emo_feature2 = []
trans_emo_feature3 = []
trans_emo_feature4 = []
trans_emo_feature5 = []

with open('Dataset & Code/dataset/sentiment/us-ai-ai-collection-senti-test.json', 'r') as f:
    data = json.load(f)

    for i in data:
        x = i
        trans_senti = {}
        trans_senti2 = {}
        trans_senti3 = {}
        trans_senti4 = {}
        trans_senti5 = {}
        trans_emotion = {}
        trans_emotion2 = {}
        trans_emotion3 = {}
        trans_emotion4 = {}
        trans_emotion5 = {}
        
        for j in range(len(i['messages']) - 1):
            current = i['messages'][j]['senti_label']
            next = i['messages'][j+1]['senti_label']
            key = f"{current}-to-{next}"
            if key not in trans_senti_feature:
                trans_senti_feature.append(key)

            if key in trans_senti:
                trans_senti[key] += 1
            else:
                trans_senti[key] = 1

            current = i['messages'][j]['senti_label2']
            next = i['messages'][j+1]['senti_label2']
            key = f"{current}-to-{next}"
            if key not in trans_senti_feature2:
                trans_senti_feature2.append(key)

            if key in trans_senti2:
                trans_senti2[key] += 1
            else:
                trans_senti2[key] = 1

            current = i['messages'][j]['senti_label3']
            next = i['messages'][j+1]['senti_label3']
            key = f"{current}-to-{next}"
            if key not in trans_senti_feature3:
                trans_senti_feature3.append(key)

            if key in trans_senti3:
                trans_senti3[key] += 1
            else:
                trans_senti3[key] = 1

            current = i['messages'][j]['senti_label4']
            next = i['messages'][j+1]['senti_label4']
            key = f"{current}-to-{next}"
            if key not in trans_senti_feature4:
                trans_senti_feature4.append(key)

            if key in trans_senti4:
                trans_senti4[key] += 1
            else:
                trans_senti4[key] = 1

            current = i['messages'][j]['senti_label5']
            next = i['messages'][j+1]['senti_label5']
            key = f"{current}-to-{next}"
            if key not in trans_senti_feature5:
                trans_senti_feature5.append(key)

            if key in trans_senti5:
                trans_senti5[key] += 1
            else:
                trans_senti5[key] = 1

            
            current = i['messages'][j]['emotion_label']
            next = i['messages'][j+1]['emotion_label']
            key = f"{current}-to-{next}"
            if key not in trans_emo_feature:
                trans_emo_feature.append(key)

            if key in trans_emotion:
                trans_emotion[key] += 1
            else:
                trans_emotion[key] = 1

            current = i['messages'][j]['emotion_label2']
            next = i['messages'][j+1]['emotion_label2']
            key = f"{current}-to-{next}"
            if key not in trans_emo_feature2:
                trans_emo_feature2.append(key)

            if key in trans_emotion2:
                trans_emotion2[key] += 1
            else:
                trans_emotion2[key] = 1

            current = i['messages'][j]['emotion_label3']
            next = i['messages'][j+1]['emotion_label3']
            key = f"{current}-to-{next}"
            if key not in trans_emo_feature3:
                trans_emo_feature3.append(key)

            if key in trans_emotion3:
                trans_emotion3[key] += 1
            else:
                trans_emotion3[key] = 1

            current = i['messages'][j]['emotion_label4']
            next = i['messages'][j+1]['emotion_label4']
            key = f"{current}-to-{next}"
            if key not in trans_emo_feature4:
                trans_emo_feature4.append(key)

            if key in trans_emotion4:
                trans_emotion4[key] += 1
            else:
                trans_emotion4[key] = 1

            current = i['messages'][j]['emotion_label5']
            next = i['messages'][j+1]['emotion_label5']
            key = f"{current}-to-{next}"
            if key not in trans_emo_feature5:
                trans_emo_feature5.append(key)

            if key in trans_emotion5:
                trans_emotion5[key] += 1
            else:
                trans_emotion5[key] = 1

        x["senti-trans"] = trans_senti
        x["senti-trans2"] = trans_senti2
        x["senti-trans3"] = trans_senti3
        x["senti-trans4"] = trans_senti4
        x["senti-trans5"] = trans_senti5
        x["senti-emotion"] = trans_emotion
        x["senti-emotion2"] = trans_emotion2
        x["senti-emotion3"] = trans_emotion3
        x["senti-emotion4"] = trans_emotion4
        x["senti-emotion5"] = trans_emotion5

        sent_message.append(x)

    features = {
            "senti-trans-feature": [],
            "senti-trans-feature2": [],
            "senti-trans-feature3": [],
            "senti-trans-feature4": [],
            "senti-trans-feature5": [],
            "trans-emo-feature": [],
            "trans-emo-feature2": [],
            "trans-emo-feature3": [],
            "trans-emo-feature4": [],
            "trans-emo-feature5": [],
        }
    
    file_path = "Dataset & Code/dataset/featured/features.json"
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as f:
            features = json.load(f)
        for j in trans_senti_feature:
            if j not in features["senti-trans-feature"] :
                features["senti-trans-feature"].append(j)
        for j in trans_senti_feature2:
            if j not in features["senti-trans-feature2"] :
                features["senti-trans-feature2"].append(j)
        for j in trans_senti_feature3:
            if j not in features["senti-trans-feature3"] :
                features["senti-trans-feature3"].append(j)
        for j in trans_senti_feature4:
            if j not in features["senti-trans-feature4"] :
                features["senti-trans-feature4"].append(j)
        for j in trans_senti_feature5:
            if j not in features["senti-trans-feature5"] :
                features["senti-trans-feature5"].append(j)
        for j in trans_emo_feature:
            if j not in features["trans-emo-feature"] :
                features["trans-emo-feature"].append(j)
        for j in trans_emo_feature2:
            if j not in features["trans-emo-feature2"] :
                features["trans-emo-feature2"].append(j)
        for j in trans_emo_feature3:
            if j not in features["trans-emo-feature3"] :
                features["trans-emo-feature3"].append(j)
        for j in trans_emo_feature4:
            if j not in features["trans-emo-feature4"] :
                features["trans-emo-feature4"].append(j)
        for j in trans_emo_feature5:
            if j not in features["trans-emo-feature5"] :
                features["trans-emo-feature5"].append(j)
    
    with open("Dataset & Code/dataset/featured/features.json", "w") as f:
        json.dump(features, f, indent=4)

    trans_senti_feature = features['senti-trans-feature']
    trans_senti_feature2 = features['senti-trans-feature2']
    trans_senti_feature3 = features['senti-trans-feature3']
    trans_senti_feature4 = features['senti-trans-feature4']
    trans_senti_feature5 = features['senti-trans-feature5']
    trans_emo_feature = features['trans-emo-feature']
    trans_emo_feature2 = features['trans-emo-feature2']
    trans_emo_feature3 = features['trans-emo-feature3']
    trans_emo_feature4 = features['trans-emo-feature4']
    trans_emo_feature5 = features['trans-emo-feature5']
        
    
    for i in sent_message:
        for j in trans_senti_feature:
            if j not in i['senti-trans']:
                i['senti-trans'][j] = 0

        for j in trans_senti_feature2:
            if j not in i['senti-trans2']:
                i['senti-trans2'][j] = 0

        for j in trans_senti_feature3:
            if j not in i['senti-trans3']:
                i['senti-trans3'][j] = 0

        for j in trans_senti_feature4:
            if j not in i['senti-trans4']:
                i['senti-trans4'][j] = 0

        for j in trans_senti_feature5:
            if j not in i['senti-trans5']:
                i['senti-trans5'][j] = 0

        for j in trans_emo_feature:
            if j not in i['senti-emotion']:
                i['senti-emotion'][j] = 0

        for j in trans_emo_feature2:
            if j not in i['senti-emotion2']:
                i['senti-emotion2'][j] = 0

        for j in trans_emo_feature3:
            if j not in i['senti-emotion3']:
                i['senti-emotion3'][j] = 0

        for j in trans_emo_feature4:
            if j not in i['senti-emotion4']:
                i['senti-emotion4'][j] = 0

        for j in trans_emo_feature5:
            if j not in i['senti-emotion5']:
                i['senti-emotion5'][j] = 0

    # print(trans_senti_feature)
    # print(json.dumps(sent_message[1]['senti-trans'], indent=4))

# print(trans_senti_feature)
# print(trans_senti_feature2)
# print(trans_senti_feature3)
# print(trans_senti_feature4)
# print(trans_senti_feature5)
# print(trans_emo_feature)
# print(trans_emo_feature2)
# print(trans_emo_feature3)
# print(trans_emo_feature4)
# print(trans_emo_feature5)

with open("Dataset & Code/dataset/featured/fus-ai-ai-collection-senti-test.json", "w") as f:
    json.dump(sent_message, f, indent=4)



        

        
            
            