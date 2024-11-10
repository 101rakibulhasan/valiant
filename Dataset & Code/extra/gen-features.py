import json

model6_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

model7_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

model8_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

model9_labels = ["anger", "disgust", "fear", "joy", "neutrality", "sadness", "surprise"]

model10_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

model1_labels = ["negative", "neutral", "positive"]

model2_labels = ["POSITIVE", "NEGATIVE"]

model3_labels = ["positive", "neutral", "negative"]

model4_labels = ["positive", "negative", "neutral"]

model5_labels = ["positive", "negative", "neutral"]

def gen_trans_labels(label_list):
    v = []
    for i in label_list:
        for j in label_list:
            x = f"{i}-{j}"
            v.append(x)

    return v

trans_senti_labels = gen_trans_labels(model1_labels)
trans_senti_labels2 = gen_trans_labels(model2_labels)
trans_senti_labels3 = gen_trans_labels(model3_labels)
trans_senti_labels4 = gen_trans_labels(model4_labels)
trans_senti_labels5 = gen_trans_labels(model5_labels)

trans_emo_labels = gen_trans_labels(model6_labels)
trans_emo_labels2 = gen_trans_labels(model7_labels)
trans_emo_labels3 = gen_trans_labels(model8_labels)
trans_emo_labels4 = gen_trans_labels(model9_labels)
trans_emo_labels5 = gen_trans_labels(model10_labels)

features = {
            "senti-trans-feature": trans_senti_labels,
            "senti-trans-feature2": trans_senti_labels2,
            "senti-trans-feature3": trans_senti_labels3,
            "senti-trans-feature4": trans_senti_labels4,
            "senti-trans-feature5": trans_senti_labels5,
            "trans-emo-feature": trans_emo_labels,
            "trans-emo-feature2": trans_emo_labels2,
            "trans-emo-feature3": trans_emo_labels3,
            "trans-emo-feature4": trans_emo_labels4,
            "trans-emo-feature5": trans_emo_labels5
        }


with open("Dataset & Code/dataset/featured/features.json", "w") as f:
    json.dump(features, f, indent=4)
