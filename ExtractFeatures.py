import General
import re
import sys
"""
Igal Zaidman 311758866
Tal Pogorelis 318225349
"""
corpus_file_name = sys.argv[1]
features_file_name = sys.argv[2]
"""
corpus_file_name = "ass1-tagger-train"
features_file_name = "features_file"

corpus_file_name = "dev"
features_file_name = "ner_features_file"
"""
words = {}
input_file = open(corpus_file_name,"r")
input_content = input_file.read().split("\n")
input_file.close()
lst = list(map(lambda couple: list(map(General.bslash_split, couple.split(' '))), input_content))
##########################################
# Functions
##########################################
def createFeatures(w1,w2,w3,w4,w5,w1_t,w2_t):
    features = {}
    
    if words[w3] < 5:
        w3_len = len(w3)
        for i in range(4):
            if w3_len > i:
                features['prefix' + str(i + 1)] = w3[:i + 1]
                features['suffix' + str(i + 1)] = w3[w3_len - i - 1:]
        features['contains_number'] = bool(re.search("\d",w3))
        features['contains_hyphen'] = bool(re.search("-",w3))
        features['contains_uppercase'] = bool(re.search("[A-Z]",w3))
    else:
        features["W3"] = w3

    features['T2'] = w2_t
    features['T1T2'] = w1_t + '/' + w2_t
    if w1 is not None:
        features['W1'] = w1
    if w2 is not None:
        features['W2'] = w2
    if w4 is not None:
        features['W4'] = w4
    if w5 is not None:
        features['W5'] = w5

    return features

def formatFeatures(obj):
    fProp = []
    for key, feature in obj.items():
        fProp.append(key + "=" + str(feature))
    return " ".join(fProp)
##########################################
# Logic
##########################################
for l in lst:
    for [w,t] in l:
        if w not in words:
            words[w] = 0
        words[w] += 1    
        
output = []

for l in lst:
    for i in range(len(l)):
        w1 = l[i - 2][0] if i > 1 else None
        w1_t = l[i - 2][1] if i > 1 else "SS"
        w2 = l[i - 1][0] if i > 0 else None        
        w2_t = l[i - 1][1] if i > 0 else "SS"
        w3 = l[i][0]
        w3_t = l[i][1]
        w4 = l[i + 1][0] if i < (len(l) - 1) else None
        w5 = l[i + 2][0] if i < (len(l) - 2) else None

        output.append(w3_t + " " + formatFeatures(createFeatures(w1,w2,w3,w4,w5,w1_t,w2_t)))

out_file = open(features_file_name, "w")
out_file.write("\n".join(output))
out_file.close()