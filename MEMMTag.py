import sys
import pickle
import numpy
import General
import re
"""
#Igal Zaidman 311758866
Tal Pogorelis 318225349
"""
input_file_name = sys.argv[1]
model_file_name = sys.argv[2]
feature_map_file_name = sys.argv[3]
output_file_name = sys.argv[4]
e_mle = sys.argv[5]
toCompare = True if len(sys.argv) > 6 else False
"""
input_file_name = "ass1-tagger-test-input"
model_file_name = "model_file"
feature_map_file_name = "feature_map_file"
output_file_name = "memm-viterbi-predictions.txt"
toCompare = False
e_mle = "e.mle"#use the e.mle file from the HMM model

input_file_name = "test.blind"
model_file_name = "ner_model_file"
feature_map_file_name = "ner_feature_map_file"
output_file_name = "ner.memm.pred"
toCompare = False
e_mle = "ner_e.mle"
"""
###############
# Definitions
###############
def predict(w1,w2,w3,w4,w5,w1_t,w2_t):
    f_vec = numpy.zeros(len(features_m) - 1)
    f = getFeature(w1,w2,w3,w4,w5,w1_t,w2_t)
    for k, feature in f.items():
        full = k + "=" + str(feature)
        if full in features_m:
            f_vec[features_m[full] - 1] = 1

    return model.predict_log_proba([f_vec])

def getScore(w1,w2,w3,w4,w5,w1_t,w2_t):
    k = str(w1) + str(w2) + w3 + str(w4) + str(w5) + w1_t + w2_t
    if k not in cache:
        cache[k] = (predict(w1,w2,w3,w4,w5,w1_t,w2_t)[0])
    return cache[k]
def getFeature(w1,w2,w3,w4,w5,w1_t,w2_t):
    features = {}
    if ("W3="+w3) not in features_m:       
        for i in range(4):
            if len(w3) > i:
                features['prefix' + str(i + 1)] = w3[:i + 1]
                features['suffix' + str(i + 1)] = w3[len(w3) - i - 1:]
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
def splitSlash(s):
    #if not toCompare:
    #    return [s]
    i = s.rfind('/')
    return [s[:i], s[i + 1:]]
###############
# Logic
###############
input_file = open(input_file_name, "r").read().split("\n")
input_f = list(map(lambda couple: list(map(splitSlash, couple.split(" "))), input_file))
feature_map = open(feature_map_file_name, "r").read().split('\n')    
feature_m = list(map(lambda x: x.split(" "), feature_map))
model = pickle.load(open(model_file_name, 'rb'))
e_file = open(e_mle, "r").read().split("\n")

#fill e from file of     
e={}
for row in e_file:
    a = row.split("\t")
    b = a[0].split(" ")
    if b[0] not in e:
        e[b[0]] = {}
    e[b[0]][b[1]] = float(a[1])

cache = {}
features_m = {}
features_rm = {}
pruning = {}
t_set = set([])
for row in feature_m:
    features_m[row[0]] = int(row[1])
    features_rm[row[1]] = row[0]
    if '=' not in row[0] or row[0] == '=':
        t_set.add(row[0])
    else:
        t_type, t_val = row[0].split('=', 1)
        if t_type == 'T1T2':
            t1, t2 = t_val.split('/')
            if t1 not in pruning:
                pruning[t1] = set([])
            pruning[t1].add(t2)
            
count_total = 0
count_good = 0
taged_to_file = []
# main logic
for row in input_f:
    row_l = len(row)
    vt_set = set(t_set)
    vt_set.add("SS")
    V = [{} for w in row] + [{}]
    for t in vt_set:
        V[0][t] = {}
        for r in t_set:
            V[0][t][r] = 0
    V[0]["SS"]["SS"] = 1
    bp = [{} for w in row] + [{}]
    tags_p2 = ["SS"]
    tags_p = ["SS"]
    for i in range(row_l):
        word = row[i][0]
        tags_curr = t_set
        if word in e:
            tags_curr = list(e[word].keys())
        w1 = row[i - 2][0] if i > 1 else None
        w2 = row[i - 1][0] if i > 0 else None
        w4 = row[i + 1][0] if i < (row_l - 1) else None
        w5 = row[i + 2][0] if i < (row_l - 2) else None
        V[i + 1] = {}
        bp[i + 1] = {}
        for t in tags_p:
            w2_t = t
            V[i + 1][t] = {}
            bp[i + 1][t] = {}
            l = {}
            for r in tags_curr:
                l[r] = {}
            for tT in tags_p2:
                w1_t = tT
                score = (V[i][tT][t]) + getScore(w1,w2,word,w4,w5,w1_t,w2_t)
                for r in tags_curr:
                    l[r][tT] = score[features_m[r]]
            for r in tags_curr:
                V[i + 1][t][r] = max(list(l[r].values()))
                bp[i + 1][t][r] = General.argmax(l[r])
        tags_p2 = tags_p
        tags_p = tags_curr
    V.pop(0)
    bp.pop(0)
    endMatrix = map(lambda x: x.values(), V[row_l-1].values())
    maxEnd = list(map(max, endMatrix))
    maxV = max(maxEnd)
    maxTIndex = maxEnd.index(maxV)
    maxT = list(V[row_l-1].keys())[maxTIndex]
    maxR = General.argmax(V[row_l-1][maxT])
    y = [0 for i in range(0, row_l)]
    y[row_l-1] = maxR
    y[row_l-2] = maxT
    for i in reversed(range(0, row_l-2)):
        y[i] = bp[i + 2][y[i + 1]][y[i + 2]]            
    if toCompare:
        for i,w in enumerate(row):
            count_total += 1
            if w[1] == y[i]:
                count_good += 1      
        print("Rate is: " + str(count_good) + "/" + str(count_total) + " = " + str(float(count_good) / count_total))            
    taged_to_file.append(" ".join(map(lambda i: str(row[i][0]) + "/" + str(y[i]), range(row_l))))    

out_file = open(output_file_name, 'w')
out_file.write('\n'.join(taged_to_file))
out_file.close() 
            