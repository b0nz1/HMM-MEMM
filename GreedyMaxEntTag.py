import sys
import pickle
import numpy
import General
import re
"""
Igal Zaidman 311758866
Tal Pogorelis 318225349
"""

input_file_name = sys.argv[1]
model_file_name = sys.argv[2]
feature_map_file_name = sys.argv[3]
output_file_name = sys.argv[4]
toCompare = True if len(sys.argv) > 5 else False
"""
input_file_name = "ass1-tagger-test-input"
model_file_name = "model_file"
feature_map_file_name = "feature_map_file"
output_file_name = "memm-greedy-predictions.txt"
toCompare = False


input_file_name = "test.blind"
model_file_name = "ner_model_file"
feature_map_file_name = "ner_feature_map_file"
output_file_name = "ner.memm.greedy.pred"
toCompare = False
"""
###############
# ALL THE FUNCTIONS
###############
def predict(w1,w2,w3,w4,w5,w1_t,w2_t):
    f_vec = numpy.zeros(len(features_m) - 1)
    f = General.getFeatures(w1,w2,w3,w4,w5,w1_t,w2_t)
    for k, feature in f.items():
        full = k + "=" + str(feature)
        if full in features_m:
            f_vec[features_m[full] - 1] = 1

    return model.predict([f_vec])

def splitSlash(s):
    if not toCompare:
        return [s]
    i = s.rfind('/')
    return [s[:i], s[i + 1:]]
###############
# LOAD ALL THE FILE DATA
###############
#read all the data from files
input_file = open(input_file_name, "r").read().split("\n")
input_f = list(map(lambda couple: list(map(splitSlash, couple.split(" "))), input_file))
feature_map = open(feature_map_file_name, "r").read().split('\n')    
feature_m = list(map(lambda x: x.split(" "), feature_map))
model = pickle.load(open(model_file_name, 'rb'))

features_m = {}
features_rm = {}
for row in feature_m:
    features_m[row[0]] = int(row[1])
    features_rm[row[1]] = row[0]

total_count = 0
total_good = 0
taged_to_file = []

for row in input_f:
    tags = ["SS", "SS"]
    output = []
    row_l = len(row)
    for i,t in enumerate(row):
        w3 = t[0]
        w1 = row[i - 2][0] if i > 1 else None
        w2 = row[i - 1][0] if i > 0 else None
        w4 = row[i + 1][0] if i < (row_l - 1) else None
        w5 = row[i + 2][0] if i < (row_l - 2) else None
        w1_t = tags[-2]
        w2_t = tags[-1]

        predicted_index = predict(w1,w2,w3,w4,w5,w1_t,w2_t)
        predicted_tag = features_rm[str(int(predicted_index[0]))]
        tags.append(predicted_tag)
        output.append(w3 + "/" + predicted_tag)
        if toCompare:
            total_count += 1
            if predicted_tag == t[1]:
                total_good += 1
    if toCompare:
        print("Rate is: " + str(total_good) + "/" + str(total_count) + " = " + str(float(total_good) / total_count))
    taged_to_file.append(" ".join(output))


out_file = open(output_file_name, 'w')
out_file.write('\n'.join(taged_to_file))
out_file.close() 