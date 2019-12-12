from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import sys
import pickle
"""
Igal Zaidman 311758866
Tal Pogorelis 318225349
"""
features_file_name = sys.argv[1]
model_file_name = sys.argv[2]
"""
features_file_name = "features_file"
model_file_name = "model_file"

features_file_name = "ner_features_file"
model_file_name = "ner_model_file"
"""
input_file = open(features_file_name, "r")
input_content = input_file.read().split("\n")
input_file.close()

features_file = list(map(lambda x: x.split(" "), input_content))

def featureMap(i, f):
    if i == 0:
        return str(f)
    else:
        return str(f) + ":1"

count = 0
features = {}
#create the features numbering
for row in features_file:
    if row[0] not in features:
        features[row[0]] = count
        count += 1
for row in features_file:
    for f in row:
        if f not in features:
            features[f] = count
            count += 1
            
f_maps = [[features[f] for f in row] for row in features_file]
for row in f_maps:
    row.sort()
f_maps = [[featureMap(i, f) for i, f in enumerate(row)] for row in f_maps]

output_vec_file = open("feature_vec_file", "w")
output_vec_file.write('\n'.join(list(map(lambda i: " ".join(i), f_maps))))
output_vec_file.close()
output_map_file = open("feature_map_file", "w")
output_map_file.write("\n".join(list(map(lambda i: i + " " + str(features[i]), list(features.keys())))))
output_map_file.close()

X, Y = load_svmlight_file("feature_vec_file")

model = LogisticRegression(penalty='l2')
model.fit(X, Y)
pickle.dump(model, open(model_file_name, 'wb'))