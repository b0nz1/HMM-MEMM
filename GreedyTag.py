import sys
import re
import General

"""
Igal Zaidman 311758866
Tal Pogorelis 318225349
"""

input_file_name = sys.argv[1]
e_mle = sys.argv[2]
q_mle = sys.argv[3]
greedy_hmm_output = sys.argv[4]
extra_file = sys.argv[5]
toCompare = True if len(sys.argv) > 6 else False
"""
input_file_name = "ass1-tagger-test-input"
e_mle = "e.mle"
q_mle = "q.mle"
greedy_hmm_output = "greedy_hmm_output.txt"
extra_file = "extra_file.txt"
toCompare = False

input_file_name = "dev"
e_mle = "ner_e.mle"
q_mle = "ner_q.mle"
greedy_hmm_output = "ner_greedy_hmm.txt"
extra_file = "extra_file.txt"
toCompare = True
"""
###############
# ALL THE FUNCTIONS
###############
def splitSlash(s):
    if not toCompare:
        return [s]
    i = s.rfind('/')
    return [s[:i], s[i + 1:]]

def getE(word):
    if word not in e:
        word = General.assignWrodClass(word)
    return e[word]

def getQ(tag,tag_p,tag_p2):
    one = 0
    two = 0
    three = 0
    if tag in q:
       one = float(q[tag])
    
    tag2 = tag_p + " " + tag
    if tag2 in q:
        two = float(q[tag2])
    
    tag3 = tag_p2 + " " + tag2
    if tag3 in q:
        three = float(q[tag3])
    
    return 0.9 * three + 0.09 * two + 0.01 * one    

def getScore(word, tag_p, tag_p2):
    scores = {}
    word_tags = getE(word)
    for word_tag in word_tags:
        scores[word_tag] = float(word_tags[word_tag]) * getQ(word_tag,tag_p,tag_p2)
    return scores    

###############
# LOAD ALL THE FILE DATA
###############
#read all the data from files
input_file = open(input_file_name, "r").read().split("\n")
input_f = list(map(lambda couple: list(map(splitSlash, couple.split(" "))), input_file))
q_file = open(q_mle, "r").read().split("\n")
e_file = open(e_mle, "r").read().split("\n")

#fill q from file from file
q={}
for row in q_file:
    a = row.split("\t")
    q[a[0]] = float(a[1])
    
#fill e from file    
e={}
for row in e_file:
    a = row.split("\t")
    b = a[0].split(" ")
    if b[0] not in e:
        e[b[0]] = {}
    e[b[0]][b[1]] = float(a[1])
    
 
total_count = 0
total_good = 0
taged_to_file = []

for row in input_f:
    tag_seq = ["SS","SS"]
    output = []
    for i,word in enumerate(row):
        predict_tag = General.argmax(getScore(word[0], tag_seq[-1], tag_seq[-2]))
        tag_seq.append(predict_tag)
        output.append(word[0] + "/" + predict_tag)
        if toCompare:
            total_count += 1
            if predict_tag == word[1]:
                total_good += 1
    if toCompare:
        print("Rate is: " + str(total_good) + "/" + str(total_count) + " = " + str(float(total_good) / total_count))
    taged_to_file.append(" ".join(output))        
    
out_file = open(greedy_hmm_output, 'w')
out_file.write('\n'.join(taged_to_file))
out_file.close()    
    
    
    
    
    
    