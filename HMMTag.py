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
viterbi_hmm_output = sys.argv[4]
extra_file = sys.argv[5]
toCompare = True if len(sys.argv) > 6 else False
"""
input_file_name = "ass1-tagger-test-input"
e_mle = "e.mle"
q_mle = "q.mle"
viterbi_hmm_output = "viterbi_hmm_output.txt"
extra_file = "extra_file.txt"
toCompare = False

input_file_name = "test.blind"
e_mle = "ner_e.mle"
q_mle = "ner_q.mle"
viterbi_hmm_output = "ner.hmm.pred"
extra_file = "extra_file.txt"
toCompare = False
"""
e_cache = {}
q_cache = {}

###############
# ALL THE FUNCTIONS
###############
def getE(word, tag):
    curr_score = word + " " + tag
    if curr_score in e_cache:
        return e_cache[curr_score]
    #word not seen in the train
    if word not in e:
        word = General.assignWrodClass(word)
        
    if tag in q:
        tag_amt = q[tag]
    else:
        tag_amt = 100000

    if tag in e[word]:
        e_cache[curr_score] = e[word][tag] / tag_amt
        return e_cache[curr_score]
    
    #if tag was not seen for the word in training than return a low score
    e_cache[curr_score] = 0.5 / tag_amt
    return e_cache[curr_score]

def getQ(c, a, b):
    abc = a + " " + b + " " + c
    if abc in q_cache:
        return q_cache[abc]

    count_bc = 0
    count_abc = 0

    count_c = 0
    if c in q:
        count_c = (q[c])

    bc = b + " " + c
    if bc in q:
        count_bc = (q[bc])

    if abc in q:
        count_abc = (q[abc])

    count_ab = 1
    ab = a + " " + b
    if ab in q:
        count_ab = (q[ab])

    count_b = 1
    if b in q:
        count_b = (q[b])

    #q_cache[abc] = (count_abc / count_ab) * 0.91 + (count_bc / count_b) * 0.08 + (count_c / len(e_file)) * 0.01
    q_cache[abc] = (count_abc / count_ab) * 0.84 + (count_bc / count_b) * 0.15 + (count_c / chars_count) * 0.01

    return q_cache[abc]

def getScore(word, tag, tag_p2, tag_p):
    return getE(word, tag) * getQ(tag, tag_p2, tag_p)
    #curr_e = getE(word, tag)
    #curr_q = getQ(tag, tag_p2, tag_p)
    #return curr_e * curr_q


def prun(word):
    if word[-3:] == 'ing':
        return(["VBG"])
    elif bool(re.search("-",word)):
        return(["JJ"])
    elif len(word) > 1 and word.isupper():
        return(["NNP"])
    elif sum(map(lambda c: (1 if c.isdigit() else 0), word)) > float(len(word)) / 2:
        return(["CD"])
    elif len(word) > 0 and word[0].isupper():
        return(["NNP", "NN", "NNS"])
    elif word[-4:] == 'able':
        return(["JJ"])
    elif word[-2:] == 'ly':
        return(["RB"])
    elif word[-3:] == 'ers':
        return(["NNS"])
    elif word[-4:] == 'tion' or word[-3:] == 'ist' or word[-2:] == 'ty':
        return(["NN"])
    else:
        word = General.assignWrodClass(word)
        return(list(e[word].keys()))

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
q_file = open(q_mle, "r").read().split("\n")
e_file = open(e_mle, "r").read().split("\n")

count_total = 0
count_good = 0
taged_to_file = []

#fill e from file from file
tags = set(["SS"])
chars_count = 0
e = {}
for row in e_file:
    a = row.split("\t")
    b = a[0].split(" ")
    if b[0] not in e:
        e[b[0]] = {}
    e[b[0]][b[1]] = float(a[1])
    chars_count += len(row)

#fill q from file from file
q={}
for row in q_file:
    a = row.split("\t")
    q[a[0]] = float(a[1])
    
#main logic:
for row in input_f:
    n = len(row) - 1

    V = [{} for w in row] + [{}]
    for t in tags:
        V[0][t] = {}
        for r in tags:
            V[0][t][r] = 0

    V[0]["SS"]["SS"] = 1
    bp = [{} for w in row] + [{}]
    tags_p2 = ["SS"]
    tags_p = ["SS"]
    
    for i in range(n + 1):
        word = row[i][0]
        
        if word not in e:
            tags_curr = prun(word)
            word = General.assignWrodClass(word)
        else:
            tags_curr = list(e[word].keys()) 

        V[i + 1] = {}
        bp[i + 1] = {}
        for t in tags_p:
            V[i + 1][t] = {}
            bp[i + 1][t] = {}
            for r in tags_curr:
                l = {}
                for tT in tags_p2:
                    l[tT] = (V[i][tT][t]) * getScore(word, r, tT, t)

                V[i + 1][t][r] = max(list(l.values()))
                bp[i + 1][t][r] = General.argmax(l)

        tags_p2 = tags_p
        tags_p = tags_curr
        
    V.pop(0)
    bp.pop(0)
    endMatrix = map(lambda x: x.values(), V[n].values())
    maxEnd = list(map(max, endMatrix))
    maxV = max(maxEnd)
    maxTIndex = maxEnd.index(maxV)
    maxT = list(V[n].keys())[maxTIndex]
    maxR = General.argmax(V[n][maxT])
    y = [0 for i in range(0, n + 1)]
    y[n] = maxR
    y[n - 1] = maxT

    for i in reversed(range(0, n - 1)):
        y[i] = bp[i + 2][y[i + 1]][y[i + 2]]

    if toCompare:
        for i,w in enumerate(row):
            count_total += 1
            if w[1] == y[i]:
                count_good += 1
        print("Rate is: " + str(count_good) + "/" + str(count_total) + " = " + str(float(count_good) / count_total))            
    
    taged_to_file.append(" ".join(map(lambda i: str(row[i][0]) + "/" + str(y[i]), range(0, n + 1))))

out_file = open(viterbi_hmm_output, 'w')
out_file.write('\n'.join(taged_to_file))
out_file.close() 