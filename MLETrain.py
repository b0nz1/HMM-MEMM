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
#input_file_name = "train"
#e_mle = "ner_e.mle"
#q_mle = "ner_q.mle"
low_freq_cutoff = 5
e={}
e_lf = {}
q={}
output_e = []
output_q = []

input_file = open(input_file_name,"r")
input_content = input_file.read().split("\n")
input_file.close()

lst = list(map(lambda couple: list(map(General.bslash_split, couple.split(' ') )), input_content))

#fill the data of e before the low frequency treatment and q  
for line_i in range(len(lst)):
    for i in range(len(lst[line_i])):
        #Fill e data
        #first appearrance of word
        if lst[line_i][i][0] not in e:
            e[lst[line_i][i][0]] = {}
                
        # first appearance of tag to this particular word    
        if lst[line_i][i][1] not in e[lst[line_i][i][0]]:
            e[lst[line_i][i][0]][lst[line_i][i][1]] = 0
            
        e[lst[line_i][i][0]][lst[line_i][i][1]] += 1
        
        #fill q data
        for j in range(3):
            triple = ' '.join(map(lambda n: 'SS' if n < 0 else lst[line_i][n][1], range(i - j, i + 1)))
                #triple = ' '.join(map(lambda n: 'SS' if n < 0 else line[n][1], range(i - j, i + 1)))
            if triple not in q:
                q[triple] = 0
            q[triple] += 1
            
#change low frequency data of e
for k in e:
    for t in e[k]:
        #assign word class from pattern if less than frequency cuttoff 
        if e[k][t] < low_freq_cutoff:
            word_class = General.assignWrodClass(k)
        else:
            word_class = k
        
        if word_class not in e_lf:
            e_lf[word_class] = {}
            
        if t not in e_lf[word_class]:
            e_lf[word_class][t] =e[k][t]
        else:
            e_lf[word_class][t] += e[k][t]

#prepare format of e for output file        
for k in e_lf:
    for t in e_lf[k]:
        output_e.append(k + " " + t + "\t" + str(e_lf[k][t]))
       
e_file = open(e_mle,"w")
e_file.write("\n".join(output_e))
e_file.close()

output_q = []
for k in q:
    output_q.append(k + '\t' + str(q[k]))

q_file = open(q_mle, "w")
q_file.write("\n".join(output_q))
q_file.close()