import re

"""
Igal Zaidman 311758866
Tal Pogorelis 318225349
"""
def bslash_split(s):
    i = s.rfind("/")
    return [s[:i],s[i+1:]]
def assignWrodClass(s):
    if s.isnumeric() and len(s) == 2:
        return("^2Dig")
    elif s.isnumeric() and len(s) == 4:
        return("^4Dig")
    elif bool(re.search(r'\d',s)) and bool(re.search("[a-zA-Z]",s)):
        return("^DigAlpha")
    elif bool(re.search('[0-9]+-[0-9]+', s)):
        return("^DigDash")
    elif bool(re.search('[0-9]+/[0-9]+/[0-9]+', s)):
        return("^DigSlash")
    elif bool(re.search(r'\d',s)) and bool(s.find(",")):
        return("^DigComma")
    elif bool(re.search(r'\d',s)) and bool(s.find(".")):
        return("^DigPeriod")
    elif s.isnumeric():
        return("^othernum")
    elif bool(re.search("^[A-Z]+$",s)):
        return("^allCaps")
    elif bool(re.search("^[A-Z]\.",s)):
        return("^capPeriod")    
    elif bool(re.search("^[A-Z]",s)):
        return("^initCap")
    elif bool(re.search("^[a-z]+$",s)):
        return("^lowercase")
    else:
        return("^UNK")
        
def argmax(d):
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

def getFeatures(w1,w2,w3,w4,w5,w1_t,w2_t):
    features = {}
    for i in range(4):
        if len(w3) > i:
            features['prefix' + str(i + 1)] = w3[:i + 1]
            features['suffix' + str(i + 1)] = w3[len(w3) - i - 1:]
    features['contains_number'] = bool(re.search("\d",w3))
    features['contains_hyphen'] = bool(re.search("-",w3))
    features['contains_uppercase'] = bool(re.search("[A-Z]",w3))
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
        