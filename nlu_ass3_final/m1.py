import numpy
import nltk
from nltk.tokenize import word_tokenize
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
dk_da = open("/home1/e1-246-19/nlu/assign3/nlu_ass3_final/ner.txt",'r').readlines()
dk_da = [i.strip().split(' ') for i in dk_da]
len(dk_da)
%%time
dk_sents = []
dk_ls = []
dk_word = set()
dk_tag = set()
for i in dk_da:
    if i == ['']:
        dk_sents.append(dk_ls)
        dk_ls = []
    else:
        dk_tag.add(i[1])
        dk_word.add(i[0])
        dk_tex = word_tokenize(i[0])
        k = [i[0],nltk.pos_tag(dk_tex)[0][1],i[1]]
        dk_ls.append(tuple(k))
        
train_sents = dk_sents[0:int(0.8*len(dk_sents))]
test_sents = dk_sents[int(0.8*len(dk_sents)):]
def word2features(dk_snt, i):
    dk_wrd = dk_snt[i][0]
    dk_post = dk_snt[i][1]
    
    dk_featrs = {
        'bias': 1.0,
        'dk_wrd.lower()': dk_wrd.lower(),
        'dk_wrd[-3:]': dk_wrd[-3:],
        'dk_wrd[-2:]': dk_wrd[-2:],
        'dk_wrd.isupper()': dk_wrd.isupper(),
        'dk_wrd.istitle()': dk_wrd.istitle(),
        'dk_wrd.isdigit()': dk_wrd.isdigit(),
        'dk_wrd.length()' : len(dk_wrd),
        'dk_post': dk_post,
        #'dk_post[:2]': dk_post[:2],        
    }
    if i > 0:
        dk_wrd1 = dk_snt[i-1][0]
        dk_post1 = dk_snt[i-1][1]
        dk_featrs.update({
            '-1:dk_wrd.lower()': dk_wrd1.lower(),
            '-1:dk_wrd.istitle()': dk_wrd1.istitle(),
            '-1:dk_wrd.isupper()': dk_wrd1.isupper(),
            '-1:dk_wrd.length()' : len(dk_wrd1),
            '-1:dk_post': dk_post1,
            #'-1:dk_post[:2]': dk_post1[:2],
        })
    else:
        dk_featrs['BOS'] = True
        
    if i < len(dk_snt)-1:
        dk_wrd1 = dk_snt[i+1][0]
        dk_post1 = dk_snt[i+1][1]
        dk_featrs.update({
            '+1:dk_wrd.lower()': dk_wrd1.lower(),
            '+1:dk_wrd.istitle()': dk_wrd1.istitle(),
            '+1:dk_wrd.isupper()': dk_wrd1.isupper(),
            '+1:dk_wrd.length()' : len(dk_wrd1),
            '+1:dk_post': dk_post1,
            #'+1:dk_post[:2]': dk_post1[:2],
        })
    else:
        dk_featrs['EOS'] = True
        
    if i > 1:
        dk_wrd1 = dk_snt[i-2][0]
        dk_post1 = dk_snt[i-2][1]
        dk_featrs.update({
            '-2:dk_wrd.lower()': dk_wrd1.lower(),
            '-2:dk_wrd.istitle()': dk_wrd1.istitle(),
            '-2:dk_wrd.isupper()': dk_wrd1.isupper(),
            '-2:dk_wrd.length()' : len(dk_wrd1),
            '-2:dk_post': dk_post1,
            #'-1:dk_post[:2]': dk_post1[:2],
        })
    else:
        dk_featrs['BOS1'] = True
        
    if i < len(dk_snt)-2:
        dk_wrd1 = dk_snt[i+2][0]
        dk_post1 = dk_snt[i+2][1]
        dk_featrs.update({
            '+2:dk_wrd.lower()': dk_wrd1.lower(),
            '+2:dk_wrd.istitle()': dk_wrd1.istitle(),
            '+2:dk_wrd.isupper()': dk_wrd1.isupper(),
            '+2:dk_wrd.length()' : len(dk_wrd1),
            '+2:dk_post': dk_post1,
            #'+1:dk_post[:2]': dk_post1[:2],
        })
    else:
        dk_featrs['EOS1'] = True
                
    return dk_featrs


def sent2features(dk_snt):
    return [word2features(dk_snt, i) for i in range(len(dk_snt))]

def sent2labels(dk_snt):
    return [label for token,p, label in dk_snt]

def sent2tokens(dk_snt):
    return [token for token,p, label in dk_snt]



%%time
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]



%%time
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)
crf.fit(X_train, y_train)


dk_lbls = list(crf.classes_)

y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred,average='weighted', dk_lbls=dk_lbls)

