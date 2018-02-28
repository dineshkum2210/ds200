from nltk.corpus import gutenberg
import re
import nltk
import sys

import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
lem = WordNetLemmatizer()
f=open("/home/saran-10/Desktop/SEM2/NLU/assign1/gutenberg/cats1.txt","r")
lem = WordNetLemmatizer()
new_token2=[]  # check the open source for writing an array that can store n values i.e. list
main_cat1=[]
main1=[]
cat_dict1= {}
for i in f.readlines():
    
    i = word_tokenize(i)
    #print(i[0])
    main1.append(i)
#print(main[1])
#print(len(main))
#print(main)
for k in range(0,len(main1),1):
    main_cat1.append(main1[k][1])
    if main1[k][1] in cat_dict1:
        cat_dict1[main1[k][1]] += 1
    else:
        cat_dict1[main1[k][1]] = 1

        
#L='news'
#print(cat_dict[L])
print(len(cat_dict1))
print(cat_dict1['gutenberg'])
print(cat_dict1)

length=0
lent=0
val1=0
train_sample1=[]
test_sample1=[]
dev_sample1=[]
tok1 = []
pos1 = []
tok_dict1 = {}


size1=[]   ## that keeps the track of length of each line by line readings used later
#for val1 in range(0,len(main1),val1):

R=main1[lent][1]
    #print(R)
length1=cat_dict1[R]
    #print(length)
val=0
for val in range(val,length1,1):
    emma = gutenberg.sents(""+main1[val][0]+".txt")
    sents_list = [" ".join(sent) for sent in emma]
    #print(len(sents_list))
    #print(sents_list[3])
    for k in range(0,len(sents_list)):


        l=re.sub(r'[^\w\s/-]','',sents_list[k].lower())
        sep=l.split(' ')
        line=[] ## for a list of each line that can be used for 'size' and also for forming n-grams line by line

        for j in range(0,len(sep)):
            if sep[j]=='':
                continue;
            else:
                line.append(lem.lemmatize(sep[j], "v"))
                tok1.append(lem.lemmatize(sep[j], "v"))
                if sep[j] in tok_dict1:
                    tok_dict1[sep[j]] += 1
                else:
                    tok_dict1[sep[j]] = 1
        size1.append(len(line))


L=len(size1)   ###this L=sum of the each indexed elements of size###
print(len(size1))

print(len(tok1))
L=len(size1)   
train_len1=int(L*0.8)
test_len1=int(L*0.1)
train_uni1=[]
train_uni_dict1={}
train_uni_1st1=[]
train_uni_1st_dict1={}
test_uni1=[]
test_uni_dict1={}
dev_uni1=[]
dev_uni_dict1={}
add=0
add1=0
for x in range(0,train_len1,1):
    train_uni_1st1.append(tok1[add])    ##for uni_first letters
    add=add+size1[x]
    
    
print(len(train_uni_1st1))   ##for uni_first letters
print(add)
#for uni_first letters  counts of it in the dictionary
for i in range(0,len(train_uni_1st1),1):
    if train_uni_1st1[i] in train_uni_1st_dict1:
        train_uni_1st_dict1[train_uni_1st1[i]] +=1
    else:
        train_uni_1st_dict1[train_uni_1st1[i]]=1
    
        


for y in range(train_len1,train_len1+test_len1,1):
    add1=add1+size1[y]
    
print(add1)

for k in range(0,add):
    if tok1[k] in train_uni_dict1:
        train_uni_dict1[tok1[k]] +=1
    else:
        train_uni_dict1[tok1[k]]=1
    train_uni1.append(tok1[k])        ########D2-TRAIN SAMPLE####
    
print(len(train_uni1))

for k in range(add,add1+add):
    if tok1[k] in test_uni_dict1:
        test_uni_dict1[tok1[k]] +=1
    else:
        test_uni_dict1[tok1[k]]=1
    test_uni1.append(tok1[k])       ########D2-TEST SAMPLE####
    
print(len(test_uni1))

for z in range(add1+add,len(tok1),1):
    if tok1[k] in dev_uni_dict1:
        dev_uni_dict1[tok1[k]] +=1
    else:
        dev_uni_dict1[tok1[k]]=1
    dev_uni1.append(tok1[k])
    
print(len(dev_uni1))####DEV SAMPLE####

def bigram(lineby,u):
    new_token1=[]
    tok1=[]
    for i in range(1,u):
        new_token1=lineby[i-1]+'_'+lineby[i]
        tok1.append(new_token1)
        
    return tok1

def trigram(lineby2,u):
    new_token2=[]
    tok2=[]
    for i in range(2,u):
        new_token2=lineby2[i-2]+ '_' +lineby2[i-1]+'_'+lineby2[i]
        tok2.append(new_token2)
        
    return tok2

def quadgram(lineby3,u):
    new_token3=[]
    tok3=[]
    for i in range(3,u):
        new_token3=lineby3[i-3]+ '_' +lineby3[i-2]+ '_' +lineby3[i-1]+'_'+lineby3[i]
        tok3.append(new_token3)
        
    return tok3

L=len(size1)

train_len1=int(L*0.8)
print(train_len1)
test_len1=int(L*0.1)
###FOR THE TRAINING SAMPLES 
train_bi1=[]
train_bi_dict1={}
train_bi_1st1=[]
train_bi_1st_dict1={}
train_tri1=[]
train_tri_dict1={}
train_tri_1st1=[]
train_tri_1st_dict1={}
train_quad1=[]
train_quad_dict1={}
train_quad_1st1=[]
train_quad_1st_dict1={}
j=0
for x in range(0,train_len1,1):
    train1=[]  ## only for storing a particular sentence or line
    r=size1[x]
    #print(r)
    i=0
    while i<r:
        train1.append(tok1[j+i])
        i=i+1
    ##train_uni_1st.append(train[0])##for uni_first letters
     
        
    j=j+r
      
    #print(train)
    #print(j)
    train_b1=[]
    train_b1=bigram(train1,r)
    #print(train_b)
    
    for k in range(0,len(train_b1),1):
        if train_b1[k] in train_bi_dict1:
            train_bi_dict1[train_b1[k]] +=1
        else:
            train_bi_dict1[train_b1[k]]=1
        train_bi1.append(train_b1[k])
        if k==0:
            train_bi_1st1.append(train_b1[k])        ###for bi_first letters list and its occurence in sample
        
        
    
            
    
    train_t1=[]
    train_t1=trigram(train1,r)
    #print(train_b)
    
    for l in range(0,len(train_t1),1):
        if train_t1[l] in train_tri_dict1:
            train_tri_dict1[train_t1[l]] +=1
        else:
            train_tri_dict1[train_t1[l]]=1
        train_tri1.append(train_t1[l])
        if l==0:
            train_tri_1st1.append(train_t1[l])  ###for tri_first letters list and its occurence in sample
        
        
    
    
    train_q1=[]
    train_q1=quadgram(train1,r)
    #print(train_b)
    
    for o in range(0,len(train_q1),1):
        if train_q1[o] in train_quad_dict1:
            train_quad_dict1[train_q1[o]] +=1
        else:
            train_quad_dict1[train_q1[o]]=1
        train_quad1.append(train_q1[o])
        if o==0:
            train_quad_1st1.append(train_q1[o])  ###for quad_first letters list and its occurence in sample
        
        

    
for i in range(0,len(train_bi_1st1),1):##for bi_first letters  counts of it in the dictionary
    if train_bi_1st1[i] in train_bi_1st_dict1:
        train_bi_1st_dict1[train_bi_1st1[i]] +=1
    else:
        train_bi_1st_dict1[train_bi_1st1[i]]=1   
#for tri_first letters  counts of it in the dictionary
for i in range(0,len(train_tri_1st1),1):
    if train_tri_1st1[i] in train_tri_1st_dict1:
        train_tri_1st_dict1[train_tri_1st1[i]] +=1
    else:
        train_tri_1st_dict1[train_tri_1st1[i]]=1
        
#for quad_first letters  counts of it in the dictionary
for i in range(0,len(train_quad_1st1),1):
    if train_quad_1st1[i] in train_quad_1st_dict1:
        train_quad_1st_dict1[train_quad_1st1[i]] +=1
    else:
        train_quad_1st_dict1[train_quad_1st1[i]]=1
        
    
'''print(len(train_bi1))
print(len(train_bi_dict1))

print(len(train_tri1))
print(len(train_tri_dict1))

print(len(train_quad1))
print(len(train_quad_dict1))
##print(len(train_uni_1st))
print(len(train_bi_1st1))
print(len(train_tri_1st1))
print(len(train_quad_1st1))
print(len(train_bi_1st_dict1))
print(len(train_tri_1st_dict1))
print(len(train_quad_1st_dict1))'''
import math
import numpy as np
def add_1_probability(dataset,train_class,v,u,l):
    log_prob=0.0
    
    for i in range(0,len(dataset)):
        if dataset[i] in train_class:
            r=train_class[dataset[i]]
        else:
            r=0
        log_prob =float(log_prob + np.log(((r+1.0)*1.0/(float(v)+1.0*u))))
                           
    tot_prob=(-1.0)*float(log_prob/l)
    perplexity=float(np.exp(tot_prob))
    #print log_prob
    return perplexity
    #return log_prob


def bigram_add_1(dataset,train_class,train_1st_dict,u,l):
    log_prob=0.0
    if dataset[0] not in train_1st_dict:
        train_1st_dict[dataset[0]]=0.0
        
    
    if dataset[0] not in train_class:
        train_class[dataset[0]]=0.0
                    
    m=float((train_1st_dict[dataset[0]]+1.0)/((train_class[dataset[0]]+1.0*u)))
    log_prob=np.log(m)
    
    
    for i in range(1,len(dataset)):
        if dataset[i] not in train_class:
            train_class[dataset[i]]=0.0
            
        if dataset[i-1] not in train_class:
            train_class[dataset[i-1]]=0.0
            
        
        log_prob =float(log_prob + np.log((float( train_class[dataset[i]]+1.0)*1.0/( train_class[dataset[i-1]]+1.0*u))))
                           
    tot_prob=(-1.0)*float(log_prob/l)
    perplexity=float(np.exp(tot_prob))
    #print log_prob
    return perplexity


def trigram_add_1(dataset,train_class,train_bi_di,test_b1,train_1st_dict,u,l):
    log_prob=0.0
    if dataset[0] not in train_1st_dict:
        train_1st_dict[dataset[0]]=0.0
        
    
    if dataset[0] not in train_class:
        train_class[dataset[0]]=0.0
                    
    m=float((train_1st_dict[dataset[0]]+1.0)/((train_class[dataset[0]]+1.0*u)))
    log_prob=np.log(m)
    #print(test_b1[0])
    '''if test_b1[0] not in train_bi_dict:
        train_bi_dict[test_b1[0]]=0.0
    
    if dataset[0] not in train_1st_dict:
        train_1st_dict[dataset[0]]=0.0
    
    
    
    k =log_prob + np.log((float( train_bi_di[test_b1[0]]+1.0)*1.0/( train_1st_dict[dataset[0]]+1.0*u)))
    log_prob=np.log(k)'''
    for i in range(1,len(dataset)-1):
        if test_b1[i] not in train_bi_di:
            train_bi_di[test_b1[i]]=0.0
            
        if dataset[i-1] not in train_class:
            train_class[dataset[i-1]]=0.0
            
        
        log_prob =float(log_prob + np.log((float( train_bi_di[test_b1[i]]+1.0)*1.0/( train_class[dataset[i-1]]+1.0*u))))
                           
    tot_prob=(-1.0)*float(log_prob/l)
    perplexity=float(np.exp(tot_prob))
    #print log_prob
    return perplexity
L=len(size1)
##print(L)
train_len1=int(L*0.8)
test_len1=int(L*0.1) ### can be used for calculating the no of sentences in test data
##print(test_len)

###FOR THE TESTING SAMPLES AND FOR CALCULATING PERPLEXITY USING ADD-K IN ONE GO
test_bi1=[]
test_bi_dict1={}
test_bi_1st1=[]
test_bi_1st_dict1={}
test_tri1=[]
test_tri_dict1={}
test_tri_1st1=[]
test_tri_1st_dict1={}
test_quad1=[]
test_quad_dict1={}
test_quad_1st1=[]
test_quad_1st_dict1={}
j=0
per_un1=0.0
per_b1=0.0
per_t1=0.0

for x in range(train_len1,train_len1+test_len1,1):
    test1=[]  ## only for storing a particular sentence or line
    r=size1[x]
    #print(r)
    i=0
    while i<r:
        
        test1.append(test_uni1[j+i])
        i=i+1
    ##train_uni_1st.append(train[0])##for uni_first letters
     
        
    j=j+r
    ###forming test bigrams
    test_b1=[]
    test_b1=bigram(test1,r)
    #print(train_b)
    
    for k in range(0,len(test_b1),1):## DONT change train and test in this line
        if test_b1[k] in test_bi_dict1:
            test_bi_dict1[test_b1[k]] +=1
        else:
            test_bi_dict1[test_b1[k]]=1
        test_bi1.append(test_b1[k])

        
    ###forming test trigrams
    test_t1=[]
    test_t1=trigram(test1,r)
    #print(train_b)
    
    
    for k in range(0,len(test_t1),1):## DONT change train and test in this line
        if test_t1[k] in test_tri_dict1:
            test_tri_dict1[test_t1[k]] +=1
        else:
            test_tri_dict1[test_t1[k]]=1
        test_tri1.append(test_t1[k])

    
    #print(len(test))
    #c=0
    #for x in range(0,len(test)):
        #if test[x] in train_uni_dict:
            #continue;
        #else:
            #c=c+1
    if len(test1)!=0:  ###as the lent(test) might go to zero
        #if c!=0:
            #print('out')
        per_uni1=add_1_probability(test1,train_uni_dict1,len(train_uni1),len(train_uni_dict1),len(test1))
        #else:
            #print('hello')
           # per_uni=probability(test,train_uni_dict,len(train_uni),len(train_uni_dict),len(test))###helps in better perplexity
        per_un1=per_un1 + per_uni1
    
    ## for BI-GRAMS OF TESTING AND ITS PERPLEXITY CALCULATION 
    #print(len(test_b))
    if len(test1)!=0:###as the lent(test) might go to zero
        per_bi1=bigram_add_1(test1,train_uni_dict1,train_uni_1st_dict1,len(train_uni_dict1),len(test1))
        per_b1=per_b1 + per_bi1
    
    ## for TRI-GRAMS OF TESTING AND ITS PERPLEXITY CALCULATION
    if len(test1)!=0:###as the lent(test) might go to zero
        
        per_tri1=trigram_add_1(test1,train_uni_dict1,train_bi_dict1,test_b1,train_uni_1st_dict1,len(train_uni_dict1),len(test1))
        per_t1=per_t1 + per_tri1



#print(len(train_uni_1st)) ###BOTH THE LENGTHS ARE SAME
#print(train_len)
print("PERPLEXITY of unigram=",per_un1/len(train_tri_dict1)) ###train_len==len(train_uni_1st) sice,total no of
print("PERPLEXITY of bigram=",per_b1/len(test_bi_dict1)) ###train_len==len(train_uni_1st) sice,total no of
print("PERPLEXITY of trigram=",per_t1/len(train_tri_dict1)) ###train_len==len(train_uni_1st) sice,total no of


