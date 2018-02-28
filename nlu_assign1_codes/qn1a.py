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
g=open("/home/saran-10/Desktop/SEM2/NLU/assign1/brown/cats.txt","r")
lem = WordNetLemmatizer()
new_token=[]  # check the open source for writing an array that can store n values i.e. list
main_cat=[]
main=[]
cat_dict= {}
for i in g.readlines():
    
    i = word_tokenize(i)
    #print(i[0])
    main.append(i)

for k in range(0,len(main),1):
    main_cat.append(main[k][1])
    if main[k][1] in cat_dict:
        cat_dict[main[k][1]] += 1
    else:
        cat_dict[main[k][1]] = 1
print(len(cat_dict))
#print(cat_dict['mystery'])
print(cat_dict)
length=0
lent=0
val1=0
train_sample=[]
test_sample=[]
dev_sample=[]
tok = []
pos = []
tok_dict = {}


size=[]   ## that keeps the track of length of each line by line readings used later
#for val1 in range(0,len(main),val1):
while val1<len(main):
    #print(val1)
    #print(lent)
    R=main[lent][1]
    #print(R)
    length=cat_dict[R]
    #print(length)
    val=0

    

    
    
    for val in range(val+lent,length+lent,1):
        f=open("/home/saran-10/Desktop/SEM2/NLU/assign1/brown/"+main[val][0],"r")
        for i in f.readlines():
            if(i=='\n'):
                continue;
            sent=""
            l=re.sub(r'[^\w\s/-]','',i)
            n=re.sub(r'\d','',l)
            p=re.sub(r'\t\n\s/\s','',n.lower().strip())
            sep=p.split(' ')
            line=[] ## for a list of each line that can be used for 'size' and also for forming n-grams line by line
            for j in sep:
                k=j.split('/')
                if k[0]=='':
                    continue;
                else:
                    line.append(lem.lemmatize(k[0], "v"))
                    tok.append(lem.lemmatize(k[0], "v")) ##only 1st words
                    pos.append(k[1].upper())##only for parts of speech
                    if k[0] in tok_dict:
                        tok_dict[k[0]] += 1
                    else:
                        tok_dict[k[0]] = 1
            size.append(len(line))  ##stores & appends the len of each line
    
    lent=val+1
    #print(lent)
    val1=lent
    #print(len(size))
    #print(len(line))
    #print(line)
    #print(size)
    
L=len(size)   ###this L=sum of the each indexed elements of size###
L=len(size)   
train_len=int(L*0.8)
test_len=int(L*0.1)
train_uni=[]
train_uni_dict={}
train_uni_1st=[]
train_uni_1st_dict={}
test_uni=[]
test_uni_dict={}
dev_uni=[]
dev_uni_dict={}
add=0
add1=0
for x in range(0,train_len,1):
    train_uni_1st.append(tok[add])    ##for uni_first letters
    add=add+size[x]
    
    
print(len(train_uni_1st))   ##for uni_first letters
print(add)
#for uni_first letters  counts of it in the dictionary
for i in range(0,len(train_uni_1st),1):
    if train_uni_1st[i] in train_uni_1st_dict:
        train_uni_1st_dict[train_uni_1st[i]] +=1
    else:
        train_uni_1st_dict[train_uni_1st[i]]=1
    
        


for y in range(train_len,train_len+test_len,1):
    add1=add1+size[y]
    
print(add1)

for k in range(0,add):
    if tok[k] in train_uni_dict:
        train_uni_dict[tok[k]] +=1
    else:
        train_uni_dict[tok[k]]=1
    train_uni.append(tok[k])        ########D1-TRAIN SAMPLE####
    
print(len(train_uni))

for k in range(add,add1+add):
    if tok[k] in test_uni_dict:
        test_uni_dict[tok[k]] +=1
    else:
        test_uni_dict[tok[k]]=1
    test_uni.append(tok[k])       ########D1-TEST SAMPLE####
    
print(len(test_uni))

for z in range(add1+add,len(tok),1):
    if tok[k] in dev_uni_dict:
        dev_uni_dict[tok[k]] +=1
    else:
        dev_uni_dict[tok[k]]=1
    dev_uni.append(tok[k])
    
print(len(dev_uni))####DEV SAMPLE####

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
L=len(size)

train_len=int(L*0.8)
print(train_len)
test_len=int(L*0.1)
###FOR THE TRAINING SAMPLES 
train_bi=[]
train_bi_dict={}
train_bi_1st=[]
train_bi_1st_dict={}
train_tri=[]
train_tri_dict={}
train_tri_1st=[]
train_tri_1st_dict={}
train_quad=[]
train_quad_dict={}
train_quad_1st=[]
train_quad_1st_dict={}
j=0
for x in range(0,train_len,1):
    train=[]  ## only for storing a particular sentence or line
    r=size[x]
    #print(r)
    i=0
    while i<r:
        train.append(tok[j+i])
        i=i+1
    ##train_uni_1st.append(train[0])##for uni_first letters
     
        
    j=j+r
      
    #print(train)
    #print(j)
    train_b=[]
    train_b=bigram(train,r)
    #print(train_b)
    
    for k in range(0,len(train_b),1):
        if train_b[k] in train_bi_dict:
            train_bi_dict[train_b[k]] +=1
        else:
            train_bi_dict[train_b[k]]=1
        train_bi.append(train_b[k])
        if k==0:
            train_bi_1st.append(train_b[k])        ###for bi_first letters list and its occurence in sample
        
        
    
            
    
    train_t=[]
    train_t=trigram(train,r)
    #print(train_b)
    
    for l in range(0,len(train_t),1):
        if train_t[l] in train_tri_dict:
            train_tri_dict[train_t[l]] +=1
        else:
            train_tri_dict[train_t[l]]=1
        train_tri.append(train_t[l])
        if l==0:
            train_tri_1st.append(train_t[l])  ###for tri_first letters list and its occurence in sample
        
        
    
    
    train_q=[]
    train_q=quadgram(train,r)
    #print(train_b)
    
    for o in range(0,len(train_q),1):
        if train_q[o] in train_quad_dict:
            train_quad_dict[train_q[o]] +=1
        else:
            train_quad_dict[train_q[o]]=1
        train_quad.append(train_q[o])
        if o==0:
            train_quad_1st.append(train_q[o])  ###for quad_first letters list and its occurence in sample
        
        

    
for i in range(0,len(train_bi_1st),1):##for bi_first letters  counts of it in the dictionary
    if train_bi_1st[i] in train_bi_1st_dict:
        train_bi_1st_dict[train_bi_1st[i]] +=1
    else:
        train_bi_1st_dict[train_bi_1st[i]]=1   
#for tri_first letters  counts of it in the dictionary
for i in range(0,len(train_tri_1st),1):
    if train_tri_1st[i] in train_tri_1st_dict:
        train_tri_1st_dict[train_tri_1st[i]] +=1
    else:
        train_tri_1st_dict[train_tri_1st[i]]=1
        
#for quad_first letters  counts of it in the dictionary
for i in range(0,len(train_quad_1st),1):
    if train_quad_1st[i] in train_quad_1st_dict:
        train_quad_1st_dict[train_quad_1st[i]] +=1
    else:
        train_quad_1st_dict[train_quad_1st[i]]=1
        
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
                    
    m=float((train_1st_dict[dataset[0]]+0.01)/((train_class[dataset[0]]+0.01*u)))
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

L=len(size)
##print(L)
train_len=int(L*0.8)
test_len=int(L*0.1) ### can be used for calculating the no of sentences in test data
##print(test_len)

###FOR THE TESTING SAMPLES AND FOR CALCULATING PERPLEXITY USING ADD-K IN ONE GO
test_bi=[]
test_bi_dict={}
test_bi_1st=[]
test_bi_1st_dict={}
test_tri=[]
test_tri_dict={}
test_tri_1st=[]
test_tri_1st_dict={}
test_quad=[]
test_quad_dict={}
test_quad_1st=[]
test_quad_1st_dict={}
j=0
per_un=0.0
per_b=0.0
per_t=0.0

for x in range(train_len,train_len+test_len,1):
    test=[]  ## only for storing a particular sentence or line
    r=size[x]
    #print(r)
    i=0
    while i<r:
        
        test.append(test_uni[j+i])
        i=i+1
    ##train_uni_1st.append(train[0])##for uni_first letters
     
        
    j=j+r
    ###forming test bigrams
    test_b=[]
    test_b=bigram(test,r)
    #print(train_b)
    
    for k in range(0,len(test_b),1):## DONT change train and test in this line
        if test_b[k] in test_bi_dict:
            test_bi_dict[test_b[k]] +=1
        else:
            test_bi_dict[test_b[k]]=1
        test_bi.append(test_b[k])

        
    ###forming test trigrams
    test_t=[]
    test_t=trigram(test,r)
    #print(train_b)
    
    
    for k in range(0,len(test_t),1):## DONT change train and test in this line
        if test_t[k] in test_tri_dict:
            test_tri_dict[test_t[k]] +=1
        else:
            test_tri_dict[test_t[k]]=1
        test_tri.append(test_t[k])

    
    #print(len(test))
    #c=0
    #for x in range(0,len(test)):
        #if test[x] in train_uni_dict:
            #continue;
        #else:
            #c=c+1
    if len(test)!=0:  ###as the lent(test) might go to zero
        #if c!=0:
            #print('out')
        per_uni=add_1_probability(test,train_uni_dict,len(train_uni),len(train_uni_dict),len(test))
        #else:
            #print('hello')
           # per_uni=probability(test,train_uni_dict,len(train_uni),len(train_uni_dict),len(test))###helps in better perplexity
        per_un=per_un + per_uni
    
    ## for BI-GRAMS OF TESTING AND ITS PERPLEXITY CALCULATION 
    #print(len(test_b))
    if len(test)!=0:###as the lent(test) might go to zero
        per_bi=bigram_add_1(test,train_uni_dict,train_uni_1st_dict,len(train_uni_dict),len(test))
        per_b=per_b + per_bi
    
    ## for TRI-GRAMS OF TESTING AND ITS PERPLEXITY CALCULATION
    if len(test)!=0:###as the lent(test) might go to zero
        
        per_tri=trigram_add_1(test,train_uni_dict,train_bi_dict,test_b,train_uni_1st_dict,len(train_uni_dict),len(test))
        per_t=per_t + per_tri



#print(len(train_uni_1st)) ###BOTH THE LENGTHS ARE SAME
#print(train_len)
print("PERPLEXITY of unigram=",per_un/len(test_uni)) ###train_len==len(train_uni_1st) sice,total no of
print("PERPLEXITY of bigram=",per_b/len(test_bi_dict)) ###train_len==len(train_uni_1st) sice,total no of
print("PERPLEXITY of trigram=",per_t/len(train_tri_dict)) ###train_len==len(train_uni_1st) sice,total no of




