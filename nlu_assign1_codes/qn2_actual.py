
# coding: utf-8

# In[1]:


##D1-TRAIN + D2-TRAIN
###D2-TRAIN
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


# In[2]:


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


# In[3]:


###D1-TRAIN  
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

        
#L='news'
#print(cat_dict[L])
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
#print(L)
#print(0.8*L)

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



# In[4]:


train_uni13= train_uni + train_uni1
print(len(train_uni13))
test_uni13=test_uni + test_uni1
print(len(test_uni13))
size13= size + size1
print(len(size13))
tok13=tok + tok1
print(len(tok13))


# In[5]:


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


# In[7]:


L=len(size13)   
train_len13=int(L*0.8)
test_len13=int(L*0.1)
train_uni13=[]
train_uni_dict13={}
train_uni_1st13=[]
train_uni_1st_dict13={}
test_uni13=[]
test_uni_dict13={}
dev_uni13=[]
dev_uni_dict13={}
add=0
add1=0
for x in range(0,train_len13,1):
    train_uni_1st13.append(tok13[add])    ##for uni_first letters
    add=add+size13[x]
    
    
print(len(train_uni_1st13))   ##for uni_first letters
print(add)
#for uni_first letters  counts of it in the dictionary
for i in range(0,len(train_uni_1st13),1):
    if train_uni_1st13[i] in train_uni_1st_dict13:
        train_uni_1st_dict13[train_uni_1st13[i]] +=1
    else:
        train_uni_1st_dict13[train_uni_1st13[i]]=1
    
        


for y in range(train_len13,train_len13+test_len13,1):
    add1=add1+size13[y]
    
print(add1)

for k in range(0,add):
    if tok13[k] in train_uni_dict13:
        train_uni_dict13[tok13[k]] +=1
    else:
        train_uni_dict13[tok13[k]]=1
    train_uni13.append(tok13[k])        ########D2-TRAIN SAMPLE####
    
print(len(train_uni13))

for k in range(add,add1+add):
    if tok13[k] in test_uni_dict13:
        test_uni_dict13[tok13[k]] +=1
    else:
        test_uni_dict13[tok13[k]]=1
    test_uni13.append(tok13[k])       ########D2-TEST SAMPLE####
    
print(len(test_uni13))

for z in range(add1+add,len(tok1),1):
    if tok13[k] in dev_uni_dict13:
        dev_uni_dict13[tok13[k]] +=1
    else:
        dev_uni_dict13[tok13[k]]=1
    dev_uni13.append(tok13[k])
    
print(len(dev_uni1))####DEV SAMPLE####


# In[8]:


L=len(size13)

train_len13=int(L*0.8)
print(train_len13)
test_len13=int(L*0.1)
###FOR THE TRAINING SAMPLES 
train_bi13=[]
train_bi_dict13={}
train_bi_1st13=[]
train_bi_1st_dict13={}
train_tri13=[]
train_tri_dict13={}
train_tri_1st13=[]
train_tri_1st_dict13={}
train_quad13=[]
train_quad_dict13={}
train_quad_1st13=[]
train_quad_1st_dict13={}
j=0
for x in range(0,train_len13,1):
    train13=[]  ## only for storing a particular sentence or line
    r=size13[x]
    #print(r)
    i=0
    while i<r:
        train13.append(tok13[j+i])
        i=i+1
    ##train_uni_1st.append(train[0])##for uni_first letters
     
        
    j=j+r
      
    #print(train)
    #print(j)
    train_b13=[]
    train_b13=bigram(train13,r)
    #print(train_b)
    
    for k in range(0,len(train_b13),1):
        if train_b13[k] in train_bi_dict13:
            train_bi_dict13[train_b13[k]] +=1
        else:
            train_bi_dict13[train_b13[k]]=1
        train_bi13.append(train_b13[k])
        if k==0:
            train_bi_1st13.append(train_b13[k])        ###for bi_first letters list and its occurence in sample
        
        
    
            
    
    train_t13=[]
    train_t13=trigram(train13,r)
    #print(train_b)
    
    for l in range(0,len(train_t13),1):
        if train_t13[l] in train_tri_dict13:
            train_tri_dict13[train_t13[l]] +=1
        else:
            train_tri_dict13[train_t13[l]]=1
        train_tri13.append(train_t13[l])
        if l==0:
            train_tri_1st13.append(train_t13[l])  ###for tri_first letters list and its occurence in sample
        
        
    
    
    train_q13=[]
    train_q13=quadgram(train13,r)
    #print(train_b)
    
    for o in range(0,len(train_q13),1):
        if train_q13[o] in train_quad_dict13:
            train_quad_dict13[train_q13[o]] +=1
        else:
            train_quad_dict13[train_q13[o]]=1
        train_quad13.append(train_q13[o])
        if o==0:
            train_quad_1st13.append(train_q13[o])  ###for quad_first letters list and its occurence in sample
        
        

    
for i in range(0,len(train_bi_1st13),1):##for bi_first letters  counts of it in the dictionary
    if train_bi_1st13[i] in train_bi_1st_dict13:
        train_bi_1st_dict13[train_bi_1st13[i]] +=1
    else:
        train_bi_1st_dict13[train_bi_1st13[i]]=1   
#for tri_first letters  counts of it in the dictionary
for i in range(0,len(train_tri_1st13),1):
    if train_tri_1st13[i] in train_tri_1st_dict13:
        train_tri_1st_dict13[train_tri_1st13[i]] +=1
    else:
        train_tri_1st_dict13[train_tri_1st13[i]]=1
        
#for quad_first letters  counts of it in the dictionary
for i in range(0,len(train_quad_1st13),1):
    if train_quad_1st13[i] in train_quad_1st_dict13:
        train_quad_1st_dict13[train_quad_1st13[i]] +=1
    else:
        train_quad_1st_dict13[train_quad_1st13[i]]=1
        
    
print(len(train_bi13))
print(len(train_bi_dict13))

print(len(train_tri13))
print(len(train_tri_dict13))

print(len(train_quad13))
print(len(train_quad_dict13))
##print(len(train_uni_1st))
print(len(train_bi_1st13))
print(len(train_tri_1st13))
print(len(train_quad_1st13))
print(len(train_bi_1st_dict13))
print(len(train_tri_1st_dict13))
print(len(train_quad_1st_dict13))


# In[9]:


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


# In[11]:


L=len(size13)
##print(L)
train_len13=int(L*0.8)
test_len13=int(L*0.1) ### can be used for calculating the no of sentences in test data
##print(test_len)

###FOR THE TESTING SAMPLES AND FOR CALCULATING PERPLEXITY USING ADD-K IN ONE GO
test_bi13=[]
test_bi_dict13={}
test_bi_1st13=[]
test_bi_1st_dict13={}
test_tri13=[]
test_tri_dict13={}
test_tri_1st13=[]
test_tri_1st_dict13={}
test_quad13=[]
test_quad_dict13={}
test_quad_1st13=[]
test_quad_1st_dict13={}
j=0
per_un13=0.0
per_b13=0.0
per_t13=0.0

for x in range(train_len13,train_len13+test_len13,1):
    test13=[]  ## only for storing a particular sentence or line
    r=size13[x]
    #print(r)
    i=0
    while i<r:
        
        test13.append(test_uni13[j+i])
        i=i+1
    ##train_uni_1st.append(train[0])##for uni_first letters
     
        
    j=j+r
    ###forming test bigrams
    test_b13=[]
    test_b13=bigram(test13,r)
    #print(train_b)
    
    for k in range(0,len(test_b13),1):## DONT change train and test in this line
        if test_b13[k] in test_bi_dict13:
            test_bi_dict13[test_b13[k]] +=1
        else:
            test_bi_dict13[test_b13[k]]=1
        test_bi13.append(test_b13[k])

        
    ###forming test trigrams
    test_t13=[]
    test_t13=trigram(test13,r)
    #print(train_b)
    
    
    for k in range(0,len(test_t13),1):## DONT change train and test in this line
        if test_t13[k] in test_tri_dict13:
            test_tri_dict13[test_t13[k]] +=1
        else:
            test_tri_dict13[test_t13[k]]=1
        test_tri13.append(test_t13[k])

    
    #print(len(test))
    #c=0
    #for x in range(0,len(test)):
        #if test[x] in train_uni_dict:
            #continue;
        #else:
            #c=c+1
    if len(test13)!=0:  ###as the lent(test) might go to zero
        #if c!=0:
            #print('out')
        per_uni13=add_1_probability(test13,train_uni_dict13,len(train_uni13),len(train_uni_dict13),len(test13))
        #else:
            #print('hello')
           # per_uni=probability(test,train_uni_dict,len(train_uni),len(train_uni_dict),len(test))###helps in better perplexity
        per_un13=per_un13 + per_uni13
    
    ## for BI-GRAMS OF TESTING AND ITS PERPLEXITY CALCULATION 
    #print(len(test_b))
    if len(test13)!=0:###as the lent(test) might go to zero
        per_bi13=bigram_add_1(test13,train_uni_dict13,train_uni_1st_dict13,len(train_uni_dict13),len(test13))
        per_b13=per_b13 + per_bi13
    
    ## for TRI-GRAMS OF TESTING AND ITS PERPLEXITY CALCULATION
    if len(test13)!=0:###as the lent(test) might go to zero
        
        per_tri13=trigram_add_1(test13,train_uni_dict13,train_bi_dict13,test_b13,train_uni_1st_dict13,len(train_uni_dict13),len(test13))
        per_t13=per_t13 + per_tri13


print(per_un13)
print(per_b13)
print(per_t13)
#print(len(train_uni_1st)) ###BOTH THE LENGTHS ARE SAME
#print(train_len)
print("PERPLEXITY of unigram=",per_un13/len(train_tri_dict13)) ###train_len==len(train_uni_1st) sice,total no of
print("PERPLEXITY of bigram=",per_b13/len(test_bi_dict13)) ###train_len==len(train_uni_1st) sice,total no of
print("PERPLEXITY of trigram=",per_t13/len(train_tri_dict13)) ###train_len==len(train_uni_1st) sice,total no of
print(len(train_tri_dict13))
print(len(test_bi_dict13))
print(len(train_tri_dict13))


# In[12]:


### generation of unigram by eliminating stopwords

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
filtered_train13=[]
for i in range(0,len(train_uni13)):
    if train_uni13[i] not in stop_words:
        filtered_train13.append(train_uni13[i])
        
        
print(len(train_uni13))
print(len(filtered_train13))### new filtered neg


# In[13]:


filtered_train_dict13={}

for k in range(0,len(filtered_train13)):
    if filtered_train13[k] in filtered_train_dict13:
        filtered_train_dict13[filtered_train13[k]] += 1
    else:
        filtered_train_dict13[filtered_train13[k]] = 1
print(len(filtered_train_dict13))


# In[14]:


import sys
from collections import Counter, defaultdict
from itertools import groupby
from operator import itemgetter
from timeit import timeit
def maximum(my_dict):
    a1 = my_dict
    max=[]
    max1=[]
    a1_sorted_keys = sorted(a1, key=a1.get, reverse=True)
    for r in a1_sorted_keys:
        max.append(a1[r])
        #print (r, a1[r])
        #for i in range(0,5):

    for i in range(0,11):
        max1.append(max[i])
        
    return max1


# In[52]:


def maximum1(my_dict):
    a1 = my_dict
    max=[]
    max1=[]
    #a1_sorted_keys = sorted(a1, key=a1.get, reverse=True)
    for r in range(0,a1):
        max.append(a1[r])
        #print (r, a1[r])
        #for i in range(0,5):

    for i in range(0,10):
        max1.append(max[i])
        
    return max1


# In[45]:


r=maximum(filtered_train_dict13)
#print(r)

res13 = {v:k for k,v in filtered_train_dict13.items()}
#print(res13)
re13=[]
for i in range(0,len(r)):
    if (res13[r[i]]!='--'):
        print(res13[r[i]])
        re13.append(res13[r[i]])
#print(re13)


# In[65]:


def bigram(lineby,u):
    new_token1=[]
    tok1=[]
    for i in range(0,len(u)):
        new_token1=lineby+'_'+u[i]
        tok1.append(new_token1)
        
    return tok1


# In[66]:


w=maximum(train_uni_1st_dict1)
biterm=[]
res13_1 = {v:k for k,v in train_uni_1st_dict1.items()}
#for i in range(0,len(r)):
    #if (res13_1[r[i]]!='--'):
print(res13_1[w[0]])
biterm.append(res13_1[w[0]])


# In[75]:


bi=bigram(res13_1[w[0]],re13)
print(bi)
maxbi=[]
for i in range(0,len(bi)):
    if bi[i] in train_bi_dict13:
        maxbi.append(train_bi_dict13[bi[i]])
print(maxbi)


# In[76]:


'''w=maxbi.sort(reverse=True)


w=maximum1(maxbi)
res13_1 = {v:k for k,v in maxbi.items()}
print(res13_1[w[0]])
biterm.append(res13_1[w[0]])'''

