#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:52:16 2018

@author: slytherin
"""

import os
import fnmatch
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag,pos_tag_sents
import regex as re
import operator
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split  
from sklearn import metrics
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import pickle
from nltk.corpus import stopwords

path="op_spam/"
label=[]
configfiles=[os.path.join(subdir,f) for subdir,dirs,files in os.walk(path) for f in fnmatch.filter(files,'*.txt')]

for f in configfiles:
    c=re.search('(trut|deceptiv)\w',f)
    label.append(c.group())
    
labels=pd.DataFrame(label,columns=['Labels'])
review=[]
directory=os.path.join(path)
for subdirs,dirs,files in os.walk(path):
    for file in files:
        if fnmatch.filter(files,'*.txt'):
            f=open(os.path.join(subdirs,file),'r')
            a=f.read()
            review.append(a)
            
reviews=pd.DataFrame(review,columns=['Reviews'])
result=pd.merge(reviews,labels,right_index=True,left_index=True)
result['Reviews']=result['Reviews'].map(lambda x:x.lower())

stop=stopwords.words('english')
result['review_without_stopwords']=result['Reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

def pos(review_without_stopwords):
    return TextBlob(review_without_stopwords).tags

os=result.review_without_stopwords.apply(pos)
os1=pd.DataFrame(os)

os1['pos']=os1['review_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))
result= pd.merge(result, os1,right_index=True,left_index = True)

review_train, review_test, label_train, label_test = train_test_split(result['pos'],result['Labels'], test_size=0.2,random_state=40)

tf_vect=TfidfVectorizer(lowercase=True,use_idf=True,smooth_idf=True,sublinear_tf=False)
X_train_tf=tf_vect.fit_transform(review_train)
X_test_tf=tf_vect.transform(review_test)

def svm_param_selection(X,y,nfolds):
    c=[0.001,0.01,0.1,1,10]
    gamma=[0.001,0.01,0.1,1]
    param_grid={'C':c,'gamma':gamma}
    grid_search=GridSearchCV(svm.SVC(kernel='linear'),param_grid,cv=nfolds)
    grid_search.fit(X,y)
    return grid_search.best_params_

clf=svm.SVC(C=10,gamma=0.001,kernel='linear')
clf.fit(X_train_tf,label_train)
pred=clf.predict(X_test_tf)

with open('vectorizer.pickle','wb') as fin:
    pickle.dump(tf_vect,fin)
    
with open('mlmodel.pickle','wb') as f:
    pickle.dump(clf,f)
    
