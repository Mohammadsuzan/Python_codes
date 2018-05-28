# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:52:41 2018

@author: Mohammadsuzan.Shaikh
"""

import urllib2
from bs4 import BeautifulSoup

url = "https://de.dariah.eu/tatom/topic_model_python.html"

soup = BeautifulSoup(urllib2.urlopen(url))

with open('E:\\My Python codes\\Web scrapping\\paragraph scrapping\\ctp_output.txt', 'w') as f:
    for tag in soup.find_all('p'):
        f.write(tag.text.encode('utf-8') + '\n')

'''Cleaning the text document'''
import re

f= open('E:\\My Python codes\\Web scrapping\\paragraph scrapping\\ctp_output.txt', 'r')
string=""
for line in f.readlines():
    string=string+line

string1=string.replace('\\n', "") #This brings line in continuation
string=string1.replace('\n', " ") #Will replace |n with spaces
splited=string.split('.')

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(str(splited)).split() for doc in splited]

# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 

dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=10, num_words=5))
