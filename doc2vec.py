# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:57:03 2019
# A sample of Doc2Vec implementation on documents. 
@author: sawid
"""

import pandas as pd
import numpy as np
import sklearn
import collections
import random
from sklearn.model_selection import train_test_split
import gensim
from gensim.test.utils import common_texts
from gensim.models import Phrases
import smart_open
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

data=pd.read_csv('sentences.csv')
res=open('train_results.csv', 'a+')

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])



train_corpus = list(read_corpus('sentences.csv'))

X_train, X_test, = train_test_split(train_corpus, test_size=.5, random_state=0)

#test_corpus = list(read_corpus('sentences2.csv', tokens_only=True))[1404:]



modelv = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
modelv.build_vocab(X_train)
%time modelv.train(X_train, total_examples=modelv.corpus_count, epochs=modelv.epochs)


ranks = []
second_ranks = []
third_ranks=[]
for doc_id in range(len(X_train)):
    inferred_vector = modelv.infer_vector(X_train[doc_id].words)
    sims = modelv.docvecs.most_similar([inferred_vector], topn=len(modelv.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])
    third_ranks.append(sims[2])


for doc_id in range(len(X_train)):
    #res.write(str('Document')+str(doc_id)+ str((train_corpus[doc_id].words)))
    out1=' '.join(train_corpus[doc_id].words)
    res.write(str(doc_id)+': '+str(out1))
   
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % modelv)
    
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD', 2), ('LEAST', len(sims) - 1)]:
        #res.write(str(label)+ str( sims[index])+str((train_corpus[sims[index][0]].words))+'\n')
        #id2=print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
        out2=' '.join(train_corpus[sims[index][0]].words)
        res.write(str(label)+','+str(out2)+'\n')
        res.close()
        #res.write(str(id2))


 #Validation   
doc_id = random.randint(0, len(X_train) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(X_train[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(X_train[sim_id[0]].words)))    


#testing

doc_id = random.randint(0, len(X_test) - 1)
inferred_vector = modelv.infer_vector(X_test[doc_id].words)
sims = modelv.docvecs.most_similar([inferred_vector], topn=len(modelv.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(X_test[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % modelv)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(X_train[sims[index][0]].words)))

