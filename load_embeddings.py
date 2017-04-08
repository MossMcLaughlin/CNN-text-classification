# 3/2017 | Moss McLaughlin


import sys
import csv
import itertools
import time
import operator
import io
import array
from datetime import datetime

import numpy as np
import nltk



SENTENCE_START_TOKEN = "sentence_start"
SENTENCE_END_TOKEN = "sentence_end"
UNKNOWN_TOKEN = "<UNK>"
ARTICLE_END_TOKEN = "</ARTICLE_END>"
NUM_TOKEN = "<NUM>"

similarity_threshold = 0.95
replace_similar_words = True

embedding_file = 'glove.6B/glove.6B.50d.txt'
data_file = '../txtgen/data/IMDB_Data.txt'


def load_embeddings():
  print("Loading word embeddings...")
  with open(embedding_file) as f:
    words = {}
    w = [line.split(' ') for line in f]
    v = [x[1:] for x in w]
    w = [x[0] for x in w]
    print("Word embedding vocab size: ",len(v),'\n')
    

    for i in range(len(w)):
      words[w[i]] = v[i]

    return(words)



def create_embedding(vocab_size,itw):
    print("Building word embedding matrix...")
    E = [None] * vocab_size
    embedding_dict = load_embeddings()
    for i in range(vocab_size): E[i] = embedding_dict[itw[i]]
    return E
        



def cos_similarity(x,y):
    x = np.array(x).astype(np.float)
    y = np.array(y).astype(np.float)
    d = np.dot(x,y) / (np.sqrt(np.dot(x,x))*(np.sqrt(np.dot(y,y))))
    return d



def find_similar_words(word,word_list,embeddings,similarity_threshold):
    similarity = [cos_similarity(embeddings[word],embeddings[w[0]]) for w in word_list]
    similarity = np.array(similarity)
    if similarity[similarity.argmax()] > similarity_threshold:
        return word,similarity.argmax()
    



def load_data(filename, vocabulary_size=2000):
    word_to_index = []
    index_to_word = []
    print("Reading text file...")
    with open(filename, 'rt') as f:
        txt = f.read()
        txt = txt.split(ARTICLE_END_TOKEN)
        txt = [line.split('.') for line in txt]
        txt.pop()
        txt.pop()
        for line in txt: line.pop()
        print('Raw training data:')
        print(txt[0][:2])
        print('\n')
        print(txt[-1][-2:])
        print('\n')

        
        
    print("Parsed %d sentences.\n" % (np.sum([len(article) for article in txt])))
    
    print("Tokenizing sentences...")
    tokenized_sentences = [[nltk.word_tokenize(line.replace('<br /><br />',' ').lower()) for line in article] for article in txt]
    print("Done.\n")
    
    
    for i,article in enumerate(tokenized_sentences):
        a = []
        for sent in article: a += sent
        tokenized_sentences[i] = a
    
    
    
    # Filter Words.
    print("Filtering words...")
    tokenized_sentences = [[w for w in line if not w==''] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if '\\' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if '*' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if '[' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if ']' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if '"' not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if "'" not in w] for line in tokenized_sentences]
    tokenized_sentences = [[w for w in line if "`" not in w] for line in tokenized_sentences]
    
    
    # Replace all numbers with num_token
    for i,line in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w.isnumeric()==False else NUM_TOKEN for w in line]
    print("Done.\n")
    
    print(tokenized_sentences[:5])
    
    # Count word frequencies and build vocab
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    n_data_words = len(word_freq.items())
    print("Found %d unique words tokens.\n" % n_data_words)

    
    embeddings = load_embeddings()
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    # Only consider words that are in GloVe embeddings and appear at least twice.
    vocab = [w for w in vocab if w[0] in embeddings]
    n_glove_words = len(vocab)
    print("Found %d out of %d words in GloVe embeddings." % (n_glove_words,n_data_words))
    vocab = [w for w in vocab if w[1] > 1]
    
    # We take the [vocabulary_size] most frequent words and build our word embedding matrix (or lookup table for now).  
    # Words in dataset are now either inside or outside embedding matrix.
    inside_words = sorted(vocab[:vocabulary_size], key=operator.itemgetter(1))
    outside_words = sorted(vocab[vocabulary_size:], key=operator.itemgetter(1))
    print("%d out of %d words appears more than once.\n" % (len(vocab),n_glove_words))
    
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN,SENTENCE_END_TOKEN] + [x[0] for x in inside_words]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocablary is '%s' and appeared %d times in this dataset.\n"           % (inside_words[1][0], inside_words[1][1]))
    
    # Find similar words that are in the data set but outside of our vocabulary
    if replace_similar_words:
        print("Searching for similar words...")
        similar_words = {}
        for w in outside_words:
            try: 
                similar_word,similar_index = find_similar_words(w[0],inside_words,embeddings,similarity_threshold)
                print("Replacing %s with %s" % (similar_word,inside_words[similar_index][0]))
                similar_words[similar_word] = inside_words[similar_index]
            except: None
            for line in tokenized_sentences:
                for x in line:
                    if x in similar_words: x = similar_words[x] 
                    
    
    # Save vocab in a file with one words in each line, from most to least frequent 
    #         (if same vocab is to be used for training and later evaluation)

    
    # Filter sentences 
    for i,line in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in line]
    tokenized_sentences = [s for s in tokenized_sentences if (len(s) > 1)]
    

    print('Filtered training data:')
    print(tokenized_sentences[:5])
    print('\n')

    # Build training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X_train,y_train,word_to_index,index_to_word
