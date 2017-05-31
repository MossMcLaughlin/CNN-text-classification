# 3/2017 | Moss McLaughlin


import sys
import csv
import itertools
import time
import operator
import io
import array
from datetime import datetime

#import matplotlib.pyplot as plt
import numpy as np
import nltk



SENTENCE_START_TOKEN = "sentence_start"
SENTENCE_END_TOKEN = "sentence_end"
UNKNOWN_TOKEN = "<UNK>"
ARTICLE_END_TOKEN = "</ARTICLE_END>"
NUM_TOKEN = "<NUM>"

similarity_threshold = 0.95
replace_similar_words = False
padding_len = 60

embedding_dim = 300
embedding_file = 'glove.6B/glove.6B.%sd.txt' % embedding_dim
#train_file = 'data/train.csv'
#test_file = 'data/test.csv'




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



def create_embedding(embedding_dict,vocab_size,itw):
    print("Building word embedding matrix...")
    E = [None] * (vocab_size+3)
    for i in range(3): E[i] = [0]*embedding_dim 
    for i in range(vocab_size): E[i+3] = embedding_dict[itw[i+3]]
    return E
        
def embedded_batch(batch,data_shape,E):
    batch = [list(tup) for tup in batch]
    embedded_batch = [[[E[word] for word in question] for question in line] for line in batch]
    X_train = np.zeros(data_shape)
    for i,line in enumerate(embedded_batch):
        for j,w in enumerate(line[0]):
            for k,e in enumerate(w):
                X_train[i][j][k][0] = e
        for j,w in enumerate(line[1]):
            for k,e in enumerate(w):
                X_train[i][j][k][1] = e
    return X_train 


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
    


def load_data(filename,vocabulary_size=12000):
    word_to_index = []
    index_to_word = []
    print("Reading CSV file...")
    with open(filename) as f:
        dat = csv.reader(f,delimiter=',',quotechar='"')
        tokenized_sentences1,tokenized_sentences2,is_duplicate = ([],[],[])
        for line in dat:
            tokenized_sentences1.append(line[3])
            tokenized_sentences2.append(line[4])
            is_duplicate.append(line[5])
    print("Parsed %d Questions.\n" % (len(tokenized_sentences1)*2))
    print("Tokenizing sentences...")
    tokenized_sentences1 = [nltk.word_tokenize(line) for line in tokenized_sentences1]
    tokenized_sentences2 = [nltk.word_tokenize(line) for line in tokenized_sentences2]
    print("Done.\n")

    tokenized_sentences = [tokenized_sentences1[1:],tokenized_sentences2[1:]] 
    txt = [a+b for a,b in zip(tokenized_sentences[0],tokenized_sentences[1])]    
    '''    
    # Replace all numbers with num_token
    for i,line in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w.isnumeric()==False else NUM_TOKEN for w in line]
    print("Done.\n")
    '''

    
    # Count word frequencies and build vocab
    word_freq = nltk.FreqDist(itertools.chain(*txt))
    n_data_words = len(word_freq.items())
    print("Found %d unique words tokens.\n" % n_data_words)
    
    
    embeddings = load_embeddings()
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    # Only consider words that are in GloVe embeddings.

    vocab = [w for w in vocab if w[0] in embeddings]
    n_glove_words = len(vocab)
    print("Found %d out of %d words in GloVe embeddings." % (n_glove_words,n_data_words))
    
    # We take the [vocabulary_size] most frequent words and build our word embedding matrix (or lookup table for now).  
    # Words in dataset are now either inside or outside embedding matrix.
    inside_words = sorted(vocab, key=operator.itemgetter(1))
    #outside_words = sorted(vocab[vocabulary_size:], key=operator.itemgetter(1))
    
    index_to_word = ["<PAD>", UNKNOWN_TOKEN,SENTENCE_END_TOKEN] + [x[0] for x in inside_words]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    E = create_embedding(embeddings,len(inside_words),index_to_word)
    print("Using vocabulary size %d." % len(E))
    print("The least frequent word in our vocablary is '%s' and appeared %d times in this dataset.\n" % (inside_words[1][0], inside_words[1][1]))
    print("The most most frequent words in our dataset: ",index_to_word[-10:])
    '''
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
    '''

    # Replace unknown words with unkown token (FOR NOW) 
    #  IN PROGRESS
    print("Replacing unknown words...")
    tokenized_sentences = [[[w if w in word_to_index else UNKNOWN_TOKEN for w in question] for question in channel] for channel in tokenized_sentences]
    print(tokenized_sentences[0][0:2])
    print("Done.\n")
    # Find max sentence length.
    q_len = []
    for channel in tokenized_sentences:
        for q in channel:
            q_len.append(len(q))
    #q_len = np.array([[len(q) for q in channel] for channel in tokenized_sentences])
    print("Max question length: ",q_len[np.array(q_len).argmax()])
    print("Mean question length: ",np.mean([q_len]))


    ### VIEW DATA ###
    '''
    import matplotlib.pyplot as plt
    # View histogram of question length to determine how to pad 
    bins = np.linspace(0,200,50)
    plt.hist(q_len,bins)
    plt.show()
    
    # See how question length affects score, this is a factor in where to pad 
    dups = is_duplicate[1:]*2
    q_len_short = []
    dups_short =[]
    for i,l in enumerate(q_len):
        if i % 12 == 0:
            q_len_short.append(l)
    for i,l in enumerate(dups):
        if i % 12 == 0:
            dups_short.append(l)    
    plt.plot(q_len_short,dups_short,'bo')
    plt.show()
    '''
    # This showed us that longest question that is similar to another question is 57 words long.
    # We can pad to 60 without loosing any duplicate examples.

    # Pad Data, assume all examples over the padding limit are NOT the same question for the TEST data.
    # On the train data we throw out all examples over the padding limit.

    print("Padding to questions to length %d..." % padding_len) 
    removed_q = []
    for channel in tokenized_sentences:
        for i,question in enumerate(channel):
            if not question or (len(question) > padding_len): 
                tokenized_sentences[0][i] = 'remove'
                tokenized_sentences[1][i] = 'remove'
                removed_q.append(i)

    # We need to make sure to remove the same rows from is_duplicate.
    is_duplicate = [d for i,d in enumerate(is_duplicate) if i not in removed_q] 


    tokenized_sentences = [[question for question in channel if not question=='remove'] for channel in tokenized_sentences]    

    for i,channel in enumerate(tokenized_sentences):
        for j,question in enumerate(channel):
            while len(question) < padding_len: question.append("<PAD>")    
        tokenized_sentences[i][j] = question
    print("Done.\n")
    avg_len = np.mean([len(q) for q in tokenized_sentences[0]])

    print(np.array(tokenized_sentences).shape)

    x_train = [[[word_to_index[w] for w in q] for q in channel] for channel in tokenized_sentences]   
    y_train = np.asarray([[1,0] if j == '1' else [0,1] for j in is_duplicate[1:]])
    x_train = list(zip(*x_train))
    return x_train,y_train,word_to_index,index_to_word,E


def load_test_data(test_data,vocabulary_size,word_to_index,E):

    print("Reading validation file...")

    with open(test_data) as f:
        dat = csv.reader(f,delimiter=',',quotechar='"')
        tokenized_sentences1,tokenized_sentences2 = ([],[])
        for line in dat[1:]:
            tokenized_sentences1.append(line[1])
            tokenized_sentences2.append(line[2])
            idx.append(line[0])
    print("Parsed %d Questions.\n" % (len(tokenized_sentences1)*2))
    print("Tokenizing sentences...")
    tokenized_sentences1 = [nltk.word_tokenize(line) for line in tokenized_sentences1]
    tokenized_sentences2 = [nltk.word_tokenize(line) for line in tokenized_sentences2]
    tokenized_sentences = [tokenized_sentences1,tokenized_sentences2]
    print("Done.\n")
    
    tokenized_sentences = [[[w if w in word_to_index else UNKNOWN_TOKEN for w in question] for question in channel] for channel in tokenized_sentences]

    X_data = np.asarray([[[E[word_to_index[w]] for w in question] for question in channel] for channel in tokenized_sentences])

    X_test = np.array([[[np.array([e1,e2]) for e1,e2 in zip(w1,w2)] for w1,w2 in zip(q1,q2)] for q1,q2 in zip(X_data[0],X_data[1])])

    return X_test

def create_batches(x_data,y_data,num_epochs,batch_size,data_shape,E):
    data_size = len(y_data)
    print("Data Size: ", data_size)
    shuffled_indices = np.random.permutation(np.arange(data_size))
    print("Shuffling data...")
    shuffled_x = [x_data[shuffled_indices[i]] for i in range(data_size)]
    shuffled_y = [y_data[shuffled_indices[i]] for i in range(data_size)]
    print("Done.\n")
    num_batches = int((np.floor(data_size/batch_size)))
    print("batch size: ",batch_size)
    print('Number of batches',num_batches)
    for j in range(num_epochs):
        print("\n-------------------------")
        print("Starting Epoch %d / %d...\n-------------------------" % (j,num_epochs)) 
        for i in range(num_batches):
            #print("batching batch number ",i)
            start_index = i * batch_size
            end_index = start_index + batch_size
            X_batch = shuffled_x[start_index:end_index]
            Y_batch = np.array(shuffled_y[start_index:end_index])
            X_batch = np.array(embedded_batch(X_batch,data_shape,E))
            zip_batches = list(zip(X_batch,Y_batch))
            yield zip_batches

def create_dev(x_data,y_data,E):
    data_size = len(y_data)
    print("Dev Data Size: ", data_size)
    shuffled_indices = np.random.permutation(np.arange(data_size))
    shuffled_x = [x_data[shuffled_indices[i]] for i in range(data_size)]
    shuffled_y = [y_data[shuffled_indices[i]] for i in range(data_size)]
    shuffled_x = [[[E[w] for w in question] for question in line] for line in shuffled_x] 
    X_dev = np.zeros((len(x_data),60,300,2))
    for i,line in enumerate(shuffled_x):
        for j,w in enumerate(line[0]):
            for k,e in enumerate(w):
                X_dev[i][j][k][0] = e
        for j,w in enumerate(line[1]):
            for k,e in enumerate(w):
                X_dev[i][j][k][1] = e    
    Y_dev = shuffled_y
    print("dev shape", np.array(X_dev).shape)
    return X_dev,Y_dev
