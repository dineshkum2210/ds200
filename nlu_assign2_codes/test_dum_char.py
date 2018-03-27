
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from nltk.corpus import gutenberg
import numpy as np
from string import digits
from itertools import chain
import regex as re
import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


def load():

    test1=[]    
    train1=[]
    train=[]
    validation=[]
    test=[]  

    for fileid in gutenberg.fileids():
        sent=gutenberg.sents(fileid)
#        sent=gutenberg.words(fileid)
        s=[]
        for str1 in sent:
            s.append(str1)

        
        str2=[]
        for i in s:
            str2.append(' '.join(i))
        
        str3=''
        for i in str2:
            str3= str3 + i.translate(str.maketrans('', '', digits)).lower()
                
        str3=re.sub("[^\P{P}]+", "", str3)           
        punctuation={' s ',' d ',' t ',' ve ',' ll ',' \'', ' st ', ' nd ', ' rd ', '`' , '$'}
        for c in punctuation:
            str3= str3.replace(c,"")

        str3=' '.join(str3.split())
        words = str3.split(' ')
        length_1=int(len(words)*0.6)
        train1.append(words[:round(length_1*0.7)])
        test1.append(words[-round(length_1*0.3):])

    
    train = [item for sublist in train1 for item in sublist]
    test = [item for sublist in test1 for item in sublist]
    
    validation = train[-round(len(train)*0.3):]
    train = train[:round(len(train)*0.7)]
    
    raw_text=' '
    raw_text=' '.join(train)

    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print ("Total Characters: ", n_chars)
    print ("Total Vocab: ", n_vocab)
    
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 80
    dataX = []
    dataY = []
    
    for i in range(0, n_chars - seq_length, 1):
        	seq_in = raw_text[i:i + seq_length]
        	seq_out = raw_text[i + seq_length]
        	dataX.append([char_to_int[char] for char in seq_in])
        	dataY.append(char_to_int[seq_out])
    
    n_patterns = len(dataX)
    print ("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    
    
        
        #from tensorflow.python.client import device_lib
        #print(device_lib.list_local_devices())
        # define the LSTM model
    with tf.device('/gpu:0'):
        model = Sequential()
        model.add(LSTM(250, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='relu'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        

    # define the checkpoint
    filepath="/home1/e1-246-19/nlu/nlu_assign1_codes/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    tf.Session(config=tf.ConfigProto(log_device_placement=True))
    model.fit(X, y, epochs=2, batch_size=20, callbacks=callbacks_list)
    
    # load the network weights
    filename = "/home1/e1-246-19/nlu/nlu_assign1_codes/assign1weights-improvement-19-1.9435.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(1000):
        	x = numpy.reshape(pattern, (1, len(pattern), 1))
        	x = x / float(n_vocab)
        	prediction = model.predict(x, verbose=0)
        	index = numpy.argmax(prediction)
        	result = int_to_char[index]
        	seq_in = [int_to_char[value] for value in pattern]
        	print(result)
        	pattern.append(index)
        	pattern = pattern[1:len(pattern)]
load()  
    
