import pandas as pd
import numpy as np

dk_da = open("/home1/e1-246-19/nlu/assign3/nlu_ass3_final/ner.txt",'r').readlines()
dk_da = [i.strip().split(' ') for i in dk_da]
dk_sen = []
dk_le = []
dk_word = set()
dk_tag = set()
for i in dk_da:
    if i == ['']:
        dk_sen.append(dk_le)
        dk_le = []
    else:
        dk_tag.add(i[1])
        dk_word.add(i[0])
        dk_le.append(tuple(i))
        
dk_tag = list(dk_tag)
dk_word = list(dk_word)
dk_word.append("ENDPAD")
dk_tag
n_words = len(dk_word); n_words
n_tags = len(dk_tag); n_tags
len(dk_sen)
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.hist([len(s) for s in dk_sen], bins=50)
plt.show()
max_len = 50
dk_wor2id = {w: i for i, w in enumerate(dk_word)}
dk_ta2id = {t: i for i, t in enumerate(dk_tag)}
dk_wor2id["assessed"]
dk_ta2id["D"]
from keras.preprocessing.sequence import pad_sequences
X = [[dk_wor2id[w[0]] for w in s] for s in dk_sen]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)
X[1]
y = [[dk_ta2id[w[1]] for w in s] for s in dk_sen]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=dk_ta2id["O"])
y[-2]
from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]
y[-2]
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
len(X_te)
from keras.models import dk_mdl, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
input = Input(shape=(max_len,))
dk_mdl = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)
dk_mdl = Dropout(0.1)(dk_mdl)
dk_mdl = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(dk_mdl)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(dk_mdl)  # softmax output layer
dk_mdl = dk_mdl(input, out)
dk_mdl.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
history = dk_mdl.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)
hist = pd.DataFrame(history.history)
plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()
