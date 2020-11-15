import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM, TimeDistributed
from keras.layers.merge import add
from tensorflow.keras.callbacks import ModelCheckpoint

from data_loader import NER


ner_data = NER(os.path.join('data', 'msra_train_bio'), os.path.join('data', 'tags.txt'), 'bert')
ner_data.get_train(maxlen=128)
print(ner_data.train_X.shape)

model = Sequential()
model.add(Embedding(len(ner_data.vocab), 300))
model.add(Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)))
model.add(TimeDistributed(Dense(len(ner_data.tag), activation="softmax")))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(ner_data.train_X, ner_data.train_Y, verbose=1, batch_size=16, epochs=3, validation_split=0.2, callbacks=[checkpoint])