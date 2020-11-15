import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from data_loader import NER, parse_from_raw_txt


model = load_model('model.h5')
ner_data = NER(os.path.join('data', '2499_104.txt'), os.path.join('data', 'tags.txt'), 'bert')
ner_data.get_test(model)

score_104 = {}
for key, value in ner_data.tag.items():
    score_104[key] = 0

for sentence in ner_data.test_label:
    for label in sentence:
        score_104[label] += 1

model = load_model('model.h5')
ner_data = NER(os.path.join('data', '2499_108.txt'), os.path.join('data', 'tags.txt'), 'bert')
ner_data.get_test(model)

score_108 = {}
for key, value in ner_data.tag.items():
    score_108[key] = 0

for sentence in ner_data.test_label:
    for label in sentence:
        score_108[label] += 1

print('2499 104: ', score_104)
print('2499 108: ', score_108)
