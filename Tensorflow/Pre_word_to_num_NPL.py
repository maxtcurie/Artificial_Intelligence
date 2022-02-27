import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

sentences=[
          'I love my dog',
          'I love my cat',
          'You love my dog!',
          'I love cat-dog!'
          ]
#set the number of the words as 100
#OOV is out of vercabulary
tokenizer=Tokenizer(num_words=100,oov_token="<OOV>")

#fit the text using tokenizer
tokenizer.fit_on_texts(sentences)

sentences=np.concatenate((sentences,['I really love my dog and cat']))

#the word will be numbered as dictionary(not case sensitive) 
word_index=tokenizer.word_index
print(word_index)
#Word index starts from 0 !!!!!!

#map the sentences to sequence
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

print('**********************')

for i in range(len(sentences)):
    print(sentences[i])
    print('---->')
    print(sequences[i])
    print('**********************')
