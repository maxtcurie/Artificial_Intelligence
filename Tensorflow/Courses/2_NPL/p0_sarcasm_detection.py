import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

if not os.path.exists('mkdir.py'):
    import wget #pip install wget
    print('Beginning file download with wget module')
    
    url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    wget.download(url, './1cats_and_dogs_filtered.zip')

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
print('*********sequences**********')
print(sequences)
print('*********padded**********')
#padding adds '0' at the front so that all array has the same length
padded=pad_sequences(sequences)
print(padded)

print('*********padded2**********')
#padding adds '0' at the end so that all array has the same length
padded=pad_sequences(sequences,padding='post')
print(padded)


print('*********padded3**********')
#padding adds '0' at the end so that all array has the same length
#anything longer at will be deleted, front by default
padded=pad_sequences(sequences,padding='post',
                        truncating='post',#now delete at the end
                        maxlen=5)
print(padded)