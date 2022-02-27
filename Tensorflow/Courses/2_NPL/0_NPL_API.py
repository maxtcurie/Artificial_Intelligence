import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer

sentences=[
          'I love my dog',
          'I love my cat',
          'You love my dog!',
          'I love cat-dog!'
          ]
#set the number of the words as 100
tokenizer=Tokenizer(num_words=100)

#fit the text using tokenizer
tokenizer.fit_on_texts(sentences)

#the word will be numbered as dictionary(not case sensitive) 
word_index=tokenizer.word_index

print(word_index)