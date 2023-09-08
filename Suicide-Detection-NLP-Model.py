import tensorflow as tf
import zipfile
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import sys

location = 'suicide_tweets.zip'
zip_ref = zipfile.ZipFile(location, 'r')
zip_ref.extractall()
zip_ref.close()


#! Explore the dataset: 
df = pd.read_csv('suicide_dataset.csv')
print(df.head(10))

index = random.randrange(0, df.shape[0])
print("Tweet is:\n", df.iloc[index]['Tweet'])
print("Sentiment is:\n", df.iloc[index]['Suicide'])

num_tweets = df.shape[0]
print("Total number of tweets in the dataset: ", num_tweets)

not_suicide = df[df['Suicide']=='Not Suicide post']['Suicide'].count()
potential_suicide = df[df['Suicide']=='Potential Suicide post ']['Suicide'].count()
print("Number of not suicide tweets: ", not_suicide)
print("Number of potential suicide tweets: ", potential_suicide)

#? Bar graph:
x = ['not suicide post', 'potential suicide post', ]
y = [not_suicide, potential_suicide]
plt.bar(x, y, color=['g', 'r'])
plt.title('Bar graph of tweets class')

#? Pie chart:
plt.pie(y, labels=x, colors=['b', 'r'])
plt.title("Pie chart of tweets class")
plt.show()

#? one hot encoding:
df['Suicide'] = df['Suicide'].replace({'Not Suicide post': 0, 'Potential Suicide post ': 1})
print("Not Suicide post: ", df[df['Suicide']==0]['Suicide'].count())
print("Potential Suicide post:", df[df['Suicide']==1]['Suicide'].count())

#! Preprocessing text: 
tweets = df['Tweet'].astype('str')
tweets = tweets.tolist()
print("Type is: ", type(tweets))

labels = df['Suicide'].tolist()
print("Type is: ", type(labels))

labels = np.array(labels)
rand_index = random.randrange(0, len(tweets))
print("Tweet is: \n", tweets[rand_index])
if labels[rand_index]==0:
    print("Not Suicide(0)")
else:
    print("Potential Suicide(1)")

#? Spliting the entire dataset:
from sklearn.model_selection import train_test_split
train_tweet, test_tweet, train_label, test_label = train_test_split(tweets, labels, random_state=42, test_size=0.2)

#? PREPROCESSING THE TEXTUAL DATA INTO NUMERICAL FORMAT:
words_count = []
for i in tweets:
    words_count.append(len(i.split()))
max_length = max(words_count) # maximum number of words in a sentence, max length our sequence will be
vocab_size = 1000 # maximum number of unique words in the dataset
pad_type = 'post'
trunc_type = 'pre'
oov_tok = "<OOV>"
embedding_dim = round(np.sqrt(vocab_size))

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_tweet)
word_indexes = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(train_tweet)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                               padding=pad_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test_tweet)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, 
                              padding=pad_type, truncating=trunc_type)

#! Model:
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(training_padded, train_label, epochs=30,
                    validation_data=(testing_padded, test_label), verbose=2)

#! Accuracy Curve:
def accuracy_plot(model):
    accuracy = model.history['accuracy']
    val_accuracy = model.history['val_accuracy']
    epochs = range(len(history.history['accuracy']))
    
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.legend()

#! Loss curve:
def loss_plot(model):
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(len(history.history['loss']))
    
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('loss')
    plt.xlabel('epochs')
    plt.legend()

#! Testing on custom data:
def load(count=50, filler="=", delay=0.02):
  for i in range(count + 1):
    sys.stdout.write('\r')
    sys.stdout.write("[%s%s]" % (filler * i, ' ' * (count - i)))
    sys.stdout.flush()
    time.sleep(delay)

def userdef_tweet():
    custom_tweet = input("Enter a tweet: ")
    print()
    print("Your tweet is: ", custom_tweet)
    sentence = []
    sentence.append(custom_tweet)
    test_s = tokenizer.texts_to_sequences(sentence)
    test_p = pad_sequences(test_s, maxlen=max_length,
                      padding=pad_type, truncating=trunc_type)
    pred_value = model.predict(test_p)
    print("Predict Value is: ", pred_value)
    load(filler=">")
    print()
    if(pred_value>0.5):
        print("Potential Suicide Tweet")
    else:
        print("Not a suicide Tweet")


