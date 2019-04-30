'''
Text Generator - Generates text word by word. 

TRIAL ONE : 	Epochs = 100; 	Headlines = 1100;	Time = 607.2s
				.	Loss - 1.0940
				.	Results - "govt to work case before to" repeated everywhere.
				+50 epochs:							Time = ?
				.	Loss - 0.5324
				.	Results - "govt canvassing ways to improve child development rate" repeats everywhere
				+75 epochs:							Time = ?
				.	Loss - 0.2862
				.	Results - "govt paper discusses preservation of adelaide..."
				Final Takeaway - Any seed word that the machine does not know, it says the same "govt.." thing.
				.				Some words give good headlines, others just start with the govt rant.

TRIAL TWO : 	Epochs = 100; 	Headlines = 5000; 	Time = 3176.3s
				.	Loss - 1.26420
				.	Results - Certainly made a difference, there's barely any repetition of phrases.
				+100 epochs:						Time = 3088.5s
				.	Loss - 0.8083
				.	Results - Found the repeating sentence! - "to offer on broken away from asylum city attack..." Might need to train more.
'''

from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
import keras.utils as ku

from tensorflow import set_random_seed
from numpy.random import seed 
set_random_seed(2)
seed(1)
import pandas as pd 
import numpy as np
import string, os, random

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
	modes = ['train', 'generate', 'retrain', 'none']
	mode = modes[3]
	num_epochs = 100
	print("You are now in mode: {0}".format(mode))
	
	#load dataset of million ABC news headlines
	path = "abcnews-millionheadlines.csv"
	all_headlines = []
	headlines_df = pd.read_csv(path)
	all_headlines = list(headlines_df.headline_text.values)[:5000]
	print("Number of headlines: {0}".format(len(all_headlines)))

	#clean the text of each headline and print a sample
	corpus = [clean_text(x) for x in all_headlines]
	print("\nCorpus: ")
	for x in corpus[:11]:
		print(x)

	#create a tokenizer to get every word into the dictionary of words (vocabulary)
	tokenizer = Tokenizer()
	#split each headline into input sequences and print a sample
	input_sequences, total_words = get_sequence_of_tokens(corpus, tokenizer)
	print("\ninput_sequences: ")
	for x in input_sequences[:11]:
		print(x)
	print("\nTotal words: {0}".format(total_words))

	#print samples of the input and output
	predictors, label, max_sequence_len = get_padded_sequences(input_sequences, total_words)
	print("\npadded sequences (input):") 
	for x in predictors[:11]:
		print(x)
	print("\nlabels (output):")
	for x in label[:11]:
		print(x)
	print("\nmax_sequence_len: {0}".format(max_sequence_len))

	#create the model and print summary
	print("\nModel Summary:") 
	model = create_model(max_sequence_len, total_words)
	print(model.summary())

	if mode == 'train':
		#TRAIN
		savepath = "second_weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
		checkpoint = ModelCheckpoint(savepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
		callbacks_list = [checkpoint]
		model.fit(predictors, label, epochs=num_epochs, verbose=5, callbacks=callbacks_list)
		print("\n\t\t~Fin~\n")
	elif mode == 'generate':
		#GENERATE
		best_file = "second_weights-improvement-100-0.8083.hdf5"
		model.load_weights(best_file)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		seed_texts = ['House', 'Houses', 'Prime', 'Mass', 'Britain', 'Brexit', 'national', 'govt', 'advertisers']
		i = 1
		for seed_text in seed_texts:
			print("Seed {0}".format(i))
			next_words = random.randint(6, max_sequence_len)
			generated_headline = generate_text(tokenizer, seed_text, next_words, model, max_sequence_len)
			print(generated_headline, end="\n\n")
			i += 1
			print("\n\t\t~Fin~\n")
	elif mode == 'retrain':
		#RETRAIN
		best_file = "weightsFile_for_words\\second_weights-improvement-100-1.2642.hdf5"
		model.load_weights(best_file)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		savepath = "second_weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
		checkpoint = ModelCheckpoint(savepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
		callbacks_list = [checkpoint]
		model.fit(predictors, label, epochs=num_epochs, verbose=5, callbacks=callbacks_list)
		print("\n\t\t~Fin~\n")
	else:
		print("\n\t\t~Fin~\n")

def clean_text(txt):
	#Remove all punctuation and convert to lower case
	txt = "".join(v for v in txt if v not in string.punctuation).lower()
	txt = txt.encode("utf8").decode("ascii", "ignore")
	return txt 

def get_sequence_of_tokens(corpus, tokenizer):
	#create a dictionary of every word corresponding to a unique number. By default keras.tokenizer class also creates 3 other objects that it may use.
	tokenizer.fit_on_texts(corpus)
	total_words = len(tokenizer.word_index) + 1				#word_index is the dictionary ^
	#map each word to an integer value and then create the input_sequences
	input_sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			input_sequences.append(n_gram_sequence)
	return input_sequences, total_words

def get_padded_sequences(input_sequences, total_words):
	#pad every input sequence so that we have uniform length inputs.
	max_sequence_len = max([len(x) for x in input_sequences])
	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
	#split the sequences taking the first n-1 columns as input and the last column as the label / output
	predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
	label = ku.to_categorical(label, num_classes=total_words)
	return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
	#Create a sequential model with one LSTM unit 
	input_len = max_sequence_len - 1
	model = Sequential()
	model.add(Embedding(total_words, 10, input_length=input_len))
	model.add(LSTM(100))
	model.add(Dropout(0.1))
	model.add(Dense(total_words, activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	return model

def generate_text(tokenizer, seed_text, next_words, model, max_sequence_len):
	#predict the next word for the desired number of times. model.predict will output an integer. 
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)
		#map the integer output to the word in the tokenizer dictionary. Append the word to seed_text and continue.
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text

if __name__ == "__main__":
	main()