'''
TextGenerator - character by character. 

Notes:- 
	The two text samples (small and large) provided right at the bottom, correspond to the TRIALs having 818 characters and 1535 characters respectively. 
	All other TRIALs are run on the first article in the MediumArticles.csv dataset.

	To change total_chars you can change the article selected from the dataset or use any other text data and feed it to the variable 'raw_text'.

TRIAL ONE:  	total chars = 818  &&  epoch_num = 20  &&  seq_len = 40 -
.						best LOSS = 2.9146
.						results in only blank spaces - no words generated, no chars generated.
            
TRIAL TWO:  	total chars = 1535  &&  epoch_num = 50  &&  seq_len = 40 -
.					  	best LOSS = 2.4977
.					  	results in lots of chars separated by spaces. Most 'words' are two letters, some are three and very few are four.
            
TRIAL THREE: 	total chars = 818  &&  epoch_num = 50  &&  seq_len = 40 -
.					    best LOSS = 2.7519
.					    results in some chars. However it is just a sequence of "oo t" repeated over and over till the end.
              
TRIAL FOUR: 	total chars = 1535  &&  epoch_num = 20  &&  seq_len = 40 -
.					    best LOSS = 2.8652
.					    results in only blank spaces - no words generated, no chars generated.
              
TRIAL FIVE: 	total chars = 11932  &&  epoch_num = 20  &&  seq_len = 40 -
.					    best LOSS = 2.6772
.					    results in repeated sequence of "ah thet aod thet ao" which actually started with "th tee".
              
TRIAL SIX: 		total chars = 11932  &&  epoch_num = 50  &&  seq_len = 40 - 
.   					best LOSS = 0.9772!!!
.		    			obviously results here are the best. Words of all sizes and barely any repitition. There are a few English words but most words are just gibberish or close attempts at words. Upto 11 letter words!
              
TRIAL SEVEN: 	total chars = 11932  &&  epoch_num = 20  &&  seq_len = 16 - 
. 		  				best LOSS = 2.6503
.	    				results in "aod toet aod toet aod toet " repeatedly.

TRIAL EIGHT: 	total chars = 11932  &&  epoch_num = 20  &&  seq_len = 100 -
.					    best LOSS = 2.6431
.					    epochs take longer to run, no definite repitition of words, but there is a lot of repitition of characters. All words are 2 or 3 letters long with a few 4 letter words here and there.
'''

import sys
import numpy as np
import pandas as pd
import string
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def main():
	global model
	modes = ['exp', 'train', 'generate']												#mode is either 'exp', 'train' or 'generate'
	mode = modes[0]
	
	#Variable parameters
	#num = the number of the article from the dataset ~ Size of the learning data | epoch_num = number of times the algorithm trains over the learning data | seq_len = window size of characters.
	num = 1
	epoch_num = 20
	seq_len = 100

	
	#MAIN CODE
	#load file data
	path = "MediumArticles.csv"
	df = pd.read_csv(path)
	if mode == 'exp':
		print("\nThe Journey of the Data!\n\n1. Data loaded from dataset-\n", df.head(3))
	
	#Extract the article from the df, remove paragraphing and punctuation
	data = get_text(num, df, mode)
	raw_text = prepare(data)
	if mode == 'exp':
		print("\n4. Final Cleaned Data-\n", raw_text[:100])

	#create a mapping from char to int and reverse
	chars = sorted(list(set(raw_text)))														#Set creates a set of unique chars and sorted sorts the list in ascending order
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	#Summarize the data
	n_chars = len(raw_text)																	#No. of characters in the text
	n_vocab = len(chars)																	#No. of unique characters in the text
	print("\nTotal chars = ", n_chars)
	print("Total vocab = ", n_vocab)
	#prepare input and output pair sequences
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_len):
		seq_in = raw_text[i:i + seq_len]
		seq_out = raw_text[i + seq_len]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append([char_to_int[char] for char in seq_out])
	n_patterns = len(dataX)
	print("Total patterns = ", n_patterns)
	
	if mode == 'exp':
		print("Last input sequence is - ", seq_in, end="\n\n")
		print("dataX (input mapped to ints). The integer mapped values of the 100 (seq_len) characters in the first 22 input sequences:\n", dataX[:22], end="\n\n")
		print("dataY (output mapped to ints). Integer mapped values of the expected output to each of the 22 inpupt sequences:\n", dataY[:22], end="\n\n")

	#Modify the data
	#Reshape
	X = np.reshape(dataX, (n_patterns, seq_len, 1))
	#Normalize
	X = X / float(n_vocab)
	#One hot encode the output
	y = np_utils.to_categorical(dataY)

	if mode == 'exp':
		print("After dividing dataX[[]] by n_vocab(44):\n", X[:22], end="\n\n")
		print("X shape is: ", X.shape, end="\n\n")

	#Create Model - 
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.1))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	print(model.summary())

	if mode is 'train': 
		#Define checkpoint to create a weight-file after every epoch
		filepath = "weights-improvement-9-{epoch:02d}-{loss:.4f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]
		#fit the model with callbacks to the checkpoint
		model.fit(X, y, epochs=epoch_num, batch_size=64, callbacks=callbacks_list)
		print("\n\t\t~Fin~\n")
	elif mode is 'generate':
		best_file = "weights-improvement-9-20-2.6431.hdf5"
		model.load_weights(best_file)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		#Pick random sequence from the input-sequences as a seed value to act as input for the prediction
		start = np.random.randint(0, len(dataX)-1)
		pattern = dataX[start]
		print("Seed: ", end="")
		#Input-sequences are stored as ints - convert them to chars
		print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
		#generate the characters, arbitrarily chosen 900 characters to generate, can be any number
		for i in range(900):
			#preprocess steps as done before training as well - reshape and divide by len(vocab)
			x = np.reshape(pattern,(1, len(pattern), 1))
			x = x / float(n_vocab)
			prediction = model.predict(x, verbose=0)
			index = np.argmax(prediction)													#Finds the value with 1 in one hot encoding
			result = int_to_char[index]
			seq_in = [int_to_char[value] for value in pattern]
			sys.stdout.write(result)														#print(result, end="")
			pattern.append(index)															#Add generated character to the input sequence
			pattern = pattern[1:len(pattern)]												#Maintain constant length of input sequence
		print("\n\t\t~Fin~\n")
	elif mode is 'exp':
		print("\n\t\t~Fin~\n")
	else:
		print("[ModeError] Select mode from: [generate / train / exp], all case-sensitive.")
		print("\n\t\t~Fin~\n")

def prepare(txt):
	#Change to lowercase and remove punctuations.
	txt = txt.lower()
	punc = string.punctuation.translate({ord(c): None for c in "!.?"})						#Remove terminating symbols and create a string of all other punctuations
	txt = txt.translate({ord(c): None for c in punc})										#Remove these punctuations from the text
	return txt 																				#(ord() returns the UTF code for the character)

def get_text(num, df, mode):
	#num is the number of the article to extract from the dataset. Example done on the first article.
	for i in range(num - 1, num):
		data = df.text[i]
		if mode == 'exp':
			print("\n2. Raw data to train on-\n", data[:100], "...")
		data = " ".join([line.strip() for line in data.split("\n")])						#replace newline characters with a space ("\n"  ->  " ")
		if mode == 'exp':
			print("\n3. Raw data after removing newline characters-\n", data[:100], "...")
	return data

if __name__ == "__main__":
	main()

'''
Text Samples - 

	small_text = ("My name is Samuel L. Jameson and I will show you how to start blogging. I have been building blogs and websites since 2002. "
	"Since then I have launched several of my own blogs, and helped hundreds of others do the same. I know that starting a blog can seem overwhelming and intimidating to many. "
	"For that, I have created this guide on blogging for beginners, which will teach you guys how to be blog with just the most basic computer skills. "
	"So whether you’re yound or old, you can create your own blog in less than 30 minutes. "
	"I am not ashamed to admit that when I first started building my own blog, I made a ton of mistakes. You will too, and that is all part of the process! "
	"Additionally, you can benefit from more than a decade of my experience so that you don’t repeat these same mistakes when you make your blog. "
	"I created this blog guide so that anyone can learn how to blog quickly and fairly easily. ")
	
	large_text = ("My name is Samuel L. Jameson and I will show you how to start blogging. I have been building blogs and websites since 2002. "
	"Since then I have launched several of my own blogs, and helped hundreds of others do the same. I know that starting a blog can seem overwhelming and intimidating to many. "
	"For that, I have created this guide on blogging for beginners, which will teach you guys how to be blog with just the most basic computer skills. "
	"So whether you’re yound or old, you can create your own blog in less than 30 minutes. "
	"I am not ashamed to admit that when I first started building my own blog, I made a ton of mistakes. You will too, and that is all part of the process! "
	"Additionally, you can benefit from more than a decade of my experience so that you don’t repeat these same mistakes when you make your blog. "
	"I created this blog guide so that anyone can learn how to blog quickly and fairly easily. "
	"Blogging is an art that must be learned. It is not necessary to have a lot of coding knowledge to be able to start your own blog. There are several online "
	"websites today like Word Press and Wix that handle all the computer related work for you so all you need to focus on is your writing. An important thing to "
	"keep in mind is that before starting your blog, you must keep a number of articles ready locally so that you always have something to upload to your blog. "
	"At any moment you may experience a dry run and your productivity may dip. This happens to the best of us and is nothing to worry about. But for situations such "
	"as this, we must be prepared and ready with a buffer of articles to post online nevertheless.")

'''