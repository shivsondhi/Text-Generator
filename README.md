# Text-Generator
# Takes input and generates text character by character

## Dataset
The dataset is taken from Kaggle. [Here](https://www.kaggle.com/sangarshanan/medium-articles-tagged-in-mldlai) is a link to the dataset, which can be downloaded there itself. The dataset basically consists of Medium articles scraped from the web. I have used one such article to train my data on. 

## Files
### Main file 
This is the text_generator.py file. This contains the training code as well as the text generating (predicting) code. The file has 3 modes: exp, train and generate; determined by the variable 'mode'. Below I explain the difference between each of these.

exp - If the mode variable is set to this, the file will neither train nor generate any text. In this mode, the different steps of data processing are shown by printing the data at every step of its preprocessing. This helped me to understand the process better as it helps to visualise the data and the way it is being transformed along the way. 

train - Selecting this as the mode will start training of the model on whatever input is provided at the top of the file. The training process consists of callbacks each time the model's loss improves (decreases) at the end of an epoch. Therefore the weights that produce the best loss are iteratively saved as .hdf5 files. 

generate - This mode takes the specified .hdf5 file and loads the weights from there. Once loaded and compiled, the model is used to make predictions given a random seed of 'n' characters (n set to 100 by default).

### Weight files
These are the best .hdf5 files for every trial of the program that I ran on my computer. Loading these files in 'generate' mode will load the weights to the model which can directly start prediction without having to train. I might update these and provide more when I work on this further.

## TRIAL Runs
Below is a description of each of the trials I have run with this model which shows the effect of training data size, number of epochs and window-length to the loss. Training data size is determined by total chars, number of epochs by epoch_num and window size by seq_len. All of these can be found in text_generator.py and changed. Total chars must be changed by changing the training data. The trial numbers correspond to the numbers on the weight-files.

TRIAL ONE:  total chars = 818  &&  epoch_num = 20  &&  seq_len = 40 -
					  best LOSS = 2.9146
					  results in only blank spaces - no words generated, no chars generated.
            
TRIAL TWO:  total chars = 1535  &&  epoch_num = 50  &&  seq_len = 40 -
					  best LOSS = 2.4977
					  results in lots of chars separated by spaces. Most 'words' are two letters, some are three and very few are four.
            
TRIAL THREE: 	total chars = 818  &&  epoch_num = 50  &&  seq_len = 40 -
					    best LOSS = 2.7519
					    results in some chars. However it is just a sequence of "oo t" repeated over and over till the end.
              
TRIAL FOUR: 	total chars = 1535  &&  epoch_num = 20  &&  seq_len = 40 -
					    best LOSS = 2.8652
					    results in only blank spaces - no words generated, no chars generated.
              
TRIAL FIVE: 	total chars = 11932  &&  epoch_num = 20  &&  seq_len = 40 -
					    best LOSS = 2.6772
					    results in repeated sequence of "ah thet aod thet ao" which actually started with "th tee".
              
TRIAL SIX: 		total chars = 11932  &&  epoch_num = 50  &&  seq_len = 40 - 
    					best LOSS = 0.9772!!!
		    			obviously results here are the best. Words of all sizes and barely any repitition. There are a few English words but most words are just gibberish or close attempts at words. Upto 11 letter words!
              
TRIAL SEVEN: 	total chars = 11932  &&  epoch_num = 20  &&  seq_len = 16 - 
  		  			best LOSS = 2.6503
	    				results in "aod toet aod toet aod toet " repeatedly.
              
TRIAL EIGHT: 	total chars = 11932  &&  epoch_num = 20  &&  seq_len = 100 -
					    best LOSS = 2.6431
					    epochs take longer to run, no definite repitition of words, but there is a lot of repitition of characters. All words are 2 or 3 letters long with a few 4 letter words here and there.
