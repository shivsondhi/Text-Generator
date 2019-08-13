# Text-Generator
Takes an input word or character and generates text either character-by-character or word-by-word. There are two different files for each technique (char-by-char and word-by-word).

# Implementation Details
The code is implemented using keras and tensorflow in python 3. 
The two main modes in both textGenerator.py files are train and generate. The word-by-word file has an extra mode called retrain and the char-by-char file has an extra mode called exp. These modes are explained in the Files section. 

# Background
The text generation problem is one that took a little while to solve and perfect. Unlike images, dealing with textual data and natural language needs some sort of temporal memory because human languages contain several instances of implication, references and other subtleties. A simple example of this is the use of pronouns like he, she and it, which replace names of people and things in common speech. There are no rules for this, most of these things are just _understood_.

For machines however, this is no trivial task. This problem is solved by RNNs, which are a class of Neural Networks used for most language tasks (as opposed to CNNs for image tasks). The specialty of RNNs lies in the fact that they have a mechanism that helps them retain certain bits of information for later use. The LSTM unit is a kind of RNN and is what is most widely used in textual tasks today. In the two algorithms I use only a single LSTM layer, although in most problems it is very common to use two LSTM units consecutively or with a dropout layer in between. I have used only one unit because RNNs in general take a lot of time to train to reflect acceptable results. 

# Files
## Main files
The two textGenerator.py files contain the training code as well as the text generating (predicting) code. The mode in each file can be changed by changing the 'mode' variable in the respective files. The model I have used for text generation is sequential with a single LSTM layer having 256 units in the char-to-char file and 100 units in the word-by-word file. A Dropout layer and a Dense layer with softmax activation are added after the LSTM unit.

### The char-by-char file has 3 modes: exp, train and generate.
**exp** - If the mode variable is set to this, the file will neither train nor generate any text. In this mode, the different steps of data processing are printed to the console at every step. This helped me understand the process better as it helps to visualise the data and the way it is being transformed along the way. 

**train** - Selecting this as the mode will start training the model on whatever input string is provided at the top of the file. The training process consists of a callback each time the model's loss improves (decreases) at the end of an epoch. Therefore the weights that produce the best loss are iteratively saved to .hdf5 files. 

**generate** - This mode takes the specified .hdf5 file and loads the weights from there. Once loaded and compiled, the model is used to make predictions given a random seed of 'n' characters (n set to 100 by default).

### The word-by-word file has 4 modes: train, retrain, generate, none.
**none** - In the none mode, the model does nothing. It just prints the data preprocessing in steps and prints the model summary. Similar to the "exp" mode in the char-by-char file.

**train** - The train mode is more or less the same as in the char-by-char file.  

**retrain** - The retrain mode, loads the specified weights file to the model and after compilation, it continues training from where it left off. The epoch numbers restart from 1 everytime you retrain the model. Beware that the new files do not overwrite older ones and once the retraining is done, you can always rename your files. Keras does provide a start_from argument in its fit / fit_generator methods, but I could not get it to work without throwing an error so I just decided to work around this manually. 

**generate** - This mode loads the specified weights file and predicts the next word in the headline. You can pass a list of seed words to the generate block to generate several headlines at once. 

## Weight files
These are the .hdf5 files for the best trials of the program that I ran on my computer. Loading these files in 'generate' mode will load the weights to the model which can directly start prediction without having to train. The weight files are found in the weightsFiles folders, one for each .py file. 

# Datasets
The datasets are taken from Kaggle. [Here](https://www.kaggle.com/sangarshanan/medium-articles-tagged-in-mldlai) is a link to the dataset used in the character by character text generator. The dataset consists of Medium articles scraped from the web. I have used just one of these articles to train my model. [Here](https://www.kaggle.com/therohk/million-headlines) is a link to the dataset used in the word by word text generator. This dataset consists of a million news headlines from ABC News over a period of 15 years. I have used only a few headlines (~5000) to train my model. You will need to sign-up to download the datasets.

# Results
The loss and numbmer of epochs for each trial is included in the headers of both of the files. Both of the techniques, have their advantages and disadvantages. The results of the char-by-char algorithm are slightly better than the word-by-word algorithm in the sense that after a point of training the repitition of certain characters completely disappears. In the word-by-word model, repition is a problem even with relatively low loss. On the flip side, the word-by-word generator is guaranteed to make at least some sense, since we are dealing directly with words. The char-by-char model does not really make much sense even at its lowest loss. 

Having said that, both algorithms do produce a lot of repeting sequences of words or characters. A good fix for both algorithms to improve results might be to add a second LSTM layer. I ran the char-by-char model for a maximum of 100 epochs and the word-by-word for a maximum of 200 epochs. I have a NVIDIA GeForce 940 MX GPU.
