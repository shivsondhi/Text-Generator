# Text-Generator
Takes an input word or character and generates text either character-by-character or word-by-word. There is one file for each technique (`char-by-char` and `word-by-word`).


## Implementation Details
The two modes in both `textGenerator.py` files are train and generate. The `word-by-word` file also has a retrain mode and the `char-by-char` file has an extra explain mode - see the `Files` section below. This project is meant as a beginner's exercise and impressive results was not the objective. Although I discuss ways to improve performance, there are better alternatives now to the tools used here. 



## Background
The text generation problem took some time to solve and perfect. Unlike images, dealing with text and natural language needs some temporal memory because human languages contain various subtleties. A simple example is the use of pronouns like he, she or it instead of nouns, like somebody's name. These things are just _understood_ by humans but machines need to learn which name can be replaced by "she" and when. This is done using RNNs - a class of neural nets used for many language tasks. Their specialty lies in a mechanism that helps them retain bits of information to use later. The LSTM unit is a kind of RNN, and was one of the most popular tool for natural language processing before transformers emerged. 

In both files I've used one LSTM layer because RNNs take a lot of time to train. Most real-world solutions used multiple, back-to-back LSTM units (or added a dropout layer in between) to hit better accuracy scores.



## Files
### Main files
The two `textGenerator.py` files first train the model and then generating / predict new text. Each file can be in different modes which are changed by changing the `mode` variable. I have used a sequential model with a single LSTM layer (having 256 units in `char-to-char` and 100 units in `word-by-word`. A dropout and a dense layer with softmax activation are added after the LSTM unit.

#### Modes: `char-by-char`
**exp** - The file neither trains the model nor generates text. This mode captures the journey of the data as it is processed, by printing to the console. Visualising changes in the data is a helpful way to understand the transformations in the preprocess step. 

**train** - The file trains the model on the input string provided by the user. Training includes a callback each time the model's loss decreases at the end of an epoch. The best weights (that produce the best loss) are iteratively saved to .hdf5 files. 

**generate** - The file generates text from a random seed of n characters. It takes a .hdf5 file, loads weights from it, compiles the model and finally makes predictions. The default seed is 100 characters long.

#### Modes: `word-by-word` 
**none** - Same as the `exp` mode above.

**train** - Same as the `train` mode above.  

**retrain** - The file continues training the model from a specific checkpoint. The weights are loaded from file and after compilation the model will continue training from here. Some things to note - the epoch numbers restart from 1 when you retrain the model. Keras provides a `start_from` argument in its `fit` and `fit_generator` methods, but it wouldn't work without throwing an error so I worked around this manually. As a consequence, new weight files may overwrite older ones if they have the same name, so it is best to rename weight files after retraining. 

**generate** - The file generates the next word in a headline. You can pass a list of seed-words to generate more than one headline at once. 

### Weight files
The `weightsFiles` folder contains some .hdf5 weight files from my runs. Loading these in 'generate' mode loads the weights, and the model is ready to generate text. 



## Datasets
The datasets are taken from Kaggle. [Here](https://www.kaggle.com/sangarshanan/medium-articles-tagged-in-mldlai) is the dataset used in `char-by-char`. The dataset consists of Medium articles scraped from the web and I have used just one of these to train my model. [Here](https://www.kaggle.com/therohk/million-headlines) is the dataset used in `word-by-word`. This consists of a million news headlines from ABC News over a period of 15 years. I used around 5000 to train my model. You may need to sign-up with Kaggle to download datasets.



## Result
Both techniques have their advantages and disadvantages - the results of `char-by-char` are better in the sense that repitition of characters completely disappears after a point. In `word-by-word`, repition is a problem even when the loss is low. On the flip side, `word-by-word` is guaranteed to make at least some sense (since we're dealing directly with words); `char-by-char` doesn't make much sense even at peak performance. The loss and numbmer of epochs for each trial is included in the header of both files. 

So, both algorithms produce a lot of repeting sequences of words or characters. Some obvious ways to improve results would be to add more LSTM layers and increase the training data. Google collab is a friendly dev environment that offers free compute resources (CPU and GPU). 
