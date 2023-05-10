A little exploratory dive into NLP using just the decoder part of the transformer architecture.

It is trained using the little shakespear data set found here: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Included is also a Lewis Carrol dataset of his peoms, a smaller alternative training set. That dataset is a lot smaller though. So the results aren't nearly as good.

The model takes in a sequence of letters and tries to predict the next one, which is a on-hot vector with the max entry representing a letter in the ALPHABET. 
I replaced new line /n to # so it could be recognised by the model as just another character in the sequence.

After about half an hour training on a RTX 3090 it gets to an output like this. Where 2 generations of kings have a dialogue:
The output of the model starts after "->>>"

'''
queen margaret:
why, brother, by sing say yet of the chall our me?

king richard iii:
suchad by the night, what a gage!

king richard ii:
have may, best me heads
is never of he bark be had thee crys
is it so seember'd in this let such them,
and way, and thee with shall is thy tow you
they ask yet thou, where what is suffent
'''

It's shakespearan gibrish, but it does keep the structure of <name person speaking>: <what he/she said>

To run it:
Download the shakespear data set, or some other text file you'd like to train on. In the file letter_loader.py, at line 25 change set the variabel to the file that needs to be loaded. This is now LITTLE_SHAKESPEARE_PATH. Or change the path for that variable. And then run main.py. It doesn't take arguments.

Make sure there is a folder called ~/models. Or change where to save the model save path variabel MODEL_SAVE_PATH in main.py to where you want to save the weights.
