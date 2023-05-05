A little exploratory dive into NLP using just the decoder part of the transformer architecture.

It is trained using the little shakespear data set found here: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Included is also a Lewis Carrol dataset of his peoms, a smaller alternative training set.

The model takes in a sequence of letters and tries to predict the next one, which is a on-hot vector with the max entry representing a letter in the ALPHABET. 
I replaced new line /n to # so it could be recognised by the model as just another character in the sequence.

After about an hour training on a RTX 3090 it gets to an output like this:
The output of the model starts after "->>>"

'''
that the very hour#you take it off again?##sicinius:->>>#the call the cannot a canpressities and the wars#and the warst the stand the wars the wars#the phe states a can the should the one the should#the one the one the wars and the one the one of the othere#the one the one the other the one the other-#the people or the people the people out the othere#the people the bear the state,#and the stand the shall be sount of the one the othere#the strong and the wars the one the one of the other#the people 
'''

It's shakespearan gibrish and it repeats a few words quite a bit, e.g. the people/ wars.
It does keep the structure of <name person speaking>: <what he/she said>

