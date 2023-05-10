from os.path import expanduser, dirname, realpath, join
from math import sqrt, ceil
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import random

ALPHABET = "abcdefghijklmnopqrstuvwxyz .,:'?!#-¬"
LITTLE_SHAKESPEARE_PATH = expanduser('~/data/little_shakespeare.txt')
file_loc = dirname(realpath(__file__))
LEWIS_CARROL_PATH = join(file_loc, 'lewis_carrol.txt')
MIN_TEXT_LEN = 1000000
TRAIN_TEST_SPLIT = 0.9


class LetterLoader(Dataset):
    """Load text data and return batches of input words represented as ints correxponding to their 
    location in the alphabet and target one hot vectors"""
    balance_prop = 0.15

    def __init__(self, token_len, nr_vec_out, set_type='train') -> None:
        super().__init__()
        self.context_len = token_len
        self.nr_vec_out = nr_vec_out
        with open(LITTLE_SHAKESPEARE_PATH, 'r') as f:
            text = f.read().lower()
        # Make sure the text is long enough to be split into batches
        correct_len = MIN_TEXT_LEN/len(text)
        if correct_len > 1:
            text = text*ceil(correct_len)

        # In the text change new lines to #
        text = text.replace('\n', '#')
        text = text.replace('"', '\'')
        if set_type == 'train':
            self.data = text[:int(len(text)*TRAIN_TEST_SPLIT)]
        else:
            self.data = text[int(len(text)*TRAIN_TEST_SPLIT):]

        self.is_train = set_type == 'train'

        self.get_all_locations_per_char()
        # self.data = self.gen_toy_data()
        # new target data always starts with a space/ full word

        self.alph_size = len(ALPHABET) - 1
        # Alphabet to dict
        self.alphabet_dict = {c: i for i, c in enumerate(ALPHABET)}
        self.nr_of_bytes = ceil(sqrt(self.alph_size))
        self.sample_chars = ALPHABET[:-1]
        # self.sample_weights = [100.0]*len(self.sample_chars)
        # self.def_sample_weights = {
        #     c: len(it) for c, it in self.target_char_dict.items()}
        # weights = np.array(
        #     list(self.def_sample_weights.values()), dtype=np.float32)
        # self.sample_weights = (100*weights)/weights.max()
        print('resetting data')
        self.sampled_chars = {c: 0 for i, c in enumerate(ALPHABET)}

    def __len__(self):
        return len(self.data)

    def get_all_locations_per_char(self):
        """Get all locations of each character in the alphabet, 
        so we can sample from it later and balance the train data"""
        self.target_char_dict = dict({})
        self.nr_chars = dict({})
        for alph_char in ALPHABET[:-1]:
            self.target_char_dict[alph_char] = [
                i for i, c in enumerate(self.data) if c == alph_char]
            self.nr_chars[alph_char] = len(self.target_char_dict[alph_char])

    def get_spaces(self):
        space_indx = [i for i, char in enumerate(self.data) if char == ' ']
        return space_indx

    def gen_toy_data(self):
        """Generate toy data for quick testing. loss should drop to less then 0.2 in 1 or 2 epochs"""
        main_string = 'and then he said: "hello world! '
        main_string = main_string*10000
        self.balance_prop = 0
        return main_string

    def char_to_int(self, char):
        return self.alphabet_dict.get(char, self.alph_size-1)

    def int_arr_from_text(self, text):
        return torch.tensor([self.char_to_int(c) for c in text], dtype=torch.int32)

    def one_hot_from_int(self, int_arr):
        len_ints = len(int_arr)
        arr = torch.zeros((len_ints, self.alph_size))
        for i, c in enumerate(int_arr):
            arr[i][int(c)] = 1

        return arr

    def vec_to_char(self, one_hot_vec):
        indx = list(torch.argmax(one_hot_vec, dim=2).numpy())
        new_vec = self.one_hot_from_int(indx)
        return indx, new_vec.unsqueeze(0)

    def __getitem__(self, index):

        index += 1
        if self.is_train and torch.rand(1) < self.balance_prop:
            # Pick a random target charachter
            rand_char = random.sample(ALPHABET[:-1], k=1)[0]
            # Pick a random index from the list of all locations of that charachter
            index = random.sample(self.target_char_dict[rand_char][10:], 1)[0]
            # Count back from where to start the context
            index = index-self.context_len-self.nr_vec_out+1
            self.sampled_chars[rand_char] += 1
        if (index+self.context_len+2) >= len(self.data):
            index = 0
        stop_start = index+self.context_len
        letter = self.data[index:stop_start]
        target = self.data[stop_start:stop_start+self.nr_vec_out]

        target_ints = self.int_arr_from_text(target)
        input_byte_arr = self.int_arr_from_text(letter)
        return input_byte_arr, self.one_hot_from_int(target_ints)
