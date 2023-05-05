from os.path import expanduser, dirname, realpath, join
from math import sqrt, ceil
import torch
from torch.utils.data import Dataset

ALPHABET = "abcdefghijklmnopqrstuvwxyz.,:'?!# -¬"
LITTLE_SHAKESPEAR_PATH = expanduser('~/data/little_shakespear.txt')
file_loc = dirname(realpath(__file__))
LEWIS_CARROL_PATH = join(file_loc,'lewis_carrol.txt')
MIN_TEXT_LEN = 1000000

class LetterLoader(Dataset):
    """Load text data and return batches of input words represented as ints correxponding to their 
    location in the alphabet and target one hot vectors"""

    def __init__(self, token_len, nr_vec_out, nr_as_bytes=True) -> None:
        super().__init__()
        self.batch_size = token_len
        self.nr_vec_out = nr_vec_out
        self.return_nrs = nr_as_bytes
        with open(LEWIS_CARROL_PATH, 'r') as f:
            text = f.read().lower()
        #Make sure the text is long enough to be split into batches
        correct_len = MIN_TEXT_LEN/len(text)
        if correct_len > 1:
            text = text*ceil(correct_len)
        # In the text change new lines to #
        self.txt_data = text.replace('\n', '#').replace('"','\'')

        # self.txt_data = self.gen_toy_data()
        # new target data always starts with a space/ full word
        self.data = self.get_spaces()
        self.alph_size = len(ALPHABET) - 1
        # Alphabet to dict
        self.alphabet_dict = {c: i for i, c in enumerate(ALPHABET)}
        self.nr_of_bytes = ceil(sqrt(self.alph_size))

    def __len__(self):
        return len(self.data)

    def get_spaces(self):
        space_indx = [i for i, char in enumerate(self.txt_data) if char == ' ']
        return space_indx

    def gen_toy_data(self):
        """Generate toy data for quick testing"""
        main_string = 'and then he said: "hello world! '
        main_string = main_string*10000
        return main_string

    def char_to_int(self, char):
        return self.alphabet_dict.get(char, self.alph_size-1)

    def int_arr_from_text(self, text):
        return torch.tensor([self.char_to_int(c) for c in text], dtype=torch.int32)

    def one_hot_from_int(self, int_arr):
        len_ints = len(int_arr)
        arr = torch.zeros((len_ints, self.alph_size))
        for i, c in enumerate(int_arr):
            arr[i][c] = 1

        return arr

    def vec_to_char(self, one_hot_vec):
        indx = list(torch.argmax(one_hot_vec, dim=2).numpy())
        new_vec = self.one_hot_from_int([indx])
        return indx, new_vec.unsqueeze(0)

    def __getitem__(self, index):
        if (index+self.batch_size+2) >= len(self.data):
            index = 0
        index += 1
        stop_start = index+self.batch_size
        letter = self.txt_data[index:stop_start]
        target = self.txt_data[stop_start:stop_start+self.nr_vec_out]

        target_ints = self.int_arr_from_text(target)
        input_byte_arr = self.int_arr_from_text(letter)
        return input_byte_arr, self.one_hot_from_int(target_ints)
