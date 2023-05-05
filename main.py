from os.path import expanduser
from torch.utils.data import DataLoader
from letter_loader import LetterLoader, ALPHABET
from llm import LLM
from torch.optim import Adam
from torch import nn, concatenate
import torch

from torch.utils.tensorboard import SummaryWriter

MODEL_SAVE_PATH = expanduser('~/models/llm.pt')


class Main:
    def __init__(self):
        self.token_len = 128
        self.out_vec_len = len(ALPHABET)-1
        self.token_depth = 52
        nr_heads = 8
        nr_vec_out = 1
        print('token_len', self.token_len)
        self.char_as_int = True

        self.letter_loader = LetterLoader(
            self.token_depth, nr_vec_out, self.char_as_int)

        self.llm = LLM(self.token_len, self.token_depth, nr_heads, self.out_vec_len, nr_vec_out).cuda()
        self.llm.load_model(MODEL_SAVE_PATH)
        self.optimizer = Adam(self.llm.parameters(), lr=0.000031, weight_decay=0.0001)
        self.loss = nn.CrossEntropyLoss()
        self.bin_loss = nn.BCELoss()
        self.tensorboard = SummaryWriter(
            expanduser('~/code/transformer/tensorboard'))
        # Count the number of parameters in the model
        self.count_parameters()

    def count_parameters(self):
        nr_params = 0
        for param in self.llm.parameters():
            nr_params += param.numel()
        print(f'Number of parameters: {nr_params}')


    def train_epoch(self):
        b_size = 100
        train_loader = DataLoader(
            self.letter_loader, batch_size=b_size, shuffle=True, num_workers=4)
        test_data = LetterLoader(self.token_depth, self.char_as_int)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

        avg_loss = 0
        avg_acc = 0
        steps_till_test = 200
        for batch_idx, (data, target) in enumerate(train_loader):

            input = data
            self.llm.train()

            self.optimizer.zero_grad()
            out = self.llm(input)
            target = target.cuda()

            loss = self.loss(out.reshape(-1, self.out_vec_len),
                             target.cuda().reshape(-1, self.out_vec_len))
            loss.backward()
            pred_char = out.argmax(dim=2)
            target_char = target.argmax(dim=2)
            accuracy = (pred_char == target_char).sum().item() / \
                (b_size * target.shape[1])
            avg_acc += accuracy
            avg_loss += loss.item()
            self.optimizer.step()
            if batch_idx % steps_till_test == 0:
                print('average loss:', avg_loss / steps_till_test)
                avg_loss = 0
                print('average accuracy: ', avg_acc/steps_till_test)

                self.llm.save_model(MODEL_SAVE_PATH)
                self.test_model(test_loader, avg_acc/steps_till_test)
                avg_acc = 0

            self.tensorboard.add_scalar('loss', avg_loss, batch_idx)

    def test_model(self, test_loader, avg_acc):
        """Test the model by starting to give a sample input 
            and then let the model predict the rest of the text"""
        data, target = test_loader.__iter__().__next__()

        results = []
        for indx in data[0]:

            results.append(ALPHABET[indx])

        results += list('->>>')
        input_enc = data
        self.llm.eval()
        test_len = int((40*avg_acc)**2)
        test_len = max(test_len, 10)
        print('test_len', test_len)
        for i in range(test_len):

            data_out_m = self.llm(input_enc).detach().cpu()

            indx, _ = self.letter_loader.vec_to_char(data_out_m)
            indx_len = len(indx[0])
            if indx == len(ALPHABET):
                print(indx)
            indx = indx

            for i in indx[0]:
                results.append(ALPHABET[i])
            input = concatenate((input_enc, torch.tensor(indx)), dim=1)
            input_enc = input[:, indx_len:]

        print('example result: ', ''.join(results))

    def run(self):
        for i in range(1000):
            print('running epoch: ', i)
            self.train_epoch()


if __name__ == '__main__':
    main = Main()
    main.run()
