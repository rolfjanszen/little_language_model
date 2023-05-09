from os.path import expanduser
from torch.utils.data import DataLoader, IterableDataset
from letter_loader import LetterLoader, ALPHABET
from llm import LLM
from torch.optim import Adam
from torch import nn, concatenate
import torch
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter

MODEL_SAVE_PATH = expanduser('~/models/llm.pt')


class Main:
    def __init__(self):
        self.token_len = 128*2
        self.out_vec_len = len(ALPHABET)-1
        self.token_depth = 32
        nr_heads = 8
        nr_vec_out = 1
        self.batch_size = 100
        print('token_len', self.token_len)
        self.char_as_int = True

        self.letter_loader = LetterLoader(self.token_depth, nr_vec_out=1,set_type='train')

        self.llm = LLM(self.token_len, self.token_depth, nr_heads, self.out_vec_len, nr_vec_out).cuda()
        self.llm.load_model(MODEL_SAVE_PATH)
        #Maximum learning rate is 0.00003. Max. weight decay is 0.00001
        self.optimizer = Adam(self.llm.parameters(), lr=0.000031, weight_decay=0.01)
        self.loss = nn.CrossEntropyLoss()
        self.bin_loss = nn.BCELoss()
        self.tensorboard = SummaryWriter(expanduser('~/code/little_language_model/tensorboard'))
        # Count the number of parameters in the model
        self.count_parameters()
        self.tensorboard.add_graph(self.llm, torch.randint(0,26, (1,self.token_depth)).cuda())

    def count_parameters(self):
        nr_params = 0
        for param in self.llm.parameters():
            nr_params += param.numel()
        print(f'Number of parameters: {nr_params}')


    def train_epoch(self):
        train_loader = DataLoader(self.letter_loader, batch_size=self.batch_size, shuffle=True, num_workers=0)
        test_data = LetterLoader(self.token_depth,nr_vec_out=1,set_type='test')
        test_loader = DataLoader(test_data, batch_size=self.batch_size,shuffle=True, num_workers=0)

        summed_loss = 0
        summed_acc = 0
        
        steps_till_test = 300
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
            #Genereate a histogram of the predicted characters
            # self.tensorboard.add_histogram('pred_char', pred_char, batch_idx)
            #Get the modal from the input array

            target_char = target.argmax(dim=2)
            accuracy = (pred_char == target_char).sum().item() / \
                (self.batch_size * target.shape[1])
            summed_acc += accuracy
            summed_loss += loss.item()
            self.optimizer.step()

            if batch_idx % steps_till_test == 0 and batch_idx != 0:
                avg_train_loss = summed_loss / steps_till_test
                print('average train loss:', avg_train_loss)
                print('average char accuracy: ', summed_acc/steps_till_test)

                self.llm.save_model(MODEL_SAVE_PATH)
                test_len = int((40*summed_acc/steps_till_test)**2)

                self.test_model( test_len)
                self.get_avg_test_loss(test_loader)
                self.tensorboard.add_scalar('Loss/train', avg_train_loss, batch_idx)
                self.tensorboard.add_scalar('char accuracy', summed_acc/steps_till_test, batch_idx)
                summed_acc = 0
                summed_loss = 0
            

            self.tensorboard.add_scalar('loss', summed_loss, batch_idx)

    def get_avg_test_loss(self, test_loader):
        """Get the loss of the model on the test set"""
        summed_loss = 0
        self.llm.eval()
        max_test_runs = 20

        for batch_idx, (data, target) in enumerate(test_loader):
            input = data
            out = self.llm(input)
            target = target.cuda()
            loss = self.loss(out.reshape(-1, self.out_vec_len),
                             target.cuda().reshape(-1, self.out_vec_len))
            summed_loss += loss.item()
            if batch_idx > max_test_runs:
                break
        averaged_loss = summed_loss / (max_test_runs)
        self.tensorboard.add_scalar('Loss/test', averaged_loss, batch_idx)
        print('average test loss: ', averaged_loss)

    
    def test_model(self,  test_len):
        """Test the model by starting to give a sample input 
            and then let the model predict the rest of the text"""
        data, _ = self.letter_loader.__getitem__(random.randint(0, len(self.letter_loader)-1))
        disp_results = []
        results = []
        for indx in data:
            disp_results.append(ALPHABET[indx])

        disp_results += list('->>>')
        disp_results = ''.join(disp_results)
        input_enc = data.unsqueeze(0)
        self.llm.eval()
        test_len = max(test_len, 10)
        print('test_len', test_len)
        for i in range(test_len):

            data_out_m = self.llm(input_enc).detach().cpu()
            char_probs = torch.functional.F.softmax(data_out_m[0,0],dim=0).numpy()
            # char_probs = char_probs.round(1) #Cut off anyhting below 0.1
            new_char = random.choices(ALPHABET[:-1], weights=list(char_probs), k=1)
            # indx, _ = self.letter_loader.vec_to_char(data_out_m)
            indx=ALPHABET.index(new_char[0])
            indx_len = 1#len(indx[0])

            results.append(new_char[0])
            input = concatenate((input_enc, torch.tensor([[indx]])), dim=1)
            input_enc = input[:, indx_len:]

        print('example result: ',disp_results, ''.join(results).replace('#','\n'))

    def run(self):
        for i in range(1000):
            print('running epoch: ', i)
            self.train_epoch()


if __name__ == '__main__':

    main = Main()
    # main.test_model(500)
    main.run()

