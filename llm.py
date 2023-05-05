import torch
from torch import nn
import torch.nn.functional as F
from attention_block import MultiHeadAttention


class AddAndNorm(nn.Module):

    def __init__(self, token_len) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(token_len)
    
    def forward(self, x, y):
        x = x+y
        x = self.layer_norm(x)
        return x
    
class FeedForward(nn.Module):

    def __init__(self,token_len):
        super().__init__()
        self.linear1 = nn.Linear(token_len, token_len,bias=False)
        self.linear2 = nn.Linear(token_len, token_len,bias=False)
        self.drop_out1 = nn.Dropout(0.2)
        self.drop_out2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        self.drop_out1(x)
        x = self.linear2(x)
        x= F.relu(x)
        x = self.drop_out2(x)
        return x

class Block(nn.Module):
    def __init__(self,token_len, token_depth,nr_heads,masked):
        super().__init__()
        self.multi_head1 = MultiHeadAttention(token_len,token_depth,nr_heads,masked)
        self.add_norm = AddAndNorm(token_len)
        self.multi_head2 = MultiHeadAttention(token_len,token_depth,nr_heads,False)
        self.add_norm2 = AddAndNorm(token_len)
        self.feed_forward = FeedForward(token_len)
        self.add_norm3 = AddAndNorm(token_len)

    def forward(self,x):
        x_out = self.multi_head1(x)
        x = self.add_norm(x,x_out)
        x_out = self.multi_head2(x)
        x = self.add_norm2(x,x_out)
        x_out = self.feed_forward(x)
        x = self.add_norm3(x,x_out)
        return x
    
class Decoder(nn.Module):

    def __init__(self, token_len, token_depth,nr_heads):
        super().__init__()
       
        nr_blocks = 6
        module_list = [Block(token_len,token_depth,nr_heads,masked=True)]
        for i in range(nr_blocks):
            module_list.append(Block(token_len,token_depth,nr_heads,masked=False))

        self.model_seq=nn.Sequential(*module_list)

   
    def forward(self, x):
        x=self.model_seq(x)

        return x
    
class LLM(nn.Module):

    def __init__(self, token_len, token_depth,nr_heads,out_len,nr_vec_out):
        super().__init__()
        self.embedding = nn.Embedding(token_len, token_len)
        self.pos_embedding = nn.Embedding( token_depth, 1)
        self.decoder = Decoder(token_len, token_depth,nr_heads)
        self.linear1 = nn.Linear(token_len, token_len)
        self.linear2 = nn.Linear(token_len, out_len)
        self.linear3 = nn.Linear(token_depth, nr_vec_out)
        self.drop_out1 = nn.Dropout(0.2)
        self.drop_out2 = nn.Dropout(0.1)

    def forward(self, x):
        x=x.cuda()
        tok_embed = self.embedding(x)
        pos_embed = self.pos_embedding(torch.arange(x.shape[1]).cuda()).unsqueeze(0)
        x = tok_embed+pos_embed
        x=self.decoder(x)
        x=F.relu(x) 
        

        x = self.linear1(x)
        x=F.relu(x)
        self.drop_out1(x)
        x = self.linear2(x)
        x=F.relu(x)
        self.drop_out2(x)   
        x = self.linear3(x.transpose(1,2))
        return x.transpose(1,2)
    
    def save_model(self, path):
        try:
            torch.save(self.state_dict(), path)
            print('successfully saved model to ', path)
        except Exception as e:
            print('Could not save model to ', path, e)
    
    def load_model(self, path):
        try:
            self.load_state_dict(torch.load(path))
            print('successfully loaded model from ', path)
        except Exception as e:
            print('Could not load model from ', path, e)
