from torch import nn, matmul, sqrt, tensor
from torch.nn import functional as F
import torch


class Head(nn.Module):

    def __init__(self,token_len, d_head, token_depth,d_k,masked=True) -> None:
        super().__init__()
        self.q = nn.Linear(token_len, d_head, bias=False)
        self.k = nn.Linear(token_len, d_head, bias=False)
        self.v = nn.Linear(token_len, d_head, bias=False)
        self.masked = masked
        if masked:
            self.mask = torch.triu(torch.ones(token_depth, token_depth), diagonal=1).cuda()
            self.mask = self.mask.masked_fill(self.mask == 1, -1e9)

        self.sqrt_dk = sqrt(tensor(d_k))

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        matmul_q_v = matmul(q, k.transpose(-2, -1))
        if self.masked:
            matmul_q_v = matmul_q_v + self.mask

        x = F.softmax(matmul_q_v/self.sqrt_dk, dim=1)
        x = matmul(x, v)
        return x

# class MultiHeadAttention(nn.Module):

#     def __init__(self, token_len, token_depth, nr_heads, masked=True) -> None:
#         super().__init__()
#         assert token_len % nr_heads == 0
#         d_k = token_len//nr_heads

#         self.heads = nn.ModuleList([Head(token_len, token_len//nr_heads, token_depth, d_k,masked) for _ in range(nr_heads)])
#         self.combine = nn.Linear(token_len, token_len, bias=False)
#         self.drop_out = nn.Dropout(0.2)

#     def forward(self, x):
#         x = [head(x) for head in self.heads]
#         x = torch.cat(x, dim=-1)
#         x = self.combine(x)
#         x = self.drop_out(x)
#         return x
    
class MultiHeadAttention(nn.Module):

    def __init__(self, token_len, token_depth, nr_heads, masked=True) -> None:
        super().__init__()
        assert token_len % nr_heads == 0
        
        self.q = nn.Linear(token_len, token_len, bias=False)
        self.k = nn.Linear(token_len, token_len, bias=False)
        self.v = nn.Linear(token_len, token_len, bias=False)
        self.combine = nn.Linear(token_len, token_len, bias=False)
        self.drop_out = nn.Dropout(0.2)
       
        self.nr_heads = nr_heads
        self.d_k = token_len//nr_heads
        self.dimension = sqrt(tensor(self.d_k))

        self.masked = masked
        if masked:
            self.mask = torch.triu(torch.ones(token_depth, token_depth), diagonal=1).cuda()
            self.mask = self.mask.masked_fill(self.mask == 1, -1e9)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.nr_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return (x.transpose(1, 2).contiguous()).view(batch_size, seq_length, self.nr_heads * self.d_k)

    def forward(self, x):
        q = self.split_heads(self.q(x))
        k = self.split_heads(self.k(x))
        v = self.split_heads(self.v(x))

        matmul_q_v = matmul(q, k.transpose(-2, -1))
        if self.masked:
            matmul_q_v = matmul_q_v + self.mask

        x = F.softmax(matmul_q_v/self.dimension, dim=1)
        x = matmul(x, v)
        x = self.combine_heads(x)
        x = self.combine(x)
        x = F.relu(x)
        x = self.drop_out(x)

        return x
