import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class DAN(nn.Module):

    def __init__(self, embedding, args, max_pool_over_time=False):
        super(DAN, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False


        self.W_hidden = nn.Linear(embed_dim, 200)
        self.W_out = nn.Linear(200, 1)



    def forward(self, x):
        all_x = self.embedding_layer(x_indx.squeeze(1))
        avg_x = torch.mean(all_x) # Check dim
        hidden = F.relu( self.W_hidden(avg_x) )
        out = F.relu( self.W_out(hidden) )
        return out




