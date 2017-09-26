import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import tqdm
import datetime
import pdb


# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")

    if args.model_name == 'dan':
        return DAN(embeddings, args)
    elif args.model_name == 'rnn':
        return RNN(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))


class DAN(nn.Module):

    def __init__(self, embeddings, args):
        super(DAN, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.W_hidden = nn.Linear(embed_dim, 200)
        self.W_out = nn.Linear(200, 1)



    def forward(self, x_indx):
        all_x = self.embedding_layer(x_indx)
        avg_x = torch.mean(all_x, dim=1)
        hidden = F.relu( self.W_hidden(avg_x) )
        out = self.W_out(hidden)
        return out


class RNN(nn.Module):

    def __init__(self, embeddings, args):
        super(RNN, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embed_dim = embed_dim

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )


        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=200,
                          num_layers=1, batch_first=True)

        self.W_o = nn.Linear(200,1)



    def forward(self, x_indx):
        all_x = self.embedding_layer(x_indx)
        h0 = autograd.Variable(torch.randn(1, self.args.batch_size, 200))
        output, h_n = self.rnn(all_x, h0)
        h_n = h_n.squeeze(0)
        out = self.W_o(h_n )
        return out




