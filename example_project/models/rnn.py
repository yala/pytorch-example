import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class RNN(nn.Module):

    def __init__(self, embedding, args, max_pool_over_time=False):
        super(RNN, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embed_dim = embed_dim

        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False


        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=200,
                          num_layers=1, batch_first=True)

        self.W_o = nn.Linear(200,1)



    def forward(self, x):
        all_x = self.embedding_layer(x_indx.squeeze(1))
        h0 = Variable(torch.randn(self.args.batch_size,
                                  self.embed_dim, 200))
        output, h_n = self.rnn(all_x)

        out = self.W_o(output ) #TODO figure out



        return out




