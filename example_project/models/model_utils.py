import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
import example_project.data.dataset_utils as data_utils
import example_project.train.train_utils as train_utils
import tqdm
import datetime
import pdb


# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")

    if args.model_name == 'dan':
        return example_project.models.dan.DAN(embeddings, args)
    elif args.model_name == 'rnn':
        return example_project.models.dan.RNN(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))