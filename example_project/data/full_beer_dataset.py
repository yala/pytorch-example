import torch
import torch.utils.data as data
import gzip
import tqdm
import numpy as np

PATH="beer_review/reviews.aspect0.{}.txt.gz"
DATASET_SIZE = 800

class FullBeerDataset(data.Dataset):

    def __init__(self, name, word_to_indx, max_length=50, stem='beer_review/reviews.aspect'):
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length

        with gzip.open(PATH.format(name)) as gfile:
            lines = gfile.readlines()[:DATASET_SIZE]
            for line in tqdm.tqdm(lines):
                sample = self.processLine(line)
                self.dataset.append(sample)
            gfile.close()

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, line):
        labels = [ float(v) for v in line.split()[:5] ]

        label = float(labels[0])
        text = line.split('\t')[-1].split()[:self.max_length]
        x =  getIndicesTensor(text, self.word_to_indx, self.max_length)
        sample = {'x':x, 'y':label}
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample


def getIndicesTensor(text_arr, word_to_indx, max_length):
    nil_indx = 0
    text_indx = [ word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
    if len(text_indx) < max_length:
        text_indx.extend( [nil_indx for _ in range(max_length - len(text_indx))])

    x =  torch.LongTensor(text_indx)

    return x