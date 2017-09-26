import gzip
import numpy as np
import torch
import cPickle as pickle
import example_project.data.full_beer_dataset as dataset

def getEmbeddingTensor():

    embedding_path='beer_review/review+wiki.filtered.200.txt.gz'
    lines = []
    with gzip.open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        vector = [float(x) for x in emb ]
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) )
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)


    return embedding_tensor, word_to_indx



# Depending on arg, build dataset
def load_dataset(args):
    print("\nLoading data...")
    embeddings, word_to_indx = getEmbeddingTensor()
    args.embedding_dim = embeddings.shape[1]

    train_data = dataset.FullBeerDataset('train', word_to_indx)
    dev_data = dataset.FullBeerDataset('heldout', word_to_indx)

    return train_data, dev_data, embeddings
