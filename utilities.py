import gc
import scipy
import torch
import random
import string
import sklearn
import scipy as sp
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from nltk.corpus import stopwords
from sklearn.utils import check_random_state
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

##### Genral utils

def report_results_mlp(gold, pred, name_set, n_classes):
    # Print Accuracy and F1-Score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = BinaryAccuracy().to(device)
    f1score_macro = BinaryF1Score(average='macro').to(device)
    f1score_micro = BinaryF1Score(average='micro').to(device)

    print("%s macro: %s %s " % (name_set, 'mlp', f1score_macro(pred, gold)))
    print("%s micro: %s %s " % (name_set, 'mlp', f1score_micro(pred, gold)))
    print("%s Accuracy: %s  %s" % (name_set, 'mlp', accuracy(pred, gold)))

def get_one_hot(labels, num_classes, times=None):
    # One hot-encoding of a list of values

    y = torch.eye(num_classes)
    times_list = []
    if times != None:
        for el in times:
            times_list += torch.full((1,num_classes), el)
        times = torch.stack((times_list))
        y = torch.mul(y[labels], times)
        return y
    return y[labels]


def preprocess(dataset, tokenize_func, vocab):
    # Preprocessing of the dataset
    # Create a graph for each document
    # If a token is in vocab it creates a node and connect
    # it to the next and previous node.
    # Two identical words have the same node

    dataset_edges = []
    dataset_emb = []

    for doc in dataset['data']:
        counter = 0
        last = 0
        nodes = {}
        edges = []
        embeddings = []
        times = []
        tokens = tokenize_func(doc)
        for t in tokens:
            if t in vocab.keys():
                if t in nodes.keys():
                    node = nodes[t]
                    times[node] += 1
                else:
                    node = counter
                    nodes[t] = counter
                    times += [1]
                    embeddings += [vocab[t]]
                    counter += 1
                    edges += [[node, node]]
                if len(nodes.keys()) > 1:
                    edges += [[last, node]]
                    edges += [[node, last]]
                last = node

        # if there is no token, it creates a graph with a node initialized to zero
        if len(edges) == 0:
            dataset_emb += [torch.zeros(1, len(vocab.keys()))]
            dataset_edges += [torch.tensor([[0, 0]])]
        else:
            dataset_emb += [get_one_hot(embeddings, len(vocab.keys()))]
            dataset_edges += [torch.tensor(edges)]

    return dataset_emb, dataset_edges

def transform_dataset_to_graph(dataset, n_classes, tokenize_func, vocab):
    # Create an object graph for each document in the dataset

    dataset_emb, dataset_edges = preprocess(dataset, tokenize_func, vocab)
    final_dataset = []

    for i in range(len(dataset_emb)):
        x = dataset_emb[i]
        edge_index = dataset_edges[i]
        y = torch.tensor(dataset.iloc[i]['label']).float()

        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)

        final_dataset += [data]

    return final_dataset


###### Interpretability methods utils

def all_possible_sentences(doc, split_func, vocab):
    # create all possible sentence of a doc without all occurences of a token

    token_split = split_func(doc)
    possible_sentences = [[doc, ""]]
    for index, word in enumerate(list(set(token_split))):
        if word in vocab.keys():
            token_split_temp = token_split.copy()
            token_split_temp = list(filter(lambda a: a != word, token_split_temp))
            new_sentence = " ".join(token_split_temp)
            possible_sentences += [[new_sentence, word]]
    return possible_sentences

def pred_mlp_sentences(input, vectorizer, mlp, device):
    # Preprocess of a document and the mlp does the prediction on it

    input = pd.DataFrame(input, columns = ['data', 'word'])
    tokenized_i = vectorizer.transform(input['data'])
    model_input = torch.tensor(scipy.sparse.csr_matrix.todense(tokenized_i)).float()
    output = mlp(model_input.to(device))
    tmp_result = [float(i) for i in output]

    return tmp_result

def pred_graph_sentences(input_a, model, tokenize_func, vocab, device):
    # Preprocess of a document and the gnn does the prediction on it

    input = pd.DataFrame(input_a, columns = ['data', 'word'])
    input['label'] = 0
    model_input = transform_dataset_to_graph(input, 1, tokenize_func, vocab)
    data_loader = DataLoader(model_input, batch_size=10, shuffle=False)
    del model_input
    gc.collect()
    result = []
    for data in data_loader:
        output = model(data.x.to(device), data.edge_index.to(device))
        output = [float(i) for i in output]
        result += output
        del data
        gc.collect()

    return result


def preprocess_with_token(dataset, tokenize_func, vocab):
    # same as preprocessing but returns also nodes
    dataset_edges = []
    dataset_emb = []

    for doc in dataset['data']:
        counter = 0
        last = 0
        nodes = {}
        edges = []
        embeddings = []
        tokens = tokenize_func(doc)
        for t in tokens:
            if t in vocab.keys():
                if t in nodes.keys():
                    node = nodes[t]
                else:
                    node = counter
                    nodes[t] = counter
                    embeddings += [vocab[t]]
                    counter += 1
                    edges += [[node, node]]
                if len(nodes.keys()) > 1:
                    edges += [[last, node]]
                    edges += [[node, last]]
                last = node

        if len(edges) == 0:
            dataset_emb += [torch.zeros(1, len(vocab.keys()))]
            dataset_edges += [torch.tensor([[0, 0]])]
        else:
            dataset_emb += [get_one_hot(embeddings, len(vocab.keys()))]
            dataset_edges += [torch.tensor(edges)]
    return dataset_emb, dataset_edges, nodes

def transform_dataset_to_graph_with_token(dataset, n_classes, tokenize_func, vocab):
    # same as transform_dataset_to_graph but it returns also tokens used
    dataset_emb, dataset_edges, tokens = preprocess_with_token(dataset, tokenize_func, vocab)
    final_dataset = []
    for i in range(len(dataset_emb)):
        x = dataset_emb[i]
        edge_index = dataset_edges[i]
        data = Data(x=x, edge_index=edge_index.t().contiguous())

        final_dataset += [data]
    return final_dataset, tokens

##### metric utilities

def pred_mlp_aopc(input, mlp, vectorizer, device):
    input = pd.DataFrame([input], columns = ['data'])
    tokenized_i = vectorizer.transform(input['data'])
    model_input = torch.tensor(scipy.sparse.csr_matrix.todense(tokenized_i)).float()
    output = mlp(model_input.to(device))
    tmp_result = float(output)

    return tmp_result

def pred_graph_aopc(input, model, tokenize_func, vocab, device):
    input = pd.DataFrame([input], columns = ['data'])
    input['label'] = 0
    model_input = transform_dataset_to_graph(input, 1, tokenize_func, vocab)
    data_loader = DataLoader(model_input, batch_size=250, shuffle=False)
    result = []
    for data in data_loader:
        output = model(data.x.to(device), data.edge_index.to(device))
        for i in output:
            tmp_result = float(output)
            result += [tmp_result]

    return result[0]

def pred(doc, m, vectorizer, tokenize_func, vocab, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'mlp':
        prob_orig = pred_mlp_aopc(doc, m, vectorizer, device)
    elif mode == 'graph':
        prob_orig = pred_graph_aopc(doc, m, tokenize_func, vocab, device)
    return prob_orig

def remove_words(doc, tokens_to_remove, tokenize_func):
    token_list = tokenize_func(doc)
    result = []
    for index, token in enumerate(token_list):
        if token not in tokens_to_remove:
            result += [token]
    result = ' '.join(result)
    return result

def count_token_used(doc, tokenize_func, vocab):
    token_list = tokenize_func(doc)
    token_list = [t for t in token_list if t in vocab.keys()]
    return len(token_list)

def diff_tokens(old_doc, new_doc, tokenize_func, vocab):
    token_list_old = tokenize_func(old_doc)
    token_list_old = [t for t in token_list_old if t in vocab.keys()]
    token_list_new = tokenize_func(new_doc)
    token_list_new = [t for t in token_list_new if t in vocab.keys()]
    return len(token_list_old) - len(token_list_new)

