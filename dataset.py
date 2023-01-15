import scipy
import pandas as pd
from utilities import *
from torch_geometric.loader import DataLoader
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_20_news_group_dataset():
    cats = ['soc.religion.christian', 'alt.atheism']

    newsgroups_train_dev_raw = fetch_20newsgroups(subset='train',
                                                  categories=cats)

    newsgroups_train_dev = list(zip(newsgroups_train_dev_raw.data,
                                    newsgroups_train_dev_raw.target))
    newsgroups_train, newsgroups_dev = train_test_split(newsgroups_train_dev,
                                                        test_size=209,
                                                        random_state=42)

    newsgroups_test = fetch_20newsgroups(subset='test',
                                         categories=cats)


    newsgroups_test = list(zip(newsgroups_test.data,
                               newsgroups_test.target))

    df_newsgroups_train = pd.DataFrame(newsgroups_train, columns=['data', 'label'])
    df_newsgroups_dev = pd.DataFrame(newsgroups_dev, columns=['data', 'label'])
    df_newsgroups_test = pd.DataFrame(newsgroups_test, columns=['data', 'label'])

    return df_newsgroups_train, df_newsgroups_dev, df_newsgroups_test

def load_movie_dataset():
    dataset = load_dataset("movie_rationales")

    train_movie, extra = train_test_split(dataset['train'],
                                          train_size=1072,
                                          random_state=42)

    df_train_movie = pd.DataFrame(train_movie, columns=['review', 'label', 'evidences'])
    df_dev_movie_tmp = pd.DataFrame(dataset['validation'], columns=['review', 'label', 'evidences'])
    df_test_movie_tmp = pd.DataFrame(dataset['test'], columns=['review', 'label', 'evidences'])
    extra = pd.DataFrame(extra, columns=['review', 'label', 'evidences'])

    extra_dev, extra_test = train_test_split(extra,
                                             train_size=158,
                                             random_state=42)

    extra_dev = pd.DataFrame(extra_dev, columns=['review', 'label', 'evidences'])

    df_dev_movie = pd.concat([df_dev_movie_tmp, extra_dev])
    df_test_movie = extra_test

    df_train_movie.rename(columns={'review': 'data'}, inplace=True, errors='raise')
    df_dev_movie.rename(columns={'review': 'data'}, inplace=True, errors='raise')
    df_test_movie.rename(columns={'review': 'data'}, inplace=True, errors='raise')

    return df_train_movie, df_dev_movie, df_test_movie

def tf_idf_on_dataset(train, dev, test):
    vectorizer = TfidfVectorizer(lowercase=True, min_df=10, max_df=0.25, norm=None)

    vectorizer.fit(train['data'])
    tokenize_func = vectorizer.build_analyzer()
    vocab = vectorizer.vocabulary_
    inv_vocab = {v: k for k, v in vocab.items()}

    X_train = vectorizer.transform(train['data'])
    Y_train = train['label']

    X_dev = vectorizer.transform(dev['data'])
    Y_dev = dev['label']

    X_test = vectorizer.transform(test['data'])
    Y_test = test['label']

    return (vectorizer, tokenize_func, vocab, inv_vocab,
            X_train, Y_train,
            X_dev, Y_dev,
            X_test, Y_test)

def prepare_for_mlp(X_train, Y_train,
                    X_dev, Y_dev,
                    X_test, Y_test):
    X_train_mlp = torch.tensor(scipy.sparse.csr_matrix.todense(X_train)).float()
    Y_train_mlp = torch.tensor(Y_train.to_numpy()).float()

    X_dev_mlp = torch.tensor(scipy.sparse.csr_matrix.todense(X_dev)).float()
    Y_dev_mlp = torch.tensor(Y_dev.to_numpy()).float()

    X_test_mlp = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()
    Y_test_mlp = torch.tensor(Y_test.to_numpy()).float()

    return (X_train_mlp, Y_train_mlp,
            X_dev_mlp, Y_dev_mlp,
            X_test_mlp, Y_test_mlp)

def prepare_for_graph(df_train_, df_dev, df_test, tokenize_func, vocab,
                      batch_size_train=50, batch_size_dev=50, batch_size_test=50):
    train_graph = transform_dataset_to_graph(df_train_, 1, tokenize_func, vocab)
    dev_graph = transform_dataset_to_graph(df_dev, 1, tokenize_func, vocab)
    test_graph = transform_dataset_to_graph(df_test, 1, tokenize_func, vocab)
    train_loader = DataLoader(train_graph, batch_size=batch_size_train, shuffle=True)
    dev_loader = DataLoader(dev_graph, batch_size=batch_size_dev, shuffle=False)
    test_loader = DataLoader(test_graph, batch_size=batch_size_test, shuffle=False)

    return train_loader, dev_loader, test_loader
