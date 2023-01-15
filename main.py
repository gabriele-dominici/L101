from dataset import *
from interpretability_methods import *
from metrics import *
from models import *
import os

if not os.path.exists('./results'):
    os.makedirs('./results')

datasets = ['20_news_group', 'movie']

print(f'Dataset used: {datasets}')
models = {'MLP': [MLP, (20, 0.001), (6, 0.001)],
          'GCN2': [GCN2, (20, 0.01), (11, 0.001), (11, 0.01), (7, 0.001)],
          'GCN3': [GCN3, (17, 0.01), (5, 0.001)],
          'GAT2': [GAT2, (19, 0.01), (7, 0.01), (17, 0.01), (6, 0.001)],
          'GAT3': [GAT3, (14, 0.01), (4, 0.01)],
          'SAGE2': [GraphSAGE2, (23, 0.01), (6, 0.001), (14, 0.01), (6, 0.001)],
          'SAGE3': [GraphSAGE3, (17, 0.01), (5, 0.001)],
          'Simple2': [SimpleConv2, (20, 0.01), (11, 0.01), (20, 0.01), (13, 0.01)],
          'Simple3': [SimpleConv3, (45, 0.01), (11, 0.01)]
          }
print(f'Model used: {models.keys()}')

readout = ['mean', 'max']
print(f'Readout function used: {readout}')
interpretability_methods = [(random_baseline, 'random'),
                            (top_k_words, 'omission'),
                            (saliency, 'saliency'),
                            (top_words_graph, 'gnne')]
print(f'Interpretability methods used: {[el[1] for el in interpretability_methods]}')
for d in datasets:
    print(f'DATASET: {d}')
    if d == '20_news_group':
        df_train, df_dev, df_test = load_20_news_group_dataset()
        index_dataset = 1
    elif d == 'movie':
        df_train, df_dev, df_test = load_movie_dataset()
        index_dataset = 2

    (vectorizer, tokenize_func, vocab, inv_vocab,
    X_train, Y_train,
    X_dev, Y_dev,
    X_test, Y_test) = tf_idf_on_dataset(df_train, df_dev, df_test)

    (X_train_mlp, Y_train_mlp,
     X_dev_mlp, Y_dev_mlp,
     X_test_mlp, Y_test_mlp) = prepare_for_mlp(X_train, Y_train,
                                               X_dev, Y_dev,
                                               X_test, Y_test)

    train_loader, dev_loader, test_loader = prepare_for_graph(df_train, df_dev, df_test, tokenize_func, vocab)

    for m in models.keys():
        if d == '20_news_group':
            index_dataset = 1
        elif d == 'movie':
            index_dataset = 2

        if m == 'MLP':
            print(f'MODEL: {m}')
            print('Training...')
            model = train_and_evaluate_mlp(X_train_mlp, Y_train_mlp, X_dev_mlp, Y_dev_mlp,
                                           vocab, models[m][index_dataset][0],
                                           models[m][index_dataset][1])
            print('Testing...')
            test_mlp(model, X_test_mlp, Y_test_mlp, dataset=d)

            for im in interpretability_methods:
                if im[1] == 'gnne':
                    continue
                print(f'Interpretability method: {im[1]}')
                output_file = f'{im[1]}_{m}_{d}.txt'
                print(f'Writing to {output_file} ...')
                compute_all_metrics(df_test, im[0], output_file, model,
                                    vectorizer, tokenize_func, vocab, inv_vocab,
                                    'mlp')
        else:
            for mode in readout:
                if mode == 'max' and m[-1] == '3':
                    continue
                print(f'MODEL: {m}')
                print(f'Readout: {mode}')
                if mode == 'mean':
                    if d == '20_news_group':
                        index_dataset = 1
                    elif d == 'movie':
                        index_dataset = 2
                elif mode == 'max':
                    index_dataset += 2
                print('Training...')
                model = train_and_evaluate_graph(train_loader, dev_loader, models[m][0],
                                                 vocab, models[m][index_dataset][0],
                                                 models[m][index_dataset][1])
                print('Testing...')
                test_graph(model, test_loader, m, mode, d)

                for im in interpretability_methods:
                    print(f'Interpretability method: {im[1]}')
                    output_file = f'{im[1]}_{m}_{mode}.txt'
                    print(f'Writing to {output_file} ...')
                    compute_all_metrics(df_test, im[0], output_file, model,
                                        vectorizer, tokenize_func, vocab, inv_vocab,
                                        'graph')



