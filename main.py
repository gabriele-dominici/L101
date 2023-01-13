from dataset import *
from interpretability_methods import *
from metrics import *
from models import *

datasets = ['20_news_group', 'movie']
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

readout = ['mean', 'max']
interpretability_methods = [(random_baseline, 'random'),
                            (top_k_words, 'omission'),
                            (saliency, 'saliency'),
                            (top_words_graph, 'gnne')]

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
    X_test, Y_test) = tf_idf_on_dataset(tf_idf_on_dataset)

    (X_train_mlp, Y_train_mlp,
     X_dev_mlp, Y_dev_mlp,
     X_test_mlp, Y_test_mlp) = prepare_for_mlp(X_train, Y_train,
                                               X_dev, Y_dev,
                                               X_test, Y_test)

    train_loader, dev_loader, test_loader = prepare_for_graph(df_train, df_dev, df_test)

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
        else:
            for mode in readout:
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
                test_graph(model, test_loader, models[m][0], mode, d)

        for im in interpretability_methods:
            print(f'Interpretability method: {im[i]}')
            output_file = f'{im[1]}_{m}_{mode}.txt'
            print(f'Writing to {output_file} ...')
            compute_all_metrics(d, im[0], output_file, model,
                                vectorizer, tokenize_func, vocab, inv_vocab,
                                mode)



