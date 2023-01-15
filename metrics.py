from tqdm import tqdm
from utilities import *
import numpy as np

def compute_perturbation_curve(doc,  m, vectorizer, tokenize_func, vocab,
                               words_to_remove_all, mode, L=10):
    # Compute AOPC https://arxiv.org/pdf/1509.06321.pdf
    values = []
    words_to_remove = []
    prob_orig = pred(doc,  m, vectorizer, tokenize_func, vocab, mode)
    i = 0
    # remove k words
    for i, word in enumerate(words_to_remove_all):
        if i == L:
            break
        words_to_remove.append(word)
        new_text = remove_words(doc, words_to_remove, tokenize_func)
        prob = pred(new_text, m, vectorizer, tokenize_func, vocab, mode)

        if prob_orig > 0.5:
            values.append(prob_orig - prob)
        elif prob_orig < 0.5:
            values.append(prob - prob_orig)

    L = min(i, L)
    return np.array(values).sum()/(L + 1)

def switching_point(doc, m, vectorizer, tokenize_func, vocab, words_to_remove_all, mode):
    # compute switching point https://aclanthology.org/N18-1097.pdf
    words_to_remove = []

    prob_orig = pred(doc, m, vectorizer, tokenize_func, vocab, mode)
    total_token_used = count_token_used(doc, tokenize_func, vocab)
    # remove words until the prediction changes
    for i, word in enumerate(words_to_remove_all):
        words_to_remove.append(word)
        new_text = remove_words(doc, words_to_remove, tokenize_func)
        prob = pred(new_text, m, vectorizer, tokenize_func, vocab, mode)
        if int(round(prob_orig)) != int(round(prob)):
            diff = diff_tokens(doc, new_text, tokenize_func, vocab)
            return diff/total_token_used
    # return 1 if the prediction never change
    return 1

def jaccard_similarity(top_words_a, top_words_b, k):
    # compute jaccard similarity https://aclanthology.org/2021.findings-acl.327.pdf
    # check how much agreement there is between the two sets of words
    A = set(top_words_a[:k])
    B = set(top_words_b[:k])
    if len(A) == 0 and len(B) == 0:
        return 1
    return len(A & B) / len(A | B)

def compute_all_metrics(dataset, interpretability_methods, output_file,
                        model, vectorizer, tokenize_func, vocab, inv_vocab, mode):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_set_for = []
    graph_aopc_for = []
    graph_sp_for = []
    for doc in tqdm(dataset['data']):
        graph_for, _ = interpretability_methods(doc, model, vectorizer, tokenize_func, vocab, inv_vocab, device, mode)
        graph_set_for += [graph_for]
        graph_aopc_for += [compute_perturbation_curve(doc, model, vectorizer, tokenize_func, vocab, graph_for, mode)]
        graph_sp_for += [switching_point(doc, model, vectorizer, tokenize_func, vocab, graph_for, mode)]

    print(f'AOPC+@10: {np.array(graph_aopc_for).mean()}')
    print(f'SP: {np.array(graph_sp_for).mean()}')

    with open('./results/'+output_file, 'w+') as f:
        for i, el in enumerate(graph_set_for):
            f.write(f'{el} \n')