import scipy
import pandas as pd
from utilities import *
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer

def random_baseline(doc, m, vectorizer, tokenize_func, vocab, inv_vocab, device, mode, max_num_words_to_remove=False, random_seed=42):

    # Select random words of a document

    random.seed(random_seed)

    # first get all the words that are in the document
    selected_words = []
    token_list = tokenize_func(doc)
    for i, token in enumerate(token_list):
        if token in vocab.keys():
            selected_words.append(token)

    # randomly select words
    random.shuffle(selected_words)
    random_to_remove_for = selected_words.copy()
    random.shuffle(selected_words)
    random_to_remove_against = selected_words.copy()

    if max_num_words_to_remove:
        random_to_remove_for = random_to_remove_for[:max_num_words_to_remove]
        random_to_remove_against = random_to_remove_against[:max_num_words_to_remove]

    return random_to_remove_for, random_to_remove_against


def top_k_words(doc, m,  vectorizer, split_func, vocab, inv_vocab, device, mode, k=False):
    # omission interpretability method

    # create all possible sentences from a doc deleting
    # each time all occurrences of a word
    possible_sentences = all_possible_sentences(doc, split_func, vocab)
    # predictions for all the possibilities
    if mode == 'mlp':
        result = pred_mlp_sentences(possible_sentences, vectorizer, m, device)
    elif mode == 'graph':
        result = pred_graph_sentences(possible_sentences, m, split_func, vocab, device)

    # diff between prediction = importance of a token
    diff = []
    for i in result:
        diff += [result[0] - i]

    final_df = pd.DataFrame(possible_sentences, columns=['data', 'word'])
    final_df['prob'] = result
    final_df['diff'] = diff
    # select pos and negative words
    if result[0] < 0.5:
        pos_words = final_df[final_df['diff'] < 0]
        neg_words = final_df[final_df['diff'] > 0]
    elif result[0] > 0.5:
        pos_words = final_df[final_df['diff'] > 0]
        neg_words = final_df[final_df['diff'] < 0]
    pos_words = list(pos_words.sort_values(by=["diff"])['word'])
    neg_words = list(neg_words.sort_values(by=["diff"], ascending=False)['word'])

    # select top k
    if k:
        pos_words = pos_words[:k]
        neg_words = neg_words[:k]

    return (pos_words, neg_words)

def saliency(doc, m, vectorizer, tokenize_func, vocab, inv_vocab, device, mode, k=False):
    # saliency method

    input = pd.DataFrame([doc], columns = ['data'])
    input['label'] = 0

    # saliency for mlp
    if mode == 'mlp':
        # do the prediction step on the document
        tokenized_i = vectorizer.transform(input['data'])
        model_input = torch.tensor(scipy.sparse.csr_matrix.todense(tokenized_i)).float()
        indexes = [ind for ind, i in enumerate(model_input[0]) if i > 0]
        model_input.requires_grad_()
        scores = m(model_input.to(device))
        # compute gradients
        scores.backward()
        saliency = model_input.grad.data[0]
        # saliency score
        saliency = [(float(score), inv_vocab[index]) for index, score in enumerate(saliency) if index in indexes]
        if scores > 0.5:
            saliency_for = [score for score in saliency if score[0] > 0]
            saliency_against =  [score for score in saliency if score[0] < 0]
        elif scores < 0.5:
            saliency_for = [score for score in saliency if score[0] < 0]
            saliency_against =  [score for score in saliency if score[0] > 0]

    # saliency for graph
    elif mode == 'graph':
        model_input, tokens_used = transform_dataset_to_graph_with_token(input, 1, tokenize_func, vocab)
        data_loader = DataLoader(model_input, batch_size=250, shuffle=False)
        # prediction step on document
        m.eval()
        for data in data_loader:
            data = data.to(device)
            data['x'].requires_grad_()
            scores = m(data.x.to(device), data.edge_index.to(device))
            # compute gradients
            scores.backward()
        # saliency score
        saliency = data['x'].grad.data#.sum(dim=1)

        node_index = {i[1]: i[0] for i in tokens_used.items()}
        # map saliency score to word
        saliency = [x[vocab[node_index[ind]]] for ind, x in enumerate(saliency)]
        if scores > 0.5:
            saliency_for = [(float(score), node_index[index]) for index, score in enumerate(saliency)
                            if score > 0]
            saliency_against = [(float(score), node_index[index]) for index, score in enumerate(saliency)
                            if score < 0]
        elif scores < 0.5:
            saliency_for = [(float(score), node_index[index]) for index, score in enumerate(saliency)
                            if score < 0]
            saliency_against = [(float(score), node_index[index]) for index, score in enumerate(saliency)
                            if score > 0]

    saliency_for = sorted(saliency_for, key=lambda tup: tup[0], reverse=True)
    saliency_against = sorted(saliency_against, key=lambda tup: tup[0])

    if k:
        saliency_for = saliency_for[:k]
        saliency_against = saliency_against[:k]

    saliency_for = [i[1] for i in saliency_for]
    saliency_against = [i[1] for i in saliency_against]

    return saliency_for, saliency_against

def top_words_graph(doc, model, vectorizer, tokenize_func, vocab, inv_vocab, device, mode, k = False):

    # use GNNExplainer to generate local explanation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = pd.DataFrame([doc], columns = ['data'])
    input['label'] = 0
    gnne_cuda = GNNExplainer(epochs=200).to(device)
    #prepare the graph
    model_input, tokens_used = transform_dataset_to_graph_with_token(input, 1, tokenize_func, vocab)

    explainer = Explainer(
        model=model.to(device),
        algorithm=gnne_cuda,
        explanation_type='model',
        node_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw'
        ),
    )

    explanation = explainer(model_input[0].x.to(device), model_input[0].edge_index.to(device))
    # importance score
    importance = explanation.node_mask
    token_score = []
    for index, el in enumerate(importance):
        token_score += [(list(tokens_used.keys())[index], el)]
    token_score = sorted(token_score, key=lambda tup: tup[1], reverse=True)
    return [i[0] for i in token_score]
