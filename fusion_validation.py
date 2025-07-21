import torch
from utils import topk_accuracy, topk_accuracy_multiple_timesteps, get_marginal_indexes, marginalize, softmax, \
    topk_recall_multiple_timesteps
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import json


def get_many_shot(args):
    """Get many shot verbs, nouns and actions for class-aware metrics (Mean Top-5 Recall)"""
    # read the list of many shot verbs
    many_shot_verbs = pd.read_csv(
        join(args.path_to_data, args.dataset + '_many_shot_verbs.csv'))['verb_class'].values
    # read the list of many shot nouns
    many_shot_nouns = pd.read_csv(
        join(args.path_to_data, args.dataset + '_many_shot_nouns.csv'))['noun_class'].values

    # read the list of actions
    actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
    # map actions to (verb, noun) pairs
    a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
               for a in actions.iterrows()}

    # create the list of many shot actions
    # an action is "many shot" if at least one
    # between the related verb and noun are many shot
    many_shot_actions = []
    for a, (v, n) in a_to_vn.items():
        if v in many_shot_verbs or n in many_shot_nouns:
            many_shot_actions.append(a)

    return many_shot_verbs, many_shot_nouns, many_shot_actions

def get_scores_fusion_maat(args, model, loader):
    model.eval()
    predictions = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            # import ipdb; ipdb.set_trace()
            x = batch['past_features']  # if args.task == 'anticipation' else 'action_features']
            if type(x) == list:
                x = [xx.to(args.device) for xx in x]
            else:
                x = x.to(args.device)

            y = batch['label'].numpy()
            ids.append(batch['id'].numpy())

            preds = model(x)
            preds = preds.cpu().numpy()[:, -args.S_ant:, :]

            predictions.append(preds)
            labels.append(y)

    # import ipdb; ipdb.set_trace()
    action_scores = np.concatenate(predictions)

    labels = np.concatenate(labels)

    ids = np.concatenate(ids)

    actions = pd.read_csv(
        join(args.path_to_data, 'actions.csv'), index_col='id')
    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')
    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))

    verb_scores = marginalize(action_probs, vi).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)
    noun_scores = marginalize(action_probs, ni).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)

    if labels.max() > 0:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2]
    else:
        return verb_scores, noun_scores, action_scores, ids
def get_scores_fusion_add_w(args, models, loaders):
    verb_scores = 0
    noun_scores = 0
    action_scores = 0
    print(args.Fusion_weight)
    wei=[0.31,0.22,0.25]
    # import ipdb; ipdb.set_trace() wei[i]#
    for i, (model, loader) in enumerate(zip(models, loaders)):
        outs = get_scores(args, model, loader)
        verb_scores += outs[0] * args.Fusion_weight[i]
        noun_scores += outs[1] * args.Fusion_weight[i]
        action_scores += outs[2] * args.Fusion_weight[i]

    verb_scores /= len(models)
    noun_scores /= len(models)
    action_scores /= len(models)

    return [verb_scores, noun_scores, action_scores] + list(outs[3:])


def get_scores(args, model, loader):
    model.eval()
    predictions = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            #import ipdb; ipdb.set_trace()
            x = batch['past_features']# if args.task == 'anticipation' else 'action_features']
            if type(x) == list:
                x = [xx.to(args.device) for xx in x]
            else:
                x = x.to(args.device)

            y = batch['label'].numpy()
            ids.append(batch['id'].numpy())
            
            preds = model(x)
            preds = preds[0].cpu().numpy()[:, -args.S_ant:, :]

            predictions.append(preds)
            labels.append(y)

    #import ipdb; ipdb.set_trace()
    action_scores = np.concatenate(predictions)

    labels = np.concatenate(labels)

    ids = np.concatenate(ids)


    actions = pd.read_csv(
        join(args.path_to_data, 'actions.csv'), index_col='id')
    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')
    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))


    verb_scores = marginalize(action_probs, vi).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)
    noun_scores = marginalize(action_probs, ni).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)

    
    if labels.max()>0:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2]
    else:
        return verb_scores, noun_scores, action_scores, ids

def fusion_validation(args, model, loader):
    verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels = get_scores_fusion_maat(args, model,
                                                                                                         loader)

    verb_accuracies = topk_accuracy_multiple_timesteps(
        verb_scores, verb_labels)
    noun_accuracies = topk_accuracy_multiple_timesteps(
        noun_scores, noun_labels)
    action_accuracies = topk_accuracy_multiple_timesteps(
        action_scores, action_labels)

    many_shot_verbs, many_shot_nouns, many_shot_actions = get_many_shot(args)

    verb_recalls = topk_recall_multiple_timesteps(
        verb_scores, verb_labels, k=5, classes=many_shot_verbs)
    noun_recalls = topk_recall_multiple_timesteps(
        noun_scores, noun_labels, k=5, classes=many_shot_nouns)
    action_recalls = topk_recall_multiple_timesteps(
        action_scores, action_labels, k=5, classes=many_shot_actions)

    all_accuracies = np.concatenate(
        [verb_accuracies, noun_accuracies, action_accuracies, verb_recalls, noun_recalls, action_recalls])
    all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
    indices = [
        ('Verb', 'Top-1 Accuracy'),
        ('Verb', 'Top-5 Accuracy'),
        ('Verb', 'Mean Top-5 Recall'),
        ('Noun', 'Top-1 Accuracy'),
        ('Noun', 'Top-5 Accuracy'),
        ('Noun', 'Mean Top-5 Recall'),
        ('Action', 'Top-1 Accuracy'),
        ('Action', 'Top-5 Accuracy'),
        ('Action', 'Mean Top-5 Recall'),
    ]

    cc = np.linspace(args.alpha * args.S_ant, args.alpha, args.S_ant, dtype=str)
    scores = pd.DataFrame(all_accuracies * 100, columns=cc, index=pd.MultiIndex.from_tuples(indices))
    print(scores)


def predictions_to_json(verb_scores, noun_scores, action_scores, action_ids, a_to_vn, top_actions=100, version='0.1',
                        sls=None):
    """Save verb, noun and action predictions to json for submitting them to the EPIC-Kitchens leaderboard"""
    predictions = {'version': version,
                   'challenge': 'action_anticipation', 'results': {}}

    if sls is not None:
        predictions['sls_pt'] = 1
        predictions['sls_tl'] = 4
        predictions['sls_td'] = 3

    row_idxs = np.argsort(action_scores)[:, ::-1]
    top_100_idxs = row_idxs[:, :top_actions]

    action_scores = action_scores[np.arange(
        len(action_scores)).reshape(-1, 1), top_100_idxs]

    for i, v, n, a, ai in zip(action_ids, verb_scores, noun_scores, action_scores, top_100_idxs):
        predictions['results'][str(i)] = {}
        predictions['results'][str(i)]['verb'] = {str(
            ii): float(vv) for ii, vv in enumerate(v)}
        predictions['results'][str(i)]['noun'] = {str(
            ii): float(nn) for ii, nn in enumerate(n)}
        predictions['results'][str(i)]['action'] = {
            "%d,%d" % a_to_vn[ii]: float(aa) for ii, aa in zip(ai, a)}
    return predictions


def fusion_test(args, model, loader, m):
    discarded_ids = loader[0].dataset.discarded_ids
    verb_scores, noun_scores, action_scores, ids = get_scores_fusion_add_w(args, model, loader)

    idx = -4 if args.task == 'anticipation' else -1
    ids = list(ids) + list(discarded_ids)
    verb_scores = np.concatenate((verb_scores, np.zeros((len(discarded_ids), *verb_scores.shape[1:]))))[:, idx, :]
    noun_scores = np.concatenate((noun_scores, np.zeros((len(discarded_ids), *noun_scores.shape[1:]))))[:, idx, :]
    action_scores = np.concatenate((action_scores, np.zeros((len(discarded_ids), *action_scores.shape[1:]))))[:,
                    idx, :]

    actions = pd.read_csv(join(args.path_to_data, 'actions.csv'))
    # map actions to (verb, noun) pairs
    a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
               for a in actions.iterrows()}

    preds = predictions_to_json(verb_scores, noun_scores, action_scores, ids, a_to_vn,
                                version='0.1', sls=True)

    with open(args.exp_name + f"_{m}.json", 'w') as f:
        f.write(json.dumps(preds, indent=4, separators=(',', ': ')))