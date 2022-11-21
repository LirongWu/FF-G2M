import dgl
import copy
import torch
import numpy as np

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training for teacher GNNs
def train(model, data, feats, labels, criterion, optimizer, idx_train):

    model.train()

    logits = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return logits, loss.item()


# Training for student MLPs
def train_mini_batch(model, edge_idx, feats, labels, out_t_all, idx_l, criterion_l, criterion_t, optimizer, param):

    model.train()

    logits = model(None, feats)
    out = logits.log_softmax(dim=1)
    loss_l = criterion_l(out[idx_l], labels[idx_l])
    loss_t = model.edge_distribution_low(edge_idx, out, out_t_all.log_softmax(dim=1), criterion_t)
    loss_s = criterion_t(model.edge_distribution_high(edge_idx, logits, param['tau']), model.edge_distribution_high(edge_idx, out_t_all, param['tau']))

    if param['ablation_mode'] == 0:
        loss = loss_l * param['lamb'] + (loss_t + loss_s) * (1 - param['lamb'])
    elif param['ablation_mode'] == 1:
        loss = loss_l
    elif param['ablation_mode'] == 2:
        loss_t = criterion_t(out, out_t_all.log_softmax(dim=1))
        loss = loss_l * param['lamb'] + loss_t * (1 - param['lamb'])
    elif param['ablation_mode'] == 3:
        loss = loss_l * param['lamb'] + loss_t * (1 - param['lamb'])
    elif param['ablation_mode'] == 4:
        loss = loss_l * param['lamb'] + loss_s * (1 - param['lamb'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_l.item() * param['lamb'], loss_t.item() * (1-param['lamb']), loss_s.item() * (1-param['lamb'])


# Testing for teacher GNNs
def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval):

    model.eval()

    with torch.no_grad():
        logits = model.forward(data, feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out[idx_eval], labels[idx_eval])
        acc = evaluator(out[idx_eval], labels[idx_eval])

    return logits, loss.item(), acc


# Testing for student MLPs
def evaluate_mini_batch(model, feats, labels, criterion, evaluator):
    
    model.eval()

    with torch.no_grad():
        logits = model.forward(None, feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, labels)
        acc = evaluator(out, labels)

    return loss.item(), acc


def train_teacher(param, model, g, feats, labels, indices, criterion, evaluator, optimizer):

    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_g = g.subgraph(idx_obs).to(device)

    g = g.to(device)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(1, param["max_epoch"] + 1):
        if param['exp_setting'] == 'tran':
            out, loss = train(model, g, feats, labels, criterion, optimizer, idx_train)
            _, train_loss, train_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_train)
            _, _, val_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_val)
            _, _, test_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_test)
        else:
            out, loss = train(model, obs_g, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
            _, train_loss, train_acc = evaluate(model, obs_g, obs_feats, obs_labels, criterion, evaluator, obs_idx_train)
            _, _, val_acc = evaluate(model, obs_g, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
            _, _, test_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_test_ind)

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == 50:
            print("Early stopping!")
            break

        if epoch % 1 == 0:
            print("\033[0;30;46m [{}] CLA: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}\033[0m".format(
                                        epoch, train_loss, train_acc, val_acc, test_acc, val_best, test_val, test_best))
     
    model.load_state_dict(state)
    if param['exp_setting'] == 'tran':
        out, _, _ = evaluate(model, g, feats, labels, criterion, evaluator, idx_val)
    else:
        obs_out, _, _ = evaluate(model, obs_g, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
        out, _, _ = evaluate(model, g, feats, labels, criterion, evaluator, idx_test_ind)
        out[idx_obs] = obs_out

    return out, test_acc, test_val, test_best


def train_student(param, model, g, feats, labels, out_t_all, indices, criterion_l, criterion_t, evaluator, optimizer):

    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
        idx_l = idx_train
    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_idx_l = obs_idx_train

        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_out_t = out_t_all[idx_obs]
        obs_edge_idx = extract_indices(g.subgraph(idx_obs))

    edge_idx = extract_indices(g)
    
    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(1, param["max_epoch"] + 1):
        if param['exp_setting'] == 'tran':
            loss_l, loss_t, loss_s = train_mini_batch(model, edge_idx, feats, labels, out_t_all, idx_l, criterion_l, criterion_t, optimizer, param)
            loss = loss_l + loss_t + loss_s

            train_loss, train_acc = evaluate_mini_batch(model, feats[idx_train], labels[idx_train], criterion_l, evaluator)
            _, val_acc = evaluate_mini_batch(model, feats[idx_val], labels[idx_val], criterion_l, evaluator)
            _, test_acc = evaluate_mini_batch(model, feats[idx_test], labels[idx_test], criterion_l, evaluator)

        else:
            loss_l, loss_t, loss_s = train_mini_batch(model, obs_edge_idx, obs_feats, obs_labels, obs_out_t, obs_idx_l, criterion_l, criterion_t, optimizer, param)
            loss = loss_l + loss_t + loss_s

            train_loss, train_acc = evaluate_mini_batch(model, obs_feats[obs_idx_train], obs_labels[obs_idx_train], criterion_l, evaluator)
            _, val_acc = evaluate_mini_batch(model, obs_feats[obs_idx_val], obs_labels[obs_idx_val], criterion_l, evaluator)
            _, test_acc = evaluate_mini_batch(model, feats[idx_test_ind], labels[idx_test_ind], criterion_l, evaluator)
        

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            es = 0
        else:
            es += 1
            
        if es == 50:
            print("Early stopping!")
            break

        if epoch % 1 == 0:
            print("\033[0;30;46m [{}] CLA: {:.5f}, KD: {:.5f}, SK: {:.5f}, Total: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}\033[0m".format(
                                        epoch, loss_l, loss_t, loss_s, loss, train_acc, val_acc, test_acc, val_best, test_val, test_best))


    return test_acc, test_val, test_best
