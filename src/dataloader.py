import dgl
import torch
import numpy as np


# Load six real-world graph datasets and Split data
def load_data(dataset):
    
    if dataset == 'cora':
        graph = dgl.data.CoraGraphDataset()[0]
    elif dataset == 'citeseer':
        graph = dgl.data.CiteseerGraphDataset()[0]
    elif dataset == 'pubmed':
        graph = dgl.data.PubmedGraphDataset()[0]
    elif dataset == 'coauthor-cs':
        graph = dgl.data.CoauthorCSDataset()[0]
    elif dataset == 'coauthor-phy':
        graph = dgl.data.CoauthorPhysicsDataset()[0]
    elif dataset == 'amazon-photo':
        graph = dgl.data.AmazonCoBuyPhotoDataset()[0]

    labels = graph.ndata['label']
    g = dgl.remove_self_loop(graph)
    g = dgl.add_self_loop(g)

    if dataset == 'cora' or dataset == 'citeseer'  or dataset == 'pubmed':
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
    else:
        n_class = int(labels.max().item() + 1)
        nrange = torch.arange(labels.shape[0])
        train_mask = torch.zeros(labels.shape[0], dtype=bool)

        for y in range(n_class):
            label_mask = (graph.ndata['label'] == y)
            train_mask[nrange[label_mask][torch.randperm(label_mask.sum())[:20]]] = True

        val_mask = ~train_mask
        val_mask[nrange[val_mask][torch.randperm(val_mask.sum())[500:]]] = False
        test_mask = ~(train_mask | val_mask)
        test_mask[nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]] = False

    return g, labels, train_mask.nonzero()[:, 0], val_mask.nonzero()[:, 0], test_mask.nonzero()[:, 0]


def load_out_t(out_t_dir):
    return torch.from_numpy(np.load(out_t_dir + "out.npz")["arr_0"])