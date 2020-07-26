import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sub_graph(adj, num):
    '''
    Monte carlo sample a number of neighbors for each node given the adjacent matrix
    adj: normalized and processed graph adjacent matrix
    num: the number of samples for each neighbor
    '''
    # nodes = adj.shape[0]
    # neighbor_number = adj.to_dense()
    # print('normalize neighbor', torch.sum(neighbor_number,dim=1)[0:10])
    # neighbor_number = torch.sum(neighbor_number>0,dim=1).float().reshape(-1,1)/float(num)
    # mask = torch.zeros(nodes,nodes).cuda()
    # for i in range(nodes):
    #     sample = torch.randint(0,nodes,(num,))
    #     mask[i,sample] = 1.0
    #     mask[sample,i] = 1.0
    num_nodes = adj.shape[0]
    dense_adj = adj.to_dense()
    mask = torch.zeros(num_nodes, num_nodes).cuda()
    normalized_coeff = torch.ones(num_nodes,1).reshape(-1,1).cuda()
    for i in range(num_nodes):
        # select neighbors for each node
        neighbors = (dense_adj[i,:]>0).nonzero().reshape(-1).cuda()
        sample = torch.unique(torch.randint(0, neighbors.shape[0], (num,)))
        sampled_neighbors = neighbors[sample]
        normalized_coeff[i,0] = neighbors.shape[0] / sample.shape[0]
        mask[i,sampled_neighbors] = 1.0
        mask[sampled_neighbors, i] = 1.0
    # renormalized the adjacent matrix based on subgraph
    dense_adj = dense_adj*mask.float()*normalized_coeff

    # return adj.to_dense()*mask.float()*neighbor_number.float()
    return dense_adj

    

