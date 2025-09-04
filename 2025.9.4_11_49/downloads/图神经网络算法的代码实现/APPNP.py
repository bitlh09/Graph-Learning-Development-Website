import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def load_cora(path="cora"):
    idx_features_labels = np.genfromtxt(f"{path}/cora.content", dtype=str)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=float)
    labels = LabelBinarizer().fit_transform(idx_features_labels[:, -1])

    idx = np.array(idx_features_labels[:, 0], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}/cora.cites", dtype=int)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=float)
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return torch.FloatTensor(features.toarray()), torch.FloatTensor(adj_normalized.toarray()), torch.FloatTensor(labels)

class APPNP(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, K=10, alpha=0.1, dropout=0.5):
        super(APPNP, self).__init__()
        self.fc1 = nn.Linear(in_feats, hidden)
        self.fc2 = nn.Linear(hidden, out_feats)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        h = x
        for _ in range(self.K):
            x = (1 - self.alpha) * torch.mm(adj, x) + self.alpha * h
        return x

def train_model(model, features, adj, labels, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    idx_train, idx_test = train_test_split(np.arange(labels.shape[0]), test_size=0.2, random_state=42)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = loss_fn(output[idx_train], labels[idx_train].max(1)[1])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = output[idx_test].max(1)[1]
                acc = pred.eq(labels[idx_test].max(1)[1]).sum().item() / len(idx_test)
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")

if __name__ == "__main__":
    features, adj, labels = load_cora()
    in_feats = features.shape[1]
    hidden = 16
    out_feats = labels.shape[1]
    model = APPNP(in_feats, hidden, out_feats, K=10, alpha=0.1)
    train_model(model, features, adj, labels)
