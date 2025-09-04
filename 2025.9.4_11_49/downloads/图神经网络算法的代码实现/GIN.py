import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# -----------------------------
# 数据加载 (Cora 数据集)
# -----------------------------
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

# -----------------------------
# GIN 卷积层
# -----------------------------
class GINConv(nn.Module):
    def __init__(self, in_feats, out_feats, eps=0.0):
        super(GINConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
        self.eps = nn.Parameter(torch.tensor([eps]))

    def forward(self, x, adj):
        out = torch.mm(adj, x) + (1 + self.eps) * x
        return self.mlp(out)

# -----------------------------
# GIN 网络
# -----------------------------
class GIN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_layers=2, dropout=0.5):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GINConv(in_feats, hidden))
        for _ in range(num_layers-2):
            self.layers.append(GINConv(hidden, hidden))
        self.layers.append(GINConv(hidden, out_feats))
        self.dropout = dropout

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i != len(self.layers)-1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x

# -----------------------------
# 训练
# -----------------------------
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
    model = GIN(in_feats, hidden, out_feats, num_layers=3)
    train_model(model, features, adj, labels)
