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
# ChebConv
# -----------------------------
class ChebConv(nn.Module):
    def __init__(self, in_feats, out_feats, K):
        super(ChebConv, self).__init__()
        self.K = K
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_feats, out_feats)) for _ in range(K+1)
        ])

    def forward(self, x, laplacian):
        Tx_0 = x
        out = torch.mm(Tx_0, self.weights[0])
        if self.K > 0:
            Tx_1 = torch.mm(laplacian, x)
            out = out + torch.mm(Tx_1, self.weights[1])
        for k in range(2, self.K+1):
            Tx_2 = 2 * torch.mm(laplacian, Tx_1) - Tx_0
            out = out + torch.mm(Tx_2, self.weights[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out

# -----------------------------
# ChebNet
# -----------------------------
class ChebNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, K=3, dropout=0.5):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_feats, hidden, K)
        self.conv2 = ChebConv(hidden, out_feats, K)
        self.dropout = dropout

    def forward(self, x, laplacian):
        x = F.relu(self.conv1(x, laplacian))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, laplacian)
        return x

# -----------------------------
# 训练函数
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

# -----------------------------
# 运行示例
# -----------------------------
if __name__ == "__main__":
    features, adj, labels = load_cora()
    in_feats = features.shape[1]
    hidden = 16
    out_feats = labels.shape[1]
    model = ChebNet(in_feats, hidden, out_feats, K=3)
    train_model(model, features, adj, labels)
