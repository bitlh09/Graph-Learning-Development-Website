import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

# -----------------------------
# 数据加载 (Cora 数据集)
# -----------------------------
def load_cora(path="cora"):
    idx_features_labels = np.genfromtxt(f"{path}/cora.content", dtype=str)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=float)
    labels = np.array(idx_features_labels[:, -1])  # 标签只用于评估

    idx = np.array(idx_features_labels[:, 0], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}/cora.cites", dtype=int)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=float)
    adj = adj + sp.eye(adj.shape[0])  # 自环
    return torch.FloatTensor(features.toarray()), torch.FloatTensor(adj.toarray()), labels

# -----------------------------
# 图卷积层
# -----------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        out = torch.mm(adj, support)
        return out

# -----------------------------
# GAE 模型
# -----------------------------
class GAE(nn.Module):
    def __init__(self, in_feats, hidden, latent):
        super(GAE, self).__init__()
        self.gc1 = GraphConvolution(in_feats, hidden)
        self.gc2 = GraphConvolution(hidden, latent)

    def encode(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        return self.gc2(h, adj)

    def decode(self, z):
        return torch.sigmoid(torch.mm(z, z.t()))

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return self.decode(z)

# -----------------------------
# 训练
# -----------------------------
def train_model(model, features, adj, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon_adj = model(features, adj)
        loss = F.mse_loss(recon_adj, adj)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    features, adj, labels = load_cora()
    in_feats = features.shape[1]
    hidden = 32
    latent = 16
    model = GAE(in_feats, hidden, latent)
    train_model(model, features, adj)
