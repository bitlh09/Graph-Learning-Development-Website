import os
import argparse
import random
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

# ---------------- utilities (same as gcn) ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def download_if_needed(root='data/cora'):
    os.makedirs(root, exist_ok=True)
    base = 'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data'
    files = {'cora.content': base + '/cora.content', 'cora.cites': base + '/cora.cites'}
    for fname, url in files.items():
        fpath = os.path.join(root, fname)
        if not os.path.exists(fpath): urllib.request.urlretrieve(url, fpath)
    print("Data ready at", root)

def load_cora(root='data/cora'):
    download_if_needed(root)
    content = os.path.join(root, 'cora.content'); cites = os.path.join(root, 'cora.cites')
    node_ids, feats, labels = [], [], []
    with open(content, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().split()
            node_ids.append(p[0]); feats.append([int(x) for x in p[1:-1]]); labels.append(p[-1])
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    x = np.array(feats, dtype=np.float32)
    label_set = sorted(list(set(labels))); label_map = {l:i for i,l in enumerate(label_set)}
    y = np.array([label_map[l] for l in labels], dtype=np.int64)
    rows, cols = [], []
    with open(cites, 'r', encoding='utf-8') as f:
        for line in f:
            s,t = line.strip().split()
            if s in id_map and t in id_map:
                i,j = id_map[s], id_map[t]; rows += [i,j]; cols += [j,i]
    adj = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(node_ids), len(node_ids)))
    adj.sum_duplicates()
    return x, y, adj.tocsr()

def build_planetoid_splits(y, num_classes=None, train_per_class=20, val_size=500, test_size=1000, seed=42):
    if num_classes is None: num_classes = int(y.max()) + 1
    rng = np.random.RandomState(seed); y_np = np.array(y)
    idx_by_class = [np.where(y_np == c)[0] for c in range(num_classes)]
    train_idx = []
    for arr in idx_by_class:
        chosen = rng.choice(arr, train_per_class, replace=False); train_idx.extend(chosen.tolist())
    rest = np.setdiff1d(np.arange(len(y_np)), train_idx); rest = rng.permutation(rest)
    val_idx = rest[:val_size].astype(np.int64); test_idx = rest[val_size:val_size+test_size].astype(np.int64)
    return np.array(train_idx, dtype=np.int64), val_idx, test_idx

def normalize_adj(adj, add_self_loops=True):
    if add_self_loops: adj = adj + sparse.identity(adj.shape[0], dtype=np.float32, format='csr')
    deg = np.array(adj.sum(1)).flatten(); deg_inv_sqrt = np.power(deg, -0.5); deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sparse.diags(deg_inv_sqrt); adj_norm = D_inv_sqrt.dot(adj).dot(D_inv_sqrt).tocoo()
    return torch.from_numpy(adj_norm.toarray()).float()

# ---------------- model (dense attention) ----------------
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, concat=True, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_feats = in_feats; self.out_feats = out_feats; self.concat = concat
        self.W = nn.Parameter(torch.empty(in_feats, out_feats))
        self.a = nn.Parameter(torch.empty(2*out_feats, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414); nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha); self.dropout = dropout
    def forward(self, h, adj):
        Wh = h @ self.W
        N = Wh.shape[0]
        Wh_i = Wh.unsqueeze(1).repeat(1, N, 1)
        Wh_j = Wh.unsqueeze(0).repeat(N, 1, 1)
        a_input = torch.cat([Wh_i, Wh_j], dim=2)
        e = self.leakyrelu(a_input @ self.a).squeeze(2)
        neg_inf = -9e15
        e = torch.where(adj > 0, e, neg_inf * torch.ones_like(e))
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        h_prime = attention @ Wh
        return F.elu(h_prime) if self.concat else h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nheads=8, dropout=0.6, alpha=0.2):
        super().__init__()
        self.dropout = dropout; self.nheads = nheads
        self.attentions = nn.ModuleList([GATLayer(nfeat, nhid, concat=True, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        self.out_att = GATLayer(nhid * nheads, nclass, concat=False, dropout=dropout, alpha=alpha)
    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

# ---------------- training ----------------
def train_loop(model, x, adj_dense, y, idx_train, idx_val, device, epochs=200, lr=0.005, weight_decay=5e-4, patience=20, save_path='gat_best.pth'):
    model.to(device); x = x.to(device); adj_dense = adj_dense.to(device); y = y.to(device)
    idx_train_t = torch.from_numpy(idx_train).to(device); idx_val_t = torch.from_numpy(idx_val).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = 0.0; best_state = None; best_epoch = 0
    for epoch in range(1, epochs+1):
        model.train(); opt.zero_grad()
        logits = model(x, adj_dense)
        loss = F.cross_entropy(logits[idx_train_t], y[idx_train_t])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, adj_dense)
            val_acc = (val_logits.argmax(dim=1)[idx_val_t] == y[idx_val_t]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc; best_epoch = epoch; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; torch.save(best_state, save_path)
        if epoch % 10 == 0 or epoch == 1:
            train_acc = (logits.argmax(dim=1)[idx_train_t] == y[idx_train_t]).float().mean().item()
            print(f"Epoch {epoch:03d} Loss {loss.item():.4f} TrainAcc {train_acc:.4f} ValAcc {val_acc:.4f}")
        if epoch - best_epoch >= patience:
            print(f"Early stop at {epoch}, best val {best_val:.4f} at epoch {best_epoch}")
            break
    if best_state is not None: model.load_state_dict(best_state)
    return model

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default='data/cora')
    parser.add_argument('--save', type=str, default='gat_best.pth')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    x_np, y_np, adj_sparse = load_cora(root=args.data_root)
    N, F = x_np.shape
    num_classes = int(y_np.max()) + 1
    print(f"Cora loaded: N={N}, F={F}, classes={num_classes}")

    row_sum = x_np.sum(1, keepdims=True); row_sum[row_sum == 0] = 1.0; x_np = x_np / row_sum
    x = torch.from_numpy(x_np).float(); y = torch.from_numpy(y_np).long()

    adj_with_self = adj_sparse + sparse.identity(N, dtype=np.float32, format='csr')
    adj_dense = torch.from_numpy(adj_with_self.toarray()).float()
    # Planetoid split
    idx_train, idx_val, idx_test = build_planetoid_splits(y, num_classes=num_classes, seed=args.seed)
    print("Split sizes:", len(idx_train), len(idx_val), len(idx_test))

    model = GAT(nfeat=F, nhid=args.hidden, nclass=num_classes, nheads=args.heads, dropout=0.6)
    model = train_loop(model, x, adj_dense, y, idx_train, idx_val, device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience, save_path=args.save)

    # test
    model.to(device); x = x.to(device); adj_dense = adj_dense.to(device); y = y.to(device)
    idx_test_t = torch.from_numpy(idx_test).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x, adj_dense)
        test_acc = (logits.argmax(dim=1)[idx_test_t] == y[idx_test_t]).float().mean().item()
    print("Test Accuracy:", test_acc)
    print("Saved best model to", args.save)

if __name__ == '__main__':
    main()
