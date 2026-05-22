import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        node_count = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(node_count, dim=0)
        Wh_repeated_alternating = Wh.repeat(node_count, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(node_count, node_count, 2 * self.out_features)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, A_QK=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, A_QK) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, A_QK))
        return x


class DwsConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super(DwsConv, self).__init__()
        self.depth_conv = nn.Conv2d(out_dim, out_dim, kernel_size, padding=kernel_size // 2, groups=out_dim)
        self.point_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, groups=1, bias=False)
        self.leakyrelu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, input):
        out = self.point_conv(self.bn(input))
        out = self.leakyrelu(out)
        out = self.depth_conv(out)
        out = self.leakyrelu(out)
        return out


class FDSM(nn.Module):
    def __init__(self, channels, freq_components=None):
        super().__init__()
        self.channels = channels
        self.freq_components = freq_components or (channels // 2 + 1)
        self.complex_weight = nn.Parameter(
            torch.randn(self.freq_components, 2, dtype=torch.float32) * 0.01
        )

    def forward(self, x):
        _, channels = x.shape
        x = x.to(torch.float32)
        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight).view(1, -1)
        x_freq = x_freq * weight
        return torch.fft.irfft(x_freq, n=channels, dim=-1, norm='ortho')


def fast_position_encoding(seq_len, d_model, encode_device='cpu'):
    position = torch.arange(seq_len, device=encode_device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=encode_device) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=encode_device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ConstructAdj(nn.Module):
    def __init__(self, in_dim, s_len, d_k=16, topk=8, tau=1.0, symmetrize=True):
        super().__init__()
        self.Wq = nn.Linear(in_dim, d_k, bias=True)
        self.Wk = nn.Linear(in_dim, d_k, bias=True)
        self.pos = fast_position_encoding(s_len, d_k, device)
        self.scale = 1.0 / math.sqrt(d_k)
        self.topk = topk
        self.tau = tau
        self.sym = symmetrize

    def forward(self, X, A_spa):
        Q = self.Wq(X) + self.pos.to(X.device)
        K = self.Wk(X) + self.pos.to(X.device)
        logits = ((Q @ K.t()) * self.scale + torch.log(A_spa + 1e-6)) / max(self.tau, 1e-6)
        A = F.softmax(logits, dim=-1)

        k = min(self.topk, A.size(1))
        vals, idx = torch.topk(A, k=k, dim=-1)
        Gs = torch.zeros_like(A).scatter_(dim=-1, index=idx, src=vals)

        if self.sym:
            Gs = torch.max(Gs, Gs.t())

        return Gs / (Gs.sum(dim=-1, keepdim=True) + 1e-6)


class GSDG(nn.Module):
    def __init__(self, height, width, changel, class_count, Q: torch.Tensor, A: torch.Tensor,
                 A_Spa: torch.Tensor, model='normal', dim=128, lama=0.95, d_k=16, topk=8,
                 layer1_kernel_size=3, layer2_kernel_size=7):
        super(GSDG, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.dim = dim
        self.Q = Q
        self.A = A
        self.A_Spa = A_Spa
        self.model = model
        self.norm_col_Q = Q / torch.sum(Q, 0, keepdim=True)
        self.Graph_Genrater = ConstructAdj(dim, A.shape[0], d_k=d_k, topk=topk)

        self.stem = nn.Sequential(
            nn.BatchNorm2d(self.channel),
            nn.Conv2d(self.channel, dim, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )

        self.conv_branch = nn.Sequential(
            DwsConv(in_dim=dim, out_dim=64, kernel_size=layer1_kernel_size),
            DwsConv(in_dim=64, out_dim=64, kernel_size=layer2_kernel_size),
        )

        self.gat_branch = GAT(nfeat=dim, nhid=30, nout=64, dropout=0.1, nheads=4, alpha=0.2)
        self.gat_branch2 = GAT(nfeat=64, nhid=60, nout=64, dropout=0.2, nheads=4, alpha=0.2)
        self.GraphProj = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.FGN = FDSM(dim)
        self.lama = lama
        self.ClassifyHead = nn.Sequential(nn.Linear(64, self.class_count))

    def forward(self, x: torch.Tensor):
        h, w, _ = x.shape
        pre_x = self.stem(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        pre_x = torch.squeeze(pre_x, 0).permute([1, 2, 0])
        clean_x_flatten = pre_x.reshape([h * w, -1])

        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)
        superpixels_flatten = self.FGN(superpixels_flatten)

        conv_out = self.conv_branch(torch.unsqueeze(pre_x.permute([2, 0, 1]), 0))
        conv_out = torch.squeeze(conv_out, 0).permute([1, 2, 0]).reshape([h * w, -1])

        A_content = self.Graph_Genrater(superpixels_flatten, self.A_Spa)
        H = self.gat_branch(superpixels_flatten, A_content)
        H = self.gat_branch2(H, A_content) + H

        gat_out = torch.matmul(self.Q, H)
        gat_out = self.GraphProj(gat_out)

        feature_z = gat_out * self.lama + (1 - self.lama) * conv_out
        logits = self.ClassifyHead(feature_z)
        return F.softmax(logits, -1)
