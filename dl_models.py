"""
dl_models.py  –  X-Vector (TDNN) and ECAPA-TDNN in PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


# ── Shared utility ─────────────────────────────────────────────────────────────
class StatsPool(nn.Module):
    def forward(self, x):           # [B, C, T]
        return torch.cat([x.mean(2), x.std(2) + 1e-9], dim=1)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── X-Vector ──────────────────────────────────────────────────────────────────
class TDNNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, context, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=context,
                              dilation=dilation,
                              padding=dilation*(context-1)//2)
        self.bn   = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class XVectorNet(nn.Module):
    def __init__(self, n_mels, num_speakers, emb_dim=config.EMBEDDING_DIM):
        super().__init__()
        self.tdnn = nn.Sequential(
            TDNNLayer(n_mels, 512, 5, 1),
            TDNNLayer(512,    512, 3, 2),
            TDNNLayer(512,    512, 3, 3),
            TDNNLayer(512,    512, 1),
            TDNNLayer(512,   1500, 1),
        )
        self.pool  = StatsPool()
        self.seg1  = nn.Sequential(nn.Linear(3000, 512), nn.ReLU(), nn.BatchNorm1d(512))
        self.embed = nn.Sequential(nn.Linear(512, emb_dim), nn.ReLU(), nn.BatchNorm1d(emb_dim))
        self.clf   = nn.Linear(emb_dim, num_speakers)

    def forward(self, x, return_embedding=False):
        x = x.squeeze(1)            # [B, n_mels, T]
        x = self.tdnn(x)
        x = self.pool(x)
        x = self.seg1(x)
        emb = self.embed(x)
        if return_embedding:
            return emb
        return self.clf(emb), emb


# ── ECAPA-TDNN ─────────────────────────────────────────────────────────────────
class Res2Block(nn.Module):
    def __init__(self, ch, kernel, dilation, scale=8):
        super().__init__()
        width = ch // scale
        self.scale = scale
        self.conv1 = nn.Conv1d(ch, ch, 1);  self.bn1 = nn.BatchNorm1d(ch)
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel, dilation=dilation,
                      padding=dilation*(kernel-1)//2)
            for _ in range(scale-1)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale-1)])
        self.conv2 = nn.Conv1d(ch, ch, 1);  self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        chunks = torch.chunk(x, self.scale, dim=1)
        out = [chunks[0]]; sp = chunks[0]
        for i, (c, b) in enumerate(zip(self.convs, self.bns)):
            sp = sp + chunks[i+1] if i > 0 else chunks[i+1]
            sp = F.relu(b(c(sp)))
            out.append(sp)
        x = F.relu(self.bn2(self.conv2(torch.cat(out, 1))))
        return x + res


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, att_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim*3, att_dim, 1), nn.Tanh(),
            nn.Conv1d(att_dim, in_dim, 1),   nn.Softmax(dim=2))
    def forward(self, x):
        gm = x.mean(2, keepdim=True).expand_as(x)
        gs = x.std(2,  keepdim=True).expand_as(x)
        w  = self.attn(torch.cat([x, gm, gs], 1))
        mean = (w*x).sum(2)
        std  = (w*(x-mean.unsqueeze(2))**2).sum(2).clamp(1e-9).sqrt()
        return torch.cat([mean, std], 1)


class ECAPATDNNNet(nn.Module):
    def __init__(self, n_mels, num_speakers, C=512, emb_dim=config.EMBEDDING_DIM):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv1d(n_mels,C,5,padding=2),
                                    nn.ReLU(), nn.BatchNorm1d(C))
        self.l1 = Res2Block(C, 3, 2)
        self.l2 = Res2Block(C, 3, 3)
        self.l3 = Res2Block(C, 3, 4)
        self.cat_conv = nn.Conv1d(C*3, C*3, 1)
        self.pool     = AttentiveStatsPool(C*3)
        self.bn_pool  = nn.BatchNorm1d(C*6)
        self.embed    = nn.Linear(C*6, emb_dim)
        self.bn_emb   = nn.BatchNorm1d(emb_dim)
        self.clf      = nn.Linear(emb_dim, num_speakers)

    def forward(self, x, return_embedding=False):
        x  = x.squeeze(1)
        x  = self.stem(x)
        x1 = self.l1(x); x2 = self.l2(x1); x3 = self.l3(x2)
        cat = self.cat_conv(torch.cat([x1,x2,x3], 1))
        p   = self.bn_pool(self.pool(cat))
        emb = self.bn_emb(self.embed(p))
        if return_embedding:
            return emb
        return self.clf(emb), emb


# ── I/O helpers ───────────────────────────────────────────────────────────────
def save_dl_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"  [DL] Saved → {path}")


def load_dl_model(model_cls, path, n_mels, num_speakers):
    device = get_device()
    m = model_cls(n_mels=n_mels, num_speakers=num_speakers)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m


def get_embedding(model, spec: np.ndarray) -> np.ndarray:
    device = get_device()
    model.eval()
    if spec.ndim == 2:
        spec = spec[np.newaxis]
    t = torch.tensor(spec[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        emb = model(t, return_embedding=True)
    return emb.cpu().numpy().squeeze()
