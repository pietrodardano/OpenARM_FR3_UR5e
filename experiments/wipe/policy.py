"""Wipe-stroke policies trained on the Eidon human strokes: a small diffusion policy (DDPM over
an action chunk) and a plain behaviour-cloning MLP with the same inputs, for comparison.

Observation: the last OBS_H tool displacements (2D, m per step) + a goal vector (2D, m).
Action: the next ACT_H displacements. Goals are hindsight-relabelled in training (where the hand
really went over the next GOAL_H steps); at run time they point at the dirt.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path("outputs/eidon")
OBS_H, ACT_H, GOAL_H = 8, 16, 24  # steps at the robot rate
RETARGET_SCALE = 0.5  # human strokes -> board size
SLOWDOWN = 2  # 24 Hz human samples played at 12 Hz robot steps (half speed)
K = 50  # diffusion steps


def make_dataset(rng: np.random.Generator, n: int) -> tuple:
    segs = np.load(OUT / "wipe_segments.npz")["segs"] * RETARGET_SCALE  # (N, 96, 2)
    d = np.diff(segs, axis=1)  # displacements
    T = d.shape[1]
    i = rng.integers(0, len(d), n)
    t = rng.integers(OBS_H, T - max(ACT_H, GOAL_H) + 1, n)
    obs = np.stack([d[a, b - OBS_H:b] for a, b in zip(i, t)])
    act = np.stack([d[a, b:b + ACT_H] for a, b in zip(i, t)])
    goal = np.stack([d[a, b:b + GOAL_H].cumsum(0).mean(0) for a, b in zip(i, t)])
    th = rng.uniform(0, 2 * np.pi, n)  # wiping has no preferred direction on the board
    c, s = np.cos(th), np.sin(th)
    Rm = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (n, 2, 2)
    obs, act = np.einsum("nij,ntj->nti", Rm, obs), np.einsum("nij,ntj->nti", Rm, act)
    goal = np.einsum("nij,nj->ni", Rm, goal)
    return obs.astype(np.float32), goal.astype(np.float32), act.astype(np.float32)


class Norm(nn.Module):
    def __init__(self, d_scale: float, g_scale: float) -> None:
        super().__init__()
        self.register_buffer("d", torch.tensor(d_scale))
        self.register_buffer("g", torch.tensor(g_scale))

    def cond(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs.flatten(1) / self.d, goal / self.g], 1)


def mlp(i: int, o: int, h: int = 512) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.Mish(), nn.Linear(h, h), nn.Mish(), nn.Linear(h, h), nn.Mish(), nn.Linear(h, o))


class Diffusion(nn.Module):
    def __init__(self, norm: Norm) -> None:
        super().__init__()
        self.norm = norm
        self.net = mlp(OBS_H * 2 + 2 + ACT_H * 2 + 16, ACT_H * 2)
        b = torch.linspace(1e-4, 0.02 * 1000 / K, K)
        self.register_buffer("beta", b)
        self.register_buffer("abar", torch.cumprod(1 - b, 0))

    def temb(self, k: torch.Tensor) -> torch.Tensor:
        f = torch.exp(torch.linspace(0, 4, 8, device=k.device))
        x = k[:, None].float() / K * f
        return torch.cat([x.sin(), x.cos()], 1)

    def eps(self, x: torch.Tensor, k: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, c, self.temb(k)], 1))

    def loss(self, obs, goal, act) -> torch.Tensor:
        c, x0 = self.norm.cond(obs, goal), act.flatten(1) / self.norm.d
        k = torch.randint(0, K, (len(x0),), device=x0.device)
        n = torch.randn_like(x0)
        a = self.abar[k][:, None]
        return ((self.eps(a.sqrt() * x0 + (1 - a).sqrt() * n, k, c) - n) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs, goal) -> torch.Tensor:
        c = self.norm.cond(obs, goal)
        x = torch.randn(len(c), ACT_H * 2, device=c.device)
        for k in reversed(range(K)):
            kk = torch.full((len(c),), k, device=c.device)
            e = self.eps(x, kk, c)
            a, ab, b = 1 - self.beta[k], self.abar[k], self.beta[k]
            x = (x - b / (1 - ab).sqrt() * e) / a.sqrt()
            if k > 0:
                x = x + b.sqrt() * torch.randn_like(x)
        return (x * self.norm.d).view(-1, ACT_H, 2)


class BC(nn.Module):
    def __init__(self, norm: Norm) -> None:
        super().__init__()
        self.norm = norm
        self.net = mlp(OBS_H * 2 + 2, ACT_H * 2)

    def loss(self, obs, goal, act) -> torch.Tensor:
        return ((self.net(self.norm.cond(obs, goal)) - act.flatten(1) / self.norm.d) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs, goal) -> torch.Tensor:
        return (self.net(self.norm.cond(obs, goal)) * self.norm.d).view(-1, ACT_H, 2)


def train(steps: int = 6000) -> None:
    rng = np.random.default_rng(0)
    obs, goal, act = make_dataset(rng, 400_000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm_args = (float(np.abs(act).std() * 3), float(np.abs(goal).std() * 3))
    data = [torch.tensor(x, device=dev) for x in (obs, goal, act)]
    for name, cls in (("diffusion", Diffusion), ("bc", BC)):
        torch.manual_seed(0)
        model = cls(Norm(*norm_args)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for s in range(steps):
            idx = torch.randint(0, len(data[0]), (512,), device=dev)
            loss = model.loss(*(x[idx] for x in data))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if s % 1000 == 0 or s == steps - 1:
                print(f"{name} step {s} loss {loss.item():.4f}")
        torch.save({"state": model.state_dict(), "norm": norm_args}, OUT / f"{name}.pt")


def load(name: str, dev: str = "cpu") -> nn.Module:
    ck = torch.load(OUT / f"{name}.pt", map_location=dev)
    model = {"diffusion": Diffusion, "bc": BC}[name](Norm(*ck["norm"]))
    model.load_state_dict(ck["state"])
    return model.to(dev).eval()


if __name__ == "__main__":
    train()
