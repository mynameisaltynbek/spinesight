import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from net import Net, image_path, keypoints_2d, prep, target

ROOT = Path(__file__).resolve().parent.parent


class Frames(Dataset):
    def __init__(self, files, kps):
        self.files, self.kps = files, kps

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        label = json.load(open(self.files[i]))
        x, s = prep(cv2.imread(str(image_path(self.files[i]))))
        return x, torch.from_numpy(target(keypoints_2d(label, self.kps) * s, len(self.kps)))


def run(net, dl, dev, opt=None):
    net.train(opt is not None)
    total = 0.0
    with torch.set_grad_enabled(opt is not None):
        for x, t in dl:
            loss = ((net(x.to(dev)) - t.to(dev)) ** 2).mean() * 1000
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(x)
    return total / len(dl.dataset)


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "synth_out"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "model" / "model.pt"
    kps = np.array(json.load(open(folder / "keypoints.json"))["keypoints_3d"])
    files = sorted(folder.glob("[0-9]*.json"))
    n_val = max(1, len(files) // 20)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(Frames(files[:-n_val], kps), batch_size=32, shuffle=True, num_workers=2)
    val_dl = DataLoader(Frames(files[-n_val:], kps), batch_size=32, num_workers=2)
    net = Net(len(kps)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = float("inf")
    print(f"{len(files) - n_val} train, {n_val} val, device {dev}")
    for ep in range(epochs):
        tr = run(net, train_dl, dev, opt)
        va = run(net, val_dl, dev)
        sched.step()
        if va < best:
            best = va
            torch.save({"state": net.state_dict(), "keypoints_3d": kps.tolist()}, out)
        print(f"epoch {ep + 1}/{epochs}  train {tr:.4f}  val {va:.4f}{'  saved' if va == best else ''}", flush=True)
