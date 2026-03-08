import os
import copy
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for k, v in ema_state.items():
            if k in model_state:
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + (1.0 - self.decay) * model_state[k].detach())
                else:
                    v.copy_(model_state[k])