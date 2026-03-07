import yaml
import math
import torch
import torch.nn.functional as F

from src.datasets.dataset import get_fixmatch_dataloaders
from src.models.wideresnet import get_model
from src.utils import set_seed, evaluate, save_checkpoint, ModelEMA


def get_cosine_lr(base_lr, step, total_steps):
    """FixMatch-style cosine decay."""
    progress = min(step / total_steps, 1.0)
    return base_lr * math.cos(progress * 7.0 * math.pi / 16.0)


def main():
    with open("configs/supervised.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    labeled_loader, _, test_loader = get_fixmatch_dataloaders(
        num_labeled=cfg["num_labeled"],
        batch_size=cfg["batch_size"],
        mu=cfg["mu"],  # loader 함수 재사용용. supervised에서는 실제로 안 씀.
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
    )

    print(f"# labeled batches: {len(labeled_loader)}")

    model = get_model(num_classes=cfg["num_classes"]).to(device)
    ema_model = ModelEMA(model, decay=cfg["ema_decay"])

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        nesterov=True,
        weight_decay=cfg["weight_decay"],
    )

    best_acc = 0.0

    num_steps_per_epoch = len(labeled_loader)
    total_steps = cfg["epochs"] * num_steps_per_epoch
    global_step = 0

    for epoch in range(cfg["epochs"]):
        model.train()

        total_loss_meter = 0.0

        for x_l, y_l in labeled_loader:
            x_l = x_l.to(device, non_blocking=True)
            y_l = y_l.to(device, non_blocking=True)

            current_lr = get_cosine_lr(cfg["lr"], global_step, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            logits = model(x_l)
            loss = F.cross_entropy(logits, y_l)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema_model.update(model)

            total_loss_meter += loss.item()
            global_step += 1

        test_acc = evaluate(ema_model.ema, test_loader, device)
        avg_loss = total_loss_meter / num_steps_per_epoch

        print(
            f"[Epoch {epoch+1:03d}/{cfg['epochs']}] "
            f"lr={current_lr:.5f} "
            f"loss={avg_loss:.4f} "
            f"test_acc={test_acc:.2f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(ema_model.ema, cfg["save_path"])

    print(f"Best supervised accuracy: {best_acc:.2f}")


if __name__ == "__main__":
    main()