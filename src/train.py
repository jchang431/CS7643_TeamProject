import yaml
import math
import torch

from src.datasets.dataset import get_fixmatch_dataloaders
from src.models.wideresnet import get_model
from src.methods.fixmatch import fixmatch_loss
from src.utils import set_seed, evaluate, save_checkpoint, ModelEMA


def get_cosine_lr(base_lr, step, total_steps):
    """Official FixMatch-style cosine decay."""
    progress = min(step / total_steps, 1.0)
    return base_lr * math.cos(progress * 7.0 * math.pi / 16.0)


def main():
    with open("configs/fixmatch.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    labeled_loader, unlabeled_loader, test_loader = get_fixmatch_dataloaders(
        num_labeled=cfg["num_labeled"],
        batch_size=cfg["batch_size"],
        mu=cfg["mu"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        augment=cfg["augment"],
    )

    print(f"# labeled batches: {len(labeled_loader)}")
    print(f"# unlabeled batches: {len(unlabeled_loader)}")

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
    num_steps_per_epoch = len(unlabeled_loader)
    total_steps = cfg["epochs"] * num_steps_per_epoch
    global_step = 0

    for epoch in range(cfg["epochs"]):
        model.train()

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        total_loss_meter = 0.0
        sup_loss_meter = 0.0
        unsup_loss_meter = 0.0
        mask_meter = 0.0

        for _ in range(num_steps_per_epoch):
            try:
                x_l, y_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_l, y_l = next(labeled_iter)

            try:
                x_uw, x_us, policies = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_uw, x_us, policies = next(unlabeled_iter)

            x_l = x_l.to(device, non_blocking=True)
            y_l = y_l.to(device, non_blocking=True)
            x_uw = x_uw.to(device, non_blocking=True)
            x_us = x_us.to(device, non_blocking=True)

            current_lr = get_cosine_lr(cfg["lr"], global_step, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            stats = fixmatch_loss(
                model,
                x_l,
                y_l,
                x_uw,
                x_us,
                threshold=cfg["threshold"],
                lambda_u=cfg["lambda_u"],
            )

            loss = stats["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema_model.update(model)

            if cfg["augment"] == "ctaugment":
                pseudo_conf = stats["pseudo_max_probs"].detach().cpu().numpy()
                pseudo_mask = stats["pseudo_mask_vec"].detach().cpu().numpy()

                cta_obj = unlabeled_loader.dataset.transform.cta
                for pol, conf, m in zip(policies, pseudo_conf, pseudo_mask):
                    if pol is not None:
                        proximity = float(conf) if m > 0 else float(conf * 0.5)
                        cta_obj.update_rates(pol, proximity)

            total_loss_meter += stats["loss"].item()
            sup_loss_meter += stats["loss_x"].item()
            unsup_loss_meter += stats["loss_u"].item()
            mask_meter += stats["mask"].item()

            global_step += 1

        test_acc = evaluate(ema_model.ema, test_loader, device)

        avg_loss = total_loss_meter / num_steps_per_epoch
        avg_sup = sup_loss_meter / num_steps_per_epoch
        avg_unsup = unsup_loss_meter / num_steps_per_epoch
        avg_mask = mask_meter / num_steps_per_epoch

        print(
            f"[Epoch {epoch+1:03d}/{cfg['epochs']}] "
            f"lr={current_lr:.5f} "
            f"loss={avg_loss:.4f} "
            f"xe={avg_sup:.4f} "
            f"xeu={avg_unsup:.4f} "
            f"mask={avg_mask:.4f} "
            f"test_acc={test_acc:.2f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(ema_model.ema, cfg["save_path"])

    print(f"Best accuracy: {best_acc:.2f}")


if __name__ == "__main__":
    main()