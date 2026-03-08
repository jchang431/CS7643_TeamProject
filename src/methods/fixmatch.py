import torch
import torch.nn.functional as F


# interleave/de_interleave: BatchNorm stabilization

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def fixmatch_loss(model, x_l, y_l, x_uw, x_us, threshold=0.95, lambda_u=1.0):
    """
    PyTorch FixMatch loss made as close as possible to the official TF version.

    Args:
        model: classifier
        x_l: labeled images, shape [B, C, H, W]
        y_l: labeled targets, shape [B]
        x_uw: unlabeled weak images, shape [B*mu, C, H, W]
        x_us: unlabeled strong images, shape [B*mu, C, H, W]
        threshold: confidence threshold
        lambda_u: unsupervised loss weight (wu in official code)

    Returns:
        dict with loss, loss_x, loss_u, mask
    """
    batch = x_l.size(0)
    uratio = x_uw.size(0) // batch

    assert x_uw.size(0) == x_us.size(0), "Weak and strong unlabeled batch sizes must match."
    assert x_uw.size(0) % batch == 0, "Unlabeled batch size must be a multiple of labeled batch size."

    # Official TF:
    # x = interleave(tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0), 2 * uratio + 1)
    inputs = torch.cat([x_l, x_uw, x_us], dim=0)
    inputs = interleave(inputs, 2 * uratio + 1)

    logits = model(inputs)

    # Official TF:
    # logits = utils.de_interleave(logits, 2 * uratio + 1)
    logits = de_interleave(logits, 2 * uratio + 1)

    logits_x = logits[:batch]
    logits_weak = logits[batch:batch + x_uw.size(0)]
    logits_strong = logits[batch + x_uw.size(0):]

    # Official TF:
    # loss_xe = mean(sparse_softmax_cross_entropy(labels=l_in, logits=logits_x))
    loss_x = F.cross_entropy(logits_x, y_l, reduction="mean")

    # Official TF:
    # pseudo_labels = stop_gradient(softmax(logits_weak))
    with torch.no_grad():
        #weak branch -> threshold -> fixmatch
        pseudo_probs = torch.softmax(logits_weak, dim=1)
        max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
        mask = (max_probs >= threshold).float()

    # Official TF:
    # loss_xeu = mean(CE(logits_strong, argmax(pseudo_labels)) * pseudo_mask)
    # strong branch CE -> mask
    loss_u_all = F.cross_entropy(logits_strong, pseudo_labels, reduction="none")
    loss_u = torch.mean(loss_u_all * mask)

    loss = loss_x + lambda_u * loss_u

    return {
    "loss": loss,
    "loss_x": loss_x,
    "loss_u": loss_u,
    "mask": mask.mean(),
    "pseudo_max_probs": max_probs.detach(),
    "pseudo_mask_vec": mask.detach(),
    }