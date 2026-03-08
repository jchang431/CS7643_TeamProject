import random
from collections import namedtuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


OPS = {}
OP = namedtuple("OP", ("f", "bins"))


def register(*bins):
    def wrap(f):
        OPS[f.__name__] = OP(f, bins)
        return f
    return wrap


def pil_to_cta_array(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image [0,255] to float32 numpy array in [-1, 1], HWC.
    """
    x = np.asarray(img).astype("f")
    x = x / 127.5 - 1.0
    return x


def cta_array_to_pil(x: np.ndarray) -> Image.Image:
    """
    Convert float32 numpy array in [-1,1], HWC to PIL image [0,255].
    """
    x = np.round(127.5 * (1.0 + x)).clip(0, 255).astype("uint8")
    return Image.fromarray(x)


def apply(x: np.ndarray, ops):
    """
    Apply CTAugment policy to image x.

    Args:
        x: numpy array in [-1, 1], shape [H, W, C]
        ops: list of OP(name, args) sampled by CTAugment.policy()

    Returns:
        numpy array in [-1, 1], shape [H, W, C]
    """
    if ops is None:
        return x

    y = Image.fromarray(np.round(127.5 * (1 + x)).clip(0, 255).astype("uint8"))
    for op, args in ops:
        y = OPS[op].f(y, *args)
    return np.asarray(y).astype("f") / 127.5 - 1.0


class CTAugment:
    """
    Official-style CTAugment controller.

    Args:
        depth: number of augmentation ops in one policy
        th: threshold for rate_to_p
        decay: exponential moving average factor for policy quality
    """
    def __init__(self, depth=2, th=0.85, decay=0.99):
        self.decay = decay
        self.depth = depth
        self.th = th
        self.rates = {}
        for k, op in OPS.items():
            self.rates[k] = tuple([np.ones(x, "f") for x in op.bins])

    def rate_to_p(self, rate):
        p = rate + (1 - self.decay)  # avoid all zeros
        p = p / p.max()
        p[p < self.th] = 0
        if np.all(p == 0):
            p = np.ones_like(p)
        return p

    def policy(self, probe=False):
        """
        Sample a policy.

        If probe=True:
            sample uniformly/randomly for probing.
        If probe=False:
            sample according to learned rates.
        """
        kl = list(OPS.keys())
        v = []

        if probe:
            for _ in range(self.depth):
                k = random.choice(kl)
                bins = self.rates[k]
                rnd = np.random.uniform(0, 1, len(bins))
                v.append((k, rnd.tolist()))
            return v

        for _ in range(self.depth):
            vt = []
            k = random.choice(kl)
            bins = self.rates[k]
            rnd = np.random.uniform(0, 1, len(bins))
            for r, bin_ in zip(rnd, bins):
                p = self.rate_to_p(bin_)
                value = np.random.choice(p.shape[0], p=p / p.sum())
                vt.append((value + r) / p.shape[0])
            v.append((k, vt))
        return v

    def update_rates(self, policy, proximity):
        """
        Update policy quality estimates.

        Args:
            policy: policy returned by self.policy(...)
            proximity: float in [0,1], higher is better
        """
        proximity = float(np.clip(proximity, 0.0, 1.0))
        for k, bins in policy:
            for p, rate in zip(bins, self.rates[k]):
                p = int(p * len(rate) * 0.999)
                rate[p] = rate[p] * self.decay + proximity * (1 - self.decay)

    def stats(self):
        return "\n".join(
            "%-16s    %s" % (
                k,
                " / ".join(
                    " ".join("%.2f" % x for x in self.rate_to_p(rate))
                    for rate in self.rates[k]
                ),
            )
            for k in sorted(OPS.keys())
        )


def _enhance(x, op, level):
    return op(x).enhance(0.1 + 1.9 * level)


def _imageop(x, op, level):
    return Image.blend(x, op(x), level)


def _filter(x, op, level):
    return Image.blend(x, x.filter(op), level)


@register(17)
def autocontrast(x, level):
    return _imageop(x, ImageOps.autocontrast, level)


@register(17)
def blur(x, level):
    return _filter(x, ImageFilter.BLUR, level)


@register(17)
def brightness(x, brightness):
    return _enhance(x, ImageEnhance.Brightness, brightness)


@register(17)
def color(x, color):
    return _enhance(x, ImageEnhance.Color, color)


@register(17)
def contrast(x, contrast):
    return _enhance(x, ImageEnhance.Contrast, contrast)


@register(17)
def cutout(x, level):
    """
    Apply cutout to PIL image at the specified level.
    """
    size = 1 + int(level * min(x.size) * 0.499)
    img_height, img_width = x.size

    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    upper_coord = (
        max(0, height_loc - size // 2),
        max(0, width_loc - size // 2),
    )
    lower_coord = (
        min(img_height, height_loc + size // 2),
        min(img_width, width_loc + size // 2),
    )

    pixels = x.load()
    for i in range(upper_coord[0], lower_coord[0]):
        for j in range(upper_coord[1], lower_coord[1]):
            pixels[i, j] = (127, 127, 127)
    return x


@register(17)
def equalize(x, level):
    return _imageop(x, ImageOps.equalize, level)


@register(17)
def invert(x, level):
    return _imageop(x, ImageOps.invert, level)


@register()
def identity(x):
    return x


@register(8)
def posterize(x, level):
    level = 1 + int(level * 7.999)
    return ImageOps.posterize(x, level)


@register(17, 6)
def rescale(x, scale, method):
    s = x.size
    scale *= 0.25
    crop = (
        scale * s[0],
        scale * s[1],
        s[0] * (1 - scale),
        s[1] * (1 - scale),
    )
    methods = (
        Image.ANTIALIAS,
        Image.BICUBIC,
        Image.BILINEAR,
        Image.BOX,
        Image.HAMMING,
        Image.NEAREST,
    )
    method = methods[int(method * 5.99)]
    return x.crop(crop).resize(x.size, method)


@register(17)
def rotate(x, angle):
    angle = int(np.round((2 * angle - 1) * 45))
    return x.rotate(angle)


@register(17)
def sharpness(x, sharpness):
    return _enhance(x, ImageEnhance.Sharpness, sharpness)


@register(17)
def shear_x(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


@register(17)
def shear_y(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


@register(17)
def smooth(x, level):
    return _filter(x, ImageFilter.SMOOTH, level)


@register(17)
def solarize(x, th):
    th = int(th * 255.999)
    return ImageOps.solarize(x, th)


@register(17)
def translate_x(x, delta):
    delta = int(np.round((2 * delta - 1) * 0.3 * x.size[0]))
    return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


@register(17)
def translate_y(x, delta):
    delta = int(np.round((2 * delta - 1) * 0.3 * x.size[1]))
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))