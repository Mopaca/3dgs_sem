#
# Per-pixel guidance maps for the semantic-guided densification experiments.
#
# Every map follows the "goodness" convention: higher = better reconstruction.
# Returned shape is [1, H, W]; maps are computed without grad and detached.
#
#   dino    : DINOv2 cosine similarity (existing DinoSimilarity.cosine_map), range [-1, 1]
#   ssim    : local SSIM map with the same 11x11 Gaussian window (sigma 1.5) and C1/C2 as
#             the training loss (utils/loss_utils.py), averaged over RGB, range [-1, 1]
#   l1      : 1 - mean_c |render - gt|, range [0, 1]
#   l1_blur : 1 - GaussianBlur(mean_c |render - gt|). The blur sigma (default 4.0 px) is
#             roughly the standard deviation of a 14 px box (14 / sqrt(12) = 4.04), i.e. the
#             spatial scale of a DINOv2 ViT-S/14 patch. Used to separate "semantic" from
#             "spatially pooled" effects.
#
# For the [9]-style mode the training loop converts these to an error map with
# error = 1 - goodness (dino, ssim: [0, 2]; l1: the plain L1 error in [0, 1]).
#

import math

import torch
import torch.nn.functional as F

from utils.loss_utils import create_window

SIGNALS = ("dino", "ssim", "l1", "l1_blur")

_SSIM_C1 = 0.01 ** 2
_SSIM_C2 = 0.03 ** 2


def ssim_map(img1, img2, window_size=11):
    """
    Local SSIM map, identical in window and constants to utils.loss_utils._ssim.

    Args:
        img1, img2: [B, C, H, W] or [C, H, W]
    Returns:
        [B, 1, H, W] (or [1, H, W] for unbatched input), averaged over channels
    """
    unbatched = img1.dim() == 3
    if unbatched:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    channel = img1.size(-3)
    window = create_window(window_size, channel).to(device=img1.device, dtype=img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    smap = ((2 * mu1_mu2 + _SSIM_C1) * (2 * sigma12 + _SSIM_C2)) / \
           ((mu1_sq + mu2_sq + _SSIM_C1) * (sigma1_sq + sigma2_sq + _SSIM_C2))
    smap = smap.mean(dim=1, keepdim=True)

    return smap.squeeze(0) if unbatched else smap


def gaussian_blur(img_bchw, sigma):
    """Separable Gaussian blur with zero padding, radius = ceil(3 sigma)."""
    if sigma <= 0:
        return img_bchw
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=img_bchw.device, dtype=img_bchw.dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    channel = img_bchw.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(channel, 1, 1, 1)
    out = F.conv2d(img_bchw, kx, padding=(0, radius), groups=channel)
    out = F.conv2d(out, ky, padding=(radius, 0), groups=channel)
    return out


class GuidanceMaps:
    """
    Computes the guidance maps named in `names` for a (render, gt) pair.

    DINOv2 is loaded only when "dino" is requested, so ssim/l1 runs do not need
    the local dinov2/ repository. Loading happens in __init__, at the same point of
    the training script where DinoSimilarity used to be created, so a dino run keeps
    the same random-number consumption as the previous code.
    """

    def __init__(self, names, device="cuda", dino_model="dinov2_vits14", blur_sigma=4.0, dino_chunk=0):
        self.names = {n for n in names if n and n != "none"}
        unknown = self.names - set(SIGNALS)
        if unknown:
            raise ValueError(f"Unknown guidance signal(s): {sorted(unknown)}; choose from {SIGNALS}")
        self.device = device
        self.blur_sigma = float(blur_sigma)
        self.dino = None
        if "dino" in self.names:
            from utils.dino_utils import DinoSimilarity
            self.dino = DinoSimilarity(model_name=dino_model, device=device, channel_chunk=dino_chunk)

    @torch.no_grad()
    def goodness(self, name, render_chw, gt_chw):
        """
        Args:
            render_chw, gt_chw: [3, H, W] in [0, 1]
        Returns:
            [1, H, W] goodness map (higher = better)
        """
        if name not in self.names:
            raise ValueError(f"Signal '{name}' was not requested at construction time")

        if name == "dino":
            if self.dino is None:
                raise RuntimeError("DINOv2 was released (densification finished) or never loaded")
            return self.dino.cosine_map(render_chw.unsqueeze(0), gt_chw.unsqueeze(0)).squeeze(0).detach()

        if name == "ssim":
            return ssim_map(render_chw, gt_chw).detach()

        err = (render_chw - gt_chw).abs().mean(dim=0, keepdim=True)  # [1, H, W]
        if name == "l1":
            return (1.0 - err).detach()

        # l1_blur
        err = gaussian_blur(err.unsqueeze(0), self.blur_sigma).squeeze(0)
        return (1.0 - err).detach()

    def release(self):
        """Free DINOv2 after the densification phase (same as the previous `del dino_sim`)."""
        if self.dino is not None:
            del self.dino
            self.dino = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
