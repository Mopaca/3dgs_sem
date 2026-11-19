#
# Projected (screen-space) footprint of each Gaussian, computed in PyTorch with the same
# formulas as the rasterizer (cuda_rasterizer/forward.cu: computeCov2D and the radius).
#
# The rasterizer only returns the integer radius of the long axis. For the visibility
# measure "contribution sum / footprint area" a disc of that radius over-estimates the
# footprint of elongated Gaussians, so the area of the projected 3-sigma ellipse is used:
#
#     area = pi * (3 sqrt(l1)) * (3 sqrt(l2)) = 9 pi sqrt(det(cov2D))
#
# The smallest (oriented) rectangle that covers this ellipse has area 36 sqrt(det(cov2D)),
# i.e. the same quantity up to the constant factor 4 / pi.
#

import math

import torch


def _quat_to_rotmat(q):
    """q: [n, 4] (w, x, y, z), normalised. Same convention as utils.general_utils.build_rotation."""
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], dim=1)
    return R.view(-1, 3, 3)


@torch.no_grad()
def projected_cov2d(xyz, scaling, rotation, world_view_transform, tanfovx, tanfovy, width, height):
    """
    2D covariance of the projected Gaussians, identical to computeCov2D (including the
    +0.3 low-pass term on the diagonal).

    Args:
        xyz:      [n, 3] centres (world)
        scaling:  [n, 3] activated scales
        rotation: [n, 4] normalised quaternions (w, x, y, z)
        world_view_transform: [4, 4] as stored by scene.cameras.Camera (row-vector convention)
    Returns:
        a, b, c: [n] elements of [[a, b], [b, c]]
    """
    M = world_view_transform
    Rm = M[:3, :3]                      # p_view = p_world @ Rm + M[3, :3]
    t = xyz @ Rm + M[3, :3]
    tz = t[:, 2]
    limx, limy = 1.3 * tanfovx, 1.3 * tanfovy
    tx = torch.clamp(t[:, 0] / tz, -limx, limx) * tz
    ty = torch.clamp(t[:, 1] / tz, -limy, limy) * tz
    fx = width / (2.0 * tanfovx)
    fy = height / (2.0 * tanfovy)

    Rc = Rm.t()                         # camera rotation (rows = camera axes in world coordinates)
    # rows of J @ Rc, one pair per Gaussian
    m1 = (fx / tz).unsqueeze(1) * Rc[0].unsqueeze(0) - (fx * tx / (tz * tz)).unsqueeze(1) * Rc[2].unsqueeze(0)
    m2 = (fy / tz).unsqueeze(1) * Rc[1].unsqueeze(0) - (fy * ty / (tz * tz)).unsqueeze(1) * Rc[2].unsqueeze(0)

    R = _quat_to_rotmat(rotation)       # Sigma = R S S^T R^T
    u1 = torch.einsum("ni,nij->nj", m1, R) * scaling
    u2 = torch.einsum("ni,nij->nj", m2, R) * scaling
    a = (u1 * u1).sum(dim=1) + 0.3
    b = (u1 * u2).sum(dim=1)
    c = (u2 * u2).sum(dim=1) + 0.3
    return a, b, c


@torch.no_grad()
def projected_ellipse_area(xyz, scaling, rotation, camera):
    """
    Returns:
        area:   [n] area (px^2) of the projected 3-sigma ellipse, 9 pi sqrt(det)
        radius: [n] long-axis radius computed like the rasterizer, ceil(3 sqrt(lambda_max));
                used only to check this function against the rasterizer's radii
    """
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    a, b, c = projected_cov2d(xyz, scaling, rotation, camera.world_view_transform,
                              tanfovx, tanfovy, int(camera.image_width), int(camera.image_height))
    det = a * c - b * b
    area = 9.0 * math.pi * torch.sqrt(det.clamp_min(1e-12))
    mid = 0.5 * (a + c)
    lam_max = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam_max))
    return area, radius
