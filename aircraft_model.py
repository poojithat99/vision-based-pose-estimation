"""Synthetic leader-aircraft model: 14 structural features, pinhole projection,
occlusion test, and pose <-> discrete label conversion.

Frames
------
Leader body frame: x forward, y left, z up (metres), origin at aircraft centre.
Follower position r = (x, y, z) is expressed in this frame. The follower camera is
gimballed to look at the leader's origin. Relative pitch/yaw between the aircraft
are ignored (as in the paper's label space); the remaining pose DOF is the roll
`phi` of the camera about its boresight.
"""
import numpy as np
import config as C

# 14 structural features on an F-16-sized aircraft (~15 m long, ~9 m span).
FEATURE_NAMES = [
    "nose", "canopy_top", "left_wingtip", "right_wingtip",
    "left_stab_tip", "right_stab_tip", "fin_tip", "exhaust",
    "left_wing_mid", "right_wing_mid", "intake_lip", "belly_fin",
    "left_pylon", "right_pylon",
]
FEATURES_BODY = np.array([
    [7.5, 0.0, 0.0],     # nose
    [3.0, 0.0, 1.2],     # canopy top
    [-2.0, 4.5, 0.0],    # left wingtip
    [-2.0, -4.5, 0.0],   # right wingtip
    [-7.0, 2.5, 0.0],    # left horizontal stabiliser tip
    [-7.0, -2.5, 0.0],   # right horizontal stabiliser tip
    [-6.0, 0.0, 3.5],    # vertical fin tip
    [-7.5, 0.0, 0.0],    # engine exhaust
    [0.0, 3.0, 0.0],     # left wing leading edge
    [0.0, -3.0, 0.0],    # right wing leading edge
    [2.5, 0.0, -1.2],    # intake lip (belly)
    [-3.0, 0.0, -1.2],   # ventral fin
    [-1.0, 2.5, -0.8],   # left wing pylon
    [-1.0, -2.5, -0.8],  # right wing pylon
])
assert FEATURES_BODY.shape == (C.NUM_FEATURES, 3)

# Fuselage approximated by an ellipsoid (semi-axes, metres); it is the occluder.
FUSELAGE_AXES = np.array([6.0, 0.9, 0.8])


# ----------------------------------------------------------------------------
# Label utilities
# ----------------------------------------------------------------------------
def pose_to_label(pose):
    """(N,4) continuous pose -> (N,) integer label in [0, 4800)."""
    idx = np.floor((pose - C.POSE_LOW) / C.BIN_WIDTH).astype(int)
    idx = np.clip(idx, 0, C.N_BINS - 1)
    nx, ny, nz, nphi = C.N_BINS
    return ((idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]) * nphi + idx[:, 3]


def label_to_indices(label):
    label = np.asarray(label)
    nx, ny, nz, nphi = C.N_BINS
    iphi = label % nphi
    iz = (label // nphi) % nz
    iy = (label // (nphi * nz)) % ny
    ix = label // (nphi * nz * ny)
    return np.stack([ix, iy, iz, iphi], axis=-1)


def label_to_pose(label):
    """Integer label(s) -> pose at the centre of the corresponding bin, shape (...,4)."""
    return C.POSE_LOW + (label_to_indices(label) + 0.5) * C.BIN_WIDTH


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def sample_poses(n, rng):
    """Uniformly sample n continuous poses, rejecting ones too close to the leader."""
    out = []
    got = 0
    while got < n:
        p = rng.uniform(C.POSE_LOW, C.POSE_HIGH, size=(2 * (n - got) + 16, 4))
        p = p[np.linalg.norm(p[:, :3], axis=1) >= C.MIN_RANGE]
        out.append(p)
        got += len(p)
    return np.concatenate(out)[:n]


def camera_basis(r, phi_deg):
    """Camera axes (right, up, forward) in the leader frame for follower position r.

    Boresight points at the leader origin; 'up' is the leader's +z projected
    orthogonal to the boresight, then rolled by phi about the boresight.
    """
    f = -r / np.linalg.norm(r, axis=1, keepdims=True)
    up_ref = np.array([0.0, 0.0, 1.0])
    right0 = np.cross(f, up_ref)
    right0 /= np.linalg.norm(right0, axis=1, keepdims=True)
    up0 = np.cross(right0, f)
    phi = np.deg2rad(phi_deg)[:, None]
    right = np.cos(phi) * right0 + np.sin(phi) * up0
    up = -np.sin(phi) * right0 + np.cos(phi) * up0
    return right, up, f


def occluded(points, r):
    """(N,K) bool: is the segment camera->feature blocked by the fuselage ellipsoid?"""
    cam = r[:, None, :] / FUSELAGE_AXES          # (N,1,3) in unit-sphere space
    pts = points[None, :, :] / FUSELAGE_AXES     # (1,K,3)
    d = pts - cam                                # (N,K,3)
    a = (d * d).sum(-1)
    b = (cam * d).sum(-1)
    c = (cam * cam).sum(-1) - 1.0                # (N,1)
    disc = b * b - a * c
    t1 = (-b - np.sqrt(np.clip(disc, 0, None))) / a
    return (disc > 0) & (t1 > 0) & (t1 < 1)


def render_features(pose, rng=None):
    """Project the 14 features into the follower image.

    pose : (N,4) [x, y, z, phi_deg]
    rng  : if given, simulates an imperfect detector (pixel noise + missed detections)

    Returns (N, 42) float32: per feature [u_norm, v_norm, visible], where u,v are
    pixel coordinates relative to the image centre, normalised to [-1, 1] by the
    half image size. Non-visible features are zeroed (u=v=0, visible=0).
    """
    pose = np.atleast_2d(pose)
    r, phi = pose[:, :3], pose[:, 3]
    right, up, fwd = camera_basis(r, phi)
    d = FEATURES_BODY[None, :, :] - r[:, None, :]            # (N,K,3)
    xc = np.einsum("nkj,nj->nk", d, right)
    yc = np.einsum("nkj,nj->nk", d, up)
    zc = np.einsum("nkj,nj->nk", d, fwd)                     # depth
    u = C.FOCAL_PX * xc / zc                                 # relative to principal point
    v = -C.FOCAL_PX * yc / zc

    if rng is not None:
        u = u + rng.normal(0, C.PIXEL_NOISE_STD, u.shape)
        v = v + rng.normal(0, C.PIXEL_NOISE_STD, v.shape)

    in_frame = (np.abs(u) < C.IMG_W / 2) & (np.abs(v) < C.IMG_H / 2) & (zc > 0)
    vis = in_frame & ~occluded(FEATURES_BODY, r)
    if rng is not None:
        vis &= rng.random(vis.shape) >= C.MISS_PROB

    un = np.where(vis, u / (C.IMG_W / 2), 0.0)
    vn = np.where(vis, v / (C.IMG_H / 2), 0.0)
    out = np.stack([un, vn, vis.astype(float)], axis=-1)     # (N,K,3)
    return out.reshape(len(pose), -1).astype(np.float32)
