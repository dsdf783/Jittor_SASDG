import jittor as jt

smpl_joints = [
    "root",  # 0
    "lhip",
    "rhip",
    "belly",  # 1 2 3
    "lknee",
    "rknee",
    "spine",  # 4 5 6
    "lankle",
    "rankle",
    "chest",  # 7 8 9
    "ltoes",
    "rtoes",
    "neck",  # 10 11 12
    "linshoulder",
    "rinshoulder",  # 13 14
    "head",
    "lshoulder",
    "rshoulder",  # 15 16 17
    "lelbow",
    "relbow",  # 18 19
    "lwrist",
    "rwrist",  # 20 21
    "lhand",
    "rhand",  # 22 23
]


def joint_indices_to_channel_indices(indices):
    out = []
    for index in indices:
        out += list(range(3 + 3 * index, 3 + 3 * index + 3))
    return out

def get_first_last_mask(posq_batch, start_width=1, end_width=1):
    mask = jt.zeros_like(posq_batch)
    mask[..., :start_width, :] = 1
    mask[..., - end_width:, :] = 1
    return mask

def get_first_mask(posq_batch, start_width=1):
    mask = jt.zeros_like(posq_batch)
    mask[..., :start_width, :] = 1
    return mask

def get_middle_mask(posq_batch, start=0, end=-1):
    mask = jt.zeros_like(posq_batch)
    mask[..., start:end, :] = 1
    return mask

def lowerbody_mask(posq_batch):
    mask = jt.zeros_like(posq_batch)
    lowerbody_indices = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    root_traj_indices = [0, 1, 2]
    lowerbody_indices = (
        joint_indices_to_channel_indices(lowerbody_indices) + root_traj_indices
    )
    mask[..., :, lowerbody_indices] = 1
    return mask

def upperbody_mask(posq_batch):
    mask = jt.zeros_like(posq_batch)
    upperbody_indices = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    root_traj_indices = [0, 1, 2]
    upperbody_indices = (
        joint_indices_to_channel_indices(upperbody_indices) + root_traj_indices
    )
    mask[..., :, upperbody_indices] = 1
    return mask