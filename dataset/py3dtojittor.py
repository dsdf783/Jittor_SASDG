import jittor as jt
from jittor import init
from scipy.spatial.transform import Rotation as R 
import numpy as np
from jittor import nn 
from scipy.signal import normalize

def _sqrt_positive_part(x):
    ret = jt.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = jt.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix): 
    
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = jt.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        jt.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = jt.stack(
        [
            jt.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            jt.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            jt.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            jt.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = jt.array(0.1).type_as(q_abs)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].maximum(flr))
    out = quat_candidates[
        nn.one_hot(jt.argmax(q_abs,dim=-1)[0], num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)



def quaternion_to_matrix(quaternions:jt.Var)->jt.Var:
    r, i, j, k = jt.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = jt.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))




def axis_angle_to_quaternion(axis_angle:jt.Var) -> jt.Var:   #zxy，ok
    angles = jt.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = jt.zeros_like(angles)
    mask = jt.logical_not(small_angles)
    sin_half_angles_over_angles = jt.where(
        mask, 
        jt.sin(half_angles) / angles, 
        sin_half_angles_over_angles
    )
    
    sin_half_angles_over_angles = jt.where(
        small_angles, 
        0.5 - (angles * angles) / 48,
        sin_half_angles_over_angles
    )
    
    quaternions = jt.concat(
        [jt.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )  
    return quaternions

def quaternion_raw_multiply(a:jt.Var, b:jt.Var)-> jt.Var: 
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = jt.unbind(a, dim=-1)
    bw, bx, by, bz = jt.unbind(b, dim=-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return jt.stack([ow, ox, oy, oz], dim=-1)

def standardize_quaternion(quaternions): 
    return jt.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_multiply(a, b): 
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def quaternion_to_axis_angle(quaternions): 
    norms = jt.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = jt.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = jt.zeros_like(angles)
    mask = jt.logical_not(small_angles)
    
    sin_half_angles_over_angles = jt.where(
        mask, 
        jt.sin(half_angles) / angles, 
        sin_half_angles_over_angles
    )
    
    sin_half_angles_over_angles = jt.where(
        small_angles, 
        0.5 - (angles * angles) / 48, 
        sin_half_angles_over_angles
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def matrix_to_rotation_6d(matrix):
    batch_dim = matrix.size()[:-2]
    
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = jt.normalize(a1, dim=-1)
    b2 = a2-(b1 * a2).sum(-1, keepdim=True) * b1
    b2 = jt.normalize(b2, dim=-1)
    b3 = jt.cross(b1, b2, dim=-1)
    return jt.stack([b1, b2, b3], dim=-2)

def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    scaling = jt.array([1, -1, -1, -1])
    return quaternion * scaling


def quaternion_apply(quaternion,point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = jt.concat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]