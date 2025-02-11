import jittor as jt
from jittor import init
from jittor import nn
from dataset.py3dtojittor import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)

def quat_to_6v(q):
    assert q.shape[- 1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat

def quat_from_6v(q):
    assert (q.shape[- 1] == 6)
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat

def ax_to_6v(q):
    assert (q.shape[- 1] == 3)
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat

def ax_from_6v(q):
    assert (q.shape[- 1] == 6)
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax

def quat_slerp(x, y, a):
    len = jt.sum(x * y, dim=-1)
    neg = (len < 0.0)
    len[neg] = - len[neg]
    y[neg] = - y[neg]
   
    a = jt.zeros_like(x[(..., 0)]) + a
   
    amount0 = jt.zeros_like(a)
    amount1 = jt.zeros_like(a)
    
    linear = (1.0 - len) < 0.01
    not_linear = jt.logical_not(linear)
    omegas = jt.arccos(len[not_linear])
    sinoms = jt.sin(omegas)
    amount0[linear] = 1.0 - a[linear]
    amount0[not_linear] = jt.sin(((1.0 - a[not_linear]) * omegas)) / sinoms

    
    amount1[linear] = a[linear]
    amount1[not_linear] = jt.sin((a[not_linear] * omegas)) / sinoms
   
    amount0 = amount0[(..., None)]
    amount1 = amount1[(..., None)]
    
    res = (amount0 * x) + (amount1 * y)
    return res
