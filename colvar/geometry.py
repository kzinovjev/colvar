import numpy as np


deg = np.rad2deg


def get_v(a, b):
    n = len(a)
    v = b - a
    return v, -np.identity(n), np.identity(n)


def _get_d(v):
    d = np.linalg.norm(v)
    return d, v / d


def get_d(a, b):
    v, grad_v_a, grad_v_b = get_v(a, b)
    d, grad_d_v = _get_d(v)
    return d, (grad_v_a.dot(grad_d_v), grad_v_b.dot(grad_d_v))


def _get_nvec(v):
    d, grad_d = _get_d(v)
    return v / d, np.identity(3) / d - np.outer(grad_d, v) / d ** 2


def _dot(v1, v2):
    return np.dot(v1, v2), v2, v1


def _get_angle(v1, v2):
    n1, grad_n1 = _get_nvec(v1)
    n2, grad_n2 = _get_nvec(v2)

    cos, grad_cos_n1, grad_cos_n2 = _dot(n1, n2)
    phi = np.arccos(cos)

    if 1 - cos ** 2 < 1E-9:
        return phi, np.zeros(3), np.zeros(3)
    d_arccos = - 1. / np.sqrt(1 - cos ** 2)
    grad_v1 = d_arccos * grad_n1.dot(grad_cos_n1)
    grad_v2 = d_arccos * grad_n2.dot(grad_cos_n2)
    return phi, grad_v1, grad_v2


def get_angle(a, b, c):
    ba, grad_ba_b, grad_ba_a = get_v(b, a)
    bc, grad_bc_b, grad_bc_c = get_v(b, c)
    phi, grad_phi_ba, grad_phi_bc = _get_angle(ba, bc)

    grad_a = grad_ba_a.dot(grad_phi_ba)
    grad_b = grad_ba_b.dot(grad_phi_ba) + grad_bc_b.dot(grad_phi_bc)
    grad_c = grad_bc_c.dot(grad_phi_bc)

    return deg(phi), (deg(grad_a), deg(grad_b), deg(grad_c))


def _get_cross(v1, v2):
    cross = np.cross(v1, v2)
    return cross, -np.cross(v2, np.identity(3)), np.cross(v1, np.identity(3))


def _get_dihedral(v1, v2, v3):
    w1, grad_w1_v1, grad_w1_v2 = _get_cross(v1, v2)
    w2, grad_w2_v2, grad_w2_v3 = _get_cross(v2, v3)
    phi, grad_phi_w1, grad_phi_w2 = _get_angle(w1, w2)

    grad_v1 = grad_w1_v1.dot(grad_phi_w1)
    grad_v2 = grad_w1_v2.dot(grad_phi_w1) + grad_w2_v2.dot(grad_phi_w2)
    grad_v3 = grad_w2_v3.dot(grad_phi_w2)

    return phi, grad_v1, grad_v2, grad_v3


def get_dihedral(a, b, c, d):
    ab, grad_ab_a, grad_ab_b = get_v(a, b)
    bc, grad_bc_b, grad_bc_c = get_v(b, c)
    cd, grad_cd_c, grad_cd_d = get_v(c, d)

    phi, grad_phi_ab, grad_phi_bc, grad_phi_cd = _get_dihedral(ab, bc, cd)

    grad_a = grad_ab_a.dot(grad_phi_ab)
    grad_b = grad_ab_b.dot(grad_phi_ab) + grad_bc_b.dot(grad_phi_bc)
    grad_c = grad_bc_c.dot(grad_phi_bc) + grad_cd_c.dot(grad_phi_cd)
    grad_d = grad_cd_d.dot(grad_phi_cd)

    return deg(phi), (deg(grad_a), deg(grad_b), deg(grad_c), deg(grad_d))


def _get_point_plane(v1, v2, v3):
    w, grad_v_v2, grad_v_v3 = _get_cross(v2, v3)
    n, grad_n = _get_nvec(w)
    grad_n_v2 = grad_v_v2.dot(grad_n)
    grad_n_v3 = grad_v_v3.dot(grad_n)

    pp, grad_pp_n, grad_pp_v1 = _dot(n, v1)
    grad_pp_v2 = grad_n_v2.dot(grad_pp_n)
    grad_pp_v3 = grad_n_v3.dot(grad_pp_n)

    return pp, grad_pp_v1, grad_pp_v2, grad_pp_v3


def get_point_plane(a, b, c, d):
    ba, grad_ba_b, grad_ba_a = get_v(b, a)
    bc, grad_bc_b, grad_bc_c = get_v(b, c)
    bd, grad_bd_b, grad_bd_d = get_v(b, d)

    pp, grad_pp_ba, grad_pp_bc, grad_pp_bd = _get_point_plane(ba, bc, bd)
    grad_a = grad_ba_a.dot(grad_pp_ba)
    grad_b = (grad_ba_b.dot(grad_pp_ba) +
              grad_bc_b.dot(grad_pp_bc) +
              grad_bd_b.dot(grad_pp_bd))
    grad_c = grad_bc_c.dot(grad_pp_bc)
    grad_d = grad_bd_d.dot(grad_pp_bd)

    return pp, (grad_a, grad_b, grad_c, grad_d)
