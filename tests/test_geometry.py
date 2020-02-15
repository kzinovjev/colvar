import numpy as np
import numpy.testing as npt
from colvar import geometry
import copy


GRADIENT_INCREMENT = 1E-6


def _test_gradient(f, *args):
    grad_analytic = f(*args)[1]

    for arg_i, arg in enumerate(args):
        for x_i in range(len(arg)):
            p_args = copy.deepcopy(args)
            p_args[arg_i][x_i] += GRADIENT_INCREMENT
            p_value = f(*p_args)[0]

            m_args = copy.deepcopy(args)
            m_args[arg_i][x_i] -= GRADIENT_INCREMENT
            m_value = f(*m_args)[0]

            grad_numeric = (p_value - m_value) / (GRADIENT_INCREMENT * 2)
            npt.assert_almost_equal(grad_analytic[arg_i][x_i], grad_numeric)


TEST_A = np.array([0., 0., 0.])
TEST_B = np.array([1., 0., 0.])
TEST_C = np.array([1., 0., 1.])
TEST_D = np.array([1., 1., 1.])


def test_d():
    npt.assert_almost_equal(geometry.get_d(TEST_A, TEST_B)[0], 1)
    _test_gradient(geometry.get_d, TEST_A, TEST_B)


def test_get_angle():
    npt.assert_almost_equal(geometry.get_angle(TEST_A, TEST_B, TEST_C)[0],
                            90)
    a = np.array([-4.62878267,    1.25606861,    0.95459788])
    b = np.array([-4.21407574,    1.75722671,   -2.12170961])
    c = np.array([-3.93964920,    2.62885961,   -2.44704966])
    _test_gradient(geometry.get_angle, a, b, c)


def test_get_dihedral():
    npt.assert_almost_equal(
        geometry.get_dihedral(TEST_A, TEST_B, TEST_C, TEST_D)[0],
        90
    )
    _test_gradient(geometry.get_dihedral, TEST_A, TEST_B, TEST_C, TEST_D)


def test_get_point_plane():
    npt.assert_almost_equal(
        geometry.get_point_plane(TEST_D, TEST_A, TEST_B, TEST_C)[0],
        -1
    )
    _test_gradient(geometry.get_point_plane, TEST_D, TEST_A, TEST_B, TEST_C)
