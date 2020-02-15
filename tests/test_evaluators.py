from colvar.compilers import compile_schema
from colvar.evaluators import evaluate_schema
import numpy as np
import numpy.testing as npt


GRADIENT_INCREMENT = 1E-6


def _test_gradient(x, schema):
    grad_analytic = evaluate_schema(x, schema)[1]
    for i in range(len(x)):
        p_x = x.copy()
        p_x[i] += GRADIENT_INCREMENT
        p_value = evaluate_schema(p_x, schema)[0]

        m_x = x.copy()
        m_x[i] -= GRADIENT_INCREMENT
        m_value = evaluate_schema(m_x, schema)[0]

        grad_numeric = (p_value - m_value) / (GRADIENT_INCREMENT * 2)
        npt.assert_almost_equal(grad_analytic[i], grad_numeric)


def test_evaluate_distance():
    schema = {
        "type": "distance",
        "params": {
            "centers": [
                [
                    {"type": "x", "params": {"index": 3}},
                    {"type": "x", "params": {"index": 4}},
                    {"type": "x", "params": {"index": 5}}
                ],
                [
                    {"type": "x", "params": {"index": 6}},
                    {"type": "x", "params": {"index": 7}},
                    {"type": "x", "params": {"index": 8}}
                ]
            ]
        }
    }

    x = np.array([0., 0., 0., 0., 0., 1., 0., 1., 1.])
    npt.assert_almost_equal(evaluate_schema(x, schema)[0], 1)
    _test_gradient(x, schema)


def test_evaluate_angle():
    schema = {
        "type": "angle",
        "params": {
            "centers": [
                [
                    {"type": "x", "params": {"index": 0}},
                    {"type": "x", "params": {"index": 1}},
                    {"type": "x", "params": {"index": 2}}
                ],
                [
                    {"type": "x", "params": {"index": 3}},
                    {"type": "x", "params": {"index": 4}},
                    {"type": "x", "params": {"index": 5}}
                ],
                [
                    {"type": "x", "params": {"index": 6}},
                    {"type": "x", "params": {"index": 7}},
                    {"type": "x", "params": {"index": 8}}
                ]
            ]
        }
    }

    x = np.array([0., 0., 0., 0., 0., 1., 0., 1., 1.])
    npt.assert_almost_equal(evaluate_schema(x, schema)[0], 90)
    _test_gradient(x, schema)


def test_evaluate_dihedral():
    schema = {
        "type": "dihedral",
        "params": {
            "centers": [
                [
                    {"type": "x", "params": {"index": 0}},
                    {"type": "x", "params": {"index": 1}},
                    {"type": "x", "params": {"index": 2}}
                ],
                [
                    {"type": "x", "params": {"index": 3}},
                    {"type": "x", "params": {"index": 4}},
                    {"type": "x", "params": {"index": 5}}
                ],
                [
                    {"type": "x", "params": {"index": 6}},
                    {"type": "x", "params": {"index": 7}},
                    {"type": "x", "params": {"index": 8}}
                ],
                [
                    {"type": "x", "params": {"index": 9}},
                    {"type": "x", "params": {"index": 10}},
                    {"type": "x", "params": {"index": 11}}
                ]
            ]
        }
    }

    x = np.array([0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1.])
    npt.assert_almost_equal(evaluate_schema(x, schema)[0], 90)
    _test_gradient(x, schema)


def test_evaluate_point_plane():
    schema = {
        "type": "point_plane",
        "params": {
            "centers": [
                [
                    {"type": "x", "params": {"index": 0}},
                    {"type": "x", "params": {"index": 1}},
                    {"type": "x", "params": {"index": 2}}
                ],
                [
                    {"type": "x", "params": {"index": 3}},
                    {"type": "x", "params": {"index": 4}},
                    {"type": "x", "params": {"index": 5}}
                ],
                [
                    {"type": "x", "params": {"index": 6}},
                    {"type": "x", "params": {"index": 7}},
                    {"type": "x", "params": {"index": 8}}
                ],
                [
                    {"type": "x", "params": {"index": 9}},
                    {"type": "x", "params": {"index": 10}},
                    {"type": "x", "params": {"index": 11}}
                ]
            ]
        }
    }

    x = np.array([0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1.])
    npt.assert_almost_equal(evaluate_schema(x, schema)[0], 1)
    _test_gradient(x, schema)


def test_evaluate_sigmoid():
    schema = {
        "type": "sigmoid",
        "params": {
            "colvar": {
                "type": "distance",
                "params": {
                    "centers": [
                        [
                            {"type": "x", "params": {"index": 3}},
                            {"type": "x", "params": {"index": 4}},
                            {"type": "x", "params": {"index": 5}}
                        ],
                        [
                            {"type": "x", "params": {"index": 6}},
                            {"type": "x", "params": {"index": 7}},
                            {"type": "x", "params": {"index": 8}}
                        ]
                    ]
                }
            },
            "L": 2,
            "k": -1,
            "x0": 2
        }
    }

    x = np.array([0., 0., 0., 0., 0., 1., 0., 1., 1.])
    npt.assert_almost_equal(evaluate_schema(x, schema)[0], 1.46211715)
    _test_gradient(x, schema)


def test_evaluate_linear():
    raw_schema = {
        "type": "linear",
        "colvars": [
            {"type": "angle", "atoms": [0, 5, 6]},
            {"type": "angle", "atoms": [2, 5, 6]},
            {"type": "angle", "atoms": [3, 5, 6]},
            {"type": "angle", "atoms": [4, 5, 6]}
        ],
        "weights": [
            {
                "type": "sigmoid",
                "colvar": {"type": "distance", "atoms": [0, 5]},
                "L": 1,
                "k": 1,
                "x0": 1.5
            },
            {
                "type": "sigmoid",
                "colvar": {"type": "distance", "atoms": [2, 5]},
                "L": 1,
                "k": 1,
                "x0": 1.5
            },
            {
                "type": "sigmoid",
                "colvar": {"type": "distance", "atoms": [3, 5]},
                "L": 1,
                "k": 1,
                "x0": 1.5
            },
            {
                "type": "sigmoid",
                "colvar": {"type": "distance", "atoms": [4, 5]},
                "L": 1,
                "k": 1,
                "x0": 1.5
            }
        ],
        "normalize": True
    }
    schema = compile_schema(raw_schema)

    x = np.array([[-4.62878267,    1.25606861,    0.95459788],
                  [-4.85261637,    2.15380812,    0.37457524],
                  [-4.27740626,    2.99438311,    0.76831501],
                  [-4.53003946,    1.95346377,   -0.88649386],
                  [-5.91912714,    2.37708958,    0.44643735],
                  [-4.21407574,    1.75722671,   -2.12170961],
                  [-3.93964920,    2.62885961,   -2.44704966]]).flatten()
    _test_gradient(x, schema)
