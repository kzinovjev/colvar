import numpy as np
from colvar.geometry import get_d, get_angle, get_dihedral, get_point_plane


def eval_constant(x, value):
    gradient = np.zeros(x.shape)
    return value, gradient


def eval_cartesian(x, index):
    gradient = np.zeros(x.shape)
    gradient[index] = 1
    return x[index], gradient


def eval_geometric(x, centers, f):
    centers_values, centers_jacobian = evaluate_schema(x, centers)
    value, grad = f(*centers_values)
    return value, np.einsum('ij,ijk->k', grad, centers_jacobian)


def geometric_evaluator(f):
    return lambda x, centers: eval_geometric(x, centers, f)


def eval_sigmoid(x, colvar, L, k, x0):
    cv_value, cv_grad = evaluate_schema(x, colvar)
    e = np.exp(k * (cv_value - x0))
    value = e / (e + 1)
    grad = k * value * (1 - value)
    return value * L, cv_grad * grad * L


def eval_linear(x, colvars, weights, normalize):
    cv_values, cv_jacobian = evaluate_schema(x, colvars)
    w_values, w_jacobian = evaluate_schema(x, weights)
    if normalize:
        sumw = np.sum(w_values)
        sumw_grad = np.sum(w_jacobian, axis=0)
    else:
        sumw, sumw_grad = eval_constant(x, 1)
    value = w_values.dot(cv_values) / sumw
    grad = (w_values.dot(cv_jacobian) + cv_values.dot(w_jacobian) -
            value * sumw_grad) / sumw
    return value, grad


EVALUATORS = {"constant": eval_constant,
              "x": eval_cartesian,
              "distance": geometric_evaluator(get_d),
              "angle": geometric_evaluator(get_angle),
              "dihedral": geometric_evaluator(get_dihedral),
              "point_plane": geometric_evaluator(get_point_plane),
              "sigmoid": eval_sigmoid,
              "linear": eval_linear}


def stack_colvars(colvars):
    return list(map(np.array, zip(*colvars)))


def evaluate_schema(x, schema):
    if isinstance(schema, list):
        return stack_colvars([evaluate_schema(x, _) for _ in schema])
    return EVALUATORS[schema["type"]](x, **schema["params"])
