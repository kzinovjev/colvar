X_INDEX_INCREMENTS = {"x": 0, "y": 1, "z": 2}


def compile_constant(schema):
    return {"type": "x",
            "params": {"value": schema["value"]}}


def compile_cartesian(schema):
    return {"type": "x",
            "params": {
                "index": schema["atom"] * 3 + X_INDEX_INCREMENTS[schema["type"]]
            }}


def compile_weight(weight):
    if isinstance(weight, dict):
        return compile_schema(weight)
    return {"type": "constant", "params": {"value": weight}}


def compile_weights(weights):
    return list(map(compile_weight, weights))


def compile_center_cartesian(x, atoms, weights):
    return {
        "type": "linear",
        "params": {
            "colvars": [compile_cartesian({"type": x, "atom": atom})
                        for atom in atoms],
            "weights": compile_weights(weights),
            "normalize": True
        }
    }


def compile_atom(atom):
    return [compile_cartesian({"type": _type, "atom": atom})
            for _type in ("x", "y", "z")]


def compile_center(center):
    if isinstance(center, list):
        return compile_schema(center)
    weights = center.get("weights", [1 for _ in center["atoms"]])
    return [compile_center_cartesian(_type, center["atoms"], weights)
            for _type in ("x", "y", "z")]


def compile_geometric(schema):
    if "centers" in schema:
        centers = list(map(compile_center, schema["centers"]))
    else:
        centers = list(map(compile_atom, schema["atoms"]))

    return {"type": schema["type"],
            "params": {"centers": centers}}


def compile_sigmoid(schema):
    return {"type": "sigmoid",
            "params": {"colvar": compile_schema(schema["colvar"]),
                       "L": schema.get("L", 1),
                       "k": schema.get("k", 1),
                       "x0": schema.get("x0", 0)}
            }


def compile_linear(schema):
    return {
        "type": "linear",
        "params": {"colvars": compile_schema(schema["colvars"]),
                   "weights": compile_weights(schema["weights"]),
                   "normalize": schema.get("normalize", False)}
    }


COMPILERS = {"constant": compile_constant,
             "x": compile_cartesian,
             "y": compile_cartesian,
             "z": compile_cartesian,
             "distance": compile_geometric,
             "angle": compile_geometric,
             "dihedral": compile_geometric,
             "point_plane": compile_geometric,
             "sigmoid": compile_sigmoid,
             "linear": compile_linear}


def compile_schema(schema):
    if isinstance(schema, list):
        return list(map(compile_schema, schema))
    if "params" in schema:
        return schema
    return COMPILERS.get(schema["type"], lambda _: _)(schema)
