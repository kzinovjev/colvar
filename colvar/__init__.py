import json
from .compilers import compile_schema
from .evaluators import evaluate_schema


class ColVar:

    def __init__(self, schema):
        self._schema = compile_schema(schema)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            return cls(json.load(f))

    def __call__(self, x):
        return evaluate_schema(x, self._schema)
