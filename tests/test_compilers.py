from colvar import compilers


def test_simple_distance():
    raw = {
        "type": "distance",
        "atoms": [1, 2]
    }

    compiled = {
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

    assert compilers.compile_schema(raw) == compiled


def test_centers_distance_no_weights():
    raw = {
        "type": "distance",
        "centers": [
            {"atoms": [1, 2, 3]},
            {"atoms": [5, 6]}
        ]
    }

    compiled = {
        "type": "distance",
        "params": {
            "centers": [
                [
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 3}},
                                {"type": "x", "params": {"index": 6}},
                                {"type": "x", "params": {"index": 9}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 1}}
                            ],
                            "normalize": True
                        }
                    },
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 4}},
                                {"type": "x", "params": {"index": 7}},
                                {"type": "x", "params": {"index": 10}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 1}}
                            ],
                            "normalize": True
                        }
                    },
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 5}},
                                {"type": "x", "params": {"index": 8}},
                                {"type": "x", "params": {"index": 11}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 1}}
                            ],
                            "normalize": True
                        }
                    }
                ],
                [
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 15}},
                                {"type": "x", "params": {"index": 18}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 1}}
                            ],
                            "normalize": True
                        }
                    },
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 16}},
                                {"type": "x", "params": {"index": 19}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 1}}
                            ],
                            "normalize": True
                        }
                    },
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 17}},
                                {"type": "x", "params": {"index": 20}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 1}}
                            ],
                            "normalize": True
                        }
                    }
                ]
            ]
        }
    }

    assert compilers.compile_schema(raw) == compiled


def test_weighted_distance():
    raw = {
        "type": "distance",
        "centers": [
            {"atoms": [1, 2, 3], "weights":  [12, 12, 16]},
            {"atoms": [5, 6], "weights":  [1, 16]}
        ]
    }

    compiled = {
        "type": "distance",
        "params": {
            "centers": [
                [
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 3}},
                                {"type": "x", "params": {"index": 6}},
                                {"type": "x", "params": {"index": 9}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 12}},
                                {"type": "constant", "params": {"value": 12}},
                                {"type": "constant", "params": {"value": 16}}
                            ],
                            "normalize": True
                        }
                    },
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 4}},
                                {"type": "x", "params": {"index": 7}},
                                {"type": "x", "params": {"index": 10}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 12}},
                                {"type": "constant", "params": {"value": 12}},
                                {"type": "constant", "params": {"value": 16}}
                            ],
                            "normalize": True
                        }
                    },
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 5}},
                                {"type": "x", "params": {"index": 8}},
                                {"type": "x", "params": {"index": 11}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 12}},
                                {"type": "constant", "params": {"value": 12}},
                                {"type": "constant", "params": {"value": 16}}
                            ],
                            "normalize": True
                        }
                    }
                ],
                [
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 15}},
                                {"type": "x", "params": {"index": 18}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 16}}
                            ],
                            "normalize": True
                        }
                    },
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 16}},
                                {"type": "x", "params": {"index": 19}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 16}}
                            ],
                            "normalize": True
                        }
                    },
                    {
                        "type": "linear",
                        "params": {
                            "colvars": [
                                {"type": "x", "params": {"index": 17}},
                                {"type": "x", "params": {"index": 20}}
                            ],
                            "weights": [
                                {"type": "constant", "params": {"value": 1}},
                                {"type": "constant", "params": {"value": 16}}
                            ],
                            "normalize": True
                        }
                    }
                ]
            ]
        }
    }

    assert compilers.compile_schema(raw) == compiled


def test_linear():
    raw = {
        "type": "linear",
        "colvars": [
            {"type": "angle", "atoms": [1, 6, 7]},
            {"type": "angle", "atoms": [3, 6, 7]},
            {"type": "angle", "atoms": [4, 6, 7]},
            {"type": "angle", "atoms": [5, 6, 7]}
        ],
        "weights": [
            {
                "type": "sigmoid",
                "colvar": {"type": "distance", "atoms": [1, 6]},
                "L": 1,
                "k": 10,
                "x0": 1.5
            },
            {
                "type": "sigmoid",
                "colvar": {"type": "distance", "atoms": [3, 6]},
                "L": 1,
                "k": 10,
                "x0": 1.5
            },
            {
                "type": "sigmoid",
                "colvar": {"type": "distance", "atoms": [4, 6]},
                "L": 1,
                "k": 10,
                "x0": 1.5
            },
            {
                "type": "sigmoid",
                "colvar": {"type": "distance", "atoms": [5, 6]},
                "L": 1,
                "k": 10,
                "x0": 1.5
            }
        ],
        "normalize": True
    }

    compiled = {
        "type": "linear",
        "params": {
            "colvars": [
                {
                    "type": "angle",
                    "params": {
                        "centers": [
                            [
                                {"type": "x", "params": {"index": 3}},
                                {"type": "x", "params": {"index": 4}},
                                {"type": "x", "params": {"index": 5}}
                            ],
                            [
                                {"type": "x", "params": {"index": 18}},
                                {"type": "x", "params": {"index": 19}},
                                {"type": "x", "params": {"index": 20}}
                            ],
                            [
                                {"type": "x", "params": {"index": 21}},
                                {"type": "x", "params": {"index": 22}},
                                {"type": "x", "params": {"index": 23}}
                            ]
                        ]
                    }
                },
                {
                    "type": "angle",
                    "params": {
                        "centers": [
                            [
                                {"type": "x", "params": {"index": 9}},
                                {"type": "x", "params": {"index": 10}},
                                {"type": "x", "params": {"index": 11}}
                            ],
                            [
                                {"type": "x", "params": {"index": 18}},
                                {"type": "x", "params": {"index": 19}},
                                {"type": "x", "params": {"index": 20}}
                            ],
                            [
                                {"type": "x", "params": {"index": 21}},
                                {"type": "x", "params": {"index": 22}},
                                {"type": "x", "params": {"index": 23}}
                            ]
                        ]
                    }
                },
                {
                    "type": "angle",
                    "params": {
                        "centers": [
                            [
                                {"type": "x", "params": {"index": 12}},
                                {"type": "x", "params": {"index": 13}},
                                {"type": "x", "params": {"index": 14}}
                            ],
                            [
                                {"type": "x", "params": {"index": 18}},
                                {"type": "x", "params": {"index": 19}},
                                {"type": "x", "params": {"index": 20}}
                            ],
                            [
                                {"type": "x", "params": {"index": 21}},
                                {"type": "x", "params": {"index": 22}},
                                {"type": "x", "params": {"index": 23}}
                            ]
                        ]
                    }
                },
                {
                    "type": "angle",
                    "params": {
                        "centers": [
                            [
                                {"type": "x", "params": {"index": 15}},
                                {"type": "x", "params": {"index": 16}},
                                {"type": "x", "params": {"index": 17}}
                            ],
                            [
                                {"type": "x", "params": {"index": 18}},
                                {"type": "x", "params": {"index": 19}},
                                {"type": "x", "params": {"index": 20}}
                            ],
                            [
                                {"type": "x", "params": {"index": 21}},
                                {"type": "x", "params": {"index": 22}},
                                {"type": "x", "params": {"index": 23}}
                            ]
                        ]
                    }
                }
            ],
            "weights": [
                {
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
                                        {"type": "x", "params": {"index": 18}},
                                        {"type": "x", "params": {"index": 19}},
                                        {"type": "x", "params": {"index": 20}}
                                    ]
                                ],
                            }
                        },
                        "L": 1,
                        "k": 10,
                        "x0": 1.5
                    }
                },
                {
                    "type": "sigmoid",
                    "params": {
                        "colvar": {
                            "type": "distance",
                            "params": {
                                "centers": [
                                    [
                                        {"type": "x", "params": {"index": 9}},
                                        {"type": "x", "params": {"index": 10}},
                                        {"type": "x", "params": {"index": 11}}
                                    ],
                                    [
                                        {"type": "x", "params": {"index": 18}},
                                        {"type": "x", "params": {"index": 19}},
                                        {"type": "x", "params": {"index": 20}}
                                    ]
                                ]
                            }
                        },
                        "L": 1,
                        "k": 10,
                        "x0": 1.5
                    }
                },
                {
                    "type": "sigmoid",
                    "params": {
                        "colvar": {
                            "type": "distance",
                            "params": {
                                "centers": [
                                    [
                                        {"type": "x", "params": {"index": 12}},
                                        {"type": "x", "params": {"index": 13}},
                                        {"type": "x", "params": {"index": 14}}
                                    ],
                                    [
                                        {"type": "x", "params": {"index": 18}},
                                        {"type": "x", "params": {"index": 19}},
                                        {"type": "x", "params": {"index": 20}}
                                    ]
                                ]
                            }
                        },
                        "L": 1,
                        "k": 10,
                        "x0": 1.5
                    }
                },
                {
                    "type": "sigmoid",
                    "params": {
                        "colvar": {
                            "type": "distance",
                            "params": {
                                "centers": [
                                    [
                                        {"type": "x", "params": {"index": 15}},
                                        {"type": "x", "params": {"index": 16}},
                                        {"type": "x", "params": {"index": 17}}
                                    ],
                                    [
                                        {"type": "x", "params": {"index": 18}},
                                        {"type": "x", "params": {"index": 19}},
                                        {"type": "x", "params": {"index": 20}}
                                    ]
                                ]
                            }
                        },
                        "L": 1,
                        "k": 10,
                        "x0": 1.5
                    }
                }
            ],
            "normalize": True
        }
    }
    assert compilers.compile_schema(raw) == compiled
