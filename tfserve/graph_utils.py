"""
Auxiliary methods that operate on tensorflow graphs.
This are private and should not be used by users.
"""

import re


def check_placeholders(graph, tensors):
    """
    Check that tensors are valid graph placeholders.
    """

    if graph is None or tensors is None:
        raise ValueError("None of the parameters can be None")

    ph = [op for op in graph.get_operations() if op.type == "Placeholder"]
    ph_names = [p.outputs[0].name for p in ph]

    for t in tensors:
        t = smart_tensor_name(t)
        if t not in ph_names:
            raise ValueError("Input node must be placeholder: {}".format(t))


def check_tensors(graph, tensors):
    """
    Check that tensors are valid graph tensors.
    """

    if graph is None or tensors is None:
        raise ValueError("None of the parameters can be None")

    for t in tensors:
        try:
            graph.get_tensor_by_name(smart_tensor_name(t))
        except (ValueError, KeyError):
            raise ValueError("Non existent tensor in graph: {}".format(t))


def check_input(provided, required, msg):
    """
    Check that provided and required set of tensors are equal.
    If not, raises ValueError with msg
    """
    if provided is None or required is None:
        raise ValueError("None of the parameters can be None")

    for x in provided:
        if x not in required:
            raise ValueError(msg)

    for x in required:
        if x not in provided:
            raise ValueError(msg)


def smart_tensor_name(t):
    """
    For check_placeholders and check_tensors functions, the tensors
    can either be the correct tensor name (finishing with :digit) or
    operation names. If user provided operations (common error in tensorflow),
    the method will assing :0 to each of them to get the tensor name.
    """
    m = re.search(r':\d+$', t)
    if m is None:
        return "{}:0".format(t)
    return t
