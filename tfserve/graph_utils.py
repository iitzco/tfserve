import re


def check_placeholders(graph, tensors):
    ph = [op for op in graph.get_operations() if op.type == "Placeholder"]
    ph_names = [p.outputs[0].name for p in ph]

    for t in tensors:
        if t not in ph_names:
            raise ValueError("Input node must be placeholder: {}".format(t))


def check_tensors(graph, tensors):
    for t in tensors:
        try:
            graph.get_tensor_by_name(smart_tensor_name(t))
        except ValueError:
            raise ValueError("Non existent tensor in graph: {}".format(t))
        except KeyError:
            raise ValueError("Non existent tensor in graph: {}".format(t))


def check_input(provided, required):
    for x in provided:
        if x not in required:
            raise ValueError("Encoded tensor that is not required: {}".format(x))

    for x in required:
        if x not in provided:
            raise ValueError("Input tensor missing: {}".format(x))


def smart_tensor_name(t):
    m = re.search(r':\d+$', t)
    if m is None:
        return "{}:0".format(t)
    return t
