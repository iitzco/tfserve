import re


def check_placeholders(graph, tensors):
    ph = [op for op in graph.get_operations() if op.type == "Placeholder"]
    ph_names = [p.outputs[0].name for p in ph]

    for t in tensors:
        t = smart_tensor_name(t)
        if t not in ph_names:
            raise ValueError("Input node must be placeholder: {}".format(t))


def check_tensors(graph, tensors):
    for t in tensors:
        try:
            graph.get_tensor_by_name(smart_tensor_name(t))
        except (ValueError, KeyError):
            raise ValueError("Non existent tensor in graph: {}".format(t))


def check_input(provided, required, msg):
    for x in provided:
        if x not in required:
            raise ValueError(msg)

    for x in required:
        if x not in provided:
            raise ValueError(msg)


def smart_tensor_name(t):
    m = re.search(r':\d+$', t)
    if m is None:
        return "{}:0".format(t)
    return t
