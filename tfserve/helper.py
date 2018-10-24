"""
Module provinding a function that estimates a model's input/output tensor names.
"""

from tfserve.loader import load_model


out_hints = ["softmax", "sigmoid", "out", "output", "prediction",
             "probability", "prob", "inference"]


def estimate_io_tensors(model_path):
    """
    Prints estimates input/output tensor names that could be used later
    with the TFServeApp class to serve the model.

    :param str model_path: can be a '.pb' model file or a checkpoint directory.
    """
    sess = load_model(model_path)

    print("Possible INPUT tensors:")
    _print_tensors(_possible_input_tensors(sess.graph))

    print()

    print("Possible OUTPUT tensors:")
    _print_tensors(_possible_output_tensors(sess.graph))

    sess.close()

def _possible_input_tensors(graph):
    return [
        op.outputs[0] for op in graph.get_operations()
        if op.type == "Placeholder"
    ]

def _possible_output_tensors(graph):
    return [
        output for op in graph.get_operations()
        for output in op.outputs
        if any(hint in op.name.lower() for hint in out_hints)
    ]

def _print_tensors(tensors):
    c1_width = len("name")
    c2_width = len("shape")
    for t in tensors:
        c1_width = max(len(t.name), c1_width)
        c2_width = max(len(str(t.shape)), c2_width)
    def print_line(c1, c2, c3):
        print("  %s  %s  %s" % (c1.ljust(c1_width), c2.ljust(c2_width), c3))
    print_line("name", "shape", "dtype")
    for t in tensors:
        print_line(t.name, str(t.shape), t.dtype.name)
