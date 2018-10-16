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
    graph = sess.graph

    print("Possible INPUT tensor names:")
    ph = [op.outputs[0] for op in graph.get_operations() if op.type == "Placeholder"]
    for p in ph:
        print("\t{}".format(p.name))

    print()
    print("Possible OUTPUT tensor names:")
    ops = [n for n in graph.get_operations()]

    for op in ops:
        if any(x in op.name.lower() for x in out_hints):
            for o in op.outputs:
                print("\t{}".format(o.name))

    sess.close()
