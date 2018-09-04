from tfserve.loader import load_model


out_hints = ["softmax", "sigmoid", "out", "output", "prediction",
             "probability", "prob", "inference"]


def estimate_io_tensors(model_path):
    sess = load_model(model_path)
    graph = sess.graph

    print("Possible INPUT tensors:")
    ph = [op.outputs[0] for op in graph.get_operations() if op.type == "Placeholder"]
    for p in ph:
        print("\t{}".format(p.name))

    print()
    print("Possible OUTPUT tensors:")
    ops = [n for n in graph.get_operations()]

    for op in ops:
        if any(x in op.name.lower() for x in out_hints):
            for o in op.outputs:
                print("\t{}".format(o.name))

    sess.close()
