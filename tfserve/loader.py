"""
Module that handles loading a tensorflow model in several different forms.
"""
import os
import tensorflow as tf


def load_model(model_path):
    """
    Loads a tensorflow model return a tf.Session running on the loaded model (the graph).

    :param str model_path: It can be a `.pb` file or directory containing checkpoint files.

    :return: tf.Session running the model graph.
    """
    if model_path is None:
        raise ValueError("model_path must not be None")

    if not os.path.exists(model_path):
        raise ValueError("model_path must exist")

    if os.path.isfile(model_path) and model_path.endswith(".pb"):
        return _load_pb(model_path)

    if os.path.isdir(model_path):
        for f in os.listdir(model_path):
            if f.endswith(".pb"):
                return _load_pb(os.path.join(model_path, f))

        return _load_ckpt(model_path)


def _load_pb(model_path):
    """
    Loads from a '.pb' model file.
    """
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with sess.graph.as_default():
            tf.import_graph_def(graph_def)

    return sess


def _load_ckpt(model_dir):
    """
    Loads from a checkpoint directory.
    """
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():
        ckpt_path = tf.train.latest_checkpoint(model_dir)
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
        saver.restore(sess, ckpt_path)
        return sess
