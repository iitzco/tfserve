import os
import tensorflow as tf


def load_model(model_path):
    if os.path.isfile(model_path) and model_path.endswith(".pb"):
        return _load_pb(model_path)

    if os.path.isdir(model_path):
        for f in os.listdir(model_path):
            if f.endswith(".pb"):
                return _load_pb(os.path.join(model_path, f))

        return _load_ckpt(model_path)


def _load_pb(model_path):
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with sess.graph.as_default():
            tf.import_graph_def(graph_def)

    return sess


def _load_ckpt(model_dir):
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
    saver.restore(sess, ckpt_path)
    return sess
