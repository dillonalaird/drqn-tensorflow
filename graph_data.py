from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob
import os


plt.style.use("ggplot")


def grab_data(directory, tag):
    data = []
    for file_path in glob.glob(os.path.join(directory, "*")):
        for e in tf.train.summary_iterator(file_path):
            for v in e.summary.value:
                if v.tag == tag:
                    data.append((e.step, v.simple_value))

    #return sorted(data, key=lambda x: x[0])
    return data


def graph_data_mul(datas, n=20):
    for i,data in enumerate(datas):
        x = [ii for ii in xrange(len(data))]
        y = [xi[1] for xi in data]
        window = np.ones((n,))/n
        smooth = np.convolve(y, window, mode="same")
        plt.plot(x, y, alpha=0.3, linewidth=2.0, color=plt.rcParams["axes.color_cycle"][i])
        plt.plot(x, smooth, linewidth=2.0, color=plt.rcParams["axes.color_cycle"][i])

    plt.show()


def graph_data(data):
    #plt.plot([x[0] for x in data], [x[1] for x in data])
    x = [i for i in xrange(len(data))]
    y = [xi[1] for xi in data]
    n = 20
    #window = signal.exponential(n, tau=25.)
    #window = window/np.sum(window)
    window = np.ones((n,))/n
    smooth = np.convolve(y, window, mode="same")
    plt.plot(x, y, alpha=0.3, linewidth=2.0, color=plt.rcParams["axes.color_cycle"][0])
    plt.plot(x, smooth, linewidth=2.0, color=plt.rcParams["axes.color_cycle"][0])
    plt.show()
