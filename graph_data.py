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


def graph_data_mul(datas, labels, title, xlabel, ylabel, n=20):
    lines = []
    for i,data in enumerate(datas):
        x = [ii for ii in xrange(len(data))]
        y = [xi[1] for xi in data]
        window = np.ones((n,))/n
        smooth = np.convolve(y, window, mode="valid")
        diff = int((len(y) - len(smooth))/2)
        plt.plot(x, y, alpha=0.2, linewidth=2.0,
                 color=plt.rcParams["axes.color_cycle"][i])
        smooth_x = [ii for ii in xrange(diff, len(smooth) + diff)]
        line, = plt.plot(smooth_x, smooth, linewidth=2.0, label=labels[i],
                         color=plt.rcParams["axes.color_cycle"][i])
        lines.append(line)

    plt.legend(handles=lines, loc=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
