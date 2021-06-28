import matplotlib.pyplot as plt
import os
import numpy as np

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def save_data_and_plot(self, data, filename, xlabel, ylabel, dumb):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = [0 for i in range(len(data))]
        max_val = [0 for i in range(len(data))]
        plt.rcParams.update({'font.size': 24})  # set bigger font size
        for index, datas in enumerate(data):
            min_val[index] = min(datas)
            max_val[index] = max(datas)
            if index==1 and dumb:
                plt.plot(datas, label="Agent "+str(index + 1)+" dumb")
            else:
                plt.plot(datas, label="Agent " + str(index + 1))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(loc="upper left")
        plt.margins(0)
        minv = min(min_val)
        maxv = max(max_val)
        plt.ylim(minv - 0.05 * abs(minv), maxv + 0.05 * abs(maxv))
        # plt.ylim(min_val)

        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                file.write("%s\n" % value)
