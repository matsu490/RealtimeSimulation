# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2016-07-26
#
# Copyright (C) 2016 Taishi Matsumura
#
from matplotlib.pyplot import subplot, Slider, axes, pause
from numpy import append, array


class Panel(object):
    def __init__(self, num, init_x, init_y, xlim=(0, 100), ylim=(0, 1), wsize=100, xlabel='Time [msec]', ylabel=''):
        self.windowsize = wsize
        self.x = array(init_x)
        self.y = array(init_y)
        self.ax = subplot(num)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.line, = self.ax.plot(self.x, self.y)

    def getPanel(self):
        return self.ax

    def updateVector(self, x_now, y_now):
        if max(self.x) < self.windowsize:
            self.x = append(self.x, x_now)
            self.y = append(self.y, y_now)
        else:
            self.x = append(self.x[1:], x_now)
            self.y = append(self.y[1:], y_now)

    def plotPanel(self):
        if max(self.x) < self.windowsize:
            self.line.set_data(self.x, self.y)
            pause(0.01)
        else:
            self.line.set_data(self.x, self.y)
            self.ax.set_xlim((self.x.min(), self.y.max()))
            pause(0.01)


class IstepSlider(object):
    def __init__(self):
        self.label = 'I_step'
        self.initval = 0.0
        self.minval = -30.0
        self.maxval = 30.0
        self.position = axes([0.15, 0.10, 0.65, 0.03])
        self.slider = Slider(
            self.position, self.label, self.minval, self.maxval, valinit=self.initval)

    def get_value(self):
        return self.slider.val
