# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2017-02-13
#
# Copyright (C) 2017 Taishi Matsumura
#
import sys
import PyQt4.QtCore
import PyQt4.QtGui
import PyQt4.uic

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg

import numpy as np

form, base = PyQt4.uic.loadUiType('./mainUI.ui')


class MainForm(base, form):
    def __init__(self):
        super(base, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Demo form')
        self.establishConnections()
        self.figman = FigureManager()
        self.figman.setCanvas(self.PlotWidget)
        NavigationToolbar2QTAgg(self.figman.canvas, self.ToolBarWidget)

        self.timer = PyQt4.QtCore.QTimer(self.figman.canvas)
	self.timer.timeout.connect(self.figman.updateFigure)

    def startTimer(self):
	self.timer.start(10)

    def stopTimer(self):
	self.timer.stop()

    def establishConnections(self):
        PyQt4.QtCore.QObject.connect(self.StartButton, PyQt4.QtCore.SIGNAL('clicked()'), self.startTimer)
        PyQt4.QtCore.QObject.connect(self.StopButton, PyQt4.QtCore.SIGNAL('clicked()'), self.stopTimer)


class FigureManager(object):
    def __init__(self):
        self.time_window = 100
        self.fig = plt.Figure((5, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.ax1.set_ylim(-80, 50)
        self.ax2.set_ylim(-50, 50)
        self.ax3.set_ylim(-11, 11)
        self.ax1.set_xlim(0, self.time_window)
        self.ax2.set_xlim(0, self.time_window)
        self.ax3.set_xlim(0, self.time_window)
        # self.ax1.set_ylabel('Membrane potential [mV]')
        # self.ax1.set_xlabel('Time [msec]')
        self.dt = 0.1
        self.t_ini = np.arange(0, self.time_window, self.dt)
        self.t = np.arange(0, self.time_window, self.dt)
        self.V = np.nan * np.zeros_like(self.t)
        self.DC = np.nan * np.zeros_like(self.t)
        self.Iwave = np.nan * np.zeros_like(self.t)
        self.line_V, = self.ax1.plot(self.t, self.V, '.', ms=1)
        self.line_DC, = self.ax2.plot(self.t, self.DC, '.', ms=1)
        self.line_Iwave, = self.ax3.plot(self.t, self.Iwave, '.', ms=1)
        self.idx = 0
        self.t_now = 0

    def setCanvas(self, widget):
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(widget)

    def updateFigure(self):
        DC = main_form.DCSlider.value()
        amp = 0.1 * main_form.AmpSlider.value()
        freq = main_form.FreqSlider.value()
        Iwave = amp * np.sin(2 * np.pi * freq * self.t_now * 0.001)
        t, X = neuron.update(DC + Iwave)
        if self.idx < len(self.t_ini):
            self.V[self.idx] = X[0]
            self.DC[self.idx] = DC
            self.Iwave[self.idx] = Iwave
            self.line_V.set_data(self.t_ini, self.V)
            self.line_DC.set_data(self.t_ini, self.DC)
            self.line_Iwave.set_data(self.t_ini, self.Iwave)
        else:
            self.t += self.dt
            self.V = np.append(self.V[1:], X[0])
            self.DC = np.append(self.DC[1:], DC)
            self.Iwave = np.append(self.Iwave[1:], Iwave)
            self.line_V.set_data(self.t, self.V)
            self.line_DC.set_data(self.t, self.DC)
            self.line_Iwave.set_data(self.t, self.Iwave)
            self.ax1.set_xlim(self.t.min(), self.t.max())
            self.ax2.set_xlim(self.t.min(), self.t.max())
            self.ax3.set_xlim(self.t.min(), self.t.max())
        self.idx += 1
        self.t_now += self.dt
        self.canvas.draw()


class Neuron(object):
    def __init__(self, N, dt=0.1):
        self.N = N
        self.t_now = 0.0
        self.dt = dt
        self.C = 1.5
        self.gl = 0.5
        self.gNa = 52.0
        self.gK = 11.0
        self.Vl = 0.0
        self.VNa = 55.0
        self.VK = -90.0

        V = -60.0 * np.ones(N)
        m = 0.0 * np.ones(N)
        h = 0.0 * np.ones(N)
        n = 0.0 * np.ones(N)
        self.x_now = np.vstack((V, m, h, n))

    def NeuronDerivs(self, t, x, I):
        V, m, h, n = x
        dxdt_V = (
            - self.gl * (V - self.Vl)
            - self.gNa * m ** 3 * h * (V - self.VNa)
            - self.gK * n ** 4 * (V - self.VK)
            + I) / self.C

        alpha_m = -0.1 * (V + 23.0) / (np.exp(-0.1 * (V + 23.0)) - 1.0)
        beta_m = 4.0 * np.exp(-(V + 48.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        tau_m = 1.0 / (alpha_m + beta_m)

        alpha_h = 0.07 * np.exp(-(V + 37.0) / 20.0)
        beta_h = 1.0 / (np.exp(-0.1 * (V + 7.0)) + 1.0)
        h_inf = alpha_h / (alpha_h + beta_h)
        tau_h = 1.0 / (alpha_h + beta_h)

        alpha_n = -0.01 * (V + 27.0) / (np.exp(-0.1 * (V + 27.0)) - 1.0)
        beta_n = 0.125 * np.exp(-(V + 37.0) / 80.0)
        n_inf = alpha_n / (alpha_n + beta_n)
        tau_n = 1.0 / (alpha_n + beta_n)

        infs = np.vstack((m_inf, h_inf, n_inf))
        taus = np.vstack((tau_m, tau_h, tau_n))
        gates = np.vstack((m, h, n))
        dxdt_ch = (infs - gates) / taus
        self.dxdt = np.vstack((dxdt_V, dxdt_ch))

        return self.dxdt

    def RungeKutta4(self, t, x, I):
        k1 = self.NeuronDerivs(t, x, I)
        k2 = self.NeuronDerivs(t + 0.5 * self.dt, x + 0.5 * k1 * self.dt, I)
        k3 = self.NeuronDerivs(t + 0.5 * self.dt, x + 0.5 * k2 * self.dt, I)
        k4 = self.NeuronDerivs(t + self.dt, x + k3 * self.dt, I)
        dx = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * self.dt / 6.0
        x_new = x + dx
        return x_new

    def update(self, I):
        self.t_now += self.dt
        self.x_now = self.RungeKutta4(self.t_now, self.x_now, I)
        return self.t_now, self.x_now

if __name__ == '__main__':
    app = PyQt4.QtGui.QApplication(sys.argv)

    neuron = Neuron(1, dt=0.1)
    main_form = MainForm()
    main_form.show()

    app.exec_()
