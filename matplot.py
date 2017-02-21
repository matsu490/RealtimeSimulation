# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    matsu490
# Created:   2017-02-13
#
# Copyright (C) 2017 matsu490
#
import sys
import PyQt4.QtCore
import PyQt4.QtGui
import PyQt4.uic

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT

import numpy as np

form, base = PyQt4.uic.loadUiType('./mainUI.ui')


class MainForm(base, form):
    def __init__(self):
        super(base, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Demo form')
        self.establishConnections()
        self.updateSliderLabels()

        self.canvas = Canvas(self.centralwidget)
        self.toolbar = NavigationToolbar2QT(self.canvas, self.centralwidget)
        self.VLayout.addWidget(self.canvas)
        self.VLayout.addWidget(self.toolbar)

        self.timer = PyQt4.QtCore.QTimer(self.canvas)
	self.timer.timeout.connect(self.canvas.updateFigure)

        self.is_running = False
        self.is_stimulating = False

    def onRunButton(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.startTimer()
        else:
            self.stopTimer()

    def pressIstimButton(self):
        self.is_stimulating = True

    def releaseIstimButton(self):
        self.is_stimulating = False

    def startTimer(self):
	self.timer.start(10)

    def stopTimer(self):
	self.timer.stop()

    def resetCanvas(self):
        self.canvas.clearAxes()
        self.canvas.initNeuron()

    def updateSliderLabels(self):
        self.DCValueLabel.setText(str(self.DCSlider.value()))
        self.AmpValueLabel.setText(str(self.AmpSlider.value()))
        self.FreqValueLabel.setText(str(self.FreqSlider.value()))
        self.StimValueLabel.setText(str(self.StimSlider.value()))

    def establishConnections(self):
        PyQt4.QtCore.QObject.connect(self.RunButton, PyQt4.QtCore.SIGNAL('clicked()'), self.onRunButton)
        PyQt4.QtCore.QObject.connect(self.ResetButton, PyQt4.QtCore.SIGNAL('clicked()'), self.resetCanvas)
        PyQt4.QtCore.QObject.connect(self.IstimButton, PyQt4.QtCore.SIGNAL('pressed()'), self.pressIstimButton)
        PyQt4.QtCore.QObject.connect(self.IstimButton, PyQt4.QtCore.SIGNAL('released()'), self.releaseIstimButton)
        self.DCSlider.valueChanged.connect(self.updateSliderLabels)
        self.AmpSlider.valueChanged.connect(self.updateSliderLabels)
        self.FreqSlider.valueChanged.connect(self.updateSliderLabels)
        self.StimSlider.valueChanged.connect(self.updateSliderLabels)


class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.initFigure()
        self.initNeuron()

        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(self,
                                   PyQt4.QtGui.QSizePolicy.Expanding,
                                   PyQt4.QtGui.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def initFigure(self):
        self.time_window = 100  # msec
        self.fig = plt.Figure((5, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(411)
        self.ax2 = self.fig.add_subplot(412)
        self.ax3 = self.fig.add_subplot(413)
        self.ax4 = self.fig.add_subplot(414)
        self.ax1.set_ylim(-81, 51)
        self.ax2.set_ylim(-51, 51)
        self.ax3.set_ylim(-11, 11)
        self.ax4.set_ylim(-1, 101)
        self.ax1.set_xlim(0, self.time_window)
        self.ax2.set_xlim(0, self.time_window)
        self.ax3.set_xlim(0, self.time_window)
        self.ax4.set_xlim(0, self.time_window)
        self.ax1.set_xticklabels([])
        self.ax2.set_xticklabels([])
        self.ax3.set_xticklabels([])
        # self.ax1.set_ylabel('Membrane potential [mV]')
        self.ax4.set_xlabel('Time [msec]')

        self.dt = 0.2
        self.t_ini = np.arange(0, self.time_window, self.dt)
        self.t = np.arange(0, self.time_window, self.dt)
        self.V = np.nan * np.zeros_like(self.t)
        self.DC = np.nan * np.zeros_like(self.t)
        self.Iwave = np.nan * np.zeros_like(self.t)
        self.Istim = np.nan * np.zeros_like(self.t)
        self.line_V, = self.ax1.plot(self.t, self.V, 'b-', ms=1)
        self.line_DC, = self.ax2.plot(self.t, self.DC, 'b-', ms=1)
        self.line_Iwave, = self.ax3.plot(self.t, self.Iwave, 'b-', ms=1)
        self.line_Istim, = self.ax4.plot(self.t, self.Istim, 'b-', ms=1)
        self.idx = 0
        self.t_now = 0

    def initNeuron(self):
        self.neuron = HHNeuron(1, dt=self.dt)
        self.inlayer = InputLayer(1, dt=self.dt)

    def clearAxes(self):
        self.line_V.remove()
        self.line_DC.remove()
        self.line_Iwave.remove()
        self.line_Istim.remove()
        self.t = np.arange(0, self.time_window, self.dt)
        self.V = np.nan * np.zeros_like(self.t)
        self.DC = np.nan * np.zeros_like(self.t)
        self.Iwave = np.nan * np.zeros_like(self.t)
        self.Istim = np.nan * np.zeros_like(self.t)
        self.line_V, = self.ax1.plot(self.t, self.V, 'b-', ms=1)
        self.line_DC, = self.ax2.plot(self.t, self.DC, 'b-', ms=1)
        self.line_Iwave, = self.ax3.plot(self.t, self.Iwave, 'b-', ms=1)
        self.line_Istim, = self.ax4.plot(self.t, self.Istim, 'b-', ms=1)
        self.idx = 0
        self.t_now = 0
        self.ax1.set_xlim(0, self.time_window)
        self.ax2.set_xlim(0, self.time_window)
        self.ax3.set_xlim(0, self.time_window)
        self.ax4.set_xlim(0, self.time_window)

    def updateFigure(self):
        DC = main_form.DCSlider.value()
        amp = 0.1 * main_form.AmpSlider.value()
        freq = main_form.FreqSlider.value()
        Iwave = amp * np.sin(2 * np.pi * freq * self.t_now * 0.001)
        t, Istim = self.inlayer.update()
        t, X = self.neuron.update(DC + Iwave + Istim)
        if self.idx < len(self.t_ini):
            self.V[self.idx] = X[0]
            self.DC[self.idx] = DC
            self.Iwave[self.idx] = Iwave
            self.Istim[self.idx] = Istim
            self.line_V.set_data(self.t_ini, self.V)
            self.line_DC.set_data(self.t_ini, self.DC)
            self.line_Iwave.set_data(self.t_ini, self.Iwave)
            self.line_Istim.set_data(self.t_ini, self.Istim)
        else:
            self.t += self.dt
            self.V = np.append(self.V[1:], X[0])
            self.DC = np.append(self.DC[1:], DC)
            self.Iwave = np.append(self.Iwave[1:], Iwave)
            self.Istim = np.append(self.Istim[1:], Istim)
            self.line_V.set_data(self.t, self.V)
            self.line_DC.set_data(self.t, self.DC)
            self.line_Iwave.set_data(self.t, self.Iwave)
            self.line_Istim.set_data(self.t, self.Istim)
            self.ax1.set_xlim(self.t.min(), self.t.max())
            self.ax2.set_xlim(self.t.min(), self.t.max())
            self.ax3.set_xlim(self.t.min(), self.t.max())
            self.ax4.set_xlim(self.t.min(), self.t.max())
        self.idx += 1
        self.t_now += self.dt
        self.draw()


class Generator(object):
    def __init__(self, N, dt=0.1):
        self.N = N
        self.dt = dt
        self.t_now = 0.0
        self.x_now = None

    def update(self, ext=0.0):
        self.t_now += self.dt
        self.x_now = self.RungeKutta4(self.Derivatives, self.t_now, self.x_now, ext)
        return self.t_now, self.x_now

    def RungeKutta4(self, f, t, x, ext=0.0):
        k1 = f(t, x, ext)
        k2 = f(t + 0.5 * self.dt, x + 0.5 * k1 * self.dt, ext)
        k3 = f(t + 0.5 * self.dt, x + 0.5 * k2 * self.dt, ext)
        k4 = f(t + self.dt, x + k3 * self.dt, ext)
        dx = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * self.dt / 6.0
        x_new = x + dx
        return x_new

    def Derivatives(self, t, x, ext=0.0):
        pass


class LIFNeuron(Generator):
    def __init__(self, N, dt=0.1):
        super(LIFNeuron, self).__init__(N, dt)
        self.tau = 1.0
        self.gl = 0.5
        self.Vl = 0.0
        self.Vth = 20.0
        self.Vreset = -40.0
        self.Vspike = 50.0
        V = -60.0 * np.ones(self.N)
        self.x_now = np.vstack((V,))

    def update(self, ext=0.0):
        self.t_now += self.dt
        if self.x_now[0] == self.Vspike:
            self.x_now = self.RungeKutta4(self.Derivatives, self.t_now, self.x_now, ext)
            self.x_now[0] = self.Vreset
        else:
            self.x_now = self.RungeKutta4(self.Derivatives, self.t_now, self.x_now, ext)
            if self.x_now[0] > self.Vth:
                self.x_now[0] = self.Vspike
        return self.t_now, self.x_now

    def Derivatives(self, t, x, I):
        V, = x
        dxdt_V = (- self.gl * (V - self.Vl) + I) / self.tau
        self.dxdt = np.vstack((dxdt_V,))
        return self.dxdt


class HHNeuron(Generator):
    def __init__(self, N, dt=0.1):
        super(HHNeuron, self).__init__(N, dt)
        self.C = 1.5
        self.gl = 0.5
        self.gNa = 52.0
        self.gK = 11.0
        self.Vl = 0.0
        self.VNa = 55.0
        self.VK = -90.0

        V = -60.0 * np.ones(self.N)
        m = 0.0 * np.ones(self.N)
        h = 0.0 * np.ones(self.N)
        n = 0.0 * np.ones(self.N)
        self.x_now = np.vstack((V, m, h, n))

    def Derivatives(self, t, x, I):
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


class InputLayer(Generator):
    def __init__(self, N, dt):
        super(InputLayer, self).__init__(N, dt)
        self.tau_Istim = 1.0  # msec
        Istim = np.zeros(self.N)
        self.x_now = np.vstack((Istim,))

    def Derivatives(self, t, x, ext=0.0):
        amp = main_form.StimSlider.value()
        delta = main_form.is_stimulating * amp
        Istim, = x
        dxdt_Istim = -Istim / self.tau_Istim + delta
        self.dxdt = np.vstack((dxdt_Istim, ))
        return self.dxdt


if __name__ == '__main__':
    app = PyQt4.QtGui.QApplication(sys.argv)

    main_form = MainForm()
    main_form.show()

    app.exec_()
