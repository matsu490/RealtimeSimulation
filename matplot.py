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
import scipy.integrate

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

        self.dt = 0.1
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
        self.neuron = AckerNeuron(dt=self.dt)
        self.inlayer = InputLayer(dt=self.dt)

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
        t, X = self.neuron.update(DC + Iwave + Istim[0])
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
    def __init__(self, dt=0.1):
        self.dt = dt
        self.t = np.array((0.0, self.dt))
        self.t_now = 0.0
        self.x_now = None

    def update(self, ext=0.0):
        self.x_now = self.integrate(ext)
        self.t_now += self.dt
        self.t += self.dt
        return self.t_now, self.x_now

    def integrate(self, ext):
        return scipy.integrate.odeint(self.dxdt, self.x_now, self.t, args=(ext,))[1]

    def dxdt(self, x, t, ext=0.0):
        pass


class LIFNeuron(Generator):
    def __init__(self, dt=0.1):
        super(LIFNeuron, self).__init__(dt)
        self.tau = 1.0
        self.gL = 0.5
        self.VL = 0.0
        self.Vth = 20.0
        self.Vreset = -40.0
        self.Vspike = 50.0
        V = -60.0
        self.x_now = [V]

    def update(self, ext=0.0):
        if self.x_now[0] == self.Vspike:
            self.x_now = self.integrate(ext)
            self.x_now[0] = self.Vreset
        else:
            self.x_now = self.integrate(ext)
            if self.x_now[0] > self.Vth:
                self.x_now[0] = self.Vspike
        self.t_now += self.dt
        self.t += self.dt
        return self.t_now, self.x_now

    def dxdt(self, x, t, ext):
        I = ext
        V, = x
        dVdt = (-self.gL * (V - self.VL) + I) / self.tau
        return [dVdt]


class HHNeuron(Generator):
    def __init__(self, dt=0.1):
        super(HHNeuron, self).__init__(dt)
        self.C = 1.5
        self.gL = 0.5
        self.gNa = 52.0
        self.gK = 11.0
        self.VL = 0.0
        self.VNa = 55.0
        self.VK = -90.0

        V = -60.0
        m = 0.0
        h = 0.0
        n = 0.0
        self.x_now = [V, m, h, n]

    def dxdt(self, x, t, ext):
        I = ext
        V, m, h, n = x
        dVdt = (
            -self.gL * (V-self.VL)
            -self.gNa * m**3 * h * (V-self.VNa)
            -self.gK * n**4 * (V-self.VK)
            +I) / self.C

        alpha_m = -0.1 * (V+23.0) / (np.exp(-0.1 * (V+23.0)) -1.0)
        beta_m = 4.0 * np.exp(-(V+48.0) / 18.0)
        m_inf = alpha_m / (alpha_m+beta_m)
        tau_m = 1.0 / (alpha_m+beta_m)
        dmdt = (m_inf - m) / tau_m

        alpha_h = 0.07 * np.exp(-(V+37.0) / 20.0)
        beta_h = 1.0 / (np.exp(-0.1 * (V+7.0)) +1.0)
        h_inf = alpha_h / (alpha_h+beta_h)
        tau_h = 1.0 / (alpha_h+beta_h)
        dhdt = (h_inf - h) / tau_h

        alpha_n = -0.01 * (V+27.0) / (np.exp(-0.1 * (V+27.0)) -1.0)
        beta_n = 0.125 * np.exp(-(V+37.0) / 80.0)
        n_inf = alpha_n / (alpha_n+beta_n)
        tau_n = 1.0 / (alpha_n+beta_n)
        dndt = (n_inf - n) / tau_n

        return [dVdt, dmdt, dhdt, dndt]


class AckerNeuron(Generator):
    def __init__(self, dt=0.1):
        super(AckerNeuron, self).__init__(dt)
        self.C = 1.5
        self.gNa = 52.0
        self.gNap = 0.5
        self.gK = 11.0
        self.gKs = 0.0
        self.gh = 1.5
        self.gL = 0.5
        self.VNa = 55.0
        self.VK = -90.0
        self.V_ha_Ks = 0.0
        self.Vh = -20.0
        self.VL = -65.0

        V = -60.0
        m = 0.0
        mNap = 0.0
        h = 0.0
        n = 0.0
        mKs = 0.0
        mhf = 0.0
        mhs = 0.0
        self.x_now = [V, m, mNap, h, n, mKs, mhf, mhs]

    def dxdt(self, x, t, ext):
        I = ext
        V, mNa, hNa, mNap, n, mKs, mhf, mhs = x
        dVdt = (
            - self.gL * (V - self.VL)
            - (self.gNa * mNa ** 3 * hNa + self.gNap * mNap) * (V - self.VNa)
            - (self.gK * n ** 4 + self.gKs * mKs) * (V - self.VK)
            - self.gh * (0.65 * mhf + 0.35 * mhs) * (V - self.Vh)
            + I) / self.C

        alpha_mNa = -0.1 * (V + 23.0) / (np.exp(-0.1 * (V + 23.0)) - 1.0)
        beta_mNa = 4.0 * np.exp(-(V + 48.0) / 18.0)
        dmNadt = alpha_mNa * (1-mNa) - beta_mNa * mNa

        alpha_hNa = 0.07 * np.exp(-(V + 37.0) / 20.0)
        beta_hNa = 1.0 / (np.exp(-0.1 * (V + 7.0)) + 1.0)
        dhNadt = alpha_hNa * (1-hNa) - beta_hNa * hNa

        alpha_mNap = 1.0 / (0.15 * (1.0 + np.exp(-(V + 38.0) / 6.5)))
        beta_mNap = np.exp(-(V + 38.0) / 6.5) / (0.15 * (1.0 + np.exp(-(V + 38.0) / 6.5)))
        dmNapdt = alpha_mNap * (1-mNap) - beta_mNap * mNap

        alpha_n = -0.01 * (V + 27.0) / (np.exp(-0.1 * (V + 27.0)) - 1.0)
        beta_n = 0.125 * np.exp(-(V + 37.0) / 80.0)
        dndt = alpha_n * (1-n) - beta_n * n

        mKs_inf = 1.0 / (1.0 + np.exp(-(V - self.V_ha_Ks) / 6.5))
        tau_mKs = 90.0
        dmKsdt = (mKs_inf - mKs) / tau_mKs

        mhf_inf = 1.0 / (1.0 + np.exp((V + 79.2) / 9.78))
        tau_mhf = 0.51 / (np.exp((V - 1.7) / 10.0) + np.exp(-(V + 340.0) / 52.0) + 1.0)
        dmhfdt = (mhf_inf - mhf) / tau_mhf

        mhs_inf = 1.0 / (1.0 + np.exp((V + 71.3) / 7.9))
        tau_mhs = 5.6 / (np.exp((V - 1.7) / 14.0) + np.exp(-(V + 260.0) / 43.0) + 1.0)
        dmhsdt = (mhs_inf - mhs) / tau_mhs

        return [dVdt, dmNadt, dhNadt, dmNapdt, dndt, dmKsdt, dmhfdt, dmhsdt]


class InputLayer(Generator):
    def __init__(self, dt):
        super(InputLayer, self).__init__(dt)
        self.tau_Istim = 1.0  # msec
        Istim = 0.0
        self.x_now = [Istim]

    def dxdt(self, x, t, ext=0.0):
        amp = main_form.StimSlider.value()
        delta = main_form.is_stimulating * amp
        Istim, = x
        dIdt = -Istim / self.tau_Istim + delta
        return [dIdt]


if __name__ == '__main__':
    app = PyQt4.QtGui.QApplication(sys.argv)

    main_form = MainForm()
    main_form.show()

    app.exec_()
