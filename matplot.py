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
        self.DCValueLabel.setText(str(0.1 * self.DCSlider.value()))
        self.AmpValueLabel.setText(str(0.1 * self.AmpSlider.value()))
        self.FreqValueLabel.setText(str(self.FreqSlider.value()))
        self.StimValueLabel.setText(str(0.1 * self.StimSlider.value()))
        self.DiffusionConstantValueLabel.setText(str(self.DiffusionConstantSlider.value()))

    def establishConnections(self):
        PyQt4.QtCore.QObject.connect(self.RunButton, PyQt4.QtCore.SIGNAL('clicked()'), self.onRunButton)
        PyQt4.QtCore.QObject.connect(self.ResetButton, PyQt4.QtCore.SIGNAL('clicked()'), self.resetCanvas)
        PyQt4.QtCore.QObject.connect(self.IstimButton, PyQt4.QtCore.SIGNAL('pressed()'), self.pressIstimButton)
        PyQt4.QtCore.QObject.connect(self.IstimButton, PyQt4.QtCore.SIGNAL('released()'), self.releaseIstimButton)
        self.DCSlider.valueChanged.connect(self.updateSliderLabels)
        self.AmpSlider.valueChanged.connect(self.updateSliderLabels)
        self.FreqSlider.valueChanged.connect(self.updateSliderLabels)
        self.StimSlider.valueChanged.connect(self.updateSliderLabels)
        self.DiffusionConstantSlider.valueChanged.connect(self.updateSliderLabels)


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
        self.time_window = 1000  # msec
        self.fig = plt.Figure((5, 4), dpi=100)
        row, clm = 5, 1
        self.ax1 = self.fig.add_subplot(row, clm, 1)
        self.ax2 = self.fig.add_subplot(row, clm, 2)
        self.ax3 = self.fig.add_subplot(row, clm, 3)
        self.ax4 = self.fig.add_subplot(row, clm, 4)
        self.ax5 = self.fig.add_subplot(row, clm, 5)
        self.ax1.set_ylim(-81, 51)
        self.ax2.set_ylim(-21, 11)
        self.ax3.set_ylim(-11, 11)
        self.ax4.set_ylim(-1, 6)
        self.ax5.set_ylim(-10, 10)
        self.ax1.set_xlim(0, self.time_window)
        self.ax2.set_xlim(0, self.time_window)
        self.ax3.set_xlim(0, self.time_window)
        self.ax4.set_xlim(0, self.time_window)
        self.ax5.set_xlim(0, self.time_window)
        self.ax1.set_xticklabels([])
        self.ax2.set_xticklabels([])
        self.ax3.set_xticklabels([])
        self.ax4.set_xticklabels([])
        self.ax5.set_xlabel('Time [msec]')

        self.dt = 0.8
        self.t_ini = np.arange(0, self.time_window, self.dt)
        self.t = np.arange(0, self.time_window, self.dt)
        self.V = np.nan * np.zeros_like(self.t)
        self.DC = np.nan * np.zeros_like(self.t)
        self.Iwave = np.nan * np.zeros_like(self.t)
        self.Istim = np.nan * np.zeros_like(self.t)
        self.Inoise = np.nan * np.zeros_like(self.t)
        self.line_V, = self.ax1.plot(self.t, self.V, 'b-', ms=1)
        self.line_DC, = self.ax2.plot(self.t, self.DC, 'b-', ms=1)
        self.line_Iwave, = self.ax3.plot(self.t, self.Iwave, 'b-', ms=1)
        self.line_Istim, = self.ax4.plot(self.t, self.Istim, 'b-', ms=1)
        self.line_Inoise, = self.ax5.plot(self.t, self.Inoise, 'b-', ms=1)
        self.idx = 0
        self.t_now = 0

    def initNeuron(self):
        self.neuron = AckerNeuron(dt=self.dt)
        self.inlayer = Stimulator(dt=self.dt)
        self.noise_source = NoiseGenerator(dt=self.dt)

    def clearAxes(self):
        self.line_V.remove()
        self.line_DC.remove()
        self.line_Iwave.remove()
        self.line_Istim.remove()
        self.line_Inoise.remove()
        self.t = np.arange(0, self.time_window, self.dt)
        self.V = np.nan * np.zeros_like(self.t)
        self.DC = np.nan * np.zeros_like(self.t)
        self.Iwave = np.nan * np.zeros_like(self.t)
        self.Istim = np.nan * np.zeros_like(self.t)
        self.Inoise = np.nan * np.zeros_like(self.t)
        self.line_V, = self.ax1.plot(self.t, self.V, 'b-', ms=1)
        self.line_DC, = self.ax2.plot(self.t, self.DC, 'b-', ms=1)
        self.line_Iwave, = self.ax3.plot(self.t, self.Iwave, 'b-', ms=1)
        self.line_Istim, = self.ax4.plot(self.t, self.Istim, 'b-', ms=1)
        self.line_Inoise, = self.ax5.plot(self.t, self.Inoise, 'b-', ms=1)
        self.idx = 0
        self.t_now = 0
        self.ax1.set_xlim(0, self.time_window)
        self.ax2.set_xlim(0, self.time_window)
        self.ax3.set_xlim(0, self.time_window)
        self.ax4.set_xlim(0, self.time_window)
        self.ax5.set_xlim(0, self.time_window)

    def updateFigure(self):
        DC = 0.1 * main_form.DCSlider.value()
        amp = 0.1 * main_form.AmpSlider.value()
        freq = main_form.FreqSlider.value()
        Iwave = amp * np.sin(2 * np.pi * freq * self.t_now * 0.001)
        _, Istim = self.inlayer.update()
        _, Inoise = self.noise_source.update()
        t, X = self.neuron.update(DC + Iwave + Istim[0] + Inoise)
        if self.idx < len(self.t_ini):
            self.V[self.idx] = X[0]
            self.DC[self.idx] = DC
            self.Iwave[self.idx] = Iwave
            self.Istim[self.idx] = Istim
            self.Inoise[self.idx] = Inoise
            self.line_V.set_data(self.t_ini, self.V)
            self.line_DC.set_data(self.t_ini, self.DC)
            self.line_Iwave.set_data(self.t_ini, self.Iwave)
            self.line_Istim.set_data(self.t_ini, self.Istim)
            self.line_Inoise.set_data(self.t_ini, self.Inoise)
        else:
            self.t += self.dt
            self.V = np.append(self.V[1:], X[0])
            self.DC = np.append(self.DC[1:], DC)
            self.Iwave = np.append(self.Iwave[1:], Iwave)
            self.Istim = np.append(self.Istim[1:], Istim)
            self.Inoise = np.append(self.Inoise[1:], Inoise)
            self.line_V.set_data(self.t, self.V)
            self.line_DC.set_data(self.t, self.DC)
            self.line_Iwave.set_data(self.t, self.Iwave)
            self.line_Istim.set_data(self.t, self.Istim)
            self.line_Inoise.set_data(self.t, self.Inoise)
            self.ax1.set_xlim(self.t.min(), self.t.max())
            self.ax2.set_xlim(self.t.min(), self.t.max())
            self.ax3.set_xlim(self.t.min(), self.t.max())
            self.ax4.set_xlim(self.t.min(), self.t.max())
            self.ax5.set_xlim(self.t.min(), self.t.max())
        self.idx += 1
        self.t_now += self.dt
        self.draw()


class Neuron(object):
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

    def integrate(self, ext, mode='scipy'):
        if mode == 'scipy':
            _, res = scipy.integrate.odeint(self.dxdt, self.x_now, self.t, args=(ext,))
            return np.array(res)
        elif mode == 'numpy':
            return self.RungeKutta4(self.dxdt, self.x_now, self.t_now, ext)

    def RungeKutta4(self, f, x, t, ext):
        # TODO: f() の引数の順序が気に食わないから直したい
        # 今のところ odeint() の仕様に合わせて、(x, t, ext) の順になっている
        k1 = f(x, t, ext)
        k2 = f(x + 0.5 * k1 * self.dt, t + 0.5 * self.dt, ext)
        k3 = f(x + 0.5 * k2 * self.dt, t + 0.5 * self.dt, ext)
        k4 = f(x + k3 * self.dt, t + self.dt, ext)
        dx = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * self.dt / 6.0
        x_new = x + dx
        return x_new

    def dxdt(self, x, t, ext=0.0):
        pass


class LIFNeuron(Neuron):
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
        return np.array([dVdt])


class HHNeuron(Neuron):
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
        self.x_now = np.array([V, m, h, n])

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

        return np.array([dVdt, dmdt, dhdt, dndt])


class AckerNeuron(Neuron):
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
        self.x_now = np.array([V, m, mNap, h, n, mKs, mhf, mhs])

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

        return np.array([dVdt, dmNadt, dhNadt, dmNapdt, dndt, dmKsdt, dmhfdt, dmhsdt])


class Stimulator(Neuron):
    def __init__(self, dt):
        super(Stimulator, self).__init__(dt)
        self.tau_Istim = 1.0  # msec
        Istim = 0.0
        self.x_now = np.array([Istim])

    def dxdt(self, x, t, ext=0.0):
        amp = 0.1 * main_form.StimSlider.value()
        delta = main_form.is_stimulating * amp
        Istim, = x
        dIdt = -Istim / self.tau_Istim + delta
        return np.array([dIdt])


class Generator(object):
    def __init__(self, dt=0.1):
        self.dt = dt
        self.t_now = 0.0
        self.x_now = None

    def update(self):
        self.x_now = self.state_updater()
        self.t_now += self.dt
        return self.t_now, self.x_now

    def state_updater(self):
        pass


class NoiseGenerator(Generator):
    def __init__(self, dt):
        super(NoiseGenerator, self).__init__(dt)

    def state_updater(self):
        D = main_form.DiffusionConstantSlider.value()
        return np.sqrt(2 * D * self.dt * 0.001) * np.random.randn()


if __name__ == '__main__':
    app = PyQt4.QtGui.QApplication(sys.argv)

    main_form = MainForm()
    main_form.show()

    app.exec_()
