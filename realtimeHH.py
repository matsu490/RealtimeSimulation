# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2016-06-29
#
# Copyright (C) 2015 Taishi Matsumura
#
from pylab import *
close('all')


class Neuron(object):
    def __init__(self, N, dt):
        self.N = N
        self.t = 0.0
        self.dt = dt
        self.C = 1.5
        self.gl = 0.5
        self.gNa = 52.0
        self.gK = 11.0
        self.Vl = 0.0
        self.VNa = 55.0
        self.VK = -90.0

        self.V = -60.0 * ones(N)
        self.m = 0.0 * ones(N)
        self.h = 0.0 * ones(N)
        self.n = 0.0 * ones(N)
        self.x = array([self.V, self.m, self.h, self.n])
        self.I = 0.0 * ones(N)

    def NeuronDerivs(self, t, x, I):
        self.V, m, h, n = x
        self.dxdt_V = (
            - self.gl * (self.V - self.Vl)
            - self.gNa * m ** 3 * h * (self.V - self.VNa)
            - self.gK * n ** 4 * (self.V - self.VK)
            + I) / self.C

        self.alpha_m = -0.1 * (self.V + 23.0) / (exp(-0.1 * (self.V + 23.0)) - 1.0)
        self.beta_m = 4.0 * exp(-(self.V + 48.0) / 18.0)
        m_inf = self.alpha_m / (self.alpha_m + self.beta_m)
        tau_m = 1.0 / (self.alpha_m + self.beta_m)

        alpha_h = 0.07 * exp(-(self.V + 37.0) / 20.0)
        beta_h = 1.0 / (exp(-0.1 * (self.V + 7.0)) + 1.0)
        h_inf = alpha_h / (alpha_h + beta_h)
        tau_h = 1.0 / (alpha_h + beta_h)

        alpha_n = -0.01 * (self.V + 27.0) / (exp(-0.1 * (self.V + 27.0)) - 1.0)
        beta_n = 0.125 * exp(-(self.V + 37.0) / 80.0)
        n_inf = alpha_n / (alpha_n + beta_n)
        tau_n = 1.0 / (alpha_n + beta_n)

        self.infs = array([m_inf, h_inf, n_inf])
        self.taus = array([tau_m, tau_h, tau_n])
        self.gates = array([m, h, n])
        self.dxdt_ch = (self.infs - self.gates) / self.taus
        self.dxdt = vstack((self.dxdt_V, self.dxdt_ch))

        return self.dxdt

    def RungeKutta4(self, t, x, I):
        self.k1 = self.NeuronDerivs(t, x, I)
        self.k2 = self.NeuronDerivs(t + 0.5 * self.dt, x + 0.5 * self.k1 * self.dt, I)
        self.k3 = self.NeuronDerivs(t + 0.5 * self.dt, x + 0.5 * self.k2 * self.dt, I)
        self.k4 = self.NeuronDerivs(t + self.dt, x + self.k3 * self.dt, I)
        dx = (self.k1 + 2.0 * self.k2 + 2.0 * self.k3 + self.k4) * self.dt / 6.0
        x_new = x + dx
        return x_new

    def update(self, I):
        self.t += self.dt
        self.x = self.RungeKutta4(self.t, self.x, I)
        return self.x

# ----------------------------------------------------------------------------
#  Neuron parameters
# ----------------------------------------------------------------------------
I_ini = -20.0
dt = 0.1
neuron = Neuron(1, dt)

t_now = 0.0
x_now = neuron.x
X = array([t_now])
Y = x_now[0:1]

# ----------------------------------------------------------------------------
#  Figure initialization
# ----------------------------------------------------------------------------
time_window = 100
fig = figure()
subplots_adjust(left=0.15, bottom=0.25)
ax = fig.add_subplot(111)
ax.set_ylim(-80, 50)
ax.set_xlim(0, time_window)
ax.set_ylabel('Membrane potential [mV]')
ax.set_xlabel('Time [msec]')
lines, = ax.plot(X, Y)

ax_I = axes([0.15, 0.15, 0.65, 0.03])
sb_I = Slider(ax_I, 'I', -30.0, 30.0, valinit=I_ini)

# ----------------------------------------------------------------------------
#  Main loop
# ----------------------------------------------------------------------------
while True:
    I = sb_I.val
    t_now = t_now + dt
    x_now = neuron.update(I)

# ----------------------------------------------------------------------------
#  Plot part
# ----------------------------------------------------------------------------
    if max(X) < time_window:
        X = append(X, t_now)
        Y = append(Y, x_now[0:1])
        lines.set_data(X, Y)
        pause(0.01)
    else:
        X += dt
        Y = append(Y[1:], x_now[0:1])
        lines.set_data(X, Y)
        ax.set_xlim((X.min(), X.max()))
        pause(0.01)
