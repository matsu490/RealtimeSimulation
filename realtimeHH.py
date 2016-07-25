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

        V = -60.0 * ones(N)
        m = 0.0 * ones(N)
        h = 0.0 * ones(N)
        n = 0.0 * ones(N)
        self.x_now = vstack((V, m, h, n))

    def NeuronDerivs(self, t, x, I):
        V, m, h, n = x
        dxdt_V = (
            - self.gl * (V - self.Vl)
            - self.gNa * m ** 3 * h * (V - self.VNa)
            - self.gK * n ** 4 * (V - self.VK)
            + I) / self.C

        alpha_m = -0.1 * (V + 23.0) / (exp(-0.1 * (V + 23.0)) - 1.0)
        beta_m = 4.0 * exp(-(V + 48.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        tau_m = 1.0 / (alpha_m + beta_m)

        alpha_h = 0.07 * exp(-(V + 37.0) / 20.0)
        beta_h = 1.0 / (exp(-0.1 * (V + 7.0)) + 1.0)
        h_inf = alpha_h / (alpha_h + beta_h)
        tau_h = 1.0 / (alpha_h + beta_h)

        alpha_n = -0.01 * (V + 27.0) / (exp(-0.1 * (V + 27.0)) - 1.0)
        beta_n = 0.125 * exp(-(V + 37.0) / 80.0)
        n_inf = alpha_n / (alpha_n + beta_n)
        tau_n = 1.0 / (alpha_n + beta_n)

        infs = vstack((m_inf, h_inf, n_inf))
        taus = vstack((tau_m, tau_h, tau_n))
        gates = vstack((m, h, n))
        dxdt_ch = (infs - gates) / taus
        self.dxdt = vstack((dxdt_V, dxdt_ch))

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
        self.t += self.dt
        self.x_now = self.RungeKutta4(self.t, self.x_now, I)
        return self.x_now

# ----------------------------------------------------------------------------
#  Neuron parameters
# ----------------------------------------------------------------------------
I_step_ini = -20.0
I_step_min = -30.0
I_step_max = 30.0
dt = 0.1
neuron = Neuron(1, dt)

t_now = 0.0
x_now = neuron.x_now
X = array([[t_now] * neuron.N])
V = x_now[0:1]
Gates = x_now[1:2]
I_step = [I_step_ini]

# ----------------------------------------------------------------------------
#  Figure initialization
# ----------------------------------------------------------------------------
time_window = 100
fig = figure()
subplots_adjust(left=0.15, bottom=0.25)

ax = fig.add_subplot(211)
ax.set_ylim(-80, 50)
ax.set_xlim(0, time_window)
ax.set_ylabel('Membrane potential [mV]')
ax.set_xlabel('Time [msec]')
lines, = ax.plot(X, V)

ax1 = fig.add_subplot(413)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlim(0, time_window)
ax1.set_ylabel('Gate variables [-]')
ax1.set_xlabel('Time [msec]')
lines1, = ax1.plot(X, Gates)

ax2 = fig.add_subplot(414)
ax2.set_ylim(I_step_min - 5, I_step_max + 5)
ax2.axhline(I_step_ini, ls='--', c='red')
ax2.set_xlim(0, time_window)
ax2.set_ylabel('I_step [uA]')
ax2.set_xlabel('Time [msec]')
lines2, = ax2.plot(X, I_step)

ax_I_step = axes([0.15, 0.10, 0.65, 0.03])
slider_I_step = Slider(
    ax_I_step, 'I_step', I_step_min, I_step_max, valinit=I_step_ini)

# ----------------------------------------------------------------------------
#  Main loop
# ----------------------------------------------------------------------------
while True:
    I_step.append(slider_I_step.val)
    I = I_step[-1]
    t_now = t_now + dt
    x_now = neuron.update(I)

# ----------------------------------------------------------------------------
#  Plot part
# ----------------------------------------------------------------------------
    if max(X) < time_window:
        X = append(X, t_now)
        V = append(V, x_now[0:1])
        Gates = append(Gates, x_now[1:2])
        lines.set_data(X, V)
        lines1.set_data(X, Gates)
        lines2.set_data(X, I_step)
        pause(0.01)
    else:
        X += dt
        V = append(V[1:], x_now[0:1])
        Gates = append(Gates, x_now[1:2])
        lines.set_data(X, V)
        ax.set_xlim((X.min(), X.max()))
        lines1.set_data(X, Gates)
        lines2.set_data(X, I_step)
        pause(0.01)
