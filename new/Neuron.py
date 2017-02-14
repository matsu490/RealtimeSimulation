# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2016-07-26
#
# Copyright (C) 2016 Taishi Matsumura
#
from numpy import ones, vstack, exp, array


class Neuron(object):
    def __init__(self, N, dt=0.1):
        self.N = N
        self.t_now = array([0.0])
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
        self.x_now = vstack((self.V, self.m, self.h, self.n))

    def NeuronDerivs(self, t, x, I):
        self.V, self.m, self.h, self.n = x
        dxdt_V = (
            - self.gl * (self.V - self.Vl)
            - self.gNa * self.m ** 3 * self.h * (self.V - self.VNa)
            - self.gK * self.n ** 4 * (self.V - self.VK)
            + I) / self.C

        alpha_m = -0.1 * (self.V + 23.0) / (exp(-0.1 * (self.V + 23.0)) - 1.0)
        beta_m = 4.0 * exp(-(self.V + 48.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        tau_m = 1.0 / (alpha_m + beta_m)

        alpha_h = 0.07 * exp(-(self.V + 37.0) / 20.0)
        beta_h = 1.0 / (exp(-0.1 * (self.V + 7.0)) + 1.0)
        h_inf = alpha_h / (alpha_h + beta_h)
        tau_h = 1.0 / (alpha_h + beta_h)

        alpha_n = -0.01 * (self.V + 27.0) / (exp(-0.1 * (self.V + 27.0)) - 1.0)
        beta_n = 0.125 * exp(-(self.V + 37.0) / 80.0)
        n_inf = alpha_n / (alpha_n + beta_n)
        tau_n = 1.0 / (alpha_n + beta_n)

        infs = vstack((m_inf, h_inf, n_inf))
        taus = vstack((tau_m, tau_h, tau_n))
        gates = vstack((self.m, self.h, self.n))
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
        self.t_now += self.dt
        self.x_now = self.RungeKutta4(self.t_now, self.x_now, I)

    def get_t(self):
        return self.t_now

    def get_V(self):
        return self.V

    def get_m(self):
        return self.m

    def get_h(self):
        return self.h

    def get_n(self):
        return self.n
