# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2016-07-26
#
# Copyright (C) 2016 Taishi Matsumura
#
from Neuron import Neuron
from Panel import Panel, IstepSlider
from matplotlib.pyplot import close, figure, subplots_adjust
close('all')

neuron = Neuron(1)

fig = figure()
subplots_adjust(left=0.15, bottom=0.25)
sl = IstepSlider()
Vm_panel = Panel(211, neuron.get_t(), neuron.get_V(), ylabel='Vm [mV]', ylim=(-80, 50))
Gates_panel = Panel(413, neuron.get_t(), neuron.get_m(), ylabel='Gate vars [-]')
Istep_panel = Panel(414, neuron.get_t(), sl.get_value(), ylabel='Istep [uA]', ylim=(sl.minval, sl.maxval))
fig.add_axes(sl.position)
fig.add_subplot(Vm_panel.getPanel())
fig.add_subplot(Gates_panel.getPanel())
fig.add_subplot(Istep_panel.getPanel())

# ----------------------------------------------------------------------------
#  Main loop
# ----------------------------------------------------------------------------
while True:
    I = sl.get_value()
    neuron.update(I)

# ----------------------------------------------------------------------------
#  Plot part
# ----------------------------------------------------------------------------
    Vm_panel.updateVector(neuron.get_t(), neuron.get_V())
    Gates_panel.updateVector(neuron.get_t(), neuron.get_m())
    Istep_panel.updateVector(neuron.get_t(), sl.get_value())
    Vm_panel.plotPanel()
    Gates_panel.plotPanel()
    Istep_panel.plotPanel()
