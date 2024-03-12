# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:19:14 2022

@author: lixuy
"""
import numpy
import scipy.io
name = '../../data/abilene-distro/Abilene.mat'
data = scipy.io.loadmat(name)


odnames = data['odnames']
X = data['X2']
X = numpy.log10(X+1)
num_points, num_nodes = X.shape
for i in numpy.arange(num_nodes):
    X[:,i] = X[:,i] - numpy.mean(X[:,i])
    X[:,i] = X[:,i]/numpy.sqrt(numpy.sum(X[:,i]**2))

timestamp = [t[0] for t in data['utc2']]
input['data'] = X.T
input['tsample'] = timestamp
input['mint'] = min(timestamp)
input['maxt'] = max(timestamp)
input['step'] = (max(timestamp) - min(timestamp)) / len(input['data'][0])

tsnames = []
for n in data['odnames']:
    narr = str(n[0][0]).split("-")
    tsnames.append(narr[1] + "." + narr[0])
input['ts_names'] = tsnames


dm = input


hiddenvars = 10
(data, reclog, sp) = run_spirit(dm, hiddenvars)
plot_recon_error_all(data, reclog, dm, 30, transformit)
hvlog = sp.gethvlog()
plot_hvs_all(hvlog, dm, hiddenvars)

