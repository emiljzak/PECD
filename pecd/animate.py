#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2021 Emil Zak <emil.zak@cfel.de>
#

from propagate import gen_input
from matelem  import mapping
from basis import radbas
import numpy as np
from scipy.special import genlaguerre,sph_harm,roots_genlaguerre,factorial,roots_hermite


import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
import matplotlib.animation as animation

params = {}

params = gen_input()
print(params['t0'])


#generate map
mymap = mapping(int(params['lmin']), int(params['lmax']), int(params['nbins']), int(params['nlobatto']))
maparray, Nbas = mymap.gen_map()


#read the wavepacket
print(int((params['tmax']-params['t0'])/params['dt']+1))


wavepacket = np.zeros( (int( (params['tmax']-params['t0']) / params['dt']+1), 2*Nbas+2))
i=0
print(params['wavepacket_file'])
with open(params['wavepacket_file'],'r') as fl: 
    for line in fl:
        wrd = line.split()
        i= i+1
        for item in range(0,2*Nbas):
            wavepacket[i-1,item] = float(wrd[item]) 
        

#generate grids and weights
rbas = radbas(params['nlobatto'], params['nbins'], params['binwidth'], params['rshift'])
rgrid = rbas.r_grid()
nlobatto = params['nlobatto']
# polar plot grid
phi0 = 0.0 #np.pi/4 #at a fixed phi value
theta0 = 0.0#np.pi/4 #at a fixed phi value
r0 = 1.0
#generate Gauss-Lobatto quadrature
xlobatto=np.zeros(nlobatto)
wlobatto=np.zeros(nlobatto)
xlobatto,wlobatto=rbas.gauss_lobatto(nlobatto,14)
wlobatto=np.array(wlobatto)
xlobatto=np.array(xlobatto) # convert back to np arrays


rang = np.linspace(params['rshift'],params['nbins']*params['binwidth'],200,True,dtype=float)
gridtheta1d = 2 * np.pi * rang
rmesh, thetamesh = np.meshgrid(rang, gridtheta1d)

y = np.zeros((len(rang),len(rang)),dtype=complex)



#================ radial-angular in real space ===============#
fig = plt.figure(figsize=(5, 5), dpi=300, constrained_layout=True)

spec = gridspec.GridSpec(ncols=1, nrows=1,figure=fig)
axradang_r = fig.add_subplot(spec[0, 0],projection='polar')

for ielem,elem in enumerate(maparray):
    for i in range(len(rang)):
            y[i,:] +=    wavepacket[0,ielem] * rbas.chi(elem[2],elem[3],rang[:],rgrid,wlobatto) * sph_harm(elem[1], elem[0],  phi0, gridtheta1d[i])

line_angrad_r = axradang_r.contourf(thetamesh,rmesh,  rmesh*np.abs(y)/np.max(rmesh*np.abs(y)),cmap = 'jet',vmin=0.0, vmax=1.0) #cmap = jet, gnuplot, gnuplot2
#line_angrad_r, = axradang_r.contourf([],[], [],cmap = 'jet',vmin=0.0, vmax=1.0) #cmap = jet, gnuplot, gnuplot2
plt.colorbar(line_angrad_r,ax=axradang_r,aspect=30)


# animation function.  This is called sequentially
def animate(iteration):
    
    for ielem,elem in enumerate(maparray):
        for i in range(len(rang)):
            y[i,:] +=    wavepacket[iteration,ielem] * rbas.chi(elem[2],elem[3],rang[:],rgrid,wlobatto) * sph_harm(elem[1], elem[0],  phi0, gridtheta1d[i])

    line_angrad_r.set_data(y)
    plt.show()   
    return line_angrad_r,

anim = animation.FuncAnimation(fig, animate,frames=10, interval=500, blit=True)



anim.save(params['plot_controls']["animation_filename"]+".gif",writer='imagemagick',fps=2)
    






