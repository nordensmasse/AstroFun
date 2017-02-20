# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:24:47 2017

@author: Mads Sørensen
"""

import os
import numpy
import math
import astropy
from astropy import units as u
from astropy import constants as const

def load_history(path):
    # Loads a MESA history file given path.
    # Returned is a array with named columns.

    #check  if last letter of path is /
    if path[-1] == '/':
        data = numpy.genfromtxt(path+'LOGS/history.data', skip_header=5, names=True)
    else:
        data = numpy.genfromtxt(path+'/LOGS/history.data', skip_header=5, names=True)
    return data

def get_model_name(n):
    # Supports function: unpack_grid_data
    model = 'BF' #Short for Binary Grid
    model_name = model+str('%08.0f'%(n))
    return model_name

def UnpackGridData(n, path_data):
    # Loads npz files made with MESA.
    model_name = get_model_name(n)
    datafile = path_data+model_name+'.npz'
    if os.path.isfile(datafile):
        data = numpy.load(datafile)
        if numpy.size(data.files) == 3:
            dB = data['B']
            dS1= data['S1']
            dS2= data['S2']
            flag = 0
            return dB,dS1,dS2
        else:
            dB = data['B']
            dS1= [1]
            dS2= [1]
            flag = 1
            return dB,dS1,dS2
    else:
        dB = [1]
        dS1= [2]
        dS2= [3]
        flag = 2
        return dB,dS1,dS2,flag

#Adopted from Tassos Frakos
def roche_lobe(m1, m2):
    if not isinstance(m1, u.Quantity):
        raise ValueError("m1 must be a Quantity with mass units")
    if not isinstance(m2, u.Quantity):
        raise ValueError("m2 must be a Quantity with mass units")

    q_mass1 = m1 / m2
    q_mass3 = q_mass1 ** (1.0 / 3.0)
    q_mass2 = q_mass1 ** (2.0 / 3.0)
    lobe = (0.49 * q_mass2) / (0.6 * q_mass2 + numpy.log(1.0 + q_mass3))
    return lobe  # Dimensionless. In units of orbital separation

#Adopted from Tassos Frakos
def separation_to_period(separation, m1, m2):
    if not isinstance(m1, u.Quantity):
        raise ValueError("m1 must be a Quantity with mass units")
    if not isinstance(m2, u.Quantity):
        raise ValueError("m2 must be a Quantity with mass units")
    if not isinstance(separation, u.Quantity):
        raise ValueError("sep must be a Quantity with length units")

    mbin = m1 + m2
    period = numpy.sqrt(4.0 * math.pi ** 2.0 * separation ** 3.0 / (const.G * mbin))
    return period.to('day')

#Adopted from Tassos Frakos
def period_to_separation(period, m1, m2):
    if not isinstance(m1, u.Quantity):
        raise ValueError("m1 must be a Quantity with mass units")
    if not isinstance(m2, u.Quantity):
        raise ValueError("m2 must be a Quantity with mass units")
    if not isinstance(period, u.Quantity):
        raise ValueError("period must be a Quantity with time units")

    mbin = m1 + m2
    separation = (period**2 * const.G * mbin / (4.0 * math.pi**2)) ** (1.0 / 3.0)
    return separation.to('Rsun')


def inv_one_part_power_law(x, xlow,xup,a1):
    #normalise
    kN1 = 1./(1.+a1)*(xup**(1.-a1)-xlow**(1.+a1))
    return (x*kN1*(1.+a1)+xlow**(1.+a1))**(1./(1.+a1))

def inv_two_part_power_law(x, xlow,xb,xup,a1,a2):

    #continuity
    kb = xb**(a1-a2)
    #normalise
    kN1 = 1./(1.+a1)*(xb**(1.+a1)-xlow**(1.+a1))
    kN2 = kN1 + kb/(1.+a2)*(xup**(1.+a2)-xb**(1.+a2))
    if type(x) != float:
        # separate between lower(small stars) an upper part of IMF
        # lower part
        x_small = x[x<kN1/kN2]
        n_small = numpy.size(x_small)
        x_1 = numpy.random.uniform(size=n_small)
        Nout1 =  (x_1*kN1*(1.+a1)+xlow**(1.+a1))**(1./(1.+a1))

        # upper part
        x_2 = x[x>=kN1/kN2]
        Nout2 = ((x_2*kN2-kN1)*(1.+a2)/kb+xb**(1.+a2))**(1./(1.+a2))
        # Collect into one array
        return numpy.concatenate((Nout1, Nout2))
    else:
        if x < kN1/kN2:
            return (x*kN1*(1.+a1)+xlow**(1.+a1))**(1./(1.+a1))
        else:
            return ((x*kN2-kN1)*(1.+a2)/kb+xb**(1.+a2))**(1./(1.+a2))

def normed_single_part_power_law(low,up,slope,x):
    N = 1./(slope+1.)*(up**(slope+1.)-low**(slope+1.))
    f = 1./N*x**slope
    return f

def normed_two_part_power_law(xlow,xup,xbreak,slope1,slope2,x):
    k = xbreak**(slope1-slope2)
    N1 = 1./(slope1+1.)*(xbreak**(slope1+1.)-xlow**(slope1+1.))
    N = N1 + k/(slope2+1.)*(xup**(slope2+1.)-xbreak**(slope2+1.))
    if type(x) == float:
        if x <= xbreak:#lower part
            return 1./(N)*x**slope1
        else:
            return k/(N)*x**slope2
    else:
        #lower part
        x1 = x[x<xbreak]
        f1 = 1./(N)*x1**slope1
        #upper part
        x2 = x[x>=xbreak]
        f2 = k/(N)*x2**slope2
        f = numpy.concatenate((f1, f2))
        return f


def radioactive_decay(N, N0, t, thalf,case):
    if case == 'forward':
        N = N0*numpy.exp(-t/thalf)
        return N
    elif case == 'backward':
        N0 = N/numpy.exp(-t/thalf)
        return N0
    else:
        print 'chose case: forward or backward'
        return 0

def Mej_Fe60(Msn):
    #Yields is taken as a straight line fit.
    Mej = Msn*(1.0-0.25)/(25.-10.)*1e-4
    return Mej

def radio_active_fluence(t,r,U,Mej,A,mp,thalf):
    #Number of atom taken up of radioactive material pr. cm*cm and incorporated into a surface layer.
    Mej = Mej*u.Msun
    Mej = Mej.to(u.kg)
    Mej= Mej.value
    F=U/4.*Mej/(4*numpy.pi*A*mp*(r*u.pc.to(u.cm))**2.)*numpy.exp(-t/numpy.log(2)/thalf)
    return F

if __name__ == "__main__":
    print 'Module AstroFun.py loaded.'
