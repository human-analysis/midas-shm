# ///////////// Copyright 2024. All rights reserved. /////////////
# //
# //   Project     : MIDAS-shm
# //   File        : geometry_functions.py
# //   Description : Utility functions related to geometry
# //
# //   Created On: 8/19/2024
# /////////////////////////////////////////////////////////////////

import numpy as np

def get_geo_info(experiment):
    if experiment == 'simu':
        px = [-30,30,30,10,-10,-30,-30];
        py = [-30,-30,10,30,30,10,-30];

        x0 = [0,0,1,6]
        y0 = [5,6,6,6]
        s0 = np.multiply(y0,7) + x0
        s1 = np.arange(4)
        index_s=s0-s1
        xv, yv = np.meshgrid(np.arange(-25,30,8), np.arange(-25,30,8))

        xs = xv.reshape(-1,1)
        ys = yv.reshape(-1,1)
        xs = np.delete(xs,s0)
        ys = np.delete(ys,s0)
        return px, py, index_s, xv, yv, xs, ys
    
    else: # experimental work
        px = [0, 36.195, 36.195, 23.495, 0, 0]
        py = [0, 0, 31.877, 44.577, 44.577, 0]
    
        s0 = [20,25,26]
        s1 = np.arange(3)
        index_s=s0-s1
        xv, yv = np.meshgrid(np.arange(6,36,6.5),np.arange(7,44,6.5))
        xv = np.flip(xv)
        
        xs = xv.reshape(-1,1)
        ys = yv.reshape(-1,1)
        xs = np.delete(xs,s0)
        ys = np.delete(ys,s0)
        return px, py, index_s, xv, yv, xs, ys

def get_experimental_crack_info():
    l = 4
    cpx = [21.195, 21.195+l, 21.195+l, 21.195, 21.195]
    cpy = [22.910, 22.910, 23.310, 23.310, 22.910]
    angle = 0
    w = 23.310-22.910
    x = 21.195 + l/2
    y = (22.910+23.310)/2
    return cpx, cpy, angle, l, w, x, y

def get_simu_crack_info(name):
    idx1 = name.find('-x') 
    idx2 = name.find('-y')
    idx3 = name.find('-l')
    idx4 = name.find('-w')
    idx5 = name.find('-a')
    idx6 = name.find('-date')
    
    x = float(name[idx1+2:idx2]) # +2 to skip ordering '-x' itself
    y = float(name[idx2+2:idx3])
    l = float(name[idx3+2:idx4])
    w = float(name[idx4+2:idx5])
    angle = int(name[idx5+2:idx6])

    x12 = x + l/2*np.cos(np.deg2rad(angle))
    y12 = y + l/2*np.sin(np.deg2rad(angle))
    
    x1 = x12 - w/2*np.sin(np.deg2rad(angle))
    y1 = y12 + w/2*np.cos(np.deg2rad(angle))
    x2 = x12 + w/2*np.sin(np.deg2rad(angle))
    y2 = y12 - w/2*np.cos(np.deg2rad(angle))
    
    x34 = x - l/2*np.cos(np.deg2rad(angle))
    y34 = y - l/2*np.sin(np.deg2rad(angle))
    
    x4 = x34 - w/2*np.sin(np.deg2rad(angle))
    y4 = y34 + w/2*np.cos(np.deg2rad(angle))
    x3 = x34 + w/2*np.sin(np.deg2rad(angle))
    y3 = y34 - w/2*np.cos(np.deg2rad(angle))
    
    cpx = np.multiply([x1,x2,x3,x4,x1],100) #100 due to unit scale
    cpy = np.multiply([y1,y2,y3,y4,y1],100)
    return cpx, cpy, angle, l, w, x, y

def find_length(filenames):
    length_list = []
    for name in filenames:
        idx3 = name.find('-l')
        idx4 = name.find('-w')
        length = float(name[idx3+2:idx4])
        length_list.append(length)
    return np.array(length_list)