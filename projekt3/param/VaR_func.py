import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import scipy.optimize as op


def add_VaR(df, distr):

    dist_type = getattr(st, distr)

    VaR_95 =[]
    VaR_99 = []
    
    for p in df['params']:
        p = eval(p)
        d = dist_type(*p)
        V95=( d.ppf(.95) )
        V99=( d.ppf(.99) )
        VaR_95.append( V95 )
        VaR_99.append( V99 )
        
    df['VaR_95'] = VaR_95
    df['VaR_99'] = VaR_99
    
    return(df)

def u(x, d):
    mu = d.stats('m')
    q = d.ppf(x)
    G = d.expect(lb=-x, ub=q)
    return( (-x * q + G) / (-mu + 2*G + (1-2*x) * q))

def u_test(x, d, alpha):
    return(u(x, d) - alpha)

def add_EVaR(df, distr):

    dist_type = getattr(st, distr)

    EVaR_95 =[]
    EVaR_99 = []
    for p in df['params']:
        p = eval(p)
        d = dist_type(*p)
        V95=( d.ppf(.95) )
        V99=( d.ppf(.99) )
        a95 = op.fsolve(u_test, 0.9, args=(d, 0.95))
        a99 = op.fsolve(u_test, 0.9, args=(d, 0.99))
        EVaR_95.append( d.ppf(a95) )
        EVaR_99.append( d.ppf(a99) )
    df['EVaR_95'] = EVaR_95
    df['EVaR_99'] = EVaR_99
    
    return(df)

def plot_vars(df,distr,data,evar=False, show=True,  save=False, name=''):
    
    plt.figure(figsize=(8*2, 6))
    #data
    ts = range(len(data))
    plt.plot(ts, data)
    plt.axvline(499, color='k')
    
    #vars
    ts1 = list(df['t_max'])[1:]
    v95 = list(df['VaR_95'])[1:]
    v99 = list(df['VaR_99'])[1:]
    plt.plot(ts1, v95, label=r'$VaR_{95}$')
    plt.plot(ts1, v99,label=r'$VaR_{99}$')
    
    if evar:
        ev95 = list(df['EVaR_95'])[1:]
        ev99 = list(df['EVaR_99'])[1:]
        plt.plot(ts1, ev95, label=r'E$VaR_{95}$')
        plt.plot(ts1, ev99,label=r'E$VaR_{99}$')
    plt.title('VaR  - {} distribution'.format(distr))
    plt.legend()
    if save:
        plt.savefig(name)
    if show:
        plt.show()