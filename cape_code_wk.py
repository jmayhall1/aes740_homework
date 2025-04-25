# coding=utf-8
'''
Python CAPE Code
Translation from George Bryan's Fortran CAPE Code.
'''
import metpy.calc as mpcalc
import numpy as np
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units
from metpy.calc.indices import WK82

def getcape(p_in, t_in, td_in, source_parc = 'mu', ml_depth = 300.0, adiabat = 'pl'):
    '''
    Ported from George Bryan's code: http://www2.mmm.ucar.edu/people/bryan/Code/getcape.F
    p_in: 1D pressure array (mb)
    t_in: 1D temperature array (C)
    td_in: 1D dewpoint array (C)

    Optional: 
    source_parc: source parcel (mu: most unstable, 
                                sb: surface, ml: mixed layer)
    ml_depth: depth of mixed layer (m)
    adiabat: Formulation of moist adiabat (pl: pseudoadiabatic, liquid only,
        rl: reversible, liquid only, pi: pseudoadiabatic, with ice, 
        ri: reversible, with ice)

    Returns:
    cape: Convective Available Potential Energy (J/kg)
    cin: Convective Inhibition (J/kg)
    '''
    cape=0
    cin = 0

    pinc = 100.0 #pressure increment (Pa)


    #defining constants:
    g     = 9.81
    p00   = 100000.0
    cp    = 1005.7
    rd    = 287.04
    rv    = 461.5
    xlv   = 2501000.0
    xls   = 2836017.0
    t0    = 273.15
    cpv   = 1875.0
    cpl   = 4190.0
    cpi   = 2118.636
    lv1   = xlv+(cpl-cpv)*t0
    lv2   = cpl-cpv
    ls1   = xls+(cpi-cpv)*t0
    ls2   = cpi-cpv

    rp00  = 1.0/p00
    eps   = rd/rv
    reps  = rv/rd
    rddcp = rd/cp
    cpdrd = cp/rd
    cpdg  = cp/g

    converge = 0.002
    debug_level=0


    #convert p, t, td to mks units, get pi, q, th, thv
    p = 100*np.array(p_in)
    t = 273.15+np.array(t_in)
    td = 273.15+np.array(td_in)
    pi = np.power(p*rp00, rddcp)
    q = getqvs(p,td)
    th = t/pi
    thv = th*(1.0+reps*q)/(1.0+q)

    #get hydrostatic heights
    z = np.zeros(len(p))

    for i in range(1,len(z)):
        dz = -cpdg*0.5*(thv[i]+thv[i-1])*(pi[i]-pi[i-1])
        z[i] = z[i-1] + dz


    if source_parc == 'sb':
        #surface parcel
        kmax = 0
    elif source_parc == 'mu':
        #use most unstable parcel
        if p[0] < 50000.0:
            #first pressure is above 500 mb, instead use the level reported.
            kmax = 0
            maxthe = getthe(p[0], t[0], td[0], q[0])
        else:
            #find max theta e below 500 mb.

            p500 = p>50000.0
            maxthe = 0.0
            the500 = list()
            for k in range(len(p)):
                if p[k] >= 50000.0:
                    the = getthe(p[k],t[k],td[k],q[k])
                    if the > maxthe:
                        maxthe = the
                        kmax = k

    elif source_parc == 'ml':
        #mixed layer.

        if (z[1] - z[0]) > ml_depth:
            #second level is above the ml depth, use the lowest level
            avgth = th[0]
            avgqv = q[0]
            kmax = 0

        elif z[-1] < ml_depth:
            #topmost level is within the mixed layer: 
            #use the upper-most level
            avgth = th[-1]
            avgqv = q[-1]
            kmax = len(th)-1

        else:
            #calculate the ml properties
            avgth = 0
            avgqv = 0
            k = 1

            while z[k] <= ml_depth and k <= len(z)-1:
                avgth = avgth + 0.5*(z[k] - z[k-1])*(th[k]+th[k-1])
                avgqv = avgqv + 0.5*(z[k] - z[k-1]) * (q[k] + q[k-1])
                k = k+1

            th2 = th[k-1] + (th[k] - th[k-1])*(ml_depth - z[k-1])/(z[k] - z[k-1])
            qv2 = q[k-1] + (q[k] - q[k-1]) * (ml_depth - z[k-1])/(z[k] - z[k-1])

            avgth = avgth + 0.5*(ml_depth-z[k-1]) * (th2+th[k-1])
            avgqv = avgqv + 0.5*(ml_depth - z[k-1])* (qv2+q[k-1])

            avgth = avgth/ml_depth
            avgqv = avgqv/ml_depth

            kmax = 0
    else:
        raise ValueError("Source Parcel can be ml, sb, or mu only.")

    narea = 0.0

    if source_parc == 'sb' or source_parc == 'mu':
        k = kmax
        th2 = th[kmax]
        pi2 = pi[kmax]
        p2 = p[kmax]
        t2 = t[kmax]
        thv2 = thv[kmax]
        qv2 = q[kmax]
        b2 = 0.0


    elif source_parc == 'ml':
        k = kmax
        th2 = avgth
        qv2 = avgqv
        thv2 = th2 * (1.0+reps*qv2)/(1.0+qv2)
        pi2 = pi[kmax]
        p2 = p[kmax]
        t2 = th2 * pi2
        b2 = g * (thv2 - thv[kmax])/thv[kmax]

    ql2 = 0.0
    qi2 = 0.0
    qt = qv2
    cape = 0.0
    cin = 0.0 
    lfc = 0.0

    doit = True
    cloud = False
    if 'i' in adiabat:
        ice=True
    else:
        ice = False

    the = getthe(p2,t2,t2,qv2)
    #begin ascent of parcel

    if debug_level >=100:
        print('Start Loop:')
        print('p2, th2, qv2', p2, th2, qv2)

    while doit and k <len(z)-1:
        k = k+1
        b1 = b2
        try:
            dp = p[k-1] - p[k]
        except IndexError:
            print(k, p[k-1], len(z))
            return
        if dp < pinc:
            nloop = 1
        else:
            nloop = 1 + int(dp/pinc)
            dp = dp/float(nloop)

        for n in range(nloop):
            p1 = p2
            t1 = t2
            pi1 = pi2
            th1 = th2
            qv1 = qv2
            ql1 = ql2 
            qi1 = qi2
            thv1 = thv2

            p2 = p2-dp
            pi2 = np.power(p2*rp00, rddcp)
            thlast = th1
            i = 0
            not_converged = True

            while not_converged:
                i = i+1
                t2 = thlast * pi2
                if ice:
                    fliq = max(min((t2-233.15)/(273.15-233.15),1.0),0.0)
                    fice = 1.0 - fliq

                else:
                    fliq = 1.0
                    fice = 0.0
                qv2 = min(qt, fliq*getqvs(p2,t2)+fice*getqvi(p2,t2))
                qi2 = max(fice*(qt-qv2),0.0)
                ql2 = max(qt-qv2-qi2, 0.0)
                tbar = 0.5 * (t1+t2)
                qvbar = 0.5 * (qv1 + qv2)
                qlbar = 0.5 * (ql1+ql2)
                qibar = 0.5 * (qi1 + qi2)

                lhv = lv1 - lv2 * tbar
                lhs = ls1-ls2*tbar
                lhf = lhs - lhv

                rm = rd+rv*qvbar
                cpm = cp+cpv*qvbar+cpl*qlbar+cpi*qibar
                
                th2 = th1 * np.exp(lhv * (ql2-ql1)/(cpm*tbar)+
                                   lhs * (qi2-qi1)/(cpm*tbar)+
                                   (rm/cpm-rd/cp)*np.log(p2/p1))

                if th2>1000:
                    #print(n,k,i)
                    pass
                    #print(th2, th1, lhv, ql2, ql1, cpm, tbar, lhs, qi2, qi1, rm, 
                    #rd, cp, p2, p1, np.log(p2/p1))

                if i>90:
                    #print(i,th2,thlast, th2-thlast)
                    pass

                if i>100:
                    raise ArithmeticError("Lack of convergence, stopping iteration")
                #print(th2, thlast, converge)
                if abs(th2-thlast) > converge:
                    thlast = thlast+0.3*(th2-thlast)
                else:
                    not_converged = False

            if ql2 >= 1.0e-10:
                cloud = True

            if 'p' in adiabat:
                #pseudoadiabat
                qt = qv2
                ql2 = 0.0
                qi2 = 0.0

        thv2 = th2 * (1.0+reps*qv2)/(1.0+qv2+ql2+qi2)
        b2 = g* (thv2-thv[k])/thv[k]
        dz = -cpdg * 0.5 * (thv[k] + thv[k-1])*(pi[k] - pi[k-1])

        the = getthe(p2,t2,t2,qv2)

        #Get contributions to CAPE and CIN
        if b2 >= 0.0 and b1 < 0.0:
            #first trip into positive area.
            ps = p[k-1]+(p[k]-p[k-1]) * (0.0-b1)/(b2-b1)
            frac = b2/(b2-b1)
            parea = 0.5*b2*dz*frac
            narea = narea - 0.5 * b1*dz*(1.0-frac)
            cin = cin+narea
        elif b2<0.0 and b1>0.0:
            #first trip into negative area
            ps = p[k-1] + (p[k]-p[k-1])*(0.0-b1)/(b2-b1)
            frac = b1/(b1-b2)
            parea = 0.5*b1*dz*frac
            narea = -0.5*b2*dz*(1.0-frac)

        elif b2<0.0:
            #still collecting negative buoyancy
            parea = 0.0
            narea = narea-0.5*dz*(b1+b2)
        else:
            #still collecting positive buoyancy
            parea = 0.5*dz*(b1+b2)
            narea = 0.0

        #print(b1,b2, parea,narea)
        cape = cape + max(0.0, parea)
        if p[k] <= 10000.0 and b2 <0.0:
            #stop if b<0 and p<100mb
            doit = False        


    return cape, cin


def getqvs(p,t):
    '''
    Gets the qv of solid?
    '''
    eps = 287.04/461.5
    es = 611.2*np.exp(17.67*(t-273.15)/(t-29.65))

    return eps*es/(p-es)

def getqvi(p,t):
    eps = 287.04/461.5

    es = 611.2*np.exp(21.8745584*(t-273.15)/(t-7.66))

    return eps*es/(p-es)


def getthe(p,t,td,q):
    if td - t >=-0.1:
        tlcl = t

    else:
        tlcl = 56.0 + np.power(np.power((td-56.0),-1) + 0.00125* np.log(t/td),-1)

    return (t*( np.power((100000.0/p),(0.2854*(1.0-0.28*q))) )   
            *np.exp( ((3376.0/tlcl)-2.54)*q*(1.0+0.81*q) ))

ds = WK82()
print(f'WK {getcape(np.array(ds.variables.get('pressure')), np.array(ds.variables.get('temperature')) - 273.15,
        np.array(ds.variables.get('dewpoint')))}')

