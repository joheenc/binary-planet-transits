import numpy as np
from scipy.optimize import minimize

def tdur_I(tdur, phi, Pbin, R_S, R_p, R_s, a_p, a_s, P=1440, a=628, b=0): #transit durations of case I (planet leads ingress, planet trails egress)
    tcor = 0
    if b != 0:
        tcor = ( -np.sqrt((R_S+R_p)**2-(b*R_S)**2) + np.sqrt((R_S+a_p*np.sin(phi)+R_p)**2-(b*R_S)**2) - a_p*np.sin(phi) \
                + np.sqrt((R_S-a_p*np.sin(phi+2*np.pi*tdur/Pbin)+R_p)**2-(b*R_S)**2) + a_p*np.sin(phi+2*np.pi*tdur/Pbin) \
                - np.sqrt((R_S+R_p)**2-(b*R_S)**2) ) / (2*np.pi*a / P)
    return (tdur - P/(2*np.pi*a) * (2*R_S+2*R_p+a_p*np.sin(phi)-a_p*np.sin(phi+2*np.pi*tdur/Pbin)) + tcor)**2

def tdur_II(tdur, phi, Pbin, R_S, R_p, R_s, a_p, a_s, P=1440, a=628, b=0): #transit durations of case II (planet leads ingress, satellite trails egress)
    tcor = 0
    if b != 0:
        tcor = ( -np.sqrt((R_S+R_p)**2-(b*R_S)**2) + np.sqrt((R_S+a_p*np.sin(phi)+R_p)**2-(b*R_S)**2) - a_p*np.sin(phi) \
                + np.sqrt((R_S+a_s*np.sin(phi+2*np.pi*tdur/Pbin)+R_s)**2-(b*R_S)**2) - a_s*np.sin(phi+2*np.pi*tdur/Pbin) \
                - np.sqrt((R_S+R_s)**2-(b*R_S)**2) ) / (2*np.pi*a / P)
    return (tdur - P/(2*np.pi*a) * (2*R_S+R_p+R_s+a_p*np.sin(phi)+a_s*np.sin(phi+2*np.pi*tdur/Pbin)) + tcor)**2

def tdur_III(tdur, phi, Pbin, R_S, R_p, R_s, a_p, a_s, P=1440, a=628, b=0): #transit durations of case II (satellite leads ingress, planet trails egress)
    tcor = 0
    if b != 0:
        tcor = ( -np.sqrt((R_S+R_s)**2-(b*R_S)**2) + np.sqrt((R_S-a_s*np.sin(phi)+R_s)**2-(b*R_S)**2) + a_s*np.sin(phi) \
                + np.sqrt((R_S-a_p*np.sin(phi+2*np.pi*tdur/Pbin)+R_p)**2-(b*R_S)**2) + a_p*np.sin(phi+2*np.pi*tdur/Pbin) \
                - np.sqrt((R_S+R_p)**2-(b*R_S)**2) ) / (2*np.pi*a / P)
    return (tdur - P/(2*np.pi*a) * (2*R_S+R_p+R_s-a_s*np.sin(phi)-a_p*np.sin(phi+2*np.pi*tdur/Pbin)) + tcor)**2

def tdur_IV(tdur, phi, Pbin, R_S, R_p, R_s, a_p, a_s, P=1440, a=628, b=0): #transit durations of case II (satellite leads ingress, satellite trails egress)
    tcor = 0
    if b != 0:
        tcor = ( -np.sqrt((R_S+R_s)**2-(b*R_S)**2) + np.sqrt((R_S-a_s*np.sin(phi)+R_s)**2-(b*R_S)**2) + a_s*np.sin(phi) \
                + np.sqrt((R_S+a_s*np.sin(phi+2*np.pi*tdur/Pbin)+R_s)**2-(b*R_S)**2) - a_s*np.sin(phi+2*np.pi*tdur/Pbin) \
                - np.sqrt((R_S+R_s)**2-(b*R_S)**2) ) / (2*np.pi*a / P)
    return (tdur - P/(2*np.pi*a) * (2*R_S+2*R_s-a_s*np.sin(phi)+a_s*np.sin(phi+2*np.pi*tdur/Pbin)) + tcor)**2

def tdur(phi, R_S=9.9512, R_p=1, R_s=1, a_sp=2, P=1500, a=615, b=0): #compute the full phase-dependence of transit duration by combining cases I-IV
    a_p = R_s**3/(R_s**3+R_p**3) * a_sp
    a_s = R_p**3/(R_s**3+R_p**3) * a_sp
    Pbin = 2*np.pi/3600 / np.sqrt(6.6743*10**-11*(R_p**3+R_s**3)*1.899*10**27/(a_sp*(R_s+R_p)*(7.149*10**7))**3)
    caseI = minimize(lambda tdur: tdur_I(tdur, phi, Pbin, R_S, R_p, R_s, a_p, a_s, b=b, a=a, P=P), x0=9)['x'][0]
    caseII = minimize(lambda tdur: tdur_II(tdur, phi, Pbin, R_S, R_p, R_s, a_p, a_s, b=b, a=a, P=P), x0=9)['x'][0]
    caseIII = minimize(lambda tdur: tdur_III(tdur, phi, Pbin, R_S, R_p, R_s, a_p, a_s, b=b, a=a, P=P), x0=9)['x'][0]
    caseIV = minimize(lambda tdur: tdur_IV(tdur, phi, Pbin, R_S, R_p, R_s, a_p, a_s, b=b, a=a, P=P), x0=9)['x'][0]
    return max(caseI, caseII, caseIII, caseIV)
