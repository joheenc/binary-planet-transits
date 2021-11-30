import numpy as np
from scipy.optimize import minimize

#transit durations of case I (planet leads ingress, planet trails egress)
def tdur_I(tdur, phi, pbin, R_S=1, R_p=1, R_s=1, a_sp=2, P=60, a=0.3, b=0, solarRad=False, jupRad=False, hrs=False):
    if not solarRad:
        R_S *= 9.9512   #convert R_S from solar radii to Jupiter radii
    if not jupRad:
        a *= 2092.51    #convert distance from AU to Jupiter radii
    if not hrs:
        P *= 24.        #convert period from days to hours
        
    a_p = R_s**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    a_s = R_p**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    return (tdur - P/(2*np.pi*a)*(2*np.sqrt((R_S+R_p)**2-(b*R_S)**2) + a_p*np.sin(phi) - a_p*np.sin(phi + 2*np.pi*tdur/pbin)))**2

#transit durations of case II (planet leads ingress, satellite trails egress)
def tdur_II(tdur, phi, pbin, R_S=1, R_p=1, R_s=1, a_sp=2, P=60, a=0.3, b=0, solarRad=False, jupRad=False, hrs=False):
    if not solarRad:
        R_S *= 9.9512   #convert R_S from solar radii to Jupiter radii
    if not jupRad:
        a *= 2092.51    #convert distance from AU to Jupiter radii
    if not hrs:
        P *= 24.        #convert period from days to hours
    
    a_p = R_s**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    a_s = R_p**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    return (tdur - P/(2*np.pi*a)*(np.sqrt((R_S+R_p)**2-(b*R_S)**2) + np.sqrt((R_S+R_s)**2-(b*R_S)**2)\
            + a_p*np.sin(phi) + a_s*np.sin(phi + 2*np.pi*tdur/pbin)))**2

#transit durations of case III (satellite leads ingress, planet trails egress)
def tdur_III(tdur, phi, pbin, R_S=1, R_p=1, R_s=1, a_sp=2, P=60, a=0.3, b=0, solarRad=False, jupRad=False, hrs=False):
    if not solarRad:
        R_S *= 9.9512   #convert R_S from solar radii to Jupiter radii
    if not jupRad:
        a *= 2092.51    #convert distance from AU to Jupiter radii
    if not hrs:
        P *= 24.        #convert period from days to hours

    a_p = R_s**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    a_s = R_p**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    return (tdur - P/(2*np.pi*a)*(np.sqrt((R_S+R_s)**2-(b*R_S)**2) + np.sqrt((R_S+R_p)**2-(b*R_S)**2)\
            - a_s*np.sin(phi) - a_p*np.sin(phi + 2*np.pi*tdur/pbin)))**2

#transit durations of case IV (satellite leads ingress, satellite trails egress)
def tdur_IV(tdur, phi, pbin, R_S=1, R_p=1, R_s=1, a_sp=2, P=60, a=0.3, b=0, solarRad=False, jupRad=False, hrs=False):
    if not solarRad:
        R_S *= 9.9512   #convert R_S from solar radii to Jupiter radii
    if not jupRad:
        a *= 2092.51    #convert distance from AU to Jupiter radii
    if not hrs:
        P *= 24.        #convert period from days to hours
        
    a_p = R_s**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    a_s = R_p**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    return (tdur - P/(2*np.pi*a)*(2*np.sqrt((R_S+R_s)**2-(b*R_S)**2) - a_s*np.sin(phi) + a_s*np.sin(phi + 2*np.pi*tdur/pbin)))**2

#compute the full phase-dependence of transit duration by combining cases I-IV
def tdur(phi, R_S=1, R_p=1, R_s=1, a_sp=2, P=60, a=0.3, b=0, solarRad=False, jupRad=False, hrs=False): 
    if not solarRad:
        R_S *= 9.9512   #convert R_S from solar radii to Jupiter radii
    if not jupRad:
        a *= 2092.51    #convert distance from AU to Jupiter radii
    if not hrs:
        P *= 24.        #convert period from days to hours

    a_p = R_s**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    a_s = R_p**3/(R_s**3+R_p**3) * a_sp * (R_p+R_s)
    pbin = Pbin(R_p, R_s, a_sp)
    caseI = minimize(lambda tdur: tdur_I(tdur, phi, pbin, R_S=R_S, R_p=R_p, R_s=R_s, a_sp=a_sp, \
                                         P=P, a=a, b=b, solarRad=True, jupRad=True, hrs=True), x0=9)['x'][0]
    caseII = minimize(lambda tdur: tdur_II(tdur, phi, pbin, R_S=R_S, R_p=R_p, R_s=R_s, a_sp=a_sp, \
                                         P=P, a=a, b=b, solarRad=True, jupRad=True, hrs=True), x0=9)['x'][0]
    caseIII = minimize(lambda tdur: tdur_III(tdur, phi, pbin, R_S=R_S, R_p=R_p, R_s=R_s, a_sp=a_sp, \
                                         P=P, a=a, b=b, solarRad=True, jupRad=True, hrs=True), x0=9)['x'][0]
    caseIV = minimize(lambda tdur: tdur_IV(tdur, phi, pbin, R_S=R_S, R_p=R_p, R_s=R_s, a_sp=a_sp, \
                                         P=P, a=a, b=b, solarRad=True, jupRad=True, hrs=True), x0=9)['x'][0] 
    return max(caseI, caseII, caseIII, caseIV)

#calculate binary orbital period assuming both planets are of Jupiter density
def Pbin(R_p, R_s, a_sp):
    Rjup = 7.149*10**7  #radius of Jupiter in m
    Mjup = 1.899*10**27 #mass of Jupiter in kg
    G = 6.6743*10**-11  #G in SI units
    return 2*np.pi/3600 / np.sqrt(G*(R_p**3+R_s**3)*Mjup/(a_sp*(R_s+R_p)*Rjup)**3)

def tdur_grid(phigrid, R_S=1, R_p=1, R_s=1, a_sp=2, P=60, a=0.3, b=0, solarRad=False, jupRad=False, hrs=False):
    if not solarRad:
        R_S *= 9.9512   #convert R_S from solar radii to Jupiter radii
    if not jupRad:
        a *= 2092.51 #convert distance from AU to Jupiter radii
    if not hrs:
        P *= 24.     #convert period from days to hours

    return np.array([tdur(phi, R_S=R_S, R_p=R_p, R_s=R_s, a_sp=a_sp, P=P, a=a, b=b, solarRad=True, jupRad=True, hrs=True) for phi in phigrid])
