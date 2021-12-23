import numpy as np
from numpy import tan
import matplotlib.pyplot as plt


def prandtl_meyer(M):
    """Prandtl-Meyer function (output in radians)"""
    return (np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt((gamma-1)*(M**2-1)/(gamma+1)))-np.arctan(np.sqrt(M**2-1)))

def inv_prandtl_meyer(nu): #Approximation de l'inverse (OK pour Mach < 7)
    """Prandtl-Meyer inverse (input in radians)"""
    # Entrée en radians
    A=1.3604
    B=0.0962
    C=-0.5127
    D=-0.6722
    E=-0.3278
    nu_0=0.5*np.pi*(np.sqrt(6)-1)
    y=(nu/nu_0)**(2/3)
    return (1 + A*y + B*y**2 + C*y**3)/(1 + D*y + E*y**2)

def moc2d(theta_max,theta_0,n):
    # Lignes = C-, colonnes = C+
    x = np.zeros((n+1,n))
    y = np.zeros((n+1,n))
    Km = np.zeros((n+1,n))
    Kp = np.zeros((n+1,n))
    theta = np.zeros((n+1,n))
    M = np.zeros((n+1,n))
    nu = np.zeros((n+1,n))
    mu = np.zeros((n+1,n))
    
    wall = theta_max
    
    #Initialisation
    dtheta = (theta_max-theta_0)/(n-1);
    for i in range(n):
        theta[i,0] = theta_0 + i*dtheta
        nu[i,0] = theta[i,0]
        M[i,0] = inv_prandtl_meyer(nu[i,0])
        mu[i,0] = np.arcsin(1/M[i,0])
        Km[i,0] = theta[i,0]+nu[i,0]
        Kp[i,0] = theta[i,0]-nu[i,0]
        
    Kp[n,0] = Kp[n-1,0]
    theta[n,0] = theta[n-1,0]
    nu[n,0] = nu[n-1,0]
    M[n,0] = inv_prandtl_meyer(nu[n,0])
    mu[n,0] = np.arcsin(1/M[n,0])
    
    #Premier éventail
    x[0,0] = -D / (tan(theta[0,0]-mu[0,0]));
    y[0,0] = 0
    for i in range(1,n): 
        x[i,0] = (D-y[i-1,0] + x[i-1,0] * tan((mu[i-1,0]+theta[i-1,0]+mu[i,0]+theta[i,0])/2)) / (tan((mu[i-1,0]+theta[i-1,0]+mu[i,0]+theta[i,0])/2) - tan(theta[i,0]-mu[i,0]))
        y[i,0] = tan(theta[i,0]-mu[i,0])*x[i,0] + D
    #Prolongement sur la paroi (i=n)
    ma = tan((theta[n,0]+wall)/2)
    mb = tan((theta[n,0]+theta[n-1,0]+mu[n,0]+mu[n-1,0])/2)
    x[n,0] = (y[n-1,0]-D - mb*x[n-1,0]) / (ma-mb)
    y[n,0] = D + (x[n,0]-x[n,0-1])*ma
        
    for j in range(1,n):
        #Nouvelles C+ à partir de l'axe
        theta[j,j] = 0 
        Km[j,j] = Km[j,j-1] #Prolongement de la C- 
        nu[j,j] = Km[j,j] - theta[j,j]
        Kp[j,j] = theta[j,j] - nu[j,j]
        M[j,j] = inv_prandtl_meyer(nu[j,j])
        mu[j,j] = np.arcsin(1/M[j,j])
        x[j,j] = x[j,j-1] - y[j,j-1] / tan((theta[j,j-1]+theta[j,j]-mu[j,j-1]-mu[j,j])/2);
        y[j,j] = 0
        for i in range(j+1,n):
            Km[i,j] = Km[i,j-1] #Prolongement de la C-
            Kp[i,j] = Kp[i-1,j] #Prolongement de la C+
            theta[i,j] = (Km[i,j] + Kp[i,j])/2
            nu[i,j] = (Km[i,j] - Kp[i,j])/2
            M[i,j] = inv_prandtl_meyer(nu[i,j])
            mu[i,j] = np.arcsin(1/M[i,j])
            #P = [i,j] 
            #A = [i,j-1] selon la C-
            #B = [i-1,j] selon la C+
            ma = tan((theta[i,j]+theta[i,j-1]-mu[i,j]-mu[i,j-1])/2)
            mb = tan((theta[i,j]+theta[i-1,j]+mu[i,j]+mu[i-1,j])/2)
            x[i,j] = (y[i-1,j]-y[i,j-1] + ma*x[i,j-1] - mb*x[i-1,j]) / (ma-mb)
            y[i,j] = y[i,j-1] + (x[i,j]-x[i,j-1])*ma
        
        # Prolongement en dehors du triangle de Prandtl-Meyer
        Kp[n,j] = Kp[n-1,j]
        theta[n,j] = theta[n-1,j]
        nu[n,j] = nu[n-1,j]
        M[n,j] = inv_prandtl_meyer(nu[n-1,j])
        mu[n,j] = np.arcsin(1/M[n-1,j])
        #Prolongement sur la paroi (i=n)
        ma = tan((theta[n,j]+theta[n,j-1])/2)
        mb = tan((theta[n,j]+theta[n-1,j]+mu[n,j]+mu[n-1,j])/2)
        x[n,j] = (y[n-1,j]-y[n,j-1] + ma*x[n,j-1] - mb*x[n-1,j]) / (ma-mb)
        y[n,j] = y[n,j-1] + (x[n,j]-x[n,j-1])*ma
        
    #Masque
    masque = np.zeros((n+1,n))
    for i in range(n+1):
        masque[i,i+1:] = [1 for j in range(i+1,n)]
        
    return [np.ma.masked_array(item,masque) for item in [x,y,Km,Kp,theta,nu]]
    


gamma = 1.2336
Me = 2
n = 10 #Nombre de caractéristiques
theta_max = prandtl_meyer(Me) / 2
theta_0 = theta_max/n
D = 1 #Taille du goulot

[x,y,Km,Kp,theta,nu] = moc2d(theta_max,theta_0,n)

M = inv_prandtl_meyer(nu)
mu = np.arcsin(1/M)


# Lignes = C-, colonnes = C+
# grille = gen_grille(theta,mu)
x = np.concatenate([np.zeros((n+1,1)),x],axis=1)
y = np.concatenate([D*np.ones((n+1,1)),y],axis=1)

#Traçage des caractéristiques
Cm = [[x[i,:i+2],y[i,:i+2]] for i in range(0,n)]
Cp = [[x[j-1:,j],y[j-1:,j]] for j in range(1,n+1)]

#Traçage de la minimal length nozzle
min_len_noz = [x[n,:],y[n,:]]

#Champ de vitesse
u = np.multiply(M,np.cos(theta))
v = np.multiply(M,np.sin(theta))



#Affichage

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

for i in range(n):
    ax.plot(Cm[i][0],Cm[i][1], 'r')
    ax.plot(Cp[i][0],Cp[i][1], 'g')
ax.plot(min_len_noz[0],min_len_noz[1],'k')
# ax.quiver(grille[0],grille[1],u,v)

plt.show()



















