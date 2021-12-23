import numpy as np
from numpy import tan
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook


# def nozzle(x):
#     if x > 4:
#         return 2.
#     return x**2 * (6-x)/32 + 1

# def slope_nozzle(x):
#     return 3*(4-x)*x/32

def nozzle(x):
    return x**2/16 + 1

def slope_nozzle(x):
    return x/8

def theta_nozzle(x):
    return np.arctan(slope_nozzle(x))

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

def inter_char(theta,mu,x,y,i,j): #A améliorer ? (Itérer la méthode)
    """Crée un point par intersection des C+ et C-"""
    #P = [i,j] 
    #A = [i,j-1] selon la C-
    #B = [i-1,j] selon la C+
    ma = tan((theta[i,j]+theta[i,j-1]-mu[i,j]-mu[i,j-1])/2)
    mb = tan((theta[i,j]+theta[i-1,j]+mu[i,j]+mu[i-1,j])/2)
    xi = (y[i-1,j]-y[i,j-1] + ma*x[i,j-1] - mb*x[i-1,j]) / (ma-mb)
    yi = y[i,j-1] + (xi-x[i,j-1])*ma
    return xi,yi

def inter_nozzle(Kp,theta,mu,x,y,n,j): #A améliorer ? (Itérer la méthode)
    """Crée un point par intersection de la C+ et de la paroi"""
    ma = tan(theta[n-1,j-1])
    mb = tan(theta[n-1,j]+mu[n-1,j])
    xi = (y[n-1,j]-y[n-1,j-1] + ma*x[n-1,j-1] - mb*x[n-1,j]) / (ma-mb)
    yi = nozzle(xi)
    return xi,yi

def moc2d(x0,y0,theta0,M0,n):
    """Méthode des caractéristiques"""
    # Lignes = C-, colonnes = C+
    x = np.zeros((n+1,n))
    y = np.zeros((n+1,n))
    Km = np.zeros((n+1,n))
    Kp = np.zeros((n+1,n))
    theta = np.zeros((n+1,n))
    M = np.zeros((n+1,n))
    nu = np.zeros((n+1,n))
    mu = np.zeros((n+1,n))
    
    #Premières C+
    for j in range(n0):
        #Initialisation sur la courbe sonique
        x[n0-1-j,j] = x0[j]
        y[n0-1-j,j] = y0[j]
        theta[n0-1-j,j] = theta0[j]
        M[n0-1-j,j] = M0[j]
        nu[n0-1-j,j] = prandtl_meyer(M0[j])
        mu[n0-1-j,j] = np.arcsin(1/M0[j])
        Km[n0-1-j,j] = theta0[j]+nu[n0-1-j,j]
        Kp[n0-1-j,j] = theta0[j]-nu[n0-1-j,j]
        
        if j>0: 
            n1 = n0+j-1
            for i in range(n0-j,n1):
                Km[i,j] = Km[i,j-1] #Prolongement de la C-
                Kp[i,j] = Kp[i-1,j] #Prolongement de la C+
                theta[i,j] = (Km[i,j] + Kp[i,j])/2
                nu[i,j] = (Km[i,j] - Kp[i,j])/2 
                M[i,j] = inv_prandtl_meyer(nu[i,j])
                mu[i,j] = np.arcsin(1/M[i,j])
                x[i,j], y[i,j] = inter_char(theta,mu,x,y,i,j)
            
            #Résolution de M sur la paroi
            Kp[n1,j] = Kp[n1-1,j] #Prolongement de la C+
            x[n1,j], y[n1,j] = inter_nozzle(Kp,theta,mu,x,y,n1,j)
            theta[n1,j] = theta_nozzle(x[n1,j]) #theta est une donnée
            nu[n1,j] = - Kp[n1,j] + theta[n1,j]
            M[n1,j] = inv_prandtl_meyer(nu[n1,j])
            mu[n1,j] = np.arcsin(1/M[n1,j])
            Km[n1,j] = 2*theta[n1,j] - Kp[n1,j]
                
    #Nouvelles C+ à partir de l'axe
    j = n0
    fini = True
    while j<n-n0:
        if x[j-n0+1,j-1]>xmax: break
    
        #Prolongation de la C- la plus proche
        Km[j-n0+1,j] = Km[j-n0+1,j-1]
        Kp[j-n0+1,j] = - Km[j-n0+1,j-1]
        theta[j-n0+1,j] = 0
        nu[j-n0+1,j] = Km[j-n0+1,j-1]
        M[j-n0+1,j] = inv_prandtl_meyer(nu[j-n0+1,j])
        mu[j-n0+1,j] = np.arcsin(1/M[j-n0+1,j])
        x[j-n0+1,j] = x[j-n0+1,j-1] - y[j-n0+1,j-1] / tan((theta[j-n0+1,j-1]+theta[j-n0+1,j]-mu[j-n0+1,j-1]-mu[j-n0+1,j])/2);
        y[j-n0+1,j] = 0
        
        n1 = n0+j-1
        for i in range(j-n0+2, n1):
            if x[i-1,j]>xmax or x[i,:j].max()>xmax: 
                fini = False
                break
            
            Km[i,j] = Km[i,j-1] #Prolongement de la C-
            Kp[i,j] = Kp[i-1,j] #Prolongement de la C+
            theta[i,j] = (Km[i,j] + Kp[i,j])/2
            nu[i,j] = (Km[i,j] - Kp[i,j])/2 
            M[i,j] = inv_prandtl_meyer(nu[i,j])
            mu[i,j] = np.arcsin(1/M[i,j])
            x[i,j], y[i,j] = inter_char(theta,mu,x,y,i,j)
        
        #Résolution de M sur la paroi
        if fini:
            Kp[n1,j] = Kp[n1-1,j] #Prolongement de la C+
            x[n1,j], y[n1,j] = inter_nozzle(Kp,theta,mu,x,y,n1,j)
            theta[n1,j] = theta_nozzle(x[n1,j]) #theta est une donnée
            nu[n1,j] = - Kp[n1,j] + theta[n1,j]
            M[n1,j] = inv_prandtl_meyer(nu[n1,j])
            mu[n1,j] = np.arcsin(1/M[n1,j])
            Km[n1,j] = theta[n1,j] + nu[n1,j]
        
        j+=1

    #Masque
    masque = np.zeros((n+1,n))
    for i in range(n+1):
        for j in range(n):
            if (x[i,j] == 0. and y[i,j] == 0.) : masque[i,j] = 1
        
    return [np.ma.masked_array(item,masque) for item in [x,y,Km,Kp,theta,nu]]


#Paramètres physiques
gamma = 1.2336 #Cp/Cv
xmax = 4 #Longueur de la tuyère
R = nozzle(0) #Taille du col

#Paramètres de la méthode numérique
n0 = 25 #Nombre de points sur la ligne sonique
n = 20*n0 #Nombre max de caractéristiques (C+)

# Conditions initiales sur la ligne sonique
x0 = np.zeros(n0)
y0 = np.linspace(R,0,n0)
M0 = np.linspace(1.05,1.01,n0)
theta0 = np.linspace(0,0,n0)

#Exécution de la méthode des caractéristiques
[x,y,Km,Kp,theta,nu] = moc2d(x0,y0,theta0,M0,n)
M = inv_prandtl_meyer(nu)
mu = np.arcsin(1/M)


#Interpolation du champ de vitesse sur une grille 
ni = 30
u = np.multiply(M,np.cos(theta))
v = np.multiply(M,np.sin(theta))
grid_x, grid_y = np.mgrid[0:4:ni*1j, 0:2:ni*1j]
grid_u = interpolate.griddata((x.compressed(),y.compressed()), u.compressed(), (grid_x, grid_y), method='cubic')
grid_v = interpolate.griddata((x.compressed(),y.compressed()), v.compressed(), (grid_x, grid_y), method='cubic')
grid_M = interpolate.griddata((x.compressed(),y.compressed()), M.compressed(), (grid_x, grid_y), method='nearest')

grid_mask = np.zeros((ni,ni))
for i in range(ni):
    for j in range(ni):
        if grid_y[i,j] > nozzle(grid_x[i,j]):
            grid_mask[i,j] = 1
    
grid_u = np.ma.masked_array(grid_u,grid_mask)            
grid_v = np.ma.masked_array(grid_v,grid_mask)    
   

### Affichage ###

fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.05)
axs = gs.subplots(sharex=True, sharey=True)
(ax1, ax2), (ax3, ax4) = axs

#Plot du profil de tuyère
X=np.linspace(-0.1,4.1,100)
for i in range(2):
    for j in range(2):
        axs[i][j].plot(X,[nozzle(u) for u in X],'k')
        axs[i][j].plot(X,np.zeros(100),'k--')
        axs[i][j].axis('scaled')
        
#Plot 1 : Conditions initiales
ax1.plot(x0,y0,'ko')
ax1.quiver(x0,y0,M0*np.cos(theta0),M0*np.sin(theta0),width=0.002,scale_units='xy')

#Plot 2 : Traçage des caractéristiques
Cm = [[x[i,:],y[i,:]] for i in range(n)]
Cp = [[x[:,j],y[:,j]] for j in range(n)]
for i in range(n):
    ax2.plot(Cm[i][0],Cm[i][1], 'r')
    ax2.plot(Cp[i][0],Cp[i][1], 'g')
ax2.plot([xmax,xmax],[0,nozzle(xmax)],'k--')

#Plot 3 : Champ de vitesse
ax3.quiver(grid_x,grid_y,grid_u,grid_v,width=0.002, scale_units='xy')

#Plot 4 : Colormap des Mach
cm = ax4.pcolormesh(grid_x, grid_y, grid_M, norm=colors.LogNorm(vmin=1, vmax=M.max()), cmap='jet', shading='gouraud')
ax4.fill_between(X,[nozzle(u) for u in X],5+np.zeros(100),color='w')
cax = ax4.inset_axes([1.03, 0, 0.03, 1], transform=ax4.transAxes)
fig.colorbar(cm, ax=ax4, cax = cax)











