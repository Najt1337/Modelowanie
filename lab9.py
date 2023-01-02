#%%
import matplotlib.pyplot as plt
import numpy as np


def R_s(n1,n2,theta):
    a=(n1*np.cos(theta)-n2*np.sqrt(1-(n1*np.sin(theta)/n2)**2))
    b=(n1*np.cos(theta)+n2*np.sqrt(1-(n1*np.sin(theta)/n2)**2))
    Rs=np.absolute(a/b)**2
    return Rs

def R_p(n1,n2,theta):
    a=(n1*np.sqrt(1-(n1*np.sin(theta)/n2)**2)-n2*np.cos(theta))
    b=(n1*np.sqrt(1-(n1*np.sin(theta)/n2)**2)+n2*np.cos(theta))
    Rp=np.absolute(a/b)**2
    return Rp

def R_eff(n1,n2,theta):
    R_eff=(R_s(n1,n2,theta)+R_p(n1,n2,theta))/2
    return R_eff

def T_eff(n1,n2,theta):
    T_eff=(1-R_s(n1,n2,theta)+1-R_p(n1,n2,theta))/2
    return T_eff

n1=1
n2=1.33
theta=np.radians(0)

R=np.absolute((n1-n2)/(n1+n2))**2

x=np.linspace(0,np.pi/2)
ys=R_s(n1,n2,x)
yp=R_p(n1,n2,x)
yR_eff=R_eff(n1,n2,x)
yT_eff=T_eff(n1,n2,x)
x1=np.linspace(0,90)
plt.figure()
plt.plot(x1,ys,label='Reflectance s-polarized light')
plt.plot(x1,yp,label='Reflectance p-polarized light')
plt.plot(x1,yR_eff,label='Effective reflectivity')
plt.plot(x1,yT_eff,label='Effective transmittance')
plt.xlim((0,90))
plt.legend()
plt.show()
#%%
l=np.random.uniform(0,1,10000000)
lac=0.000224723
r=-1*np.log(l)/lac
print('Teoretyczne',1/lac)
print('MMC',np.average(r),'std dev',np.std(r))
# %%
