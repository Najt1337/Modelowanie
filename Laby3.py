# %%
#Zadanie1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sns
a=3
b=7
plt.figure()
plt.subplot(211)
Dst = np.random.uniform(a,b,10000)
count,bins,ignored=plt.hist(Dst,15,density=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.figure()
plt.subplot(211)
sns.distplot(Dst,hist=False)
avg_t=(a+b)/2
var_t=((b-a)**2)/12
std_dev_t=np.sqrt(var_t)
avg_emp=np.average(Dst)
var_emp=np.var(Dst)
std_dev_emp=np.std(Dst)
parameters={'Parametry':['Średnia','Wariancja','Odchylenie standardowe'],'Teoretyczna':[avg_t,var_t,std_dev_t],\
    'Empiryczna':[avg_emp,var_emp,std_dev_emp]}
df=pd.DataFrame(data=parameters)
df
# %%
#Zadanie2
import numpy as np
from numpy import random
outcomes=[1,2,3,4,5,6]
results=[]
for i in range(0,100):
    throw=np.random.choice(outcomes)
    results.append(throw)

counted_results=[]
def Counter(list):
    for i in range(1,7):
        res=list.count(i)
        print("liczba"+str(i)+':',res)
        counted_results.append(res)
    return None
Counter(results)
prob_dens=[]
def Probability(list):
    for i in range(0,len(counted_results)):
        prob=list[i]/len(results)
        prob_dens.append(prob)
    return None
Probability(counted_results)
print(prob_dens)
plt.hist(outcomes,bins=[1,2,3,4,5,6,7],density=True,rwidth=0.1, align='left',weights=prob_dens )
plt.ylabel('Gęstośc prawdopodobieństwa')
plt.xlabel('liczba oczek')
yticks=np.r_[0:0.275:0.025]
plt.yticks(yticks)
# %%
#Zadanie3.1
import numpy as np
import matplotlib.pyplot as plt
var1=0.2
var2=1
var3=5
var4=0.5
mu1=0
mu2=0
mu3=0
mu4=-2
sigma1=np.sqrt(var1)
sigma2=np.sqrt(var2)
sigma3=np.sqrt(var3)
sigma4=np.sqrt(var4)
bins=300
normal1=np.random.normal(mu1,sigma1,10000)
normal2=np.random.normal(mu2,sigma2,10000)
normal3=np.random.normal(mu3,sigma3,10000)
normal4=np.random.normal(mu4,sigma4,10000)
fig=plt.figure(figsize=(20,10))
ax1=plt.subplot(221)
counts1,bins,ignored=plt.hist(normal1,300,density=True)
plt.plot(bins,1/(sigma1*np.sqrt(2*np.pi))*np.exp(-(bins-mu1)**2/(2*sigma1**2)),linewidth=2,color='r')
ax1.title.set_text(r'$\mu=0,\sigma^2=0,2$')
ax2=plt.subplot(222)
counts2,bins,ignored=plt.hist(normal2,300,density=True)
plt.plot(bins,1/(sigma2*np.sqrt(2*np.pi))*np.exp(-(bins-mu2)**2/(2*sigma2**2)),linewidth=2,color='r')
ax2.title.set_text(r'$\mu=0,\sigma^2=1$')
ax3=plt.subplot(223)
counts3,bins,ignored=plt.hist(normal3,300,density=True)
plt.plot(bins,1/(sigma3*np.sqrt(2*np.pi))*np.exp(-(bins-mu3)**2/(2*sigma3**2)),linewidth=2,color='r')
ax3.title.set_text(r'$\mu=0,\sigma^2=5$')
ax4=plt.subplot(224)
counts4,bins,ignored=plt.hist(normal4,300,density=True)
plt.plot(bins,1/(sigma4*np.sqrt(2*np.pi))*np.exp(-(bins-mu4)**2/(2*sigma4**2)),linewidth=2,color='r')
ax4.title.set_text(r'$\mu=-2,\sigma^2=0,5$')


# %%
#Zadanie3.2
fig=plt.figure(figsize=(20,10))
bx1=plt.subplot(221)
pdf1=counts1/sum(counts1)
cdf1=np.cumsum(pdf1)
plt.plot(bins[1:],cdf1)
bx1.title.set_text(r'$\mu=0,\sigma^2=0,2$')
bx2=plt.subplot(222)
pdf2=counts2/sum(counts2)
cdf2=np.cumsum(pdf2)
plt.plot(bins[1:],cdf2)
bx2.title.set_text(r'$\mu=0,\sigma^2=1$')
bx3=plt.subplot(223)
pdf3=counts3/sum(counts3)
cdf3=np.cumsum(pdf3)
plt.plot(bins[1:],cdf3)
bx3.title.set_text(r'$\mu=0,\sigma^2=5$')
bx4=plt.subplot(224)
pdf4=counts4/sum(counts4)
cdf4=np.cumsum(pdf4)
plt.plot(bins[1:],cdf4)
bx4.title.set_text(r'$\mu=-2,\sigma^2=0,5$')
# %%
#Zadanie4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import pylab
muh=111.2681351
sigmah=0.943071459
height_rand=np.random.normal(muh,sigmah,10000)
print(np.mean(height_rand))
count,bins,ignored=plt.hist(height_rand,300,density=True,label='Histogram dla generatora random.normal')

plt.plot(bins,1/(sigmah*np.sqrt(2*np.pi))*np.exp(-(bins-muh)**2/(2*sigmah**2)),linewidth=2,color='r',label='Wykres otrzymany ze wzoru teoretycznego')
plt.xlabel('Wzrost')
plt.ylabel('Gęstość prawdopodobieństwa')
plt.title('Rozkład wzrostu polskich chłopców z grupy wiekowej lat 5, rok 1985')
plt.legend()
df = pd.read_csv("wzrost.csv")
df.head()

height=df.loc[df['Age group'] == 5, 'Mean height'].tolist()
print(height)
my_data=pd.Series(height)
plt.figure()



my_data = norm.rvs(size=1000)
sm.qqplot(my_data, line='45')
pylab.show()


# %%
#Zadanie5
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import seaborn as sns
at_least_two_sixes=1-(5/6)**5-(scipy.special.comb(5,1)*(1/6)*(5/6)**4)
two_sixes=(scipy.special.comb(5,2)*(1/36)*(5/6)**3)
one_six=(scipy.special.comb(5,1)*(1/6)*(5/6)**4)
zero_sixes=(5/6)**5
sum=at_least_two_sixes+one_six+zero_sixes
print(sum)
print(at_least_two_sixes)
print(two_sixes)
print(one_six)
print(zero_sixes)
bnml=np.random.binomial(5,1/6,10000)
count,bins,ignored=plt.hist(bnml,5,density=True,rwidth=0.3,align='left')
plt.xlabel('Liczba szóstek')
plt.ylabel('Prawdopodobieństwo')
plt.xlim((0,5.1))
#%%
#Zadanie6
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


size = 1000

sns.distplot(np.random.binomial(100, 0.5, size=100000),hist=False)
sns.distplot(np.random.binomial(100, 0.5, size=10000),hist=False)
sns.distplot(np.random.binomial(100, 0.5, size=1000),hist=False)
sns.distplot(np.random.normal(loc=50,scale=5,size=1000),hist=False, label='normal')
plt.legend(["$N=100000$", 
            "$N=10000$", 
            "$N=1000$",
            'normal'])




# %%
#Zadanie7
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
def Poisson(k):
    p=(2**k)*(np.exp(-2))/(np.math.factorial(k))
    return p

print('jeden raz',Poisson(1))
print('dwa razy',Poisson(2))
print('trzy razy',Poisson(3))
print('cztery razy',Poisson(4))
sns.distplot(random.poisson(lam=2, size=1000), kde=False)


