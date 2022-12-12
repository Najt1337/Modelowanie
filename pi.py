# %%
#Zadanie1
import numpy as np


def f_avg(x):
    f=np.sqrt(1-x**2)
    return f

samples=[50,100,1000,10000]
x50=[]
x100=[]
x1k=[]
x10k=[]
for i in samples:
    x=np.random.random(i)
    if i==50:
        x50.extend(x)
    elif i==100:
        x100.extend(x)
    elif i==1000:
        x1k.extend(x)
    else:
        x10k.extend(x)

x50f=[]
x100f=[]
x1kf=[]
x10kf=[]
for i in samples:
    if i==50:
        for k in range(0,i):
            fx=f_avg(x50[k])
            x50f.append(fx)
    elif i==100:
        for k in range(0,i):
            fx=f_avg(x100[k])
            x100f.append(fx)
    elif i==1000:
        for k in range(0,i):
            fx=f_avg(x1k[k])
            x1kf.append(fx)
    else:
        for k in range(0,i):
            fx=f_avg(x10k[k])
            x10kf.append(fx)

x50f4=[item*4 for item in x50f]
x100f4=[item*4 for item in x100f]
x1kf4=[item*4 for item in x1kf]
x10kf4=[item*4 for item in x10kf]
avg50=np.average(x50f4)
avg100=np.average(x100f4)
avg1k=np.average(x1kf4)
avg10k=np.average(x10kf4)
var50=np.var(x50f4)
var100=np.var(x100f4)
var1k=np.var(x1kf4)
var10k=np.var(x10kf4) 
print('N=50',avg50,var50,'N=100',avg100,var100,'N=1000',avg1k,var1k,'N=10000',avg10k,var10k)  
# %%
#Zadanie2
#https://www.bragitoff.com/2021/05/value-of-pi-using-monte-carlo-python-program/
import random
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
 
fig = figure(figsize=(8, 8), dpi=120)
 
samples=[50,100,1000,10000]
radius = 1
nInside = 0
nDrops = 0
  

fig1 = plt.figure(1)
plt.get_current_fig_manager().window.wm_geometry("+00+00") 
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend() 

isFirst1 = True
isFirst2 = True
 
piValueI = []
nDrops_arr = []
 

insideX = []
outsideX = []
insideY = []
outsideY = []
circle_points=0
square_points=0

for i in range(samples[2]):
    x1 = random.uniform(0, 1)
    y1 = random.uniform(0, 1)
    r = x1**2 + y1**2
    nDrops = nDrops + 1
    if r <= 1:
        circle_points += 1
        insideX.append(x1)
        insideY.append(y1)              
    else:
        outsideX.append(x1)
        outsideY.append(y1)
 
    square_points += 1
 
    pi = 4 * circle_points / square_points
           
    if i%100==0:
        
        plt.figure(1)
        
        if isFirst1:
             
            
            plt.scatter(insideX,insideY,c='pink',s=50,label='Drop inside')
            isFirst1 = False
            plt.legend(loc=(0.75, 0.9))
        else:
            
           plt.scatter(insideX,insideY,c='pink',s=50)
        
        plt.figure(1)
        
        if isFirst2:
            
            plt.scatter(outsideX,outsideY,c='orange',s=50,label='Drop outside')
            isFirst2 = False
            plt.legend(loc=(0.75, 0.9))
        else:
            
           plt.scatter(outsideX,outsideY,c='orange',s=50)
         
          
        area = 4*nInside/nDrops
        plt.figure(1)
        plt.title('No. of pin drops = '+str(nDrops)+';         No. inside circle = '+str(nInside)+r';         π  ≈ $4\frac{N_\mathrm{inside}}{N_\mathrm{total}}=$ '+str(np.round(area,6)))
        piValueI.append(area)
        nDrops_arr.append(nDrops)
       
        
        plt.pause(0.1)
             
     
print(pi)
plt.show()
# %%
#Zadanie3 Orzeł-reszka
import random
import numpy as np
samples=[50,100,1000,10000]
x50=[]
x100=[]
x1k=[]
x10k=[]
for i in samples:
    for k in range(0,i):
        x=random.uniform(0,np.pi/2)
        if i==50:
            x50.append(x)
        elif i==100:
            x100.append(x)
        elif i==1000:
            x1k.append(x)
        else:
            x10k.append(x)

y50=[]
y100=[]
y1k=[]
y10k=[]
for i in samples:
    for k in range(0,i):
        y=random.uniform(0,1)
        if i==50:
            y50.append(y)
        elif i==100:
            y100.append(y)
        elif i==1000:
            y1k.append(y)
        else:
            y10k.append(y)

count50=[]
count100=[]
count1k=[]
count10k=[]

for i in samples:
    for k in range(0,i):
        if i==50:
            p=np.abs(np.cos(x50[k]))
            if p>=y50[k]:
                count50.append(1)
            else:
                count50.append(0)

        elif i==100:
            p=np.abs(np.cos(x100[k]))
            if p>=y100[k]:
                count100.append(1)
            else:
                count100.append(0)
        elif i==1000:
            p=np.abs(np.cos(x1k[k]))
            if p>=y1k[k]:
                count1k.append(1)
            else:
                count1k.append(0)
        else:
            p=np.abs(np.cos(x10k[k]))
            if p>=y10k[k]:
                count10k.append(1)
            else:
                count10k.append(0)

n50=count50.count(1)
N50=len(count50)
pi50=2*N50/n50
n100=count100.count(1)
N100=len(count100)
pi100=2*N100/n100
n1k=count1k.count(1)
N1k=len(count1k)
pi1k=2*N1k/n1k
n10k=count10k.count(1)
N10k=len(count10k)
pi10k=(2*N10k)/n10k
print(pi50,2.374/np.sqrt(samples[0]),pi100,2.374/np.sqrt(samples[1])\
    ,pi1k,2.374/np.sqrt(samples[2]),pi10k,2.374/np.sqrt(samples[3]))

# %%
#Zadanie3 podstawowe MC
import random
import numpy as np
samples=[50,100,1000,10000]
x50,y50,w50=[],[],[]
x100,y100,w100=[],[],[]
x1k,y1k,w1k=[],[],[]
x10k,y10k,w10k=[],[],[]


for i in samples:
    for k in range(0,i):
        x=random.uniform(0,np.pi/2)
        y=random.uniform(0,1)
        if i==50:
            x50.append(x)
            w=np.cos(x)
            w50.append(w)
            y50.append(y)
        elif i==100:
            x100.append(x)
            w=np.cos(x)
            w100.append(w)
            y100.append(y)
        elif i==1000:
            x1k.append(x)
            w=np.cos(x)
            w1k.append(w)
            y1k.append(y)
        else:
            x10k.append(x)
            w=np.cos(x)
            w10k.append(w)
            y10k.append(y)
                   
w50sum=sum(w50)/len(w50)
w100sum=sum(w100)/len(w100)
w1ksum=sum(w1k)/len(w1k)
w10ksum=sum(w10k)/len(w10k)
print(w50sum,w100sum,w1ksum,w10ksum)
count50=[]
count100=[]
count1k=[]
count10k=[]

for i in samples:
    for k in range(0,i):
        if i==50:           
            if w50sum>=y50[k]:
                count50.append(1)
            else:
                count50.append(0)

        elif i==100:
            if w100sum>=y100[k]:
                count100.append(1)
            else:
                count100.append(0)
        elif i==1000:
            if w1ksum>=y1k[k]:
                count1k.append(1)
            else:
                count1k.append(0)
        else:
            if w10ksum>=y10k[k]:
                count10k.append(1)
            else:
                count10k.append(0)

print(count50)

n50=count50.count(1)
N50=len(count50)
pi50=2*N50/n50
n100=count100.count(1)
N100=len(count100)
pi100=2*N100/n100
n1k=count1k.count(1)
N1k=len(count1k)
pi1k=2*N1k/n1k
n10k=count10k.count(1)
N10k=len(count10k)
pi10k=(2*N10k)/n10k
print(pi50,1.52/np.sqrt(samples[0]),pi100,1.52/np.sqrt(samples[1])\
    ,pi1k,1.52/np.sqrt(samples[2]),pi10k,1.52/np.sqrt(samples[3]))  
             

# %%
