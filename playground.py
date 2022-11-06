#Zadanie1
# %%
import matplotlib.pyplot as plt
import numpy as np

alpha=[0.5, 0.75, 1.0, 1.5, 2.0]
arguments = np.r_[0:1:0.1]
def exp(y,x):
    z=y**x
    return z

plt.figure()
plt.title('Plots of simple exponential function for different exponent values')
plt.xlabel('x')
plt.ylabel('f(x)')
for i in range(0,len(alpha)):
    plt.plot(arguments,exp(arguments,alpha[i]),label=str(alpha[i]))
plt.legend()    
    
#Zadanie2   
# %%
import pandas as pd
import matplotlib.pyplot as plt
HourFrame={'Subject Name':['Subject1','Subject2','Subject3','Subject4'],\
    'Hours':['1','2','3','4']}
df_hf=pd.DataFrame(HourFrame)
df_hf
total_hours=0
for i in range(0,len(HourFrame['Hours'])):
    total_hours=total_hours+int(HourFrame['Hours'][i])
labels=[]
for i in range(0,len(HourFrame['Subject Name'])):
    label=HourFrame['Subject Name'][i]
    labels.append(label)


sizes = []
for i in range(0,len(HourFrame['Hours'])):
    size=int(HourFrame['Hours'][i])/total_hours
    sizes.append(size)
print(sizes)    

fig1, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax[0].axis('equal')  
ax[1].bar(labels, sizes)
plt.show()

#Zadanie3
# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

df_csv1=pd.read_csv('zakazenia.csv', sep=';')
df_csv2=pd.read_csv('zakazenia2.csv', sep=',')
fig=plt.figure(figsize=(7,5))
ax=plt.gca()
ax.plot(df_csv1['Data'],df_csv1['Nowe przypadki'].astype(int))
ax.plot(df_csv2['Data'],df_csv2['Nowe przypadki'].astype(int))
fig.autofmt_xdate()

#Zadanie4
# %%
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
g=9.8
vx=20
p_values=[]
t=np.r_[0:2.1:0.1]
t_length=len(t)
for i in range(0,t_length):
    p=20-(g*t[i]**2)/2
    if p<0:
        break
    else:
        p_values.append(p)
  
xr=vx*math.sqrt(40/g)
x=np.r_[0:xr:2]   
ax.scatter(x,p_values,t) 
for angle in range(0,360):
    ax.view_init(30,angle)
    plt.draw()
    plt.pause(1)

#Zadanie5
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation,rc
from IPython.display import HTML, Image
rc('animation', html='html5')
fig, ax = plt.subplots()
ax.set_xlim((0,2))
ax.set_ylim((-2,2))
line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
line3, = ax.plot([], [], lw=2)
def animate(i):
    x=np.linspace(0,2,1000)
    y1=np.cos(2*np.pi*x-0.01*i)
    y2=np.cos(2*np.pi*x+0.01*i)
    y3=2*np.sin(2*np.pi*x)*np.cos(0.01*i)
    line1.set_data(x,y1)
    line2.set_data(x,y2)
    line3.set_data(x,y3)
    return line1, line2, line3,

anim=animation.FuncAnimation(fig,animate,frames=1000, interval=20, blit=True)
anim.save('continuousSineWave.gif', 
          writer = 'Pillow', fps = 30)

# %%
