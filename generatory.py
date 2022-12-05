# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
x=[1]
lists=[]
for i in range(1,11):
    lists.append([1])
    for k in range(1,11):
        z=i*lists[i-1][k-1]
        y=z%11
        lists[i-1].append(y)

correlation=[]
correlation_binary=[]
for k in range(0,10):
    xs=[]
    ys=[]
    xs_binary=[]
    ys_binary=[]
    ys_norm=[]
    for i in range(0,11):
        ys_i=lists[k][i]
        ys_normed=(lists[k][i])/10
        ys_i_binary=int(np.binary_repr(lists[k][i]))
        ys_binary.append(ys_i_binary)
        ys_norm.append(ys_normed)
        ys.append(ys_i)
        xs_i=lists[k][i-1]
        xs_i_binary=int(np.binary_repr(lists[k][i-1]))
        xs_binary.append(xs_i_binary)
        xs.append(xs_i)
    u=np.corrcoef(xs,ys)
    correlation.append(u)
    u_binary=np.corrcoef(xs_binary,ys_binary)
    correlation_binary.append(u_binary)     
    #plt.figure()
    #plt.scatter(xs,ys)
    #plt.figure()
    #plt.hist(ys_norm,density=True)

print(lists)
print(ys,xs)
averages=[]
variances=[]

for i in range(0,len(lists)):
    avg=np.average(lists[i])
    averages.append(avg)
    var=np.var(lists[i])
    variances.append(var)
    

print(averages,variances,correlation,correlation_binary)

# %%

# %%
