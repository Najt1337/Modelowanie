# %%
import numpy as np
import matplotlib.pyplot as plt

population=[0,0,0,0,0,0,0,0,0,0,\
    1,1,1,1,1,1,1,1,1,1]
ones_counted=[]
def NewPopulation(N):
    for i in range(0,N+1):
        new_population=[]
        for k in range(0,20):
            x=np.random.choice(population)
            new_population.append(x)
        ones=new_population.count(1)
        ones_counted.append(ones)
    return ones_counted




iterations=int(input())
at_least14=[]
six_or_less=[]
results=NewPopulation(iterations)
plt.hist(ones_counted,density=True,rwidth=0.2,align='left')
plt.xlabel('Number of males in population of 20')
plt.ylabel('Probability')
plt.xlim((0,20))
for i in range(0,len(results)):
    if results[i]>=14:
        at_least14.append(results[i])
    elif results[i]<=6:
        six_or_less.append(results[i])


prob_14=len(at_least14)/len(results)
prob_6=len(six_or_less)/len(results)
print('Probability of 14 or more males in population of 20:',prob_14)
print('Probability of 6 or less males in population of 20:',prob_6)

# %%
