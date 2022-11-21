import numpy as np
import matplotlib.pyplot as plt

population=[0,0,0,0,0,0,0,0,0,0,\
    1,1,1,1,1,1,1,1,1,1]
new_population=[]
def NewPopulation(N):
    for i in range(0,N+1):
        x=np.random.choice(population)
        new_population.append(x)
    return None

NewPopulation(100)
print(new_population)