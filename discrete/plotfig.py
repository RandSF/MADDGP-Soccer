import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

p=pd.read_csv("pi_certain.csv")
P=[]
for x in p["0"]:
    P.append(eval(x))
P=np.array(P)
x=np.linspace(1,5,5)
y=np.linspace(1,len(P),len(P))
X,Y=np.meshgrid(x,y)
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.plot_surface(X,Y,P)
plt.show()