import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
def func(x):

    # 两个高斯之和
    return (1 / 3) * ((1 / (sqrt(2 * np.pi) * 3 )) * np.exp(-(((x - 0.5) ** 2) / (2 * (3 ** 2))))) + \
           (2 / 3) * ((1 / (sqrt(2 * np.pi) * 4 )) * np.exp(-(((x - 3) ** 2) / (2 * (4 ** 2)))))


#采样的样本量
nums=10000
burn_period = 3000
count=0
points=[]

point=np.random.randn()
points.append(point)
while count<nums:
    #按照q(x,x')，采下一个点：均值为0，方差为1的标准正态分布采样+point偏置
    new_point=np.random.randn()+point
    #alpha(x,x')
    alpha=min(1.,func(new_point)/(1e-12+func(point)))
    #从(0,1)均匀采样一个u
    u=np.random.random()
    #判断是否接收新点还是旧点
    if u<alpha:
        points.append(new_point)
        point=new_point
    else:
        points.append(point)
    count+=1



x=np.linspace(-10,20,100)
plt.plot(x,func(x))
sns.distplot(points, bins=200, hist = True, norm_hist=True)
plt.legend(['original_data', 'sampling_data'])
plt.savefig('mcmc.png', dpi=250, bbox_inches='tight')
plt.show()