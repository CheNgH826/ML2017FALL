import numpy as np
from util import *

a = np.array([0.1,0.2,.3,.4,.6,.3,.8])
b = np.array([0,1,0,1,1,1,1])
print(determine_ans(a))
print(b)
print(compute_accuracy(a,b))
