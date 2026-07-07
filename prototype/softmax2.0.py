import numpy as np 
import math 

layer_outputs = [4.8,1.21,2.385]
E=math.e #E=2.7182

exp_values=np.exp(layer_outputs)

norm_values= exp_values/np.sum(exp_values)

print(norm_values)
print("--------------------------------")
print(sum(norm_values)) # sum of all the normalized values should be 1
