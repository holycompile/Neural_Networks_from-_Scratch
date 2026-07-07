import numpy as np 
import math 

layer_outputs = [ [4.8,1.21,2.385],
                  [8.9, 1.81, 0.2],
                  [1.41, 1.051, 0.026]
                ]
exp_values=np.exp(layer_outputs)

print(np.sum(layer_outputs, axis=0))

#norm_values= exp_values/np.sum(exp_values)
'''
print(norm_values)
print("--------------------------------")
print(sum(norm_values)) # sum of all the normalized values should be 1
'''
