import math 

layer_outputs = [4.8,1.21,2.385]

#E= 2.7182
E=math.e 

exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)
    
print(exp_values)

norm_base = sum(exp_values) # sum of all the exponential values 

norm_values=[]

for value in exp_values:
    norm_values.append(value/norm_base)
    
print(norm_values)
print("--------------------------------")
print(sum(norm_values)) # sum of all the normalized values should be 1  
