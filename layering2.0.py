import numpy as np 

inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]  # 3 x 4 

weights1 = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]  # 3 x 4 

biases1 = [2, 3, 0.5]   # 1 x 3 

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]   # 3 x 3 

biases2 = [-1, 2, -0.5]  # 1 x 3 

# Converting all fo them into numpy arrays 
 
input_array=np.array(inputs)# 3 x 4 
weight1_array=np.array(weights1)# 3 x 4 
weight2_array=np.array(weights2) # 3 x 3 
bias1_array=np.array(biases1)
bias2_array=np.array(biases2)

#Calculation of the 1 st layer 
layer1_output=np.dot(input_array, weight1_array.T) + bias1_array  # layer1_output shall be 3 x 3 

#Calculation of the 1 st layer 
layer2_output=np.dot(layer1_output, weight2_array.T) + bias2_array # layer2_output shall be also 3 x 3 

print(layer2_output)