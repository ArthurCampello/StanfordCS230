# needed import
import numpy as np

# AUGMENTATION OF DATA

# augments data as outlined in report
def augment_scalar(data):
    # takes in original scalar data in numpy format
    # returns augmented data as described by report (data is augmented 6x)
    
    d_X_x1 = data[0:2, 0:data.shape[1]] # input positions of mass 1
    d_X_x2 = data[2:4, 0:data.shape[1]] # input positions of mass 2
    d_X_x3 = data[4:6, 0:data.shape[1]] # input positions of mass 3

    d_X_v1 = data[6+0:6+2, 0:data.shape[1]] # input velocities of mass 1
    d_X_v2 = data[6+2:6+4, 0:data.shape[1]] # input velocities of mass 2
    d_X_v3 = data[6+4:6+6, 0:data.shape[1]] # input velocities of mass 3

    d_Y_x1 = data[12+0:12+2, 0:data.shape[1]] # output positions of mass 1
    d_Y_x2 = data[12+2:12+4, 0:data.shape[1]] # output positions of mass 2
    d_Y_x3 = data[12+4:12+6, 0:data.shape[1]] # output positions of mass 3

    d_Y_v1 = data[18+0:18+2, 0:data.shape[1]] # output velocities of mass 1
    d_Y_v2 = data[18+2:18+4, 0:data.shape[1]] # output velocities of mass 2
    d_Y_v3 = data[18+4:18+6, 0:data.shape[1]] # output velocities of mass 3

    # data with original permutation (1,2,3)
    block_1 = np.vstack((d_X_x1, d_X_x2, d_X_x3, d_X_v1, d_X_v2, d_X_v3, d_Y_x1, d_Y_x2, d_Y_x3, d_Y_v1, d_Y_v2, d_Y_v3))
    print(block_1.shape)

    # data with original permutation (1,3,2)
    block_2 = np.vstack((d_X_x1, d_X_x3, d_X_x2, d_X_v1, d_X_v3, d_X_v2, d_Y_x1, d_Y_x3, d_Y_x2, d_Y_v1, d_Y_v3, d_Y_v2))

    # data with original permutation (2,1,3)
    block_3 = np.vstack((d_X_x2, d_X_x1, d_X_x3, d_X_v2, d_X_v1, d_X_v3, d_Y_x2, d_Y_x1, d_Y_x3, d_Y_v2, d_Y_v1, d_Y_v3))

    # data with original permutation (2,3,1)
    block_4 = np.vstack((d_X_x2, d_X_x3, d_X_x1, d_X_v2, d_X_v3, d_X_v1, d_Y_x2, d_Y_x3, d_Y_x1, d_Y_v2, d_Y_v3, d_Y_v1))

    # data with original permutation (3,1,2)
    block_5 = np.vstack((d_X_x3, d_X_x1, d_X_x2, d_X_v3, d_X_v1, d_X_v2, d_Y_x3, d_Y_x1, d_Y_x2, d_Y_v3, d_Y_v1, d_Y_v2))

    # data with original permutation (3,2,1)
    block_6 = np.vstack((d_X_x3, d_X_x2, d_X_x1, d_X_v3, d_X_v2, d_X_v1, d_Y_x3, d_Y_x2, d_Y_x1, d_Y_v3, d_Y_v2, d_Y_v1))
    
    # returns augmented data
    return np.vstack((block_1, block_2, block_3, block_4, block_5, block_6))

# TRANSFORMATION OF DATA INTO POLAR COORDINATES

# find r in polar coordinates from x and y in cartesian
def r_from_xy(x,y):
    return np.sqrt(x**2+y**2)

# find theta in polar coordinates from x and y in cartesian
def t_from_xy(x,y):
    return np.arctan(y/x)

# transforms data from cartesian to polar coordinates
def cartes_to_polar(data):
    # takes in data in cartesian coordinates and returns data in polar coordinates
    
    n = data.shape[1] # number of data points
 
    for i in range(12) # loops over mass entries
        for j in range(n): # loops over data points
            data[int(2*i)][j] = r_from_xy(data[int(2*i)][j],data[int(2*i+1)][j]) # finds r values
            data[int(2*i+1)][j] = t_from_xy(data[int(2*i)][j],data[int(2*i+1)][j]) # finds theta values
        
    return data # returns data in polar coordinates

# TRANSFORMATION OF DATA INTO BINARY VECTORS

# find binary vector from scalar value as specified in report
def v_from_x(x,n):
    
    # finds and returns v as defined by equation in report
    return [np.round(2**(i-2)*np.mod(x+1,2**(2-i))) for i in range(1,n+1) ]

# transforms data from scalar values to binary vectors  
def cartes_to_binary(data, n):
    # takes in data and n, binary vector length
    
    for i in range(data.shape[0]): # loops through data features 
        for j in range(data.shape[1]): # loops through data points
            data[i][j] = v_from_x(data[i][j],n) # converts scalar value to binary vector
            
    return data # returns data with points as binary vector (requires reshaping for use)