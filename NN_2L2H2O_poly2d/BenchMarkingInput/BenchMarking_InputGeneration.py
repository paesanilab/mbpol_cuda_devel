
# coding: utf-8

# # Generate input data array for benchmarking of NN_2L2H2O_poly2d with cudnn

# In[1]:

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd

np.random.seed(2016)  # for reproducibility


# # read data (distances, E_to_fit, binding E) from data file

# In[5]:

## mapping distances
mapping = [("H1(a)","H2(a)"),
("H1(b)","H2(b)"),
("O(a)","H1(a)"),
("O(a)","H2(a)"),
("O(b)","H1(b)"),
("O(b)","H2(b)"),
("H1(a)","H1(b)"),
("H1(a)","H2(b)"),
("H2(a)","H1(b)"),
("H2(a)","H2(b)"),
("O(a)","H1(b)"),
("O(a)","H2(b)"),
("O(b)","H1(a)"),
("O(b)","H2(a)"),
("O(a)","O(b)"),
("L1(a)","H1(b)"),
("L1(a)","H2(b)"),
("L2(a)","H1(b)"),
("L2(a)","H2(b)"),
("L1(b)","H1(a)"),
("L1(b)","H2(a)"),
("L2(b)","H1(a)"),
("L2(b)","H2(a)"),
("O(a)","L1(b)"),
("O(a)","L2(b)"),
("O(b)","L1(a)"),
("O(b)","L2(a)"),
("L1(a)","L1(b)"),
("L1(a)","L2(b)"),
("L2(a)","L1(b)"),
("L2(a)","L2(b)")]


# ### mapping in mbpol
# 
# ```
#  0 :  H1(a) <===>  H2(a) : x-intra-HH
#  1 :  H1(b) <===>  H2(b) : x-intra-HH
#  2 :   O(a) <===>  H1(a) : x-intra-OH
#  3 :   O(a) <===>  H2(a) : x-intra-OH
#  4 :   O(b) <===>  H1(b) : x-intra-OH
#  5 :   O(b) <===>  H2(b) : x-intra-OH
#  6 :  H1(a) <===>  H1(b) : x-HH
#  7 :  H1(a) <===>  H2(b) : x-HH
#  8 :  H2(a) <===>  H1(b) : x-HH
#  9 :  H2(a) <===>  H2(b) : x-HH
# 10 :   O(a) <===>  H1(b) : x-OH
# 11 :   O(a) <===>  H2(b) : x-OH
# 12 :   O(b) <===>  H1(a) : x-OH
# 13 :   O(b) <===>  H2(a) : x-OH
# 14 :   O(a) <===>   O(b) : x-OO
# 15 :  L1(a) <===>  H1(b) : x-LH
# 16 :  L1(a) <===>  H2(b) : x-LH
# 17 :  L2(a) <===>  H1(b) : x-LH
# 18 :  L2(a) <===>  H2(b) : x-LH
# 19 :  L1(b) <===>  H1(a) : x-LH
# 20 :  L1(b) <===>  H2(a) : x-LH
# 21 :  L2(b) <===>  H1(a) : x-LH
# 22 :  L2(b) <===>  H2(a) : x-LH
# 23 :   O(a) <===>  L1(b) : x-OL
# 24 :   O(a) <===>  L2(b) : x-OL
# 25 :   O(b) <===>  L1(a) : x-OL
# 26 :   O(b) <===>  L2(a) : x-OL
# 27 :  L1(a) <===>  L1(b) : x-LL
# 28 :  L1(a) <===>  L2(b) : x-LL
# 29 :  L2(a) <===>  L1(b) : x-LL
# 30 :  L2(a) <===>  L2(b) : x-LL
# ```
# 

# In[6]:

data_column_names = mapping + ['2BEtofit','BindingE']


# In[7]:

data_with_header = pd.read_table('./NN_input_2LHO_correctedD6_f64.dat', sep='\s+',                                  header=0,names = data_column_names, index_col=None,lineterminator='\n') 


# In[8]:

data_with_header.shape


# In[9]:

data_with_header.columns


# In[10]:

#data_with_header.columns[:31]


# In[11]:

# Initial data
distances = data_with_header[data_with_header.columns[:31]].values
E_ref = data_with_header['2BEtofit'].values
Eb  = data_with_header['BindingE'].values


# # data preprocessing: remove Eb>60, then:
# ## X: distances -> exp => poly
# ## y: E_to_fit -> shift => log
# ## weights: f(Eb)

# In[12]:

# Remove ~400 configurations with binding energies >60 kcal/mol
idx_low = np.where(Eb < 60)
distances = distances[idx_low]
E_ref = E_ref[idx_low]
Eb = Eb[idx_low]


# In[13]:

# exp. transformation before symmetrization 
#distance_exp = np.exp(-distances)
distance_exp = pd.DataFrame(data = np.exp(-distances))


# In[14]:

def poly_2d(x):
    p = pd.DataFrame()
    p[0] = x[18] + x[19] + x[17] + x[16] + x[22] + x[21] + x[20] + x[15];
    p[1] = x[30] + x[29] + x[28] + x[27];
    p[2] = x[14];
    p[3] = x[26] + x[23] + x[24] + x[25];
    p[4] = x[12] + x[13] + x[10] + x[11];
    p[5] = x[7] + x[6] + x[9] + x[8];

    p[6] = x[15]*x[17] + x[19]*x[21] + x[20]*x[22] + x[16]*x[18];
    p[7] = x[15]*x[8] + x[17]*x[8] + x[17]*x[6] + x[22]*x[8] + x[19]*x[6] + x[15]*x[6] + x[19]*x[7] + x[18]*x[9] + x[16]*x[9] + x[20]*x[8] + x[22]*x[9] + x[21]*x[7] + x[21]*x[6] + x[16]*x[7] + x[20]*x[9] + x[18]*x[7];
    p[8] = x[0]*x[22] + x[0]*x[20] + x[18]*x[1] + x[15]*x[1] + x[17]*x[1] + x[16]*x[1] + x[0]*x[19] + x[0]*x[21];
    p[9] = x[16]*x[25] + x[19]*x[23] + x[17]*x[26] + x[15]*x[25] + x[20]*x[23] + x[22]*x[24] + x[18]*x[26] + x[21]*x[24];
    p[10] = x[1]*x[8] + x[1]*x[9] + x[0]*x[8] + x[0]*x[9] + x[1]*x[7] + x[0]*x[7] + x[0]*x[6] + x[1]*x[6];
    p[11] = x[21]*x[25] + x[22]*x[26] + x[19]*x[25] + x[15]*x[24] + x[17]*x[23] + x[17]*x[24] + x[21]*x[26] + x[20]*x[26] + x[19]*x[26] + x[20]*x[25] + x[22]*x[25] + x[16]*x[24] + x[18]*x[23] + x[16]*x[23] + x[15]*x[23] + x[18]*x[24];
    p[12] = x[4]*x[6] + x[3]*x[8] + x[2]*x[6] + x[5]*x[7] + x[3]*x[9] + x[4]*x[8] + x[2]*x[7] + x[5]*x[9];
    p[13] = x[16]*x[27] + x[15]*x[27] + x[20]*x[29] + x[15]*x[28] + x[17]*x[29] + x[19]*x[27] + x[21]*x[30] + x[19]*x[29] + x[17]*x[30] + x[22]*x[28] + x[18]*x[29] + x[21]*x[28] + x[22]*x[30] + x[18]*x[30] + x[16]*x[28] + x[20]*x[27];
    p[14] = x[15]*x[21] + x[18]*x[22] + x[15]*x[19] + x[18]*x[19] + x[17]*x[19] + x[17]*x[20] + x[18]*x[21] + x[16]*x[22] + x[15]*x[22] + x[18]*x[20] + x[16]*x[20] + x[17]*x[21] + x[15]*x[20] + x[16]*x[19] + x[16]*x[21] + x[17]*x[22];
    p[15] = x[11]*x[30] + x[12]*x[29] + x[11]*x[28] + x[13]*x[27] + x[13]*x[28] + x[12]*x[27] + x[12]*x[28] + x[10]*x[27] + x[10]*x[29] + x[12]*x[30] + x[13]*x[29] + x[11]*x[29] + x[10]*x[28] + x[11]*x[27] + x[13]*x[30] + x[10]*x[30];
    p[16] = x[29]*x[2] + x[27]*x[2] + x[28]*x[2] + x[29]*x[5] + x[30]*x[5] + x[27]*x[4] + x[29]*x[3] + x[27]*x[3] + x[28]*x[4] + x[27]*x[5] + x[29]*x[4] + x[30]*x[3] + x[28]*x[5] + x[2]*x[30] + x[28]*x[3] + x[30]*x[4];
    p[17] = x[30]*x[9] + x[29]*x[7] + x[29]*x[9] + x[29]*x[6] + x[27]*x[7] + x[27]*x[6] + x[28]*x[7] + x[28]*x[9] + x[27]*x[8] + x[30]*x[7] + x[28]*x[8] + x[27]*x[9] + x[28]*x[6] + x[30]*x[8] + x[30]*x[6] + x[29]*x[8];
    p[18] = x[13]*x[14] + x[12]*x[14] + x[11]*x[14] + x[10]*x[14];
    p[19] = x[10]*x[11] + x[12]*x[13];
    p[20] = x[6]*x[6] + x[7]*x[7] + x[8]*x[8] + x[9]*x[9];
    p[21] = x[26]*x[3] + x[25]*x[2] + x[24]*x[4] + x[23]*x[4] + x[26]*x[2] + x[23]*x[5] + x[24]*x[5] + x[25]*x[3];
    p[22] = x[11]*x[7] + x[13]*x[9] + x[12]*x[7] + x[10]*x[8] + x[10]*x[6] + x[13]*x[8] + x[12]*x[6] + x[11]*x[9];
    p[23] = x[19]*x[4] + x[15]*x[3] + x[22]*x[4] + x[19]*x[5] + x[16]*x[2] + x[17]*x[3] + x[18]*x[3] + x[22]*x[5] + x[21]*x[4] + x[18]*x[2] + x[15]*x[2] + x[17]*x[2] + x[20]*x[4] + x[16]*x[3] + x[21]*x[5] + x[20]*x[5];
    p[24] = x[12]*x[2] + x[10]*x[4] + x[13]*x[3] + x[11]*x[5];
    p[25] = x[16]*x[4] + x[21]*x[3] + x[15]*x[5] + x[18]*x[4] + x[22]*x[2] + x[20]*x[2] + x[17]*x[5] + x[19]*x[3];
    p[26] = x[12]*x[9] + x[11]*x[8] + x[10]*x[7] + x[13]*x[7] + x[10]*x[9] + x[13]*x[6] + x[12]*x[8] + x[11]*x[6];
    p[27] = x[27]*x[30] + x[28]*x[29];
    p[28] = x[23]*x[6] + x[26]*x[6] + x[26]*x[8] + x[23]*x[8] + x[24]*x[7] + x[24]*x[9] + x[25]*x[6] + x[25]*x[9] + x[23]*x[9] + x[24]*x[8] + x[24]*x[6] + x[25]*x[8] + x[26]*x[7] + x[25]*x[7] + x[23]*x[7] + x[26]*x[9];
    p[29] = x[6]*x[9] + x[7]*x[8];
    p[30] = x[11]*x[25] + x[13]*x[23] + x[12]*x[24] + x[10]*x[26] + x[11]*x[26] + x[10]*x[25] + x[13]*x[24] + x[12]*x[23];
    p[31] = x[0]*x[14] + x[14]*x[1];
    p[32] = x[12]*x[22] + x[12]*x[20] + x[11]*x[15] + x[10]*x[18] + x[13]*x[19] + x[11]*x[17] + x[13]*x[21] + x[10]*x[16];
    p[33] = x[14]*x[25] + x[14]*x[24] + x[14]*x[26] + x[14]*x[23];
    p[34] = x[25]*x[25] + x[23]*x[23] + x[26]*x[26] + x[24]*x[24];
    p[35] = x[0]*x[18] + x[1]*x[20] + x[1]*x[21] + x[0]*x[15] + x[19]*x[1] + x[0]*x[17] + x[0]*x[16] + x[1]*x[22];
    p[36] = x[19]*x[8] + x[20]*x[7] + x[15]*x[9] + x[22]*x[7] + x[22]*x[6] + x[19]*x[9] + x[21]*x[8] + x[17]*x[9] + x[17]*x[7] + x[20]*x[6] + x[18]*x[8] + x[16]*x[6] + x[18]*x[6] + x[21]*x[9] + x[15]*x[7] + x[16]*x[8];
    p[37] = x[20]*x[28] + x[18]*x[28] + x[17]*x[27] + x[22]*x[29] + x[20]*x[30] + x[19]*x[30] + x[16]*x[30] + x[22]*x[27] + x[21]*x[29] + x[17]*x[28] + x[16]*x[29] + x[21]*x[27] + x[18]*x[27] + x[19]*x[28] + x[15]*x[29] + x[15]*x[30];
    p[38] = x[22]*x[23] + x[16]*x[26] + x[20]*x[24] + x[18]*x[25] + x[21]*x[23] + x[15]*x[26] + x[19]*x[24] + x[17]*x[25];
    p[39] = x[23]*x[26] + x[24]*x[25] + x[24]*x[26] + x[23]*x[25];
    p[40] = x[0]*x[29] + x[1]*x[27] + x[1]*x[30] + x[0]*x[30] + x[1]*x[29] + x[0]*x[27] + x[0]*x[28] + x[1]*x[28];
    p[41] = x[16]*x[5] + x[21]*x[2] + x[18]*x[5] + x[19]*x[2] + x[22]*x[3] + x[15]*x[4] + x[17]*x[4] + x[20]*x[3];
    p[42] = x[10]*x[12] + x[11]*x[13] + x[11]*x[12] + x[10]*x[13];
    p[43] = x[12]*x[5] + x[13]*x[4] + x[11]*x[3] + x[11]*x[2] + x[12]*x[4] + x[10]*x[3] + x[10]*x[2] + x[13]*x[5];
    p[44] = x[7]*x[9] + x[8]*x[9] + x[6]*x[7] + x[6]*x[8];
    p[45] = x[15]*x[16] + x[17]*x[18] + x[21]*x[22] + x[19]*x[20];
    p[46] = x[14]*x[9] + x[14]*x[7] + x[14]*x[6] + x[14]*x[8];
    p[47] = x[10]*x[5] + x[13]*x[2] + x[11]*x[4] + x[12]*x[3];
    p[48] = x[25]*x[28] + x[25]*x[27] + x[26]*x[30] + x[26]*x[29] + x[24]*x[28] + x[24]*x[30] + x[23]*x[27] + x[23]*x[29];
    p[49] = x[11]*x[1] + x[0]*x[12] + x[10]*x[1] + x[0]*x[13];
    p[50] = x[17]*x[17] + x[18]*x[18] + x[22]*x[22] + x[20]*x[20] + x[16]*x[16] + x[15]*x[15] + x[21]*x[21] + x[19]*x[19];
    p[51] = x[10]*x[24] + x[11]*x[23] + x[11]*x[24] + x[12]*x[25] + x[13]*x[26] + x[12]*x[26] + x[10]*x[23] + x[13]*x[25];
    p[52] = x[14]*x[20] + x[14]*x[15] + x[14]*x[22] + x[14]*x[17] + x[14]*x[18] + x[14]*x[21] + x[14]*x[16] + x[14]*x[19];
    p[53] = x[10]*x[17] + x[12]*x[21] + x[13]*x[20] + x[11]*x[18] + x[13]*x[22] + x[10]*x[15] + x[11]*x[16] + x[12]*x[19];
    p[54] = x[1]*x[24] + x[0]*x[25] + x[1]*x[23] + x[0]*x[26];
    p[55] = x[4]*x[9] + x[3]*x[6] + x[2]*x[9] + x[5]*x[8] + x[3]*x[7] + x[4]*x[7] + x[2]*x[8] + x[5]*x[6];
    p[56] = x[12]*x[15] + x[13]*x[15] + x[10]*x[21] + x[11]*x[20] + x[13]*x[16] + x[11]*x[21] + x[12]*x[18] + x[10]*x[20] + x[10]*x[22] + x[13]*x[17] + x[11]*x[22] + x[12]*x[17] + x[12]*x[16] + x[13]*x[18] + x[10]*x[19] + x[11]*x[19];
    p[57] = x[23]*x[28] + x[24]*x[29] + x[24]*x[27] + x[26]*x[27] + x[25]*x[30] + x[26]*x[28] + x[23]*x[30] + x[25]*x[29];
    p[58] = x[23]*x[3] + x[25]*x[4] + x[25]*x[5] + x[26]*x[4] + x[24]*x[3] + x[26]*x[5] + x[23]*x[2] + x[24]*x[2];
    p[59] = x[14]*x[27] + x[14]*x[28] + x[14]*x[29] + x[14]*x[30];
    p[60] = x[29]*x[29] + x[28]*x[28] + x[27]*x[27] + x[30]*x[30];
    p[61] = x[1]*x[25] + x[0]*x[24] + x[0]*x[23] + x[1]*x[26];
    p[62] = x[14]*x[5] + x[14]*x[3] + x[14]*x[2] + x[14]*x[4];
    p[63] = x[0]*x[11] + x[13]*x[1] + x[0]*x[10] + x[12]*x[1];
    p[64] = x[28]*x[30] + x[27]*x[29] + x[27]*x[28] + x[29]*x[30];
    p[65] = x[11]*x[11] + x[10]*x[10] + x[12]*x[12] + x[13]*x[13];
    p[66] = x[23]*x[24] + x[25]*x[26];
    p[67] = x[16]*x[17] + x[19]*x[22] + x[20]*x[21] + x[15]*x[18];
    p[68] = x[14]*x[14];
    return p


# In[15]:

p = poly_2d(distance_exp)


# In[16]:

# INPUT
X = p.values

# TARGET
##data shift so, that everything can be log
shift = - np.floor(E_ref.min())
y = np.log10(E_ref + shift)

# sample weights based on the binding energies
deltaEb = 25
weight =  (deltaEb/(Eb - Eb.min() + deltaEb))**2


# In[17]:

'''shuffle input data -> Splitting training and test sets'''
idx = np.random.permutation(len(y))
y_shuffled = y[idx] #y.copy()[idx]
X_shuffled = X[idx]
weight_shuffled = weight[idx]
Eb_shuffled = Eb[idx]

test_size = 0.1
offset = int((1-test_size)* len(y))
X_train, y_train, w_train = X_shuffled[:offset], y_shuffled[:offset], weight_shuffled[:offset]
X_test, y_test, w_test , Eb_test= X_shuffled[offset:], y_shuffled[offset:], weight_shuffled[offset:], Eb_shuffled[offset:]


# In[18]:

print("The size of input data array for benchmarking is ", X.shape)


# In[19]:

filename='NN_2L2H2O_poly2d_benchmarking.in'
with open(filename,'w') as f:
    f.write("double X[][69] = { \\\n {")

with open(filename,'ab') as f:
    np.savetxt(f, X[:-1],fmt="%.18e", delimiter=" , ", newline=" } , \n {")
    np.savetxt(f, np.atleast_2d(X[-1]), fmt="%.18e", delimiter=" , ", newline=" } } ;")
    
print("Input data array for benchmarking generated !")    

