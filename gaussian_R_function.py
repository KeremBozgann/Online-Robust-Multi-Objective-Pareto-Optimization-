import numpy as np
from scipy.stats import norm

def gaussian_R_coeff(t_bar, sigma):
    B= norm.ppf(3/4)/(norm.pdf(norm.ppf(1/2+t_bar)))
    m2= np.sqrt(2/np.pi)* sigma
    return B*m2

# t_bar= 0.4
# sigma= 1
#
# B= norm.ppf(3/4)/(norm.pdf(norm.ppf(1/2+t_bar)))
# m2= np.sqrt(2/np.pi)* sigma
# print(B*m2)