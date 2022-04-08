import numpy as np

class Arm():
    def __init__(self,  dist_name, dist_params, advers_name, M, epsilon):
        self.dist_name = dist_name
        self.dist_params= dist_params
        self.epsilon =epsilon
        self.M = M
        self.advers_name= advers_name

    def sample(self):
        if self.dist_name== 'Gaussian':
            mean= self.dist_params[0]
            var= self.dist_params[1]
            samples = np.random.normal(mean, var, self.M)
        if self.advers_name == 'oblivious':
            corrupt_indicator=  np.random.binomial(size= self.M, n= 1, p =self.epsilon).nonzero()
            samples[corrupt_indicator]= 1000
        elif self.advers_name== 'prescient':
            corrupt_indicator=  np.random.binomial(size= self.M, n= 1, p =self.epsilon).nonzero()
            samples[corrupt_indicator]= 1000

        elif self.advers_name == 'malicious':
            corrupt_indicator=  np.random.binomial(size= self.M, n= 1, p =self.epsilon).nonzero()
            samples[corrupt_indicator]= 1000


