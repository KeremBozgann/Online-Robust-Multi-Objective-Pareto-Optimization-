import pandas as pd
import numpy as np
from util import pickle_save
import matplotlib.pyplot as plt
plt.ioff()

data = pd.read_csv("sw-llvm.csv")
dat = data.to_numpy()

dat_mat= np.zeros([dat.shape[0], 13])
for i, row in enumerate(dat):
    dat_mat[i, :] = (np.array(row[0].split(';'))).astype(np.float)

x = dat_mat[: , :11]
y = dat_mat[:, 11:]
y[:, 1] = -y[:, 1]

y -= np.mean(y, axis =0)
y/= np.std(y, axis =0)

x_chosen= x[:, 7:11]

sample_inds_dict= dict()
for i in range(2):
    for j in range(2):
        for k in range(2):
            for z in range(2):
                    sample_inds_dict[i*8 +j* 4 +k * 2 +z * 1] = np.where(np.all(x_chosen == np.array([i, j, k, z]),axis=1)  == True)[0]

arm_mean_list= np.zeros([0,2])
arm_std_list= np.zeros([0,2])
for arm in sample_inds_dict:
    print(arm)
    print(len(sample_inds_dict[arm]))
    arm_sample_inds= sample_inds_dict[arm]
    arm_mean_list= np.append(arm_mean_list, np.mean(y[arm_sample_inds], axis= 0).reshape(1, -1), axis =0)
    arm_std_list= np.append(arm_std_list, np.std(y[arm_sample_inds], axis= 0).reshape(1, -1), axis =0)

print(arm_std_list)
print(arm_mean_list)

print(np.max(arm_std_list, axis = 0))
print(np.min(arm_std_list, axis = 0))


fig, ax = plt.subplots(1)
ax.scatter(arm_mean_list[:, 0], arm_mean_list[:, 1])
ax.grid()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.major.pad'] = '5'
plt.rcParams['ytick.major.pad'] = '5'
plt.savefig('test.pdf', bbox_inches='tight')



llvm_dict= dict()
llvm_dict['x']= x
llvm_dict['y'] = y
llvm_dict['sample_inds_dict'] = sample_inds_dict
pickle_save('llvm_dict.pickle', llvm_dict)

