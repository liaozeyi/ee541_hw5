import pandas as pd
import numpy as np

np_total = np.load('/Users/lzy/Desktop/semeter1/541/HW5/ee541-hw5-starter-master_2/np_total.npy')
pd_data= pd.DataFrame(np_total,columns=['true_id','compatiability_id'])
pd_data.to_csv('/Users/lzy/Desktop/semeter1/541/HW5/ee541-hw5-starter-master_2/np_total.csv',index=False)
