from utils import Config
def write_func(l1,l2):
    with open(Config['category_txt'],'w') as f:
        for i in range(len(l1)):
            f.write(str(l1[i]) + ' ' + str(l2[i]))
            f.write('\n')

def write_func_1(l1):
    with open('/Users/lzy/Desktop/semeter1/541/HW5/data/compatiability_txt_hw.txt','w') as f:
        for i in range(len(l1)):
            f.write(str(l1[i]))
            f.write('\n')