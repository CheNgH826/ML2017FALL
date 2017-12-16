import pandas as pd
import numpy as np
import sys

ans = []
ans_list = sys.argv[1:-1]
for ansi in ans_list:
    np_ans = np.array(pd.read_csv(ansi)['Rating'])
    ans.append(np_ans)
ans = np.array(ans)
ans = np.mean(ans, axis=0)
with open(sys.argv[-1], 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, rate in enumerate(ans):
        f.write(str(i+1)+','+str(rate)+'\n')
