
import numpy as np
import pandas as pd

import blockmodels as bm


a, f, ca, cf, sa, sf = bm.read_css()

am = bm.blockmodel(a, k=2, iterations=100)
fm = bm.blockmodel(f, k=2, iterations=100)

alist = []
colnames = ", ".join([str(i) for i in range(1,21+1)])
for e in np.ndindex((21,21)):
    row = [e[0], e[1], ca[e], sa[e]]
    row += [str(am.groups[[e[0], e[1]]])]
    row += list(a[:,e[0],e[1]])
    alist.append(row)
adf = pd.DataFrame(alist, columns=("sender, receiver, ca, sa, block, "+colnames).split(", "))


flist = []
colnames = ", ".join([str(i) for i in range(1,21+1)])
for e in np.ndindex((21,21)):
    row = [e[0], e[1], cf[e], sf[e]]
    row += [str(fm.groups[[e[0], e[1]]])]
    row += list(f[:,e[0],e[1]])
    flist.append(row)
fdf = pd.DataFrame(flist, columns=("sender, receiver, cf, sf, block, "+colnames).split(", "))


adf.to_csv("adf.csv")
fdf.to_csv("fdf.csv")
