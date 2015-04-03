# This file has a clear purpose.

#####################################################
#####################################################
#                                                   #
#   PARAMETER ESTIMATION IN TEST AND DATA CASES     #
#                                                   #
#####################################################
#####################################################


# ASIDE FROM THAT, ALSO DO A FULL TIME PROFILE ON ONE EM LOOP.

# Also, spit these out into files so that they can be plotted elsewhere,
# meaning RUN with pypy.


# THINGS TO KEEP -- edge_accuracy, group accuracy, pfp, pfn ... the model itself
# because ... of p, groups, etc ... and the indices. right. # iterations?
# time to convergence? sure.

# From active learning

import pickle as pk
import multiprocessing as mp
import active_learning as al

import blockmodels as bm

a, f, ca, cf, sa, sf = bm.read_css()


args = [(f, cf, 2, "random")]#,
        # (f, cf, 10, "gregarious"),
        # (f, cf, 10, "popular"),
        # (f, cf, 10, "unfamiliar"),
        # (f, cf, 10, "familiar"),
        #
        # (a, ca, 10, "random"),
        # (a, ca, 10, "gregarious"),
        # (a, ca, 10, "popular"),
        # (a, ca, 10, "unfamiliar"),
        # (a, ca, 10, "familiar")]

if __name__=="__main__":

    jobs = []
    for i in range(len(args)):
        p = mp.Process(target=al.kaboom, args=args[i])
        jobs.append(p)
        p.start()
    # 
    # f = open("jobs.pkl", 'wb')
    # pickle.dump(jobs, f)
    # f.close()


    # IF ONLY I COULD TELL WHEN THEY WERE GOING TO FINISH
