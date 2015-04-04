
import pickle as pk
import multiprocessing as mp
import active_learning as al

import blockmodels as bm

a, f, ca, cf, sa, sf = bm.read_css()

#       THESE FOUR (two things each two times) TOOK 5 HOURS. O
args = [(f, cf, "friendship_2_random.pkl",   2, "random"),
        (f, cf, "friendship_2_familiar.pkl", 2, "familiar"),
        (a, ca, "advice_2_random.pkl",   2, "random"),
        (a, ca, "advice_2_familiar.pkl", 2, "familiar")]#,

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
