
import time

import multiprocessing

import blockmodels as bm

def worker(num):
    """thread worker function"""
    print 'Worker:', num
    return


def multiple(f):

    jobs = []
    for i in range(21):
        p = multiprocessing.Process(target=bm.blockmodel, args=(f[i], 2))
        jobs.append(p)
        p.start()
        #p.join()


def single(f):

    jobs = []
    for i in range(21):
        jobs.append(bm.blockmodel(f[i], 2))



if __name__ == '__main__':
    a, f, ca, cf, sa, sf = bm.read_css()

    before = time.time()
    single(f)
    print "Single Elapsed: ", time.time() - before

    before = time.time()
    print 'before: ', time.time()
    multiple(f)
    print 'Multiple Join Elapsed: ', time.time() - before
