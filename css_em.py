
import sys
import copy
import pdb

import numpy as np
from scipy.cluster.vq import kmeans, vq

import blockmodels as bm

def initial_condition(data):
    """Clusters the observation vectors into 'edge' and 'no edge' for G_0.

    Treats each length-m observation vector as data point -- clusters all the
    vectors using kmeans with k=2, then assigns each G_hat value based on the
    cluster assignment.
    """
    N = data.shape[1]

    ########
    # if add:
    #     votes = np.sum(data, axis=0)
    #     X = np.array([ np.array([votes[i,j]]) for i, j in np.ndindex((N,N))])
    # else:
    X = np.array([data[:,i,j] for i, j in np.ndindex((N,N))])
    ########

    centers, distortion = kmeans(X, 2)
    assignments, something = vq(X, centers)

    G_hat = np.zeros((N,N))
    for i, j in np.ndindex((N,N)):
        G_hat[i,j] = assignments[i*N+j]

    # It's not clear a-priori if the clusters MEAN 'yes' or 'no,'
    # so try both, and see which is closer to taking the mean value. Yeah?
    G_hat_prime = (1-G_hat)**2

    d1 = np.sum(np.abs(G_hat - np.mean(data, axis=0)))
    d2 = np.sum(np.abs(G_hat_prime - np.mean(data, axis=0)))

    if d1 < d2:
        return G_hat
    else:
        return G_hat_prime
    #return G_hat, G_hat_prime


def calculate_likelihood(data, groups, G, n, k):

    Po1e0   = np.zeros((k,)*3) # pIo_1le_0I
    Po1e1   = np.zeros((k,)*3) # pIo_1le_1I

    Pe0   = np.zeros((k,)*3)
    Pe1   = np.zeros((k,)*3)

    for o_ijk in np.ndindex(data.shape):

        e = o_ijk[1:]
        B = tuple(groups[list(o_ijk)])

        # if data[o_ijk]:
        Po1e0[B] += data[o_ijk]*(1-G[e])
        Po1e1[B] += data[o_ijk]*G[e]
        #print 'o_ijk: ', data[o_ijk], ' G[e]: ', G[e]

        Pe0[B] += 1-G[e] #*counts[B[0]]
        Pe1[B] +=   G[e] #*counts[B[0]]

    # pdb.set_trace()

    likelihood = {0:Po1e0/Pe0,
                  1:Po1e1/Pe1}
    # If n_A = 0, we'll have /0 errors in likelihood --
    # replace them as 0?
    for i in [0,1]:
        likelihood[i][np.isnan(likelihood[i])] = 0

    # print 'likelihood: \n', likelihood
    return likelihood


def calculate_posterior(data, groups, Po1_e, p, N, k):

    Pe0o = np.ones((N,N))
    Pe1o = np.ones((N,N))

    for o_ijk in np.ndindex(data.shape):

        e = o_ijk[1:]
        B = tuple(groups[list(o_ijk)])
        #       p(o_ijk=1 | e=0/1 ; B)         p(o_ijk=0 | e=0/1 ; B)
        Pe0o[e] *= Po1_e[0][B] if data[o_ijk] else (1-Po1_e[0][B])
        Pe1o[e] *= Po1_e[1][B] if data[o_ijk] else (1-Po1_e[1][B])


    # print 'before: ', np.mean(Pe1o/(Pe1o+Pe0o) == cf)

    for e in np.ndindex((N,N)):

        B = tuple(groups[list(e)]) # YEAH? This went from 3d->2d blocks ...
        Pe0o[e] *= 1-p[B] # p(e=0)
        Pe1o[e] *=   p[B] # p(e=1)

    #print 'after: ', np.mean(Pe1o/(Pe1o+Pe0o) == cf)
    #raw_input()

    return Pe1o/(Pe0o + Pe1o) # = Pe1o/sum_e{Peo] = Pe1o/Po = Pe1_o i.e. P(e=1|O)



def weigh_errors(data, best_guess, groups, k):
    """ Counts true/false positive/negatives for each cluster-tuple.

    Takes the observations, the best_guess, and the group assignments
    """

    over  = np.zeros((k,)*3) #tuple(k for i in range(3)))
    under = np.zeros((k,)*3) #tuple(k for i in range(3)))
    # Perceiver, Sender, Receiver
    for p, i, j in np.ndindex(data.shape):
        # The Truth (we're assuming)
        t = best_guess[i,j]
        # The Observation
        o = data[p,i,j]
        # The node-group index tuple
        gix = tuple(groups[[p, i, j]])

        if t <= o:
            over[gix]  = o-t
        else:
            under[gix] = t-o

    return over, under


def count_errors(data_orig, best_guess_orig, groups_orig, k, indices=None):
    """ Counts true/false positive/negatives for each cluster-tuple.

    Takes the observations, the best_guess, and the group assignments
    """

    data = copy.copy(data_orig)
    groups = copy.copy(groups_orig)
    best_guess = copy.copy(best_guess_orig)
    # DANGER
    # DO I NEED TO COPY THE DATA?
    if indices:# and g.ndim==3:
        other_indices = [i for i in range(data.shape[1]) if not i in indices]
        new_indices = np.array(indices + other_indices)
        # Reshuffles the matrices so that the first elems are the samples
        data = data[:,new_indices,:]
        data = data[:,:,new_indices]
        best_guess = best_guess[:,new_indices]
        best_guess = best_guess[new_indices,:]
        groups = groups[new_indices]

    # The structures that will hold the accuracy counts
    tp = np.zeros((k,)*3)
    fp = np.zeros((k,)*3)
    tn = np.zeros((k,)*3)
    fn = np.zeros((k,)*3)

    # Perceiver, Sender, Receiver
    for p, i, j in np.ndindex(data.shape):
        # The Truth (we're assuming)
        t = best_guess[i,j]
        # The Observation
        o = data[p,i,j]
        # The node-group index tuple
        gix = tuple(groups[[p, i, j]])

        if       t and     o:
            tp[gix] += 1
        elif not t and     o:
            fp[gix] += 1
        elif not t and not o:
            tn[gix] += 1
        elif     t and not o:
            fn[gix] += 1
        else:
            raise Exception("Bottomed out of cases for observations")

    return tp, fp, tn, fn


def estimate_error_rates(tp, fp, tn, fn):

    # # P(false_pos) = P(edge=0 | view = 1) # ... yeah? no -- no.
    # #              = c(e=0,v=1)/c(v=1)
    # #              = fp/(tp + fp)
    #
    # pfp = fp/(tp + fp)
    # pfn = fn/(tn + fn)

    # P(false_pos) = P(view = 1 | edge = 0)
    #              = c(e=0,v=1)/c(e=0)
    #              = fp/(tn + fp)

    pfp = fp/(tn + fp)
    pfn = fn/(fn + tp)

    # Div by 0 => numerator is 0 too
    pfp[np.isnan(pfp)] = 0.0
    pfn[np.isnan(pfn)] = 0.0

    # Trying to enforce an identifiability constraint ...
    # but a little arbitrarily
    #pfp[pfp >= .5] = 0.05
    #pfn[pfn >= .5] = 0.05

    return pfp, pfn


def guess(data, i, j, pfp, pfn, groups, p):

    # Preparing to build up the values:
    # P(obs | edge), P(obs | no edge)
    p1, p0 = 1, 1

    for perceiver in range(data.shape[0]):

        gix = tuple(groups[[perceiver, i, j]])
        # P(all observations | edge/no edge)
        p1 *= pfn[gix] if data[gix]==0 else 1-pfp[gix]
        p0 *= pfp[gix] if data[gix]==1 else 1-pfn[gix]

    # Multiply by the a priori possibility of the edge,
    ij = tuple(groups[[i,j]])
    # based on the blocks it would span, to finally get
    p1 *= p[ij]     # = P(edge,  obs)
    p0 *= 1 - p[ij] # = P(!edge, obs)
    # We want the posterior -- but we have all the elements.
    # pIeloI = p1/(p1+p0) # P(edge|obs) = P(edge, obs)/(P(!e,o) + P(e,o))
    # Trying to write p(e|o) only using alphanumerics. Kinda works, right?
    # return pIeloI
    return (p1 > p0) + 0 # bool => int

def accuracy(x, y):
    tp = np.sum((x+y)==2)/np.sum(y)
    fp = np.sum((x+y)==0)/np.sum((1-y)**2)
    return tp, fp


def em(data, k=2, indices=None, G_true=None, num_samples=5, iterations=100, corrected=True):

    n = data.shape[0] # The number of samples
    N = data.shape[1] # The size of the network being sampled

    # Select the subset
    # np.random.choice(np.arange(n), num_samples, replace=False)
    # s_vec = np.in1d(np.arange(n), s_nums)  # Vectorised 'in' => inclusion booleans
    # A mask for the edges on which I can see consensus
    # both_there = np.outer(s_vec, s_vec)
    #if num_samples:
    #    s_nums = np.arange(num_samples)
    #    data = data[s_nums,:,:] # Or just data[s]?

    G = initial_condition(data) #(np.sum(data, axis=0) > len(data)/5.) + 0
    G_old = copy.copy(G)

    # b = bm.blockmodel(data, k, iterations=iterations,
    #                         corrected=corrected,
    #                         indices=indices)

    em_iterations = 0
    while True:
        print "EM iteration #{}".format(em_iterations)
        em_iterations += 1

        # OR should I do this ... AS THE FULL 3D MODEL?
        b = bm.blockmodel(G, k, iterations=iterations,
                                corrected=corrected,
                                indices=indices)

        likelihood = calculate_likelihood(data, b.groups, G, n, k)
        G = calculate_posterior(data, b.groups, likelihood, b.p, N, k)

        if np.abs(np.sum(G - G_old)) < 1e-1 or em_iterations > 1000:
            b = bm.blockmodel(np.round(G), k,
                                    iterations=iterations,
                                    corrected=corrected,
                                    indices=indices)
            return np.round(G), b
        else:
            G_old = copy.copy(G)




def test_count_errors(n, k, e):

    # em.test_count_errors(20, 2, np.array([[1,2],[3,4]]))

    # I want to see if ... fp + fn == e
    # first ... necessary?
    e = copy.copy(e)
    # 1) Make a PERFECT model,
    x = bm.blocked_matrix(n, k)
    parts = bm.partition(n, k)
    # [[1,2,3],[4,5,6]] => [0,0,0,1,1,1 ...]
    groups = np.concatenate([[i]*len(parts[i]) for i in range(k)])
    # 2) Copy it, for the perceptions
    y = copy.copy(x)
    # Have two people, with the same views ... yes, counts double.
    #x = np.array(x, ndmin=3) # So the indexing over 'perceivers' works right
    x = np.array([x, x]) # in case I want several observers
    # 3) Turn the NUMBERS of errors into actual coordinates
    edges = list(np.ndindex((n,)*e.ndim))
    eix = np.zeros(y.shape, dtype=np.bool)
    for edge in edges:
        # The groups of the nodes involved in the edge ('Group IndeX')
        gix = tuple(groups[list(edge)])
        # If there are errors left to disperse from block gix,
        if e[gix]:
            # Mark the index for the error
            eix[edge] = True
            # And reduce the number of remaining errors to distribute
            e[gix] -= 1
    # 4) Flip the bits on the error spots
    y[eix] = ((1-y)**2)[eix]

    print 'y: \n', np.int64(y)

    # 4) See if count errors' recovers them.
    # data, best_guess, groups, k
    out = count_errors(x, y, groups, k)

    return out

def test_estimate_error_rates():
    pass

def test_guess():
    pass


if __name__ == "__main__":

    # Putting it down here is particularly important because it doesn't play at all with
    # pypy, so now the OTHER EM functions can be imported (via the script) withoug
    # throwing that wrench.
    from matplotlib import pyplot as plt


    # all data, idem, consensus, idem, self-reported, idem
    advice, friendship, ca, cf, sa, sf = bm.read_css()
    X = np.array([cf for i in range(21)])
    # print 'yeah? ', np.all([np.all(cf==X[i]) for i in range(len(X))])
    # raw_input()
    # G_hat, b =  em(X, np.random.randint(2,size=(21,21)), k=2, num_samples=5, iterations=20, corrected=True)

    accs = []
    bcf = bm.blockmodel(cf, k=2, corrected=True)
    for num in range(21):
        G_hat, b = em(friendship[range(num)], k=2, iterations=100, corrected=True)
        accs.append(accuracy(np.round(G_hat), cf))
        print 'accuracy after: ', accuracy(np.round(G_hat), cf)
        print 'bG.p: \n', b.p
        print 'bcf.p: \n', bcf.p

    plt.plot([i[0] for i in accs])
    plt.plot([i[1] for i in accs])
    plt.plot([np.prod(i) for i in accs])
    plt.ylim([-.1,1.1])
    plt.show()


    # THIS DOESN"T WORK PARTICULARLY WELL
    # print 'distance: ', np.sum(np.abs(G_hat-cf))
    # for t in np.linspace(0,1, 10):
    #     print 't: ', t, ' distance: ', np.sum(np.abs((G_hat > t)+0 - cf))


    # data = {'a': [advice, ca], 'f': [friendship, cf]}
    # # all_data, consensus, k, num_samples, iterations, corrected
    # defaults = ('f', 2, 10, 20, 1)
    #
    # v = sys.argv[1:]
    # args = [v[i] if len(v) > i else defaults[i] for i in range(len(defaults))]
    # #print 'args: ', args
    # args = data[args[0]] + list(np.int64(args[1:])) # array cast [str]->[int]
    #
    # # best_guess, model, pfp,  pfn =
    # # em(friendship, cf, k=3, num_samples=10, iterations=20)
    # best_guess, b, pfp, pfn = em(*args)
    #
    # print 'final accuracy:          {:.3f}'.format(np.sum(best_guess == args[1])/441.)
    # print 'pfp: \n', pfp
    # print 'pfn: \n', pfn
    # print 'b.p: \n', b.p
    # print 'b.m: \n', b.m
    # print 'b.groups: ', b.groups
    # b.plot()
