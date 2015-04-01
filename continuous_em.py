



import sys
import copy
import collections

# Idiots don't have a module sk.linear_model
from sklearn import linear_model as lm
import scipy.stats as st
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np

import blockmodels as bm


def gather_joint(data, best_guess, groups, k):

    # One list (to hold the observations) for each inter-block cluster.
    joint = {gix:[] for gix in np.ndindex((k,)*3)}
    # Iterate over all observations --
    for oix in np.ndindex(data.shape):
        # Find the groups it spans
        gix = tuple(groups[list(oix)])
        # Connect the observation with the actual (assumed) value of the edge
        # and log it in the appropriate between-block list.
        joint[gix].append((oix[1:3], best_guess[oix[1], oix[2]], data[oix]))

    # Now you have the observations necessary for
    # p(o,e|blocks)
    return joint

def log_odds(x):
    y = x+.0001
    return np.log(y/(1-y))

def logit(x):
    return 1./(1 + np.exp(-x))

def just_logistic_regression(data, g, groups, N, k):

    pre_df = []
    best_guess = np.zeros((N,N))
    # For each edge in the matrix to be predicted:
    for e in np.ndindex((N,N)):
        #     The edge index, the edge value, the edge block,
        row = [str(e),        log_odds(g[e]), str(groups[list(e)])]
        # Gather the array of votes for this particular edge
        votes = np.array(data[:,e[0],e[1]], dtype=bool)
        # Count the group memberships of the people who voted for the edge
        counts = collections.Counter(groups[votes])
        # Counter works like a dict, so iteration order is not guaranteed
        ordered = [counts[c] for c in range(k)] # Will return 0s, helpfully
        # Stick on the row -- that's one observation for the model.
        pre_df.append(row+ordered)

    # Just preparing the DataFrame
    labels = [str(i) for i in range(k)]
    # Index, LogOddsEdge, Block, #0s, #1s, etc, meaning 'votes from group x'
    colnames = "ix, loe, block, " + ", ".join(labels)
    # C(.) means 'categorical'
    formula = "loe ~ " + " + ".join(labels)
    df = pd.DataFrame(pre_df, columns=colnames.split(", "))

    # Split, apply --- point estimate.
    models = []
    # Estimate a different model for each block, due to (presumed)
    # difference in accuracy of perception for different cross-group views.
    # Which makes me think -- actually, just try one model, too.
    for name, group in df.groupby('block'):

        # JESUS CHRIST -- sm.ols LOWER case's 'predict' function is broken,
        #lm = sm.ols(formula, group).fit()
        # but sm.OLS UPPER case (WHY!?!) doesn't take formulas? So ...
        lm = sm.OLS(group['loe'], group[group.columns[3:]]).fit()
        models.append(lm)
        ehat = lm.predict(group[group.columns[3:]])
        ehat = logit(ehat) # Since this was transformed initially
        for i, e in enumerate(group['ix']):
            best_guess[eval(e)] = ehat[i]
    #print best_guess
    best_guess[np.isnan(best_guess)] = 0
    #print 'mean(best_guess): ', np.mean(best_guess)
    return best_guess, models, df



class Dummy:
    def __init__(self, value):
        self.value = [value, value]

    def __call__(self, arg):
        return 1

    def predict_proba(self, x):
        return self.value


def estimate_joint(joint, k):

    # A place to store p(o|e),p(e) for each inter-block
    models = {gix:{'p(o|e)':None, 'p(e)':None} for gix in np.ndindex((k,)*3)}
    for mix in models:
        # p(o|e)
        pIoleI = lm.LogisticRegression()
        x = [j[0] for j in joint[mix]]
        y = [j[1] for j in joint[mix]]
        # print 'x: ', x
        # print 'y: ', y
        # Each element is in its
        # own list, to make .fit() realize it's an observation, not a feature.

        # What if I get only one class, or no data in that cell?
        if len(set(y)) > 1:
            pIoleI.fit([[xi] for xi in x], y)
            pIeI = st.gaussian_kde(x)
        else:
            pIoleI = Dummy(y[0] if y else 0)
            pIeI = Dummy(0)

        models[mix]['p(o|e)'] = pIoleI
        models[mix]['p(e)']   = pIeI

    return models


def posterior(data, g, groups, m, N):
    # OR, const N = ...whatever.
    obs = np.zeros((N,N))
    # For each network edge,
    for e in np.ndindex(obs.shape):

        # For each observation of that edge,
        O = 1
        for n in range(data.shape[0]):
            gix = tuple(groups[[n]+list(e)])
            # p(o_i|e_jk)
            #                     Gives all class probabilities, in a 2D list.
            #                     GOD I hate these packages.
            #                     Just 'predict' will only give 0/1 though.
            temp = m[gix]['p(o|e)'].predict_proba(g[e])[0,1]
            #print 'p(o_i|e_jk): ', temp
            O *= temp

        # p(O|e_jk)
        #print 'O: ', O
        likelihood = np.prod(O)
        # p(e_jk)
        prior = m[gix]['p(e)'](g[e])
        # SUM_e p(O|e)*p(e)
        Z = 0
        for p in np.linspace(0,1): # default=50 points
            Z += m[gix]['p(o|e)'].predict_proba(p)[0,1]*m[gix]['p(e)'](p)

        #print 'prior: ', prior
        #print 'likelihood: ', likelihood
        #print 'Z: ', Z
        # p(e|O)
        obs[e] = prior*likelihood/Z
    return obs



def em(all_data, consensus, k=2, num_samples=5, iterations=20, corrected=True):

    N = all_data.shape[0]

    # Select the subset
    # np.random.choice(np.arange(n), num_samples, replace=False)
    # FOR NOW, RANDOM IS BAD.
    s_nums = np.arange(num_samples)
    s_vec = np.in1d(np.arange(N), s_nums)  # Vectorised 'in' => inclusion booleans

    # A mask for the edges on which I can see consensus
    # both_there = np.outer(s_vec, s_vec)

    # Boolean arrays aren't supported (!) -- they're treated as 0/1.
    data = all_data[s_nums,:,:]

    # How many perceivers 'saw' each edge
    votes = np.sum(data, axis=0, dtype=float)

    #majority = votes > len(s_nums)/2.0

    best_guess = votes/num_samples # majority + 0

    before = np.sum((best_guess>.2) & (consensus==1))/float(np.sum(consensus==1))
    print 'Initial voting recognition rate: {:.3f}'.format(before)

    # What if we START like this?
    ########################################
    ########################################
    best_guess = np.array(best_guess>.2, dtype=float)
    print 'Before: \n', np.int64(best_guess)
    ########################################
    ########################################

    # If we know some, we don't have to guess
    # best_guess[both_there] = consensus[both_there]
    # print 'After adding consensus:  {:.3f}'.format(np.sum(best_guess == consensus)/441.)

    em_iterations = 0
    old_groups = np.zeros(N)

    print """#########################################"""

    while True:
        sys.stdout.write("EM iteration #{}              ".format(em_iterations))
        sys.stdout.flush()
        em_iterations += 1

        b = bm.blockmodel(best_guess, k, iterations=iterations,
                                         corrected=corrected)

        #print 'in gather_joint'
        #gathered   = gather_joint(data, best_guess, b.groups, k)
        #print 'in estimate_joint'
        #joint      = estimate_joint(gathered, k)
        #print 'in posterior'
        #best_guess = posterior(data, best_guess, b.groups, joint, N)
        #print 'best guess: ', best_guess
        #raw_input()
        # Again, we know these
        # best_guess[both_there] = consensus[both_there]
        # FIRST, DO THIS WITHOUT --

        # OR too many em_iterations OR epsilon-idential likelihoods?
        if np.array_equal(old_groups, b.groups):
            break
        else:
            old_groups = b.groups
            best_guess, models, df = just_logistic_regression(data, best_guess, b.groups, N, k)

    return best_guess, b, models, df



if __name__ == "__main__":

    # all data, all data, consensus, consensus
    advice, friendship, ca, cf, sa, sf = bm.read_css()

    out = em(friendship, cf, k=2, num_samples=20, iterations=20, corrected=True)
    N = len(cf)

    after = np.sum((out[0]>.2) & (cf==1))/float(np.sum(cf==1))
    #print 'Before adding consensus: {:.3f}'.format(before)
    print 'AND THE FINAL VALUE: ', after
    #print (N**2 - np.sum(np.abs(out[0]-cf)))/N**2
    #print np.mean(np.round(out[0])==cf)
    print np.int64(out[0])
