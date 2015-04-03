
import os
import sys
import csv
import copy
import math
import collections

import pdb

import numpy as np
rand = np.random.random
import networkx as nx
import pygraphviz as gv
import colour

import grouping_distance as gd



def ndindex(ix):
    """The same as np.ndindex, but 20% slower. Pypy's numpy doesn't have it.
    """

    if len(ix)==2:
        for i in range(ix[0]):
            for j in range(ix[1]):
                yield (i,j)

    elif len(ix)==3:
        for i in range(ix[0]):
            for j in range(ix[1]):
                for k in range(ix[2]):
                    yield (i,j,k)




def color_range(one, two, n):
    """ color_range("red", "blue", 4) gives four colors in six digit hex

    Used because pygraphviz definitely accepts 6-digit hex codes.
    Should be noted, the two colors in the middle will be shades of lime ...
    Not much to do with red, or blue. Unfortunately, colorbrewer has NO
    documentation (!) and other things don't seem to have ranges!

    'seaborn' looks like possibly a good option (documentation!) but
    """
    col1 = colour.Color(one)
    col2 = colour.Color(two)
    r = [c.get_hex_l() for c in col1.range_to(col2, n)]
    return r

def one_round_swaps(b):
    """Finds the best group change for each node, once.

    Described in (Karrer and Newman 2011) as adapted from (Kernhigan, Lin 1970).

    The implementation holds a list of likelihoods that were acheived after
    each node was optimally placed, as well as the group into which it was
    placed. When all nodes have been examined, the system goes back to the
    state that achieved the highest likelihood, which means keeping the nodes
    that were switched before the maximum was hit, and discarding the changes
    after.
    """

    # Initialized so that there's a 'previous' value for the initial one
    new_max = [b.calculate_likelihood()]
    new_groups = copy.copy(b.groups)

    # Iterate through the nodes
    for i, group in enumerate(b.groups):
        changes = []
        # Iterate through the possible clusters
        for j in range(b.k):
            # Not worth evaluating your current position.
            if j != group:
                # Place node i in group j
                changes.append(b.change_in_likelihood(i, j))
            else:
                changes.append(0)
        # After trying all groups, set the new group to the best option
        best = changes.index(max(changes))
        new_groups[i] = best
        # Gotta change the ACTUAL assignment -- this is cumulative
        b.set_group(i, best)

        # Update the likelihood by the local change
        # Remember, the list was initialized to have a 'previous' elem
        # even for the FIRST element.
        new_max.append(new_max[-1] + max(changes))

    mx = new_max.index(max(new_max))

    # If the index isn't 0, which is the 'no changes' state,
    if mx:
        # Remove the first element
        del new_max[0]
        # and pretend it wasn't there
        mx -= 1
        # so that you can easily index into the best place.
        # With n nodes, you index easily into 1:n changes -- not 0:n changes.
        best_groups = np.concatenate((new_groups[:mx+1], b.groups[mx+1:]))
        b.set_group(best_groups)

        # Return the likelihood corresponding to the groups
        return new_max[mx] #b.calculate_likelihood()
    else:
        # The groups don't change.
        return new_max[0]



def blockmodel(g, k, iterations=100, corrected=True, indices=[]):
    """ Takes a graph and a number of clusters, returns group assignments.

    g is the graph, a 2- or 3-d binary (NOT boolean) numpy array
    Right now, treats the network as 1-mode, so there's only 1 k.
    """

    # danger
    # do i need to 1) copy g, or 2) permute the groups
    #g = copy.copy(g_orig)

    # The indices of the people whose network slices we're seeing
    revert_indices = None
    if indices and g.ndim==3:
        other_indices = [i for i in range(g.shape[1]) if not i in indices]
        new_indices = np.array(indices + other_indices)
        # Reshuffles the matrices so that the first elems are the samples
        g = g[:,new_indices,:]
        # Because numpy is INSANE, I have to permute the dimensions 1 by 1.
        g = g[:,:,new_indices]
        # Cast so that it has list's 'index' function.
        ni = list(new_indices)
        revert_indices = np.array([ni.index(i) for i in range(g.shape[1])])




    likelihoods = []
    models = []
    # Randomly initialized, so try a few times.
    for random_itn in range(iterations):
        #sys.stdout.write('Random start #{}\r'.format(random_itn+1))
        #sys.stdout.flush()

        b = BlockModel(g, k, corrected=corrected)
        models.append(b)

        lkhds = []

        old_likelihood = -np.inf
        new_likelihood = 0
        convergence_itn = 0
        while True:
            convergence_itn += 1

            new_likelihood = one_round_swaps(b)
            lkhds.append(new_likelihood)

            # How many rounds should EACH START take to converge?
            if abs(new_likelihood-old_likelihood)<1e-1 or convergence_itn>50:

                likelihoods.append(new_likelihood)
                break
            else:
                old_likelihood = new_likelihood

    model = models[likelihoods.index(max(likelihoods))]

    if not revert_indices is None:
        model.groups = model.groups[revert_indices]
        model.g = model.g[:,revert_indices,:]
        model.g = model.g[:,:,revert_indices]

    return model


class BlockModel:

    def __init__(self, g, k, corrected=False):
        """Randomly assigns nodes to clusters, and counts inter-cluster ties.

        """

        __slots__ = ['g','k','corrected','groups','counts','n','m','kappa','p']

        # The graph
        self.g = g

        # The number of clusters (assuming 1-mode)
        self.k = k

        # Whether there is a degree correction or not
        self.corrected = corrected

        # An array of cluster assignments for each node
        # g.shape[ONE] because we might have 2 < D < 3
        self.groups = np.random.randint(k, size=g.shape[1])

        # The number of nodes in each group -- indexed by group-id.
        self.counts = collections.Counter(self.groups)

        # Convenient. Will the dimensions ever DIFFER though?
        self.n = g.shape[1]

        # The counts (/weight) of between-cluster ties
        self.m = np.zeros(tuple(self.k for dim in range(self.g.ndim)))
        self.calculate_m()

        # The in/out(/through) group stub counts
        self.kappa = np.zeros((self.g.ndim, self.k))
        self.calculate_kappa()

        # The inter-cluster edge probabilities
        self.p = np.zeros(self.m.shape)
        self.calculate_p()


    def calculate_p(self):
        #Calculate edges/num_possible_edges = m[i,j]/(n[i]*n[j])
        for ix in ndindex(self.m.shape):
            # If the edge weights stay between 0/1, then this is general.
            possible_edges = np.prod([self.counts[i] for i in ix])
            if possible_edges:
                self.p[ix] = float(self.m[ix])/possible_edges
            else:
                self.p[ix] = 0
        # HOW STUPID AM I !?!
        # self.p = self.m/np.sum(self.m)
        # Renormalizing, because of edges < 1. (Right?)
        # self.p = self.p/np.sum(self.p)


    def calculate_kappa(self):
        """Counts the number of 'in' and 'out' stubs for each group

        Actually, counts the number of GENERALIZED edges --
        1) starting in each block
        2) having a MIDpoint in each block IFF the model is 3D
        3) ending in each block

        so that

            self.kappa[0][k]

        is the number of edges that START (have their 0th point) in group 'k'.

        If the model is 2D, this is just the out-degree and in-degree of each
        block. If it's 3D, then you need to think about the elements of edges
        a little more abstractly than "in" and "out" (head and tail) and rather
        as the 1st, 2nd, or even 3rd POINTS of the 'edge', because a 3D matrix
        has 3-pointed edges (which are ordered, if the edge is directed).

        If you're doing a directed AND degree-corrected model, then you also
        need to multiply the CORRECT kappa terms:

            m_ijk / kappa[0][g_i]*kappa[1][g_j]*kappa[2][g_k]

        Which has the more intuitive 2D equivalent of saying 'when you want
        to know the baseline for edges going from block A->B, you want to know
        1) how many come OUT of A, and 2) How many go IN to B -- not IN to A, or
        OUT of B. The example above generalizes this beyond 2 dimensions.

        """

        # For each set of inter-group edges:
        self.kappa = np.zeros((self.g.ndim, self.k)) # RESET!

        for ix in ndindex(self.m.shape):
            # For each group involved in it, and their index
            for i, b in enumerate(ix):
                # At the group's POSITION in the edge, tally
                # for the group to which the edge belongs.
                self.kappa[i][b] += self.m[ix]


    def calculate_m(self):

        for gix in ndindex(self.m.shape):
            self.m[gix] = 0

        #self.m = np.zeros(self.m.shape)
        for ix in ndindex(self.g.shape):
            gix = tuple(self.groups[list(ix)])
            # Allows for weighted edges, by not adding '1 if edge else 0',
            # but rather 'whatever the value is'
            self.m[gix] += self.g[ix]


    def update_m(self, i, old, new):

        if self.g.ndim == 2:

            # The array of i's out-edges
            ex = self.g[i,:]
            ey = self.g[:,i] # and in-edges
            # Issues of double counting are dealt with explicitly below
            for ie, e in enumerate(ex):
                g0 = self.groups[ie]
                # Renders this general for weighted edges, by adding
                # 'whatever was there' instead of hard-coding 0/1
                self.m[(old, g0)] -= e
                self.m[(new, g0)] += e

            for ie, e in enumerate(ey):
                g0 = self.groups[ie]
                self.m[(g0, old)] -= e
                self.m[(g0, new)] += e

            # A FEW FUNNY EDGE CASES, all due to the fact that
            # i's OLD group has already been RESET to the NEW one.
            # As a result, when the edge in question is (i,i),
            # it's treated as SPANNING old-new, instead of as the self-loop
            # it actually is. Writing these few cases out explicitly
            # keeps the loops clean, and avoids a conditional inside
            # the loops, which would be slow -- and this function is
            # written explicitly for speed.
            self.m[(old, old)] -= self.g[i,i] # Was never subtracted
            self.m[(old, new)] += self.g[i,i] # Should not have been added
            self.m[(new, old)] += self.g[i,i] #          "
            self.m[(new, new)] -= self.g[i,i] # Was added twice


        else:

            # The most efficient way I can think of for generating
            # ONLY the indices of edges in which 'i' participates --
            # which are the only relevant ones for updating m.
            ex = [(i, x, y) for x, y in ndindex((self.n, self.n))]
            ey = [(x, i, y) for x, y in ndindex((self.n, self.n))]
            ez = [(x, y, i) for x, y in ndindex((self.n, self.n))]
            # There appeared to be issues with double counting, and getting
            # a unique set simply by sophisticated indexing seemed REALLY
            # tricky in 3D.
            unique_edges = set(ex + ey + ez)

            # This means "We might only have a subset of the groups perceptions"
            # i.e. 'perceiver' might not go all the way to N.
            # To make sure the samples are also INDEXED as 0:num_samples,
            # we reshuffle things at the beginning, then shuffle them back
            # at the end. That's easier than skipping indices.
            unique_edges = [e for e in unique_edges if e[0] < self.g.shape[0]]

            for e in unique_edges:

                # Doesn't have the old/new issues above (that require the
                # edge cases) because this tests to see if the element
                # in question is 'i,' so it's never confused with
                # 'oh, just another element in group 'new'.'
                old_tuple = tuple(old if h==i else self.groups[h] for h in e)
                new_tuple = tuple(new if h==i else self.groups[h] for h in e)
                # Adding/ subtracting g[e] is the same as +1 if edge else 0
                # Except also more general, in that it allows weighted edges,
                # by just asking for 'whatever the edge value is'.
                self.m[old_tuple] -= self.g[e]
                self.m[new_tuple] += self.g[e]



    def change_in_likelihood(self, i, new_group):
        """Return the change in likelihood for moving node i to cluster new."""
        # I want to create a matrix of -- i's degrees into and out of all
        # the other clusters. Right? Anything else?

        # Save the original group
        old_group = self.groups[i]

        # Have a comparison point
        before = self.calculate_likelihood()

        # Do EVERYTHING associated with changing group assignments
        self.set_group(i, new_group)

        # And redo the likelihood, based on that change
        after =  self.calculate_likelihood()

        # BUT NOW YOU'VE GOT TO PUT EVERYTHING BACK.
        # Cheaper maybe to just -- copy the object?
        self.set_group(i, old_group)

        return after - before



    def set_group(self, i, new_group=None):
        """ Change the membership of i to k, then recalculate m, p, and counts.

        The other quantities are changed by definition when groups are
        reassigned, so it's important to make sure all the operations
        happen at once, to maintain an accurate model.
        """
        # One node reassigned can be recalculated relatively efficiently
        if not new_group is None: # Necessary because k=0 is valid
            old_group = self.groups[i]
            self.counts[old_group] -= 1
            self.groups[i] = new_group
            self.counts[new_group] += 1
            # Only do the necessary work -- this needs to be efficient
            self.update_m(i, old_group, new_group)

        else:
            # i is ALL the groups,
            # so redo EVERYTHING
            self.groups = i
            self.counts = collections.Counter(self.groups)
            self.calculate_m()

        # Kappa is trivial given m
        if self.corrected:
            self.calculate_kappa()
        # 'p' relies on 'counts', and on 'm'
        # AND, I don't think can be made efficient --
        # 'counts' and 'm' can though.
        self.calculate_p()










    def calculate_likelihood(self):#, corrected=False):
        """Returns the likelihood of the current model."""

        total = 0

        for ix in ndindex(self.m.shape):
            if self.m[ix]:
                total += self.m[ix]*np.log(self.m[ix])

                if self.corrected:
                    d = [self.kappa[iv, v] for iv, v in enumerate(ix)]
                else:
                    # Does this, and does it NEED to, take
                    # DIRECTED-ness into account?
                    d = [self.counts[v] for v in ix] #self.counts[list(ix)] #
                if all(d):
                    total -= self.m[ix]*np.log(np.prod(d))
        return total










    def plot(self, name=None, layout=None):
        """ Plots and renders the found groups and their relationships.

        If you only have two dimensions, you can plot the actual NODES.

        When you're in 3D with n^3 edges, you
        have to wire the MODEL, not the actual GRAPH,
        so that means working with 'm'.

        This makes separate networks for each layer/perceiver, and
        has edges between them with width equal to the proportional
        value of m[i,j,k], relative to the tensor maximum.

        The fancy layout is supposed to make things evenly
        distributed, but only 'neato' and 'fdp' LISTEN to that,
        and only to a limited and poorly understood degree.
        twopi and dot ignore it completely.
        """


        g = gv.AGraph(directed=True)
        g.node_attr['style']='filled'
        counts = collections.Counter(self.groups)

        #
        if self.g.ndim == 2:
            for i in range(self.n):
                g.add_node(i) #in case there are no edges with it
                for j in range(self.n):
                    if self.g[i,j] and i!=j :
                        g.add_edge(i, j)
            for i in range(self.n):
                n = g.get_node(i)
                n.attr['fillcolor']=color_range('red','blue',self.k)[self.groups[i]]

        if self.g.ndim == 3:

            # Perceiver
            for x in range(self.k):
                # For each Perceiver, make a little graph that represents
                # their perceptions.

                # For each GROUP (instead of person, now)
                for n in range(self.k):
                    # Get the index ... augmented by LAYER
                    ix = n + x*self.k
                    g.add_node(ix)
                    node = g.get_node(ix)
                    if x == n:
                        node.attr['fillcolor'] = 'black'
                    else:
                        node.attr['fillcolor']=['red','blue',"#6699FF","#339933"][n]
                    size = counts[n]/float(self.n)
                    node.attr['height'] = size
                    node.attr['width'] = size
                    node.attr['label'] = ''

                    if   self.k == 2:
                        node.attr['pos'] = "%f,%f"%(50+300*x, 50+50*n)

                    # A little weird cause I was trying to do triangles.
                    elif self.k == 3:
                        if   x == 0:
                            base = np.array([250,  50])
                        elif x == 1:
                            base = np.array([50,  300])
                        elif x == 2:
                            base = np.array([450, 300])

                        if   n == 0:
                            node.attr['pos'] = "%f,%f"%tuple(base)
                        elif n == 1:
                            node.attr['pos'] = "%f,%f"%tuple(base+[-50, 100])
                        elif n == 2:
                            node.attr['pos'] = "%f,%f"%tuple(base+[ 50, 100])


                    elif self.k == 4:

                        # Layout (for each table, and THE tables):
                        # 0 1
                        # 2 3
                        # Go right if you're an ODD index,
                        xcoord = 50 + 600*(x%2==1) + 300*(n%2==1)
                        # Go down if you're a HIGH index.
                        ycoord = 50 + 600*(x > 1)  + 300*(n > 1)
                        # (Amount depending on within/between tables.)
                        node.attr['pos'] = "%f,%f"%(xcoord, ycoord)

                    else:
                        raise Exception("No plotting for k>4 3d models, sorry.")

                # Sender
                for y in range(self.k):
                    # Receiver
                    for z in range(self.k):
                        # The nodes have multiple versions of themselves,
                        # in the multiple (perception) layers of the graph.
                        n1 = y + x*self.k
                        n2 = z + x*self.k
                        #if self.m[x,y,z]:
                            # If there ARE relations between the two groups,
                            # y and z (in this perceivers view) then
                            # get the two nodes that correspond to that
                            # perception.

                        g.add_edge(n1, n2)
                        e = g.get_edge(n1, n2)
                        w = self.m[x,y,z]/float(np.max(self.m))
                        e.attr['style'] = 'setlinewidth(%f)'%(w*10)
                        if w < .10:
                            e.attr['style'] = "invis"# "#FFFFFF"
                            # And set some other things too?

        g.layout()
        if not layout:
            layout = 'neato' if self.g.ndim==3 else 'dot'
        if name:
            file = name + "_" + layout + ".png"
        else:
            file = layout+"_blocks.png"
        g.draw(file, prog=layout)
        os.system("open "+file)


##### END of the BlockModel Class #####

# Actually, how slow would brute be? 2^21 ... two million. I can do that.

def partition(n, k):
    """ Splits range(0,n) into k sub-lists of maximally equal size.

    Used to calculate the nodes in each block, assuming perfect block structure
    """

    N = range(n)
    # Have to round up when k doesn't divide n perfectly, to avoid a
    # 'remainder' cluster
    s = int(math.ceil(float(n)/k))
    partitions = [ N[i:i+s] for i in range(0,n,s)]

    return partitions


def blocked_matrix(n, k, on=1, off=0, dim=2, corrected=False):
    """
    """

    if corrected:
        dist = rand(size=n)
        dist[0:5] = 1
        dist[5:] = .07
        # Any less severe, correction doesn't matter.
        # Tips WHEN -- it appears that p(btwn-node)
        # beats p(btwn-group) -- intuitively.
    else:
        dist = np.ones(n)

    partitions = partition(n, k)

    g = np.zeros((n,)*dim)

    # For each person ...
    for p in range(n):
        # Look at ALL the edges.
        for ix in ndindex(g.shape):
            # Is this edge associated with this person?
            if p in ix:
                # If so, see if they even CONSIDER flipping for it:
                # which is based on their personal propensity.
                # on-diagonal
                # if all the indices are in ONE of the partitions ...
                if any([ all(np.in1d(ix, part)) for part in partitions]):
                    g[ix] = int(rand() < on ) if (rand() < dist[p]) else 0
                # off-diagonal
                else:
                    g[ix] = int(rand() < off) if (rand() < dist[p]) else 0
    np.fill_diagonal(g, 0)
    return g


def show():
    plt.show(block=False)
    raw_input("Press RETURN to close the window")
    plt.close()



def read_css():

    """I may never have written a more painful function in my life.

    If you want data, and are thinking of using numpy or pandas --
    read it in by hand.
    """
    # Can't delete array elements.
    #ix = 1 #so as to skip the header row.
    n = 21
    chunk = 2*n + 3 # Both matrices, and 3 extra rows
    advice = np.zeros((n, n, n))
    friendship = np.zeros((n, n, n))

    path = "/Users/alexloewi/Documents/Data/cognitive social structures/rearranged_cogsocstr.txt"
    #pdf = pd.read_csv("/Users/alexloewi/Documents/Data/cognitive social structures/rearranged_cogsocstr.txt", sep="\t")

    data = []
    with open(path, 'rb') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            data.append(row[0:21]) #?

    # Removing the header?
    del data[0]
    #matrix_columns = pdf[pdf.columns[0:21]]
    #print 'matrix columns!!!!!!,', matrix_columns

    for perceiver in range(n):
        # This gets all the data for one person
        x = (chunk*perceiver)
        y = (chunk*(perceiver+1))-1

        #a = np.array(matrix_columns.ix[x:x+20])
        a = np.array(data[x:x+21])
        np.fill_diagonal(a, 0)
        #f = np.array(matrix_columns.ix[x+21:x+41])
        f = np.array(data[x+21:x+42])
        np.fill_diagonal(f, 0)
        advice[perceiver,:,:]     = a #np.array(matrix_columns.ix[0:20])
        friendship[perceiver,:,:] = f #np.array(matrix_columns.ix[21:41])

    # Consensus matrices (AND rule)
    ca = np.zeros((n,n))
    cf = np.zeros((n,n))

    for i,j in ndindex(ca.shape):
        if advice[i,i,j] + advice[j,i,j] == 2:
            ca[i,j] = 1

    for i,j in ndindex(cf.shape):
        if friendship[i,i,j] + friendship[j,i,j] == 2:
            cf[i,j] = 1

    # Self-proclaimed relationships (OR rule)
    sa = np.zeros((n,n))
    sf = np.zeros((n,n))

    for i,j in ndindex(sa.shape):
        if advice[i,i,j] + advice[j,i,j] >= 1:
            sa[i,j] = 1

    for i,j in ndindex(sf.shape):
        if friendship[i,i,j] + friendship[j,i,j] >= 1:
            sf[i,j] = 1


    return advice, friendship, ca, cf, sa, sf



def tweak(i, d, pfp):#, pfn):

    for e in ndindex(d.shape):

        gix = (i<10, e[0]<10, e[1]<10)

        d[e] = (1-d[e])**2 if np.random.random() < pfp[gix] else d[e]

    return d


def generate_model():

    p = np.array([[.7,.3],[.3,.7]])
    pfp = np.array([[[0,.05],[.05,.1]], [[.1,.05],[.05,0]]])
    pfn = copy.copy(pfp)
    consensus = blocked_matrix(20, 2, on=.7, off=.3)
    data = np.array([copy.copy(consensus) for row in consensus])
    for i, d in enumerate(data):
        data[i] = tweak(i, d, pfp)#, pfn)

    return p, pfp, pfn, consensus, data

# # Need to have the groups available for 'analyses.py' in use
# a, f, ca, cf, sa, sf = read_css()
# cfbc = blockmodel(cf, 2, corrected=True, iterations=100)
# # # Looks good
# # cfbc.plot()
# cabc = blockmodel(ca, 2, corrected=True, iterations=100)
# # cabc.plot()

if __name__ == '__main__':

    #import pandas as pd
    #from matplotlib import pyplot as plt




    #(Also, build a test case for the 3d corrected model --
    #BY, having one model, but having people draw different
    #numbers of edges FROM that model)

    a, f, ca, cf, sa, sf = read_css()


    #a, b, c, d, e, f = read_css()
    #print 'did one'
    #a2, b2, c2, d2, e2, f2 = read_css2()
    b = blockmodel(cf, 2)

    sys.exit()

#    sys.exit()
    # Okay --
    adata = []
    for it in range(10):
        once = []
        # Every iteration, choose a different ordering
        people = range(21)
        np.random.shuffle(people) # IN PLACE

        for n in range(1,21+1):
            print 'On n= ', n
            sample = people[0:n]
            #print 'sample: ', sample
            m = blockmodel(a[sample], 2, corrected=True,
                                         iterations=100,
                                         indices=sample)

            score = gd.d2(m.groups, ba.groups)
            once.append(score)
        adata.append(once)

    plt.plot(np.array(adata).T)
    plt.ylim([0,1.1])
    plt.savefig("a_data.png")


    bf = blockmodel(f, 2, corrected=True, iterations=100)

    # Okay --
    fdata = []
    for it in range(10):
        once = []
        # Every iteration, choose a different ordering
        people = range(21)
        np.random.shuffle(people) # IN PLACE

        for n in range(1,21+1):
            print 'On n= ', n
            sample = people[0:n]
            #print 'sample: ', sample
            m = blockmodel(f[sample], 2, corrected=True,
                                         iterations=100,
                                         indices=sample)

            score = gd.d2(m.groups, bf.groups)
            once.append(score)
        fdata.append(once)

    plt.plot(np.array(fdata).T)
    plt.ylim([0,1.1])
    plt.savefig("f_data.png")







    #print 'blockmodels.py isn\'t set up to do anything right now'
    #b = blockmodel(np.reshape(np.random.random(size=400), (20,20)), 2)
    #b.plot()

    # m =   blocked_matrix(30, 2, corrected=True, dim=3, on=.9, off=.3)
    # b3u = blockmodel(m, 2, corrected=False, iterations=10)
    # b3c = blockmodel(m, 2, corrected=True, iterations=10)
    # print 'uncorrected: ', b3u.groups
    # print 'corrected:   ', b3c.groups
    # print 'g[0]: \n', np.int64(m[0])


    #canx = nx.from_numpy_matrix(ca)
    #cfnx = nx.from_numpy_matrix(cf)
    #nx.write_gml(canx, "ca.gml")
    #nx.write_gml(cfnx, "cf.gml")
    #b = blockmodel(F, 4, iterations=10)
    #karate = "/Users/alexloewi/Documents/Data/karate/karate.gml"
    #kc = np.array(nx.to_numpy_matrix(nx.read_gml(karate)))
    #print kc
    #print kc.ndim
    #raw_input('...')
    #kb = blockmodel(kc, 2, corrected=True, iterations=30)
    #kb.plot()

    # On the consensus model

    # cfbc = blockmodel(cf, 2, corrected=True, iterations=100)
    # cfbc.plot()
    # # On the majority model
    # print 'b'
    # mfbc = blockmodel(np.int64(np.sum(f,axis=0) >= 21/2.), 2, corrected=True, iterations=100)
    # mfbc.plot(name="mf_corrected_k=2")#, "dot")
    #
    # # A 3D model --
    # print 'c'
    # fbc = blockmodel(f, 2, corrected=True, iterations=500)
    # # Put its GROUPS into a 2D model, for interpretable plotting
    # cfbc.groups = fbc.groups
    # cfbc.plot(name="f_in_cf_corrected_k=2")
    #
    # print 'd'
    # # SAME AS ABOVE, BUT FOR ADVICE
    # cabc = blockmodel(ca, 2, corrected=True, iterations=500)
    #cabc.plot(name="ca_corrected_k=3")
    # # On the majority model
    # print 'e'
    # mabc = blockmodel(np.int64(np.sum(a,axis=0) >= 21/2.), 2, corrected=True, iterations=100)
    # #
    # mabc.plot(name="ma_corrected_k=2")#, "dot")
    #
    # # A 3D model --
    # print 'f'
    # abc = blockmodel(a, 3, corrected=True, iterations=500)
    # Put its GROUPS into a 2D model, for interpretable plotting
    # cabc.groups = abc.groups
    # cabc.k = 3
    #cabc.plot(name="a_in_ca_corrected_k=3")



    # kb = blockmodel(ca, 3, corrected=True, iterations=500)
    # kb.plot("ca_corrected_k=3")
    # kb = blockmodel(ca, 4, corrected=True, iterations=500)
    # kb.plot("ca_corrected_k=4")


# def read_css():
#
#     """I may never have written a more painful function in my life.
#
#     If you want data, and are thinking of using numpy or pandas --
#     read it in by hand.
#     """
#     # Can't delete array elements.
#     #ix = 1 #so as to skip the header row.
#     n = 21
#     chunk = 2*n + 3 # Both matrices, and 3 extra rows
#     advice = np.zeros((n, n, n))
#     friendship = np.zeros((n, n, n))
#
#     #path = "/Users/alexloewi/Documents/Data/cognitive social structures/rearranged_cogsocstr.txt"
#     pdf = pd.read_csv("/Users/alexloewi/Documents/Data/cognitive social structures/rearranged_cogsocstr.txt", sep="\t")
#
#     # data = []
#     # with open(path, 'rb') as file:
#     #     reader = csv.reader(file, delimiter="\t")
#     #     for row in reader:
#     #         data.append(row[0:21]) #?
#
#
#     matrix_columns = pdf[pdf.columns[0:21]]
#     #print 'matrix columns!!!!!!,', matrix_columns
#
#     for perceiver in range(n):
#         # This gets all the data for one person
#         x = (chunk*perceiver)
#         y = (chunk*(perceiver+1))-1
#
#         a = np.array(matrix_columns.ix[x:x+20])
#         #a = np.array(data[x:x+21])
#         np.fill_diagonal(a, 0)
#         f = np.array(matrix_columns.ix[x+21:x+41])
#         #f = np.array(data[x+21:x+42])
#         np.fill_diagonal(f, 0)
#         advice[perceiver,:,:]     = a #np.array(matrix_columns.ix[0:20])
#         friendship[perceiver,:,:] = f #np.array(matrix_columns.ix[21:41])
#
#     # Consensus matrices (AND rule)
#     ca = np.zeros((n,n))
#     cf = np.zeros((n,n))
#
#     for i,j in ndindex(ca.shape):
#         if advice[i,i,j] + advice[j,i,j] == 2:
#             ca[i,j] = 1
#
#     for i,j in ndindex(cf.shape):
#         if friendship[i,i,j] + friendship[j,i,j] == 2:
#             cf[i,j] = 1
#
#     # Self-proclaimed relationships (OR rule)
#     sa = np.zeros((n,n))
#     sf = np.zeros((n,n))
#
#     for i,j in ndindex(sa.shape):
#         if advice[i,i,j] + advice[j,i,j] >= 1:
#             sa[i,j] = 1
#
#     for i,j in ndindex(sf.shape):
#         if friendship[i,i,j] + friendship[j,i,j] >= 1:
#             sf[i,j] = 1
#
#
#     return advice, friendship, ca, cf, sa, sf

    #######################################################################
    #######################################################################
    ## OKAY -- what next? ##
    ########################
    ## Symmetry/asymmetry? Am I doing that right?
    ## k > 2 (CHECK)
    ## 3D    (CHECK)
    ## Work out, analytically, some of the questions of disruption
    ## (which ARE?)
    ## Signed, (difficult)
    ## or voting, (more feasible)
    ## or bipartite, (only for signed, no?)
    ## or otherwise opinionated. (... voting)
    #######################################################################
    #######################################################################
