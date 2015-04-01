
import os
import sys
import copy
import collections

import numpy as np
import networkx as nx
import pygraphviz as gv
import colour
import pandas as pd

from matplotlib import pyplot as plt


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
                # GOTTA CHANGE IT ***BACK***
            else:
                changes.append(0)
        # After trying all groups, set the new group to the best option
        best = changes.index(max(changes))
        new_groups[i] = best
        # Gotta change the ACTUAL assignment -- this is cumulative
        b.groups[i] = best
        # Switch the matrix -- things have changed.
        b.calculate_m()

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
        b.groups = best_groups
        # Since you got new groups, rewire.
        b.calculate_m()

        # Return the likelihood corresponding to the groups
        return new_max[mx] #b.calculate_likelihood()
    else:
        # The groups don't change.
        return new_max[0]



def blockmodel(g, k, iterations=1):
    """ Takes a graph and a number of clusters, returns group assignments.

    g is the graph, a 2- or 3-d binary (NOT boolean) numpy array
    Right now, treats the network as 1-mode, so there's only 1 k.
    """

    likelihoods = []
    models = []
    # Randomly initialized, so try a few times.
    for itn in range(iterations):
        sys.stdout.write('Iteration #{}\r'.format(itn+1))
        sys.stdout.flush()

        b = BlockModel(g, k)
        models.append(b)

        lkhds = []

        old_likelihood = -np.inf
        new_likelihood = 0
        iterations = 0
        while True:

            iterations += 1

            new_likelihood = one_round_swaps(b)

            lkhds.append(new_likelihood)

            if new_likelihood == old_likelihood or iterations > 100:

                likelihoods.append(new_likelihood)
                plt.plot(lkhds)
                break
            else:
                old_likelihood = new_likelihood

    # This are comparable, no?
    return models[likelihoods.index(max(likelihoods))]


class BlockModel:

    def __init__(self, g, k, symmetric=False):
        """Randomly assigns nodes to clusters, and counts inter-cluster ties.

        """
        # The graph
        self.g = g

        # The number of clusters (assuming 1-mode)
        self.k = k

        # An array of cluster assignments for each node
        self.groups = np.random.randint(k, size=g.shape[0])

        # Is the graph symmetric? (Relevance for 3d?)
        self.symmetric = symmetric

        # Convenient. Will the dimensions ever DIFFER though?
        self.n = g.shape[0]

        # The counts of between-cluster ties
        self.m = np.zeros(tuple(self.k for dim in range(self.g.ndim)))

        # The set of node<->group edge counts needed for diff(likelihood)
        # Useful to have on hand, if it COULD be (I think)
        # also defined as-needed in the diff-step -- but that's also
        # a class function.
        self.ki = np.zeros(tuple(self.k for dim in range(self.g.ndim)))

        # Initialize this matrix
        self.calculate_m()




    def count_between_edges(self, x, node=None):#, group=None):
        """ Populates tables of group<->group ties, for one or all nodes.

        The engine under calculate_m and calculate_k.
        """



        if self.g.ndim == 2:

            # Am I actually assigning to the OBJECT?
            #x = np.zeros(tuple(self.k for i in range(self.g.ndim)))
            for i in range(self.k):
                for j in range(self.k):
                    x[i,j] = 0 # But THIS should work ... no?

            for i in range(self.n):
                for j in range(self.n):

                    # The groups the nodes belong to
                    r, s = self.groups[[i, j]]
                    # If we're only looking for ONE node's numbers ...
                    if node and node not in [r, s]:
                        break
                    else:
                        # Don't TOTALLY understand the double-diagonal,
                        # but it's in K+N
                        # PAY CLOSE ATTENTION -- we're adding to 'x',
                        # which could be m, OR k.
                        x[r,s] += self.g[i,j]*(2 if r==s else 1)


        elif self.g.ndim == 3:

            # Am I actually assigning to the OBJECT?
            #x = np.zeros(tuple(self.k for i in range(self.g.ndim)))
            for i in range(self.k):
                for j in range(self.k):
                    for k in range(self.k):
                        x[i,j,k] = 0 # But THIS should work ... no?

            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):# if self.g.ndim==3 else [None]:

                        # The groups the nodes belong to
                        #ix_tuple = tuple(x for x in (i,j,k) if x)
                        r, s, t = self.groups[[i, j, k]] #group_tuple = self.groups[list(ix_tuple)]
                        # If we're only looking for ONE node's numbers ...
                        if node and node not in [r, s, t]: #if node and node not in group_tuple:
                            break
                        else:
                            #x[group_tuple] += self.g[ix_tuple]*(2 if all([gt==group_tuple[0] for gt in group_tuple]) else 1))
                            x[r,s,t] += self.g[i,j,k]*(2 if r==s==t else 1)
        else:
            # Wrong number of dimensions. ?
            raise Exception("The dimension of the matrix was not 2 or 3")


    def calculate_m(self):
        #self.m = self.count_between_edges(self.)
        self.count_between_edges(self.m)


    def calculate_k(self, node):

        self.count_between_edges(self.ki, node=node)#, group=group)
        #self.kib, self.kbi = self.count_between_edges(self.)


    def change_in_likelihood(self, i, k):
        """Return the change in likelihood for moving node i to cluster k."""
        # I want to create a matrix of -- i's degrees into and out of all
        # the other clusters. Right? Anything else?

        # First, and trivially --
        old_group = self.groups[i]
        #old_m = copy.copy(self.m)


        before = self.calculate_likelihood()


        self.groups[i] = k
        # Redo the group matrix,
        self.calculate_m()
        # And redo the likelihood, based on it.
        after =  self.calculate_likelihood()


        # BUT NOW YOU'VE GOT TO PUT EVERYTHING BACK.
        self.groups[i] = old_group
        #self.m = old_m
        self.calculate_m()

        return after - before

        # Next, see what the difference is ... which is easier than
        # calculating the WHOLE likelihood.


    def calculate_likelihood(self, corrected=False):
        """Returns the likelihood of the current model."""

        # THIS ONLY WORKS FOR 2D MODELS.
        # Degree corrected, or not?
        # L(G|g) = SUM_rs m_rs log m_rs / n_r*n_s

        if corrected:
            raise Exception("The degree corrected version isn't implemented")

            total = 0

            if self.g.ndim == 2:

                # fuck -- directed? then it's much more complicated.


                for i in range(self.k):
                    for j in range(self.k):

                        # This is avoiding a Divide By Zero Warning
                        if self.m[i,j]:
                            total += self.m[i,j] * np.log(self.m[i,j])
                            if n[i] and n[j]:
                                total -= self.m[i,j] * np.log(n[i]*n[j])

            if self.g.ndim == 3:

                for i in range(self.k):
                    for j in range(self.k):
                        for k in range(self.k):

                            # This is avoiding a Divide By Zero Warning
                            if self.m[i,j,k]:
                                total += self.m[i,j,k] * np.log(self.m[i,j,k])
                                if n[i] and n[j] and n[k]:
                                    total -= self.m[i,j,k] * np.log(n[i]*n[j]*n[k])

        else:
            n = collections.Counter(self.groups)

            total = 0

            if self.g.ndim == 2:
                for i in range(self.k):
                    for j in range(self.k):

                        # This is avoiding a Divide By Zero Warning
                        if self.m[i,j]:
                            total += self.m[i,j] * np.log(self.m[i,j])
                            if n[i] and n[j]:
                                total -= self.m[i,j] * np.log(n[i]*n[j])
            if self.g.ndim == 3:

                for i in range(self.k):
                    for j in range(self.k):
                        for k in range(self.k):

                            # This is avoiding a Divide By Zero Warning
                            if self.m[i,j,k]:
                                total += self.m[i,j,k] * np.log(self.m[i,j,k])
                                if n[i] and n[j] and n[k]:
                                    total -= self.m[i,j,k] * np.log(n[i]*n[j]*n[k])

        return total


    def plot(self, name=None):
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
        layout = 'neato' if self.g.ndim==3 else 'dot'
        if name:
            file = name + ".png"
        else:
            file = layout+"_blocks.png"
        g.draw(file, prog=layout)
        os.system("open "+file)


##### END of the BlockModel Class #####

# Actually, how slow would brute be? 2^21 ... one million. I can do that.


if __name__ == '__main__':

    def blocked_matrix(n, k, dim=2):

        # Partitions
        # There MUST be a cleaner way to do this though.
        partitions = [tuple(i for i in range(j, j+n/k) if i < n)
                              for j in range(0, n, n/k)]
        #x = range(n)
        #splits = range(0, n, n/k)
        #[x[i:j] for

        g = np.zeros(tuple(n for i in range(dim)))
        if dim == 2:
            for i in range(n):
                for j in range(n):
                    # on-diagonal
                    if any([i in p and j in p for p in partitions]):
                        g[i,j] = int(np.random.random() > .4)
                    # off-diagonal
                    else:
                        g[i,j] = int(np.random.random() > .9)

        if dim == 3:
            for x in range(n):
                for y in range(n):
                    for z in range(n):
                        # on-diagonal
                        if any([x in p and y in p and z in p for p in partitions]):
                            g[x,y,z] = int(np.random.random() > .1)
                        # off-diagonal
                        else:
                            g[x,y,z] = int(np.random.random() > .9)

        return g

    def show():
        plt.show(block=False)
        raw_input("Press RETURN to close the window")
        plt.close()


    def read_in_css():

        """I may never have written a more painful function in my life.

        If you want data, and are thinking of use numpy or pandas --
        read it in by hand.
        """

        #f = np.genfromtxt("/Users/alexloewi/Documents/Data/cognitive social structures/rearranged_cogsocstr.txt",
        #                    delimiter="\t",
        #                    filling_values=np.nan)

        # Can't delete array elements.
        #ix = 1 #so as to skip the header row.
        n = 21
        chunk = 2*n + 3 # Both matrices, and 3 extra rows
        advice = np.zeros((n, n, n))
        friendship = np.zeros((n, n, n))

        pdf = pd.read_csv("/Users/alexloewi/Documents/Data/cognitive social structures/rearranged_cogsocstr.txt", sep="\t")

        matrix_columns = pdf[pdf.columns[0:21]]
        #print 'matrix columns!!!!!!,', matrix_columns

        for perceiver in range(n):
            # This gets all the data for one person
            x = (chunk*perceiver)
            y = (chunk*(perceiver+1))-1
            #print matrix_columns.ix[x:y]
            #print data.ix[0:20]
            #print x, y
            a = np.array(matrix_columns.ix[0:20])
            np.fill_diagonal(a, 0)
            f = np.array(matrix_columns.ix[21:41])
            np.fill_diagonal(f, 0)
            advice[perceiver,:,:]     = a #np.array(matrix_columns.ix[0:20])
            friendship[perceiver,:,:] = f #np.array(matrix_columns.ix[21:41])

        return advice, friendship


    A, F = read_in_css()
    b = blockmodel(F, 4, iterations=10)
    #blocked_matrix(12, 3, dim=3), 3, iterations=1)
    # gotta do some ... model selection on cluster number
    # ... how?
    # print b.groups
    b.plot("friendship_k=4")
    show()


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
