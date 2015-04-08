import numpy as np
import blockmodels as bm

def draw(b):
    """Draw a graph from a blockmodel.
    """
    graph = np.zeros((b.n, b.n))
    for e in bm.ndindex(graph.shape):
        gix = tuple(b.groups[list(e)])
        graph[e] = 1 if b.p[gix] > np.random.random() else 0

    return graph

def produce_stats(b, N=100):

    # The drawn graphs
    g = [draw(b) for i in range(N)]

    # The statistics
    s = []

    for i in range(N):
        # The vector of stats for the graph being examined
        v = []

        # Density
        d = np.mean(g[i])
        v.append(d)

        # In/out degrees for the nodes --- does ORDER matter?
        v += list(np.sum(g[i], axis=0))
        v += list(np.sum(g[i], axis=1))

        # Centrality distributions and shit?

        # Finally ...
        s.append(np.array(v))

    return np.array(s)

def compare(b1, b2, N=100):
    """Compare the statistics of two models.

    Used to check for identifiability issues -- do the two models produce
    graphs that look similar?

    Produce a vector of statistics for each of N drawn graphs -- then
    compare the IN-model variance to the BETWEEN-model variance.
    """

    s1 = produce_stats(b1)
    s2 = produce_stats(b2)

    # # I should -- probably -- normalize all the statistics or something, no?
    # in1  = scipy.spatial.distance(s1, s1, 'sqeuclidean')
    # in2  = scipy.spatial.distance(s2, s2, 'sqeuclidean')
    # btwn = scipy.spatial.distance(s1, s2, 'sqeuclidean')
    print "s1: \n", np.mean(s1, axis=0)
    print "s2: \n", np.mean(s2, axis=0)



if __name__=="__main__":
    pass
