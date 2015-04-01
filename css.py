import numpy as np
import panda as pd

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

    pdf = pd.read_csv("/Users/alexloewi/Documents/Data/cognitive social structures/rearranged_cogsocstr.txt", sep="\t")

    matrix_columns = pdf[pdf.columns[0:21]]
    #print 'matrix columns!!!!!!,', matrix_columns

    for perceiver in range(n):
        # This gets all the data for one person
        x = (chunk*perceiver)
        y = (chunk*(perceiver+1))-1

        a = np.array(matrix_columns.ix[x:x+20])
        np.fill_diagonal(a, 0)
        f = np.array(matrix_columns.ix[x+21:x+41])
        np.fill_diagonal(f, 0)
        advice[perceiver,:,:]     = a #np.array(matrix_columns.ix[0:20])
        friendship[perceiver,:,:] = f #np.array(matrix_columns.ix[21:41])

    # Consensus matrices (AND rule)
    ca = np.zeros((n,n))
    cf = np.zeros((n,n))

    for i,j in np.ndindex(ca.shape):
        if advice[i,i,j] + advice[j,i,j] == 2:
            ca[i,j] = 1

    for i,j in np.ndindex(cf.shape):
        if friendship[i,i,j] + friendship[j,i,j] == 2:
            cf[i,j] = 1

    # Self-proclaimed relationships (OR rule)
    sa = np.zeros((n,n))
    sf = np.zeros((n,n))

    for i,j in np.ndindex(sa.shape):
        if advice[i,i,j] + advice[j,i,j] >= 1:
            sa[i,j] = 1

    for i,j in np.ndindex(sf.shape):
        if friendship[i,i,j] + friendship[j,i,j] >= 1:
            sf[i,j] = 1


    return advice, friendship, ca, cf, sa, sf
