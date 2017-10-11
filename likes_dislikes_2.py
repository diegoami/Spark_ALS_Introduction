
def cmp(a, b):
    return (a > b) - (a < b)


from collections import Counter
import math
from functools import reduce
import random


def denormEntropy(counts):
    '''Computes the entropy of a list of counts scaled by the sum of the counts. If the inputs sum to one, this is just the normal definition of entropy'''
    counts = list(counts)
    total = float(sum(counts))
    # Note tricky way to avoid 0*log(0)
    return -sum([k * math.log(k / total + (k == 0)) for k in counts])


def llr_2x2(k11, k12, k21, k22):
    '''Special case of llr with a 2x2 table'''
    return 2 * abs(denormEntropy([k11 + k12, k21 + k22]) +
                   denormEntropy([k11 + k21, k12 + k22]) -
                   denormEntropy([k11, k12, k21, k22]))


def llr_root(k11, k12, k21, k22):
    '''Computes a score for a 2x2 contingency table, but then adds a sign according to whether k11 is larger (result is positive) or smaller (result is negative) than might be expected. The magnitude of the result can be roughly interpreted on a scale similar to standard deviations'''

    k11 = random.random() / 100000 if k11 == 0 else k11
    k12 = random.random() / 100000 if k11 == 0 else k12
    k21 = random.random() / 100000 if k11 == 0 else k21
    k22 = random.random() / 100000 if k11 == 0 else k22
    row = k11 + k21
    total = (k11 + k12 + k21 + k22)
    sign = cmp(float(k11) / (k11 + k12), float(row) / total)
    return math.copysign(math.sqrt(llr_2x2(k11, k12, k21, k22)), sign)


import numpy as np

rawdata = np.array([
    [5, 5, 0, 0, 0, 0],
    [0, 0, 5, 5, 0, 0],
    [0, 0, 0, 0, 5, 5],
    [0, 1, 5, 5, 5, 0],
    [1, 1, 5, 0, 5, 5],
    [5, 5, 0, 5, 1, 1],
    [5, 0, 0, 5, 0, 1],
    [5, 5, 5, 0, 1, 0]
])

likes = np.array([[1 if x == 5 else 0 for x in row] for row in rawdata])
dislikes = np.array([[1 if x == 1 else 0 for x in row] for row in rawdata])



def retrieve_matrixes( rawdata , likes):

    cooccurrence_matrix = np.dot(likes.transpose(), likes)
    # In[4]:
    np.fill_diagonal(cooccurrence_matrix, 0)
    size = cooccurrence_matrix.shape[0]
    sums = np.array([row.sum() for row in cooccurrence_matrix[:, 0:size]])
    total = sums.sum()

    # In[8]:
    conc_mult = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            a_b = cooccurrence_matrix[i, j].tolist()
            a_not_b = (sums[i] - a_b).tolist()
            b_not_a = (sums[j] - a_b).tolist()
            not_ab = (total - (a_b + sums[i] + sums[j])).tolist()
            conc_mult[i, j] = round(llr_root(a_b, a_not_b, b_not_a, not_ab), 2)
    return cooccurrence_matrix, conc_mult

def extract_matrixes(conc_mult, cooccurrence_matrix, likes):
    return np.dot(conc_mult, likes.T).T

cooccurrence_matrix, conc_mult = retrieve_matrixes(rawdata , likes)
cooccurrence_matrix_d, conc_mult_d = retrieve_matrixes(rawdata , dislikes)

A_liked= extract_matrixes(conc_mult, cooccurrence_matrix, likes)
A_disliked = extract_matrixes(conc_mult_d, cooccurrence_matrix_d, dislikes)


print ( " ============ RAWDATA ========================")
print(rawdata)
print()
print ( " ============ A_LIKED ========================")
print(A_liked)
print(" ============ A_DISLIKED ========================")
print(A_disliked)
print(" ============ A_LIKED - A_DISLIKED ========================")
print(A_liked - A_disliked)

