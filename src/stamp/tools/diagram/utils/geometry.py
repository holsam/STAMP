'''
STAMP: small geometry helpers shared by the scene, pick and native stages
'''

# Import external dependencies
import numpy as np

# non_max_suppression: greedily take the highest-scoring point, drop everything within radius, repeat; turns a dense score field into discrete picks
def non_max_suppression(points, scores, radius):
    order = np.argsort(scores)[::-1]
    used = np.zeros(len(points), dtype=bool)
    taken = []
    for i in order:
        if used[i]:
            continue
        taken.append(int(i))
        used |= np.linalg.norm(points - points[i], axis=1) < radius
    return np.array(taken, dtype=int)

# UnionFind: disjoint-set with path halving, for grouping nearby picker points
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
