import math

# print(- 0.5 * math.log(0.5, 2) - 0.5 * math.log(0.5, 2))
# print(- 2 * (0.5 * math.log(0.5, 2)))

# import scipy.stats

# print(scipy.stats.entropy([2, 1], base=2))

# Max info gain According to grade
entropy_child = - ((2 / 3) * math.log((2 / 3), 2) + (1 / 3) * math.log((1 / 3), 2))
# print(entropy_child)
print(1 - ((3 / 4) * entropy_child))

# Max info gain according to bumpiness
entropy_child2 = - (1 * math.log(1, 2) + 1 * math.log(1, 2))
entropy_combined = 0.5 * entropy_child2 + 0.5 * entropy_child2
print(1 - entropy_combined)
