import numpy as np
import sys

# lines = open("model/backup_ori/htm_wt.txt").readlines()

# p1 = []
# for line in lines:
# 	num = [float(n) for n in line.split(" ")[:-1]]
# 	res = [np.log(n) for n in num]
# 	p1.append(sum(res))

# # print p1
# mmin = sys.maxint
# mmax = -mmin
# for n in p1:
# 	if n > mmax:
# 		mmax = n
# 	if n < mmin:
# 		mmin = n
# print mmax, mmin
# res = [(n-mmin)/(mmax-mmin) for n in p1]
# res = [n+sorted(res)[1] for n in res]
# print res

'''
Twitter hashtags evaluation
'''
at_lines = open("model/backup_ori/tweets_at.txt")
wt_lines = open("model/backup_ori/tweets_wt.txt")

at = []
wt = []

for line in at_lines:
	num = [float(n) for n in line.split(" ")[:-1]]
