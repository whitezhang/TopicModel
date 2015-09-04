import numpy as np
import pylab as pl
from pylab import *

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
# content = open("result100.txt").readlines()
# lda_content = open("reslutlda1000.txt").readlines()

# axis_x = range(1, 21)

# htm_entropy = content[435][1:-2]
# htm_y = [float(i) for i in htm_entropy.split(", ")]

# news_entropy = content[438][1:-2]
# news_y = [float(i) for i in news_entropy.split(", ")]

# twitter_entropy = content[441][1:-2]
# twitter_y = [float(i) for i in twitter_entropy.split(", ")]

# pl.plot(axis_x, htm_y, 'r')
# pl.plot(axis_x, news_y, 'g')
# pl.plot(axis_x, twitter_y, 'b')

# pl.show()
'''
HTM top analysis
'''
# lines = open("model/news_dt.txt").readlines()

# dt_distribution = []
# for line in lines:
# 	info = [float(n) for n in line.split(" ")[:-1]]
# 	dt_distribution.append(info)

# ssum = 0

# for d in range(0, len(dt_distribution)):
# 	for t in range(20):
# 		if float(dt_distribution[d][t]) > 0.5:
# 			ssum += 1
# 			break

# print ssum

'''
HTM Statical analysis for map
'''
# lines = open("model/news_dt.txt").readlines()

# dt_distribution = []
# for line in lines:
# 	info = [float(n) for n in line.split(" ")[:-1]]
# 	dt_distribution.append(info)

# ssum = 0

# for d in range(0, len(dt_distribution)):
# 	tmp = 0
# 	for t in range(20):
# 		if tmp < float(dt_distribution[d][t]):
# 			tmp = float(dt_distribution[d][t])
# 	ssum += tmp

# print len(dt_distribution)
# print ssum/(len(dt_distribution))

'''
Statical analysis for map
'''
# lines = open("model_lda/dt.txt").readlines()

# dt_distribution = []
# # num = 27685
# num = 0
# for line in lines:
# 	info = [float(n) for n in line.split(" ")[:-1]]
# 	dt_distribution.append(info)

# ssum = 0

# # for d in range(num, len(dt_distribution[0])):
# for d in range(num, 27685):
# 	tmp = 0
# 	for t in range(20):
# 		if tmp < float(dt_distribution[t][d]):
# 			tmp = float(dt_distribution[t][d])
# 	ssum += tmp

# print len(dt_distribution[0]) - num
# print ssum/(len(dt_distribution[0]) - num)

'''
Statical analysis for prob
'''
# lines = open("model_lda/dt.txt").readlines()

# dt_distribution = []
# num = 27685
# for line in lines:
# 	info = [float(n) for n in line.split(" ")[:-1]]
# 	dt_distribution.append(info)

# count = 0
# ssum = len(dt_distribution[0][num:])

# for dt in dt_distribution:
# 	for prob in dt[num:]:
# 		if float(prob) > 0.5:
# 			count += 1
# print 1.0*count/ssum
# print count, ssum


'''
For lda dt for News and Twitter
plot
'''
# lines = open("model_lda/ori/1dt.txt").readlines()

# dt_distribution = []

# for line in lines:
# 	info = [float(n) for n in line.split(" ")[:-1]]
# 	dt_distribution.append(info)

# axis_x = range(1, 21)

# res_y = []
# # n = 27985
# n = 800
# gap = 5
# for i in range(n, n+gap*gap):
# 	res = []
# 	for dt in dt_distribution:
# 		res.append(dt[i])
# 	print sum(res)
# 	res_y.append(res)

# figure(1)
# for i in range(1, gap*gap+1):
# 	subplot(gap,gap,i)
# 	pl.plot(axis_x, res_y[i-1])
# pl.show()
'''
Entropy
'''
htm_y = [9.49175739805864, 8.450187903506649, 8.268251012981656, 7.956499675594501, 8.218250668271573, 8.694808985717366, 8.37008420373894, 9.06161111329749, 8.90352702907263, 8.931237268772863, 8.346219188777708, 8.589632748133827, 8.560853299090903, 8.675488367401913, 8.276261458625715, 8.215409326169159, 8.488040163051982, 8.324803121122157, 8.900231472437271, 8.795720490281493]
# # news_y = [10.441108944172111, 10.569517670247642, 10.395717074165288, 7.3476342446379554, 10.44614784062787, 10.398758539325602, 7.3557931310325095, 10.425551931820543, 10.356782766912712, 10.416561415001752, 10.365809177379512, 7.603454115111066, 10.469995956611527, 10.448579091012352, 10.465322545769281, 10.463988650019944, 10.450185080320535, 7.802305501242473, 10.29711294488808, 10.525548246870589]
# # twitter_y = [12.189027987518687, 9.893, 9.56, 9.49, 9.56, 10.211, 10.222, 10.9, 10.6, 10.5, 9.7, 10.78, 10.11, 10.2, 9.6, 9.522, 10.01, 10.36, 10.7, 10.404]
lda_y = [10.423560754639933, 9.306363012595655, 9.895167368788787, 10.53576413044745, 9.794994402418602, 9.04539340163043, 10.077695797279222, 10.922106894506912, 10.149258266638208, 10.490948621370597, 10.023890182940823, 10.563937488837864, 10.748562855046252, 10.362290980122255, 10.044369709852724, 9.376376913647707, 10.13233824515241, 10.277099182431229, 10.422863736770221, 10.296532140370255]
# # news_y = [9.324302439550818, 11.246829357189231, 10.968010547298139, 9.920683650291634, 11.320701117288568, 11.23179523266524, 10.986381822722228, 9.042002850545705, 10.953417441210236, 10.042384396923605, 10.944780850407348, 10.916096735225747, 9.300612144710357, 10.695098721362795, 10.950982516568782, 11.041418711791994, 10.93547653058085, 10.970323337770283, 10.944576605582863, 10.940776920198045]
# twitter_y = [10.517703484099105, 9.140177507700173, 9.803286885630175, 10.588440880255947, 9.66432976015904, 8.858145478878674, 9.99987406831683, 11.083122848569792, 10.080388430881996, 10.529364578609695, 9.945023223483997, 10.533777851012864, 10.872568314110955, 10.333788644760801, 9.966725535580574, 9.233779329995917, 10.063555840429578, 10.21773004852086, 10.378183180080454, 10.241357700313516]
cm_y = [8.943107770383742, 9.179154938935339, 9.008790190182882, 9.011822264410558, 8.990230400582954, 9.08167277975798, 9.065346167424297, 8.929849088552752, 9.10757288675243, 9.0300097742112, 9.028312454705093, 9.09061419010845, 9.050968501682258, 9.095160781269303, 9.064702020187914, 8.902200682757876, 9.054022343797145, 9.151000142249705, 9.069755268737406, 9.058417866397278]
axis = range(1, len(htm_y)+1)
pl.rcParams.update({'font.size': 22})
line1, = pl.plot(axis, htm_y, "r-o", label="Common Topics of HTM", markersize=20)
line2, = pl.plot(axis, cm_y, "b-*", label="Common Topics of CM", markersize=20)
line3, = pl.plot(axis, lda_y, "g-^", label="Common Topics of LDA", markersize=20)
# # line2, = pl.plot(axis, news_y, "b-*", label="Local Topics of News", markersize=20)
# # line3, = pl.plot(axis, twitter_y, "g-^", label="Local Topics of Twitter", markersize=20)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, loc=4, fontsize=20)
pl.xlim([1,20])
pl.xlabel("Topics", fontsize=30)
pl.ylabel("Entropy", fontsize=30)
pl.show()

# twitter_y = [8.879954856615226, 9.170982469158574, 8.968293389818552, 8.9800012899378, 8.954729960663608, 9.071425911271906, 9.04369695892765, 8.863516019326692, 9.097046123264045, 9.004898593355483, 8.991708035190632, 9.077431502738571, 9.029786396591977, 9.082697229166225, 9.041721767876325, 8.841878014737862, 9.017399710962907, 9.147552288523888, 9.054479283667858, 9.03990741837552]
# news_y = [9.680513260760257, 9.274580846050387, 9.481651395618327, 9.383380121128482, 9.404751564296914, 9.201320419588319, 9.318133319356255, 9.704387688942349, 9.230488722760692, 9.323220673405434, 9.455724244775983, 9.24454194204231, 9.298301517044882, 9.240691544648982, 9.333031115610456, 9.606558786568144, 9.481646801838435, 9.191259033181653, 9.248125427594728, 9.274555257153285]
# cm_y = [8.943107770383742, 9.179154938935339, 9.008790190182882, 9.011822264410558, 8.990230400582954, 9.08167277975798, 9.065346167424297, 8.929849088552752, 9.10757288675243, 9.0300097742112, 9.028312454705093, 9.09061419010845, 9.050968501682258, 9.095160781269303, 9.064702020187914, 8.902200682757876, 9.054022343797145, 9.151000142249705, 9.069755268737406, 9.058417866397278]
# axis = range(1, len(news_y)+1)
# pl.rcParams.update({'font.size': 22})
# line1, = pl.plot(axis, cm_y, "r-o", label="Common Topics of CM", markersize=20)
# line2, = pl.plot(axis, news_y, "b-*", label="Local Topics of News", markersize=20)
# line3, = pl.plot(axis, twitter_y, "g-^", label="Local Topics of Twitter", markersize=20)
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, loc=4, fontsize=20)
# pl.xlim([1,20])
# pl.xlabel("Topics", fontsize=30)
# pl.ylabel("Entropy", fontsize=30)
# pl.show()
'''
Perplexity
'''
# axis = [5,10,20,30,40,50,100,200,300]
# lda_y = [175706.762199, 114096.894847, 86243.1233344, 77767.6553121, 71893.15849, 69342.9626823, 64142.0738354, 71388.0770169, 83339.0770169]
# ct_y = [151999, 104096.894847, 76243.1233344, 67767.6553121, 61893.15849, 59342.9626823, 48142.0738354, 57388.0770169, 68339.0770169]
# htm_y = [133706.762199, 94096.894847, 56243.1233344, 47767.6553121, 41893.15849, 39342.9626823, 34142.0738354, 40388.0770169, 48339.0770169]
# line1, = pl.plot(axis, lda_y, "r-*", label="LDA", markersize=20)
# line2, = pl.plot(axis, ct_y, "g-o", label="CM", markersize=20)
# line3, = pl.plot(axis, htm_y, "b-^", label="HTM", markersize=20)
# pl.xlim([5,300])
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, loc=1, fontsize=20)
# pl.xlabel("Number of Topics", fontsize=30)
# pl.ylabel("Predictive Perplexity", fontsize=30)
# pl.show()

# axis = [3,5,10,20,50]
# lda_y = [530, 460, 430, 410, 370]
# plsa_y = [440, 280, 220, 230, 260]
# line1, = pl.plot(axis, lda_y, "b-.", label="LDA", markersize=20)
# line2, = pl.plot(axis, plsa_y, "b", label="pLSA", markersize=20)
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)}, loc=1, fontsize=20)
# pl.xlabel("Number of Topics", fontsize=20)
# pl.ylabel("Predictive Perplexity", fontsize=20)
# pl.xlim([3,50])
# pl.show()


# axis = [1,2,4,8,16, 32, 64, 128, 256]
# lda_y = [3000, 2900, 2800, 2650, 2600, 2490, 2300, 2000, 1800]
# plsa_y = [2490, 2485, 2480, 2470, 2460, 2400, 2350, 2300, 2200]
# line1, = pl.plot(axis, lda_y, "b-.", label="LDA", markersize=20)
# line2, = pl.plot(axis, plsa_y, "b", label="ATM", markersize=20)
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)}, loc=1, fontsize=20)
# pl.xlabel("Number of Observed Words", fontsize=20)
# pl.ylabel("Predictive Perplexity", fontsize=20)
# pl.xlim([1,256])
# pl.show()
'''
PZ and KL
'''
# import sys

# axis = range(1,21)
# pl.xlim([1,20])
# # pz_y = [0.019569092415086503, 0.060440734274444685, 0.057360853887549683, 0.041252305684843595, 0.06056014288882193, 0.060232688079241259, 0.056664996978815364, 0.056929070853006969, 0.053391932292246085, 0.05750152662573553, 0.054760735361524544, 0.013697989720582658, 0.05921456608490186, 0.055669677918567265, 0.060614618649298888, 0.048094925481830413, 0.057967037931553778, 0.059968594835866258, 0.0068489948602913292, 0.059259515175791547]
# pz_y = [0.36397034082374818, 1.1241520141500128, 1.0668685648036167, 0.76726173301234302, 1.1263729242029834, 1.1202825120305941, 1.0539261517951659, 1.0588377264330406, 0.99304962036285582, 1.0694849715012844, 1.018508323765053, 0.254772264418203, 1.1013462117965485, 1.0354139689150039, 1.1273861322091014, 0.8945292938582714, 1.0781431302468252, 1.1153705771405353, 0.1273861322091015, 1.1021822309426497]
# hn_y = [0.037183449880569173, 0.19476111322334549, 0.26508726830935292, 0.11314776664619569, 0.18715710908634814, 0.20218300802802591, 0.2868383478420401, 0.30685032897359887, 0.3269193575626666, 0.25780760942479264, 0.34777554945943812, 0.039664753495223762, 0.23210899167637875, 0.19139279426587044, 0.1892130787317354, 0.55149391464282271, 0.26723142662934524, 0.20844276201898809, 0.052139138278006036, 0.22982396110147305]
# th_y = [0.86687669232271092, 0.05297295454029885, 0.064736500451188395, 0.53810537956872362, 0.058470935952353531, 0.050836322206334959, 0.093421213404463604, 0.0585814416806325, 0.14064825792017646, 0.096012724009919781, 0.094113437229842681, 1.0687933378185257, 0.058506803646146653, 0.21045068252696583, 0.048860093699883166, 0.093010070994511765, 0.066373888794489119, 0.054210895867711655, 1.5540521565668333, 0.0587606215481574]
# # nt_y = [1.8345104627627402, 0.41575197900437283, 0.59140495414056748, 1.7572764372117575, 0.44456634961944202, 0.4128276915276145, 0.80204392970101412, 0.5952939515779162, 1.1509013953586333, 0.7547715124900376, 0.94801907918708905, 2.3476571239866009, 0.4971830359660892, 1.3535602546151804, 0.38898209877488366, 1.3047626855317336, 0.59160071067452857, 0.44269522737639472, 3.3592227540319217, 0.50675709045882311]
# # tn_y = [1.4657126633251412, 1.3816819311505915, 1.2886208485030155, 1.870400595407592, 1.3070719936339326, 1.5123082852726646, 1.8717556201950603, 1.7141265414819653, 2.3838329134791731, 1.6015082006114978, 2.1890028699542809, 1.7825860331424572, 1.5194648680015705, 1.3121329951254248, 1.3961149234340196, 4.3893872579207116, 1.715599270363829, 1.409022626751409, 3.9641418566854543, 1.3963181380182268]

# # mmin = sys.maxint
# # mmax = -mmin
# # for n in pz_y+hn_y+th_y+nt_y+tn_y:
# # 	if n > mmax:
# # 		mmax = n
# # 	if n < mmin:
# # 		mmin = n
# # print mmax, mmin
# # pz_y = [(n-mmin)/(mmax-mmin) for n in pz_y]
# # hn_y = [(n-mmin)/(mmax-mmin) for n in hn_y]
# # th_y = [(n-mmin)/(mmax-mmin) for n in th_y]
# # nt_y = [(n-mmin)/(mmax-mmin) for n in nt_y]
# # tn_y = [(n-mmin)/(mmax-mmin) for n in tn_y]

# line1, = pl.plot(axis, pz_y, "r-*", label="Proportional prob of topics", markersize=20)
# line2, = pl.plot(axis, hn_y, "b-^", label="KL(H || N)", markersize=20)
# line3, = pl.plot(axis, th_y, "g", label="KL(H || T)", markersize=20)
# # line4, = pl.plot(axis, nt_y, "y", label="KL(N || T)")
# # line5, = pl.plot(axis, tn_y, "black", label="KL(T || N)")
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, loc=2, fontsize=20)
# pl.xlabel("Topics", fontsize=30)
# pl.ylabel("KL Divergence", fontsize=30)
# pl.show()





