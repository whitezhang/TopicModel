#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import os

model = "./model/"

class LDA:
	def __init__(self, K, alpha, beta, docs, V, smartinit=True):
		print "News alpha =", alpha, ",\teta =", beta
		self.K = K
		self.alpha = alpha # parameter of topics prior
		self.beta = beta   # parameter of words prior
		self.docs = docs
		self.V = V

		self.z_m_n = [] # topics of words of documents
		self.n_m_z = np.zeros((len(self.docs), K)) + alpha	 # word count of each document and topic
		self.n_z_t = np.zeros((K, V)) + beta # word count of each topic and vocabulary
		self.n_z = np.zeros(K) + V * beta	# word count of each topic
		self.N = 0
		for m, doc in enumerate(docs):
			self.N += len(doc)
			z_n = []
			for t in doc:
				if smartinit:
					p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
					z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
				else:
					z = np.random.randint(0, K)
				z_n.append(z)
				self.n_m_z[m, z] += 1
				self.n_z_t[z, t] += 1
				self.n_z[z] += 1
			self.z_m_n.append(np.array(z_n))

	def inference(self):
		"""learning once iteration"""
		for m, doc in enumerate(self.docs):
			z_n = self.z_m_n[m]
			n_m_z = self.n_m_z[m]
			for n, t in enumerate(doc):
				# discount for n-th word t with topic z
				z = z_n[n]
				n_m_z[z] -= 1
				self.n_z_t[z, t] -= 1
				self.n_z[z] -= 1

				# sampling topic new_z for t
				p_z = self.n_z_t[:, t] * n_m_z / self.n_z
				new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

				# set z the new topic and increment counters
				z_n[n] = new_z
				n_m_z[new_z] += 1
				self.n_z_t[new_z, t] += 1
				self.n_z[new_z] += 1

	def wt_matrix(self):
		"""get word count"""
		return self.n_z_t

	def worddist(self):
		"""get topic-word distribution"""
		return self.n_z_t / self.n_z[:, np.newaxis]

	def docdist(self):
		"""get document-topic distribution"""
		n_m = np.zeros(len(self.docs))
		for i in range(len(self.n_m_z)):
			n_m[i] += sum(self.n_m_z[i])
		probs = self.n_m_z/n_m[:, np.newaxis]		
		return self.n_m_z / n_m[:, np.newaxis]

	def perplexity(self, docs=None):
		if docs == None: docs = self.docs
		phi = self.worddist()
		log_per = 0
		N = 0
		Kalpha = self.K * self.alpha
		for m, doc in enumerate(docs):
			theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
			for w in doc:
				log_per -= np.log(np.inner(phi[:,w], theta))
			N += len(doc)
		return np.exp(log_per / N)

# Author Topic Model
class ATM(object):
	def __init__(self, W, K, A, docList, authorList, alpha=0.5, eta=0.5):
		"""
		Initialize at_model

		vocab = vocabulary list
		K = number of topics
		A = number of authors
		alpha = author-topic distribution dirichlet parameter
		eta = word-topic distribution dirichlet parameter

		docList
			list of documents, constructed based on the vocab
			format = list(list(words))
			ex) [[0,2,2,3],[1,3,3,4]]
				tokens of 1st document= 0,2,2,3 (note that 2 appears twice becase word 2 used twice in the first document)
		authorList 
			format = list(list(authors))
			at least one author should be exist for each document
			ex) [[0,1],[1,2]] 
				authors of 1st doc = 0, 1
		"""
		print "Twitters alpha =",alpha, ",\teta =",eta
		self._W = W
		self._K = K
		self._A = A
		self._D = len(docList)
		self._docList = docList
		self._authorList = authorList
		self._alpha = alpha
		self._eta = eta

		self.c_wt = np.zeros([self._W, self._K]) + self._alpha
		self.c_at = np.zeros([self._A, self._K]) + self._eta
		self.topic_assigned = list()
		self.author_assigned = list()
		self.topic_sum = np.zeros(self._K)
		self.author_sum = np.zeros(self._A)

		#initialization
		for di in xrange(0, self._D):
			self.author_assigned.append(list())
			self.topic_assigned.append(list())
			doc = self._docList[di]
			authors = self._authorList[di]
			for wi in xrange(0, len(doc)):
				w = doc[wi]
				#random sampling topic
				z = np.random.choice(self._K, 1)[0]
				#random sampling author
				a = np.random.choice(len(authors),1)[0]

				#assigning sampled value (sufficient statistics)
				self.c_wt[w,z] += 1
				self.c_at[authors[a],z] += 1
				self.topic_sum[z] += 1
				self.author_sum[authors[a]] += 1

				#keep sampled value for future sampling
				self.topic_assigned[di].append(z)
				self.author_assigned[di].append(authors[a])

	def worddist(self):
		return self.c_wt/self.wt_sum

	def wt_matrix(self):
		"""get W*T matrix"""
		return self.c_wt

	def perplexity(self):
		docs = self._docList
		phi = self.worddist()
		log_per = 0
		N = 0
		Kalpha = self._K * self._alpha
		for ai, ad in enumerate(self._authorList):
			for w in docs[ai]:
				for a in ad:
					theta = self.c_at[a] / (self._A + Kalpha)			
					log_per -= np.log(np.inner(phi[w, :].T, theta))
			N += len(docs[ai])
		return np.exp(log_per / N)


	def sampling_topics(self, max_iter):
		for iter in xrange(0, max_iter):
			for di in xrange(0, len(self._docList)):
				doc = self._docList[di]
				authors = self._authorList[di]

				for wi in xrange(0, len(doc)):
					w = doc[wi]
					old_z = self.topic_assigned[di][wi]
					old_a = self.author_assigned[di][wi]

					self.c_wt[w, old_z] -= 1
					self.c_at[old_a, old_z] -= 1
					self.topic_sum[old_z] -= 1
					self.author_sum[old_a] -= 1

					wt = (self.c_wt[w, :]+ self._eta)/(self.topic_sum+self._W*self._eta) 
					at = (self.c_at[authors,:] + self._alpha)/(self.author_sum[authors].repeat(self._K).reshape(len(authors),self._K)+self._K*self._alpha)

					pdf = at*wt
					pdf = pdf.reshape(len(authors)*self._K)
					pdf = pdf/pdf.sum()

					#sampling author and topic
					idx = np.random.multinomial(1, pdf).argmax()

					new_ai = idx/self._K
					new_z = idx%self._K

					new_a = authors[new_ai]
					self.c_wt[w,new_z] += 1
					self.c_at[new_a, new_z] += 1
					self.topic_sum[new_z] += 1
					self.author_sum[new_a] += 1
					self.topic_assigned[di][wi] = new_z
					self.author_assigned[di][wi] = new_a
		self.wt_sum = np.sum(self.c_wt, axis=0)
		self.at_sum = np.sum(self.c_at, axis=1)
		self.theta = self.c_at/self.at_sum[:, np.newaxis]	# author-topic
		self.phi = self.c_wt/self.wt_sum[:np.newaxis]	# word-topic
		return self.phi

# lda Learning
def lda_learning(lda, iteration, voca):
	pre_perp = lda.perplexity()
	for i in range(iteration):
		lda.inference()
		perp = lda.perplexity()
		if pre_perp:
			if pre_perp < perp:
				# output_word_topic_dist(lda, voca)
				# pre_perp = None
				pre_perp = perp
				break
			else:
				pre_perp = perp
	return pre_perp
	# output_word_topic_dist(lda, voca)

def atm_learning(atm, iteration, voca):
	atm.sampling_topics(iteration)

class HTM(object):
	# htm = HTM(options.K, option.alpha, options.beta, docs, news_len, voca.size, options.smartinit, num_authors, author_set)
	def __init__(self, K, alpha, beta, docs, news_len, num_authors, author_set, voca):
		print "\tInitialize HTM sampler"
		self.voca = voca
		self.lda_sampler = LDA(K, alpha, beta, docs[:news_len], voca.size(), False)
		self.atm_sampler = ATM(voca.size(), K, num_authors, docs[news_len:], author_set)

	def gibbs_sampling(self, iteration):
		for i in range(iteration):
			print "Iteration", i
			lda_learning(self.lda_sampler, 1, self.voca)
			atm_learning(self.atm_sampler, 1, self.voca)
			print "Preplexity of News =", self.lda_sampler.perplexity()
			print "Preplexity of Tweets =", self.atm_sampler.perplexity()

		n_dt = self.lda_sampler.wt_matrix()
		c_wt = self.atm_sampler.wt_matrix().T
		news_sum_dt = np.array([sum(n_dt[i]) for i in range(n_dt.shape[0])])
		tweets_sum_wt = np.array([sum(c_wt[i]) for i in range(c_wt.shape[0])])
		htm_wt = n_dt + c_wt
		htm_sum_wt = np.array([sum(htm_wt[i]) for i in range(htm_wt.shape[0])])

		"""get distributions"""
		news_dt_distribution = n_dt/news_sum_dt[:, np.newaxis]
		tweets_wt_distribution = c_wt/tweets_sum_wt[:, np.newaxis]
		htm_wt_distribution = htm_wt/htm_sum_wt[:, np.newaxis]

		print "KL from news to htm"
		KL_divergence(news_dt_distribution, htm_wt_distribution)
		print "KL from tweets to htm"
		KL_divergence(tweets_wt_distribution, htm_wt_distribution)
		print "KL from news to tweets"
		KL_divergence(news_dt_distribution, tweets_wt_distribution)
		print "KL from tweets to news"
		KL_divergence(tweets_wt_distribution, news_dt_distribution)

def KL_divergence(p, q):
	row, col = p.shape
	kl = []
	for i in range(row):
		kl_row = 0
		for j in range(col):
			kl_row += p[i][j] * math.log(p[i][j] / q[i][j])
		kl.append(kl_row)
	print kl

def main():
	import optparse
	import vocabulary
	parser = optparse.OptionParser()
	parser.add_option("--newsf", dest="newsfile", help="news corpus filename")
	parser.add_option("--tweetsf", dest="tweetsfile", help="tweets corpus filename")
	parser.add_option("-a", dest="authorfile", help="author filename")
	parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
	parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
	parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
	parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
	parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
	parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=False)
	parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
	parser.add_option("--seed", dest="seed", type="int", help="random seed")
	parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
	(options, args) = parser.parse_args()
	if not (options.newsfile or options.corpus): parser.error("need corpus news file(--newsf) or corpus range(-c)")
	if not options.tweetsfile: parser.error("need corpus tweets file(--tweetsf)")
	if not options.authorfile: parser.error("need author file(-a)")

	if options.newsfile:
		news_corpus = vocabulary.load_file(options.newsfile)
		news_len = len(news_corpus)
		print "Load News data from'" + options.newsfile + "'"
		print "\t", news_len, "News in total"
	else:
		news_corpus = vocabulary.load_corpus(options.corpus)
		if not news_corpus: parser.error("corpus range(-c) forms 'start:end'")
	if options.seed != None:
		np.random.seed(options.seed)

	print "Load Twitters data from '" + options.tweetsfile + "'"
	twitter_corpus = vocabulary.load_file(options.tweetsfile)
	twitter_len = len(twitter_corpus)
	print "\t", twitter_len, "Tweets in total"

	corpus = news_corpus + twitter_corpus
	
	voca = vocabulary.Vocabulary(options.stopwords)
	docs = [voca.doc_to_ids(doc) for doc in (corpus)]

	print "Initialize the authors set"
	num_authors, author_set = vocabulary.load_author(options.authorfile)
	print "\t", num_authors, "authors in total"

	if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

	corpus_collection = list(set([w for doc in docs for w in doc]))
	# Initialization
	print "Initialize the heterogenous topic model"
	htm = HTM(options.K, options.alpha, options.beta, docs, news_len, num_authors, author_set, voca)
	htm.gibbs_sampling(options.iteration)
	

if __name__ == "__main__":
	main()
