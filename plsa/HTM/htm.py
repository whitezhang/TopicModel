#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy as np
import os

model = "./model/"

class LDA:
	def __init__(self, K, alpha, beta, docs, V, smartinit=True):
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

	def worddist(self):
		"""get topic-word distribution"""
		return self.n_z_t / self.n_z[:, np.newaxis]

	def docdist(self):
		"""get document-topic distribution"""
		os.remove(model + "news_pzd.txt")
		pzdout = open(model + "news_pzd.txt", "a")
		n_m = np.zeros(len(self.docs))
		for i in range(len(self.n_m_z)):
			n_m[i] += sum(self.n_m_z[i])

		probs = self.n_m_z/n_m[:, np.newaxis]
		for doc in probs:
			for prob in doc:
				pzdout.write(str(prob)+" ")
			pzdout.write("\n")
		pzdout.close()
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
	def __init__(self, vocab, K, A, docList, authorList, alpha=0.1, eta=0.01):
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

		self._vocab = vocab
		self._W = len(vocab)
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

	def sampling_topics(self, max_iter):
		for iter in xrange(0, max_iter):
			print "Iteration", iter
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
		# print self.theta
		# print self.phi
		os.remove(model + "tweets_pzd.txt")
		pzd_out = open(model + "tweets_pzd.txt", "a")
		self.pzd = np.ones([self._D, self._K])
		for di, doc in enumerate(self._docList):
			# Get the author set of document
			ad = self._authorList[di]
			for k in range(self._K):
				for a in ad:
					self.pzd[di, k] *= self.theta[a, k]
				for word in doc:
					self.pzd[di, k] *= self.phi[word, k]
			self.pzd[di] /= sum(self.pzd[di])
			for values in self.pzd[di]:
				pzd_out.write(str(values) + " ")
			pzd_out.write("\n")
		pzd_out.close()
		# print self.pzd
		# print self.getLikelihood()return self.n_z_t / self.n_z[:, np.newaxis]
		return self.phi
	
	def getLikelihood(self):
		likelihood = 1.0
		for doc in self._docList:
			tmp = 0
			print "\t", doc
			for wi in doc:
				for ad in self._authorList:
					for t in xrange(0, self._K):
						for ai in ad:
							tmp += self.c_wt[wi, t] * self.c_at[ai, t]
					tmp = float(tmp)/len(ad)
				likelihood *= tmp
		return likelihood


# HTM Learning
def htm_learning(lda, atm, iteration, voca):
	prep1 = lda_learning(lda, iteration, voca)
	prep2 = atm_learning(atm, iteration, voca)



def atm_learning(atm, iteration, voca):
	atm.sampling_topics(iteration)
	pre_likelihood = atm.getLikelihood()
	# print pre_likelihood

# lda Learning
def lda_learning(lda, iteration, voca):
	pre_perp = lda.perplexity()
	print ("initial perplexity=%f" % pre_perp)
	for i in range(iteration):
		lda.inference()
		perp = lda.perplexity()
		print ("-%d p=%f" % (i + 1, perp))
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

# Results
def output_word_topic_dist(lda, voca):
	zcount = np.zeros(lda.K, dtype=int)
	wordcount = [dict() for k in range(lda.K)]
	for xlist, zlist in zip(lda.docs, lda.z_m_n):
		for x, z in zip(xlist, zlist):
			zcount[z] += 1
			if x in wordcount[z]:
				wordcount[z][x] += 1
			else:
				wordcount[z][x] = 1

	phi = lda.worddist()
	theta = lda.docdist()
	for k in range(lda.K):
		print ("\n-- topic: %d (%d words)" % (k, zcount[k]))
		# for w in np.argsort(-phi[k])[:20]:
		for w in np.argsort(-phi[k]):
			# print ("%s: %f (%d)" % (voca[w], phi[k,w], wordcount[k].get(w,0)))
			pass

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
	else:
		news_corpus = vocabulary.load_corpus(options.corpus)
		if not news_corpus: parser.error("corpus range(-c) forms 'start:end'")
	if options.seed != None:
		np.random.seed(options.seed)

	twitter_corpus = vocabulary.load_file(options.tweetsfile)
	twitter_len = len(twitter_corpus)
	corpus = news_corpus + twitter_corpus
	
	voca = vocabulary.Vocabulary(options.stopwords)
	docs = [voca.doc_to_ids(doc) for doc in (corpus)]

	num_authors, author_set = vocabulary.load_author(options.authorfile)

	if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

	corpus_collection = list(set([w for doc in docs for w in doc]))
	# Initialization
	lda = LDA(options.K, options.alpha, options.beta, docs[:news_len], voca.size(), options.smartinit)
	atm = ATM(corpus_collection, options.K, num_authors, docs[news_len:], author_set)
	
	# LDA
	print "LDA training..."
	print "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta)
	lda_learning(lda, options.iteration, voca)
	lda_beta = lda.worddist()
	lda.docdist()
	
	# ATM
	print "Author Topic Model training..."
	print "words=%d, twitters=%d, authors=%d" % (len(corpus_collection), len(docs[news_len:]), num_authors)
	atm_beta = atm.sampling_topics(options.iteration)	# word-topic distribution

	atm_beta = atm_beta.T

	# Write corpus into file
	os.remove(model + "news_corpus.txt")
	ncout = open(model + "news_corpus.txt", "a")
	for doc in news_corpus:
		for word in doc:
			ncout.write(word + " ")
	ncout.close()

	os.remove(model + "tweets_corpus.txt")
	tcout = open(model + "tweets_corpus.txt", "a")
	for doc in twitter_corpus:
		for word in doc:
			tcout.write(word + " ")
	tcout.close()

	os.remove(model + "news_beta.txt")
	os.remove(model + "tweets_beta.txt")
	os.remove(model + "htm_beta.txt")
	ldaout = open(model + "news_beta.txt", "a")
	atmout = open(model + "tweets_beta.txt", "a")
	htmout = open(model + "htm_beta.txt", "a")
	htm_beta = np.zeros([options.K, len(corpus_collection)])
	for i in range(options.K):
		for j in range(len(corpus_collection)):
			htm_beta[i, j] = lda_beta[i, j] * atm_beta[i, j]
			ldaout.write(str(lda_beta[i, j]) + " ")
			atmout.write(str(atm_beta[i, j]) + " ")
			htmout.write(str(htm_beta[i, j]) + " ")
		ldaout.write("\n")
		atmout.write("\n")
		htmout.write("\n")
	ldaout.close()
	atmout.close()
	htmout.close()
	
	# ATM
	# atm = (voca, options.K, N_AUTHORS, twitter_docs, AU_LIST)
	#import cProfile
	#cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
	

if __name__ == "__main__":
	main()
