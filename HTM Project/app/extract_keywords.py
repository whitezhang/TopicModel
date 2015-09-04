# This script is used to extract top keywords from the collection
# The input file contains corpus file and probbaility file

def read_corpus(filename):
	lines = open(filename).readlines()
	corpus = []
	for line in lines:
		words = lines.split(" ")
		for w in words:
			corpus.append(w)
	return corpus

def read_probs(probname):
	lines = open(probname).readlines()
	for line in lines:
		prob = [float(p) for p in lines.split("")]
		

def main():
	import optparse
	import vocabulary
	parser = optparse.OptionParser()
	parser.add_option("-f", dest="filename", help="corpus filename")
	parser.add_option("-p", dest="probname", help="probability filename")
	(options, args) = parser.parse_args()

	if not (options.filename or options.probname): parser.error("need corpus file(-f) or probability file(-p")

	corpus = read_corpus(options.filename)
	probs = read_probs(optiosn.probname)


if __name__ == "__main__":
	main()