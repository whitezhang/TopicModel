def processing_news(foldername):
	import os
	target = 14
	fout = open("htm-news-" + str(target) + ".txt", "a")
	mapout = open("htm-id-" + str(target) + ".txt", "a")

	for root, dirs, files in os.walk(foldername):
		for name in files:
			filename = root + "/" + name
			content = open(filename).readlines()
			day, month, year = [int(x) for x in content[0].split(" ")]
			text = content[1:]
			if day == target:
				for t in text:
					fout.write(t.replace("\n", ""))
				fout.write("\n")
				mapout.write(name+"\n")
	fout.close()



def main():
	import optparse
	parser = optparse.OptionParser()
	parser.add_option("-c", dest="choice", help="news or tweets")
	parser.add_option("-f", dest="foldername", help="corpus folder")
	(options, args) = parser.parse_args()

	if not (options.choice or options.foldername):
		parser.error("need a foldername(-f)")

	if options.choice == "n":
		processing_news(options.foldername)
	

if __name__ == "__main__":
	main()