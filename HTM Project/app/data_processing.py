'''
# Script
python data_processing.py --newsf xxx
'''

import re
import json

def filter_tags(content):
	ans = re.sub(r'</?\w+[^>]*>', '', content)
	return ans.encode("utf-8")

def process_news(filename, output):
	story_filename = output + "_story.txt"
	other_filename = output + "_other.txt"
	story_output = open(story_filename, "a")
	other_output = open(other_filename, "a")
	content = open(filename).readlines()
	news_len = len(content)
	for line in content:
		info = json.loads(line)
		oid = info["_id"]["$oid"]
		time_stamp = info["timeStamp"]
		day, month, year = time_stamp.split(" ")[0].split("/")
		if month != "07" or year != "2015":
			continue
		main_story = filter_tags(info["mainStory"].replace("\n", ""))
		story_output.write(main_story + "\n")
		other_output.write(oid + "\t" + time_stamp + "\n")
	
	story_output.close()
	other_output.close()


def main():
	import optparse
	import vocabulary
	parser = optparse.OptionParser()
	parser.add_option("--newsf", dest="newsfile", help="news corpus filename")
	parser.add_option("--tweetsf", dest="tweetsfile", help="tweets corpus filename")
	parser.add_option("-o", dest="output", help="output filename")
	(options, args) = parser.parse_args()
	if not (options.newsfile or options.tweetsfile): parser.error("need corpus news file(--newsf) or corpus range(-c)")
	if not options.output: parser.error("need output filename(-o)")

	if options.newsfile:
		print "Process news", options.newsfile
		process_news(options.newsfile, options.output)
	if options.tweetsfile:
		print "Process tweets", options.tweetsfile


if __name__ == "__main__":
	main()
