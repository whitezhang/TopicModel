import pymongo
import sys
import re
import plsa_ori
import os

data_path = "./data/gap"

def writeData(file_path, file_name, content):
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	file = open(file_path + "/" + file_name, "w")
	file.write(content.encode('utf-8'))
	file.close()

def cleanStory(content):
	content = re.sub(r'</?\w+[^>]*>','', content)
	return content.replace("Daily Record", "")

def samplingFromDB(day_gap):
	start_day = 1
	my_month = "04"
	conn = pymongo.Connection("localhost", 27017)
	db = conn.gms
	while start_day < 30:
		print "Day", start_day
		results = db.newsdata.find()
		for newsdata in results:
			oid = newsdata['_id']
			time_stamp = newsdata['timeStamp']
			title = newsdata['title']
			main_story = cleanStory(newsdata['mainStory'])
			print oid

			if title == "":
				continue
			day, month, year = time_stamp.split('/')
			day = int(day)
			if month == my_month and day >= start_day and day < start_day+day_gap:
				writeData(data_path + "/gap" + str(day_gap) + "/" + str(start_day), str(oid), main_story)
		start_day += day_gap

def findFolder(data_path):
	for root, dirs, files in os.walk(data_path):
		pass

def main(argv):
# Sampling procedure
	# day_gap_list = [1, 3, 5, 7, 9]
	# day_gap_list = [1]
	# for day_gap in day_gap_list:
	# 	print "Gap...", day_gap
	# 	samplingFromDB(day_gap)
# PLSA procedure
	for number_of_topics in range(3, 10):
		for root, dirs, files in os.walk(data_path):
			model_name = 'model'+str(number_of_topics)
			if '1' in root.split('/')[-1]:
				print root
				document_path = root + "/"
				model_path = document_path + model_name + '/'
				file_path_name = model_path + "file-path.txt"
				corpus_path = document_path + "corpus.txt"
				if not os.path.exists(model_path):
					os.makedirs(model_path)
				print "Document path:", document_path
				print "File path name:", file_path_name
				print "Model path:", model_path
				print "Corpus path", corpus_path
				plsa_ori.main(document_path, file_path_name, model_path, corpus_path, number_of_topics)
	
if __name__ == "__main__":
	main(sys.argv)