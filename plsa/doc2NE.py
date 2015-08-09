'''
> DBPedia Parser
Use to get the entity from the server and parse it.
The Parser compute the probability of the entity based ont the entity graph

> Reference
https://github.com/dbpedia-spotlight/dbpedia-spotlight
'''

import xml.etree.ElementTree as ET
import requests
import sys
import os
import operator
import math
import urllib2

'''
Common Section
'''
def readFiles(document_path, mode):
	flag = 0
	for root, dirs, files in os.walk(document_path):
		for name in files:
			# This is for break point
			# if name == '553aceb1e4b08795cdc52b20':
			# 	flag = 1
			# if 0 == flag:
			# 	continue
			# ===
			file_name = root + "/" + name
			if 'DS_Store' in file_name:
				continue
			# Curl resource from DBPedia
			if 1 == mode:
				text_content = ""
				with open(file_name) as f:
					for line in f.readlines():
						text_content += line
				# curlDBPedia(root.replace("s_20_newsgroups", "r_20_newsgroups"), name, text_content)
				curlDBPedia(root.replace("1", "r_1"), name, text_content)

def write_response(file_path, file_name, text):
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	file_name = file_path+"/"+file_name
	print "Writing...", file_name
	output = open(file_name, "w")
	output.write(text.encode("utf-8"))
	output.close()

def process_text(text_content):
	PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*', '>', '<']
	CARRIAGE_RETURNS = ['\n', '\r\n', '\0']
	for punc in PUNCTUATION + CARRIAGE_RETURNS:
	# for punc in CARRIAGE_RETURNS:
		text_content = text_content.replace(punc, '').strip("'")
	return text_content

'''
Section 1
'''
def curlDBPedia(file_path, file_name, text_content):
	apikey = "3f07faf2bf9dc29f4a0d40072dfc09e6e3e2fbd9"
	text_content = process_text(text_content)

	DB_url = "http://access.alchemyapi.com/calls/text/TextGetRankedNamedEntities?apikey="+apikey+"&text="+text_content+"&outputMode=json"
	# payload = {"text": text_content, "confidence": "0.2", "support": "20"}
	# r = requests.get(DB_url, payload)
	print DB_url
	result = urllib2.urlopen(DB_url)
	write_response(file_path, file_name, result.read())

def main(argv):
	if len(argv) < 2:
		print "Usage: python doc2NE.py <option>"
		print "\t1. curl resource from DBPedia, store in r_*"
		return
	if argv[1] == '1':
		document_path = "./data/1/"
		readFiles(document_path, 1)

if __name__ == "__main__":
	main(sys.argv)
	# XMLParser('../data/research_data/r_alt.atheism/51122_s')
