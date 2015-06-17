'''
> DBPedia Parser
Use to get the entity from the server and parse it.

> Reference
https://github.com/dbpedia-spotlight/dbpedia-spotlight
'''

import xml.etree.ElementTree as ET
import requests
import sys
import os

def write_response(file_name, text):
	print "Writing...", file_name
	output = open("../data/research_data/r_alt.atheism/"+file_name, "w")
	output.write(text)
	output.close()

def process_text(text_content):
	PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*', '>', '<']
	CARRIAGE_RETURNS = ['\n', '\r\n']
	# for punc in PUNCTUATION + CARRIAGE_RETURNS:
	for punc in CARRIAGE_RETURNS:
		text_content = text_content.replace(punc, '').strip("'")
	return text_content

def curlDBPedia(file_name, text_content):
	DB_url = "http://spotlight.dbpedia.org/rest/annotate"
	text_content = process_text(text_content)
	payload = {"text": text_content, "confidence": "0.2", "support": "20"}
	r = requests.get(DB_url, payload)
	write_response(file_name, r.text)

def readFiles(document_path, mode):
	flag = 0
	for root, dirs, files in os.walk(document_path):
		for name in files:
			# This is for break point
			if name == '54165_s':
				flag = 1
			if 0 == flag:
				continue
			# ===
			file_name = root + "/" + name
			# Curl resource from DBPedia
			if 1 == mode:
				text_content = ""
				with open(file_name) as f:
					for line in f.readlines():
						text_content += line
				curlDBPedia(name, text_content)
			# Parse the xml
			elif 2 == mode:
				XMLParser(file_name)

def XMLParser(xml_file_name):
	tree = ET.parse(xml_file_name)
	root = tree.getroot()
	for resource in root:
		for child_resource in resource:
			attrib = child_resource.attrib
			entityURI = attrib['URI']
			

def main(argv):
	if len(argv) < 2:
		print "Usage: python parser.py <option>"
		print "\t1. curl resource from DBPedia, store in r_*"
		print "\t2. parse the xml"
		return
	if argv[1] == '1':
		document_path = "../data/research_data/s_alt.atheism/"
		readFiles(document_path, 1)
	elif argv[1] == '2':
		document_path = "../data/research_data/r_alt.atheism/"
		readFiles(document_path, 2)

if __name__ == "__main__":
	# main(sys.argv)
	XMLParser('../data/research_data/r_alt.atheism/51122_s')
