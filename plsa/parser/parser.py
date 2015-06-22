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

'''
Common Section
'''
def readFiles(document_path, mode):
	flag = 0
	for root, dirs, files in os.walk(document_path):
		for name in files:
			# This is for break point
			# if name == '82766_s':
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
				curlDBPedia(root.replace("s_", "r_"), name, text_content)
			# Parse the xml
			elif 2 == mode:
				XMLParser(file_name)

def write_response(file_path, file_name, text):
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	file_name = file_path+"/"+file_name
	print "Writing...", file_name
	output = open(file_name, "w")
	output.write(text)
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
	DB_url = "http://spotlight.dbpedia.org/rest/annotate"
	text_content = process_text(text_content)
	
	payload = {"text": text_content, "confidence": "0.2", "support": "20"}
	r = requests.get(DB_url, payload)
	write_response(file_path, file_name, r.text)

'''
Section 2
'''
# Compute p(ej|e) and get the largest one
def getProbE(entityURI):
	print "Computing..."
	prob1 = 0
	probDict2 = {}
	mapping_base_file_path = "/Users/wyatt/Documents/Code/Gla/Final/DB/mapping-based.nt"  # File name of entity graph
	# To remove 'entityURI_xxx'
	entityURI += ">"
	with open(mapping_base_file_path) as f:
		print 'The entity now is', entityURI
		for line in f:
			info = line.split(" ")
			if entityURI in info[0]:
				prob1 += 1
				if info[2] in probDict2:
					probDict2[info[2]] += 1	
				else:
					probDict2[info[2]] = 1
	f.close()
	# Find the largest number
	if bool(probDict2):
		maxProb = max(probDict2.iteritems(), key=operator.itemgetter(1))
		return maxProb[0], maxProb[1]/float(prob1)
	else:
		return entityURI, -1

# Get p(ej|e) based on the entity file. This one is faster
def getProbE2(entityURI):
	entity_map = open("entity_map.txt").readlines()
	for entity_item in entity_map:
		info = entity_item.split(":")
		if entityURI == info[0]:
			return entityURI, int(info[1])
	return -1

# Compute p(d|e)
def getProbD_E(file_path, entityURI):
	num_doc = 0
	for root, dirs, files in os.walk(file_path):
		for name in files:
			file_name = root+"/"+name
			with open(file_name) as f:
				for line in f:
					if entityURI in line:
						num_doc += 1
						break
	if num_doc == 0:
		return 0
	return 1/float(num_doc)

# For a specific file <xml_file_name>, make it convergent
def XMLParser(xml_file_name):
# xml_file_name = '../data/research_data/r_20_newsgroups/xxx/name' This is xml_Resource file
	resource_file_path = '../data/research_data/r_20_newsgroups/'
# Data Setting
	file_path_list = open("../file-path.txt", "r").readlines()
	p_z_d_list = open("../model/p_z_d.txt", "r").readlines()
	output_file = open("new_topic.txt", "a")
	output_file.write(xml_file_name+" ")
	p_d_e_dict = {}
	# Main Setcion
	print "Now xml file is", xml_file_name
	tree = ET.parse(xml_file_name)
	root = tree.getroot()
	for resource in root:
		for child_resource in resource:
			attrib = child_resource.attrib
			entityURI = attrib['URI']
			# Get p(ej|e), (1) or (2)
			# prob = getProbE(entityURI)
			# prob = getProbE2(entityURI)
			# p_Ej_E = prob[1] if prob[1] != -1 else 1
			# Compute p(di|ej)
			if not entityURI in p_d_e_dict:
				p_d_e_dict[entityURI] = getProbD_E(resource_file_path, entityURI)
		# Convergent procedure
		file_index = file_path_list.index(xml_file_name[1:].replace("r_", "s_")+"\n")
		# file_index = file_path_list.index(xml_file_name[1:].replace("r_20_newsgroups", "s_20_newsgroups")+"\n")

# Doc #2: 0.0394462383593 0.152423279112 0.0281086723938 0.0732624674083 0.0638212706
#  0.120287585089 0.00898699441055 0.0056926216879 0.00658182367856 0.0197639962998 
#  0.0737626612581 0.00718354943866 0.0964698172343 0.00248216013013 0.0383827150555 
#  0.0679034859763 0.0521520271707 0.0429636119073 0.0392873982509 0.0610376245382

		lamb = 0.8
		p_z_d = p_z_d_list[file_index].split(" ")
		for i in range(2,len(p_z_d)):
			sum_prob = 0
			# Use log
			if float(p_z_d[i]) == 0:
				continue
			print p_z_d[i]
			p_z_d[i] = math.log(float(p_z_d[i]))
			for iteration in range(100):
				for key in p_d_e_dict:
					if p_d_e_dict[key] == 0:
						continue
					sum_prob += p_z_d[i] - math.log(float(p_d_e_dict[key]))
				p_z_d[i] = lamb*p_z_d[i]+(1-lamb)*sum_prob
			output_file.write(str(p_z_d[i])+" ")
		output_file.write("\n")
	output_file.close()


def main(argv):
	if len(argv) < 2:
		print "Usage: python parser.py <option>"
		print "\t1. curl resource from DBPedia, store in r_*"
		print "\t2. parse the xml"
		return
	if argv[1] == '1':
		document_path = "../data/research_data/s_20_newsgroups/"
		readFiles(document_path, 1)
	elif argv[1] == '2':
		document_path = "../data/research_data/r_20_newsgroups/"
		readFiles(document_path, 2)

if __name__ == "__main__":
	main(sys.argv)
	# XMLParser('../data/research_data/r_alt.atheism/51122_s')
