#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This a pre-processing program for 20_newsgroups data. The program is extract content data from the original data set

> For 20_newsgroups
'''
import sys
import os

def extract_content(file_name):
	# MAC system file
	if "DS_Store" in file_name:
		return
	# if not "_s" in file_name:
		# return
	# Main section
	with open(file_name) as f:
		content = f.readlines()
		processed_content = ""
		has_colon = True
		for con in content:
			if (not ":" in con) or (has_colon == False):
				has_colon = False
				processed_content += con + " "
		processed_content = processed_content.lower()
		output_file = open(file_name+"_s", "w")
		output_file.write(processed_content)
		output_file.close()

def del_file_s(file_name):
	if not "_s" in file_name:
		os.remove(file_name)

def traverse_folder():
	for root, dirs, files in os.walk("/Users/wyatt/Documents/Code/Gla/Final/Sources/plsa/data/research_data/s_20_newsgroups/"):
		for name in files:
			file_name = root + "/" + name
			# Extract content and write into "*_s" file
			# extract_content(file_name)
			# print "Extract content from", file_name
			# Delete rubbish file inside
			del_file_s(file_name)
			print "Delete", file_name

def main():
	traverse_folder()

if __name__ ==  "__main__":
	main()
