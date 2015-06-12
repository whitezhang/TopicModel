#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This a pre-processing program for 20_newsgroups data. The program is extract content data from the original data set
'''
import sys
import os

def extract_content(file_name):
	# MAC system file
	if "DS_Store" in file_name:
		return
	if not "_s" in file_name:
		return
	# Main section
	with open(file_name) as f:
		content = f.readlines()
		processed_content = ""
		has_colon = True
		for con in content:
			if (not ":" in con) or (has_colon == False):
				has_colon = False
				processed_content += con + " "
		output_file = open(file_name+"_s", "w")
		output_file.write(processed_content)
		output_file.close()

def del_file_s(file_name):
	if "_s_s" in file_name:
		os.remove(file_name)

def traverse_folder():
	for root, dirs, files in os.walk("./"):
		for name in files:
			file_name = root + "/" + name
			# Extract content and write into "*_s" file
			extract_content(file_name)
			print "Extract content from", file_name
			# Delete rubbish file inside
			# del_file_s(file_name)
			# print "Delete", file_name
			

def main():
	traverse_folder()

if __name__ ==  "__main__":
	main()