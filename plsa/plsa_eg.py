def read_file_path():
	file_name = "file-path.txt"
	return open(file_name).readlines()

def main(argv):
	print "PLSA+EG approach..."
	file_name_set = read_file_path()
	for name in file_name_set:
		if "DS_Store" in name:
			continue
		


if __name__ == "__main__":
	main(sys.argv)