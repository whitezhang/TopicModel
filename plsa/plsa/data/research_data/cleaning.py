import os

source_path = "./r1_20_newsgroups/"

for root, dirs, files in os.walk(source_path):
  for name in files:
    file_name = root+"/"+name
    content = open(file_name).readlines()
    if "<title>414 Request-URI Too Large</title>\n" in content:
      print "Deleting...",
      print file_name
      os.remove(file_name)
      os.remove(file_name.replace("r1_", "s1_"))
