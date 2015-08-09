import os
import shutil

source_path = "../20_newsgroups/"
output_path = "./s1_20_newsgroups/"

num = 60

for root, dirs, files in os.walk(source_path):
  i = 0
  for name in files:
    i += 1
    source_name = root+"/"+name
    file_path = root.replace(source_path, output_path)
    if not os.path.exists(file_path):
      os.makedirs(file_path)
    file_name = file_path+"/"+name
    shutil.copy(source_name, file_name)
    if i == 50:
      break
