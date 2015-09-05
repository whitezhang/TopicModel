import json
from datetime import datetime
import os
import re

flag = 0
count = 0

fin = open("D:\\March\\output20.json", "r")
fout_path = "D:\\March\\glasgow-atm-content-14-4.txt"
fout_path2 = "D:\\March\\glasgow-atm-others-14-4.txt"

for line in fin:
	try:
		info = json.loads(line)
		if 'user' not in info or 'text' not in info or 'id_str' not in info:
			continue
		if 'id_str' not in info['user']:
			continue
		count += 1
		print count

		u_set = []
		if "entities" in info:
			if "user_mentions" in info["entities"]:
				users = info["entities"]["user_mentions"]
				for u in users:
					u_set.append(u["id_str"])

		filename = open(fout_path, 'a')
		filename.write(info['text'].encode('utf-8').replace("\n", "")+"\n")

		filename2 = open(fout_path2, 'a')
		filename2.write(info['user']['id_str'])
		for u in u_set:
			filename2.write(" "+u)
		filename2.write("\t")
		filename2.write(info['created_at']+"\t")
		filename2.write(info['id_str']+"\n")
		filename.close()
	except ValueError, e:
		continue
	except TypeError, e:
		continue
fin.close()