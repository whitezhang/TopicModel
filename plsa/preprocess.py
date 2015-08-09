text_p = open("file-path.txt").readlines()
text_pzd = open("./model/p_z_d.txt").readlines()
output = open("n_pzd.txt", "w")

label = 1
tag = "alt.atheism"

for i in range(len(text_pzd)):
	tag_ = text_p[i].split("/")[-2]
	if tag == tag_:
		output.write(text_pzd[i].replace("\n", "").split(": ")[1]+" "+str(label)+"\n")
	else:
		label += 1
		tag = tag_
		output.write(text_pzd[i].replace("\n", "").split(": ")[1]+" "+str(label)+"\n")

output.close()