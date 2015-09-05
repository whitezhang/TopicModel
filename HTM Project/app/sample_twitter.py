import os

def main():
    ofile = "D:\\March\\output1000.json"
    output = open(ofile, "a")
    i = 0
    for root, dirs, files in os.walk("D:\\March\\07"):
        for dir in dirs:
            filepath = root + "\\" + dir
            for root1, dirs1, files2 in os.walk(filepath):
                for name in files2:
                    filename = root1 + "\\" + name
                    # Sampling from json file
                    print filename
                    fp = open(filename, "r")
                    for line in fp:
                        if len(line) < 3:
                            continue
                        i += 1
                        if i == 1000:
                            # print len(line),
                            output.write(line)
                            i = 0
                    fp.close()
    output.close()

if __name__ == '__main__':
    main()