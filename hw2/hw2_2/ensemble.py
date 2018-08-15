import sys
infile1 = open('./coco_output1.txt', 'r')
infile2 = open('./coco_output2.txt', 'r')
outfile = open(sys.argv[1], 'w')

for line1, line2 in zip(infile1, infile2):
    line1 = line1.strip()
    line2 = line2.strip()

    if len(line1) > 7: print(line1, file=outfile)
    else: print(line2, file=outfile)

infile1.close()
infile2.close()
outfile.close()
