import sys
import re

filename=sys.argv[1]
outfile=sys.argv[2]
host=r"172.17.0.(\d)+\b" #change it to the host ip address for the docker container
round_base=512

lines = []
with open(filename, "r") as f:
    lines = f.readlines()

id = []
source = []
length = []
for line in lines:
    ar = line.split(" ")
    i = 0
    s = ""
    l = 0
    try:
        i = int(ar[0])
        s = ar[1]
        l = int(ar[2])
    except ValueError:
        continue
    id.append(i)
    source.append(s)
    length.append(l)

sequence = []   
for i, idx in enumerate(id):
    sign = 101
    if re.match(host, source[i]):
        #print "Match found: %s" %source[i]
	sign = -101
    #rounded_len = float(length[i])*sign
    rounded_len = max(1, int(round(float(length[i])/round_base))) 
    print "%s %s" %(length[i], rounded_len)
    seq = [sign for i in range(rounded_len)]
    sequence.extend(seq)
    #sequence.append(rounded_len)

#print sequence 
seq_out = "\n".join(map(str, sequence))
with open(outfile, "w") as f:
    f.write(seq_out)
        
