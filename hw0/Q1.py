import sys

with open(sys.argv[1]) as in_file:
    data = in_file.read()

word_dict = {}
for word in data.split():
    if word in word_dict:
        word_dict[word]+=1
    else:
        word_dict[word]=1

with open("Q1.txt", "w") as out_file:
    idx = 0
    for word in word_dict:
        out_file.write("%s %d %d\n" %(word, idx, word_dict[word]) )
        idx += 1
