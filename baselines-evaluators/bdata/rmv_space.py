import os

f=open("lyrics.txt","r")
o=open("nslyrics.txt","w")

content = f.readlines()

print(len(content))

content = [x.strip() for x in content] 

for line in content:
	if (len(line) > 3):
		line = line + '\n'
		o.write(line)

f.close()
o.close()
