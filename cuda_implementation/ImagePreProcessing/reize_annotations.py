import os

for filename in os.listdir("../ref_data/ref_csv"):
    if filename.endswith(".txt"):
        s = os.path.join(filename)
        print(s)
        f = os.open(filename,os.O_RDWR)


        continue
    else:
        continue
