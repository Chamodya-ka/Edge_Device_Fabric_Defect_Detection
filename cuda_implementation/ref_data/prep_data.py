import csv

def getLabel(c,file_name):
    with open("ref_csv/"+file_name+".txt") as text:
        txt_reader = csv.reader(text,delimiter=" ")
        for i in txt_reader:
            #print(i)
            row = [float(j) for j in i]
            x = row[1] *2048.0
            y = row[2] *2048.0
            x2 = x + row[3] *2048.0
            y2 = y + row[4] *2048.0
            dim_x = c%64 * 32
            dim_y = c//64 * 32
            if (dim_x>=x and (x+x2)>=dim_x and dim_y>=y and dim_y<=(y+y2)):
                return True
    return False

            
        

prev_img = "0"
c=0
with open("data.csv","w") as out:
    writer = csv.writer(out)
    with open("ref_csv/ref_data2.csv") as file:
        reader = csv.reader(file,delimiter= ',')
        
        for r in reader:
            file_name = r[0].split('/')[-1][0:2]
            #print(file_name)
            if (prev_img!=file_name):
                print(c)
                print("From ")
                print(prev_img)
                print("To ")
                print(file_name)

                c=0

            
            if getLabel(c,file_name):
                writer.writerow(r[1:]+["1"])
            else:
                writer.writerow(r[1:]+["0"])
            prev_img = file_name
            c+=1

            
            
