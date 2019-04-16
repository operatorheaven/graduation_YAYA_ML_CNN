import os, sys 
for num in range(53, 9566):
    os.system("mv "+str(num)+".jpg "+str(num-52)+".jpg")
