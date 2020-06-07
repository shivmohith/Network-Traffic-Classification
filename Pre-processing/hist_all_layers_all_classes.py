import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm
import pandas as pd
from skimage.feature import hog
TRAIN_DIR = r"D:\ML mini project\main codes\train_L7"
training_data = []
hist = []
temp = [] #pixels of an image
temp2 = [] #multiple images(pixel values) in one array

def label(arr):
    if("BitTorrent" in arr):
        return 0
    elif("Facetime" in arr):
        return 1
    elif("FTP" in arr):
        return 2
    elif("Gmail" in arr):
        return 3
    elif("MySQL" in arr):
        return 4
    elif("Outlook" in arr):
        return 5
    elif("Skype" in arr):
        return 6
    elif("SMB" in arr):
        return 7
    elif("Weibo" in arr):
        return 8
    elif("WorldOfWarcraft" in arr):
        return 9
    elif("Cridex" in arr):
        return -1
    elif("Geodo" in arr):
        return -2
    elif("Htbot" in arr):
        return -3
    elif("Miuref" in arr):
        return -4
    elif("Neris" in arr):
        return -5
    elif("Nsis-ay" in arr):
        return -6
    elif("Shifu" in arr):
        return -7
    elif("Tinba" in arr):
        return -8
    elif("Virut" in arr):
        return -9
    elif("Zeus" in arr):
        return -10    

for img in tqdm(os.listdir(TRAIN_DIR)):
        ppc = 8
        path = os.path.join(TRAIN_DIR, img)
        t = label(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([img],[0],None,[32],[0,256]) #4th parameter can be varied to change the bin size
        
        for i in range(len(hist)):
             temp.append(hist[i][0])
        temp.append(t)
        temp2.append(temp)
        temp = []
        
        #np.savetxt("foo.csv", temp,fmt = "%d", delimiter=",")
#print(temp2)
        
shuffle(temp2)
temp3 = pd.DataFrame(temp2) #converting into dataframe
print(temp3.shape)
temp3.to_csv('dataset_all_layers_bin_size_8.csv', index=False)
            
