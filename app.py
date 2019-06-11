# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 01:43:45 2019

@author: fabian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\AI\train.csv")


Train_orig=df.iloc[:,1:len(df.columns)].values

train=Train_orig.reshape(Train_orig.shape[0],28,28)

plt.imshow(train[3])


from scipy.misc import toimage
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk


i=10

def action():
   global i
   global appl
   global entry
   print(i)

   i=i+1

   array1 = train[i]
   image1=Image.fromarray(array1.astype('uint8'))
   image1 = image1.resize((250, 250), Image.ANTIALIAS)
   appl.img1 =  ImageTk.PhotoImage(image1)


   #entry=tk.Label(appl,image=img1)
   entry.config(image=appl.img1)
   entry.image = appl.img1
   
   entry.pack()
   appl.update_idletasks()




appl = tk.Tk()
appl.geometry("1000x500")
array = train[i]
image=Image.fromarray(array.astype('uint8'))
image = image.resize((250, 250), Image.ANTIALIAS)
img =  ImageTk.PhotoImage(image)

entry=tk.Label(appl, image=img)
entry.pack()

button=tk.Button(appl,text="Next",command=action)
button.pack(side="left")

appl.mainloop()