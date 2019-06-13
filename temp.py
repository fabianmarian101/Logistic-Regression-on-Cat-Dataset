# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:39:07 2019

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 01:43:45 2019

@author: fabian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

User_name=""


df=pd.read_csv(r"C:\personal\train.csv")


Train_orig=df.iloc[:,1:len(df.columns)].values

train=Train_orig.reshape(Train_orig.shape[0],28,28)

train_Y=df['label'].values

train_Y=train_Y.reshape(-1,1)

plt.imshow(train[3])


from scipy.misc import toimage
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

error_count=0
i=0

def action(data):
   global i
   global appl
   global entry
   global error_count
   
   orig_Y=train_Y[i,0]
   if data!=orig_Y:
       error_count=error_count+1
       
   i=i+1

   if i>=12:
       appl.destroy()
       error_rate=(error_count/11)*100
       tk.messagebox.showinfo("Information",User_name+" error _rate is "+str(error_rate)+"%")



   array1 = train[i]
   image1=Image.fromarray(array1.astype('uint8'))
   image1 = image1.resize((250, 250), Image.ANTIALIAS)
   appl.img1 =  ImageTk.PhotoImage(image1)


   #entry=tk.Label(appl,image=img1)
   entry.config(image=appl.img1)
   entry.image = appl.img1
   
   labelText.set(str(i)+"/"+str(train.shape[0]))
   
   entry.pack()
   appl.update_idletasks()
   print(error_count)




appl = tk.Toplevel()
appl.geometry('900x500')
array = train[i]
image=Image.fromarray(array.astype('uint8'))
image = image.resize((250, 250), Image.ANTIALIAS)
img =  ImageTk.PhotoImage(image)

entry=tk.Label(appl, image=img)
entry.pack()

labelText = tk.StringVar()
labelText.set("0/"+str(train.shape[0]))
text=tk.Label(appl,textvariable=labelText)
text.pack(side="left")

zero=tk.Button(appl,text="0",height=2,width=4,command=lambda: action(0))
zero.pack(side="left",padx=7,pady=7)

one=tk.Button(appl,text="1",height=2,width=4,command=lambda: action(1))
one.pack(side="left",padx=4,pady=4)

two=tk.Button(appl,text="2",height=2,width=4,command=lambda: action(2))
two.pack(side="left",padx=4,pady=4)

three=tk.Button(appl,text="3",height=2,width=4,command=lambda: action(3))
three.pack(side="left",padx=4,pady=4)

four=tk.Button(appl,text="4",height=2,width=4,command=lambda: action(4))
four.pack(side="left",padx=4,pady=4)

five=tk.Button(appl,text="5",height=2,width=4,command=lambda: action(5))
five.pack(side="left",padx=4,pady=4)

six=tk.Button(appl,text="6",height=2,width=4,command=lambda: action(6))
six.pack(side="left",padx=4,pady=4)

seven=tk.Button(appl,text="7",height=2,width=4,command=lambda: action(7))
seven.pack(side="left",padx=4,pady=4)

eight=tk.Button(appl,text="8",height=2,width=4,command=lambda: action(8))
eight.pack(side="left",padx=4,pady=4)

nine=tk.Button(appl,text="9",height=2,width=4,command=lambda: action(9))
nine.pack(side="left",padx=4,pady=4)

#button=tk.Button(appl,text="Next",height=5,width=18,command=lambda: action(0))
#button.pack(side="right")
User_name = tk.simpledialog.askstring(title="Name",prompt="Enter Your Name",parent=appl)
appl.mainloop()































