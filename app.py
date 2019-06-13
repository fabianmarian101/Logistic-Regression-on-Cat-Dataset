# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:32:19 2019

@author: fabian
"""
import tkinter
from scipy.misc import toimage
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
 
User_name=""
error_count=0
i=0
labelText = tk.StringVar()
appl = tk.Toplevel()
entry=tk.Label(appl)





def action(data):
   global i
   global appl
   global entry
   global error_count
   global entry
   orig_Y=train_Y[i,0]
   if data!=orig_Y:
       error_count=error_count+1
   if i>=train.shape[0]:
       pass

   i=i+1

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





def page_two_content():
    global appl
    global labelText
    
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


    appl.mainloop()


















def page_two():
    global User_name
    data=entry1.get()
    if data=="" or data=="Please enter your name":
        entry1.delete(0,"end")
        entry1.insert(0,"Please enter your name")
    else:
        User_name=data
        window.destroy()
        print(data)
        page_two_content()
        #print(d)

"""---------------------------------------------------------------------------------"""
df=pd.read_csv(r"C:\personal\train.csv")


Train_orig=df.iloc[:,1:len(df.columns)].values

train=Train_orig.reshape(Train_orig.shape[0],28,28)

train_Y=df['label'].values

train_Y=train_Y.reshape(-1,1)

"""-----------------------------------------------------------------------------------"""

window = tkinter.Tk()
window.title("Human error in Digit Recogniser")

window.geometry("300x100")

tkinter.Label(window, text = "Enter Name").pack(side="left") # this is placed in 0 0
# 'Entry' is used to display the input-field
entry1=tkinter.Entry(window)
entry1.pack(side="left") # this is placed in 0 1
#T = tk.Text(root, height=2, width=30)
tkinter.Button(window,text="Next",command=page_two).pack(side="left")

window.mainloop()





m1 =tkinter.PanedWindow()
m1.pack(fill="both", expand=1)

left = tkinter.Label(m1, text="left pane")
m1.add(left)

m2 = tkinter.PanedWindow( orient="vertical")
#m1.add(m2)
m2.pack(fill="both", expand=1)

top = tkinter.Label(m2, text="top pane")
m2.add(top)

bottom = tkinter.Label(m2, text="bottom pane")
m2.add(bottom)

tkinter.mainloop()








