# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:32:19 2019

@author: fabian
"""
import tkinter

User_name=""

def page_two():
    data=entry.get()
    if data=="" or data=="Please enter your name":
        entry.delete(0,"end")
        entry.insert(0,"Please enter your name")
    else:
        User_name=data
        window.destroy()
        print(data)
        #print(d)




window = tkinter.Tk()
window.title("Human error in Digit Recogniser")

window.geometry("300x100")

tkinter.Label(window, text = "Enter Name").pack(side="left") # this is placed in 0 0
# 'Entry' is used to display the input-field
entry=tkinter.Entry(window)
entry.pack(side="left") # this is placed in 0 1
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








