from tkinter import *

window = Tk()

canvas = Canvas(window, width=800, height=400)

canvas.pack()

myid = canvas.create_oval(0,0,10,10,fill="red")

id = canvas.coords(myid)
print(id)