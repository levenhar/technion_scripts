import UserInterface as gui
import tkinter as tk
from PIL import ImageTk, Image


"""student_Name - Mor Levenhar and Tchelet Lev
Student_Id - 318301124, 206351611
Course Number - 014846
Course Name - Geo-spatial Databases
Home Work number 5"""


'''
database information:
user - postgres
password - mortchelet
host - 127.0.0.1
port - 5432
database Name - Big
'''

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('720x720+0+0')
    root.title('HW5 - Mor and Tchelet')
    my_image = ImageTk.PhotoImage(Image.open("background.jpg"))
    my_label = tk.Label(root, image=my_image)
    my_label.place(x=0, y=0, relwidth=1, relheight=1)
    main = gui.UI(root)
    main.ui()
    root.mainloop()