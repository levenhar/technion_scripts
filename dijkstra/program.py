from UserInterface import UI
import tkinter
from PIL import ImageTk, Image

"""student_Name - Mor Levenhar and Tchelet Lev
Student_Id - 318301124, 206351611
Course Number - 014845
Course Name - introduction to computer mapping
Home Work number 4"""

if __name__ == '__main__':
    root = tkinter.Tk()
    root.geometry('680x680+0+0')
    my_image = ImageTk.PhotoImage(Image.open("screen.png"))
    my_label = tkinter.Label(root, image=my_image)
    my_label.place(x=0, y=0, relwidth=1, relheight=1)
    main = UI(root)
    main.ui()
    root.mainloop()
