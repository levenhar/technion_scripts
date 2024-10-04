import UserInterface as gui
import tkinter as tk
from PIL import ImageTk, Image


"""student_Name - Mor Levenhar
Student_Id - 318301124
Course Number - 014856
Home Work number 2"""

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('720x720+0+0')
    root.title('HW2 - Mor Levenhar')
    my_image = ImageTk.PhotoImage(Image.open("background.jpg"))
    my_label = tk.Label(root, image=my_image)
    my_label.place(x=0, y=0, relwidth=1, relheight=1)
    main = gui.UI(root)
    main.ui()
    root.mainloop()