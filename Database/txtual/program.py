import UserInterface as gui
import tkinter as tk


"""student_Name - Mor Levenhar and Tchelet Lev
Student_Id - 318301124, 206351611
Course Number - 014846
Course Name - Geo-spatial Databases
Home Work number 4"""


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
    root.geometry('1250x680+0+0')
    root.title('HW4 - mor and tchelet')
    main = gui.UI(root)
    main.ui()
    root.mainloop()
