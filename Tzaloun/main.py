import GUI as gui
import tkinter as tk


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('680x680+0+0')
    root.title('HW4 - mor and tchelet')
    main = gui.UI(root)
    main.ui()
    root.mainloop()


