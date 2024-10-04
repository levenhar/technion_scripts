import matplotlib.pyplot as plt
import cv2
import numpy as np


def onclick(event):
    x=event.xdata
    y=event.ydata
    ax.scatter([event.xdata],[event.ydata])
    print(f"{x} \t {y}")

img = cv2.imread("IMG_3511.JPG")
imgG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots()
ax.imshow(imgG)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
