
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

mouse_x = 0
mouse_y = 0
inaxes = False

# Create 28x28 black canvas
canvas = np.zeros((28, 28))

plt.imshow(canvas, cmap='gray')

def on_move(event):
    if event.inaxes:
        mouse_x = event.x
        mouse_y = event.y
        inaxes = True
    else:
        inaxes = False



def on_click(event):    
    if event.button is MouseButton.LEFT:
        if inaxes == True:
            print("works")



binding_id = plt.connect('motion_notify_event', on_move)
plt.connect('button_press_event', on_click)
plt.show()
