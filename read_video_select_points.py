from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw

import numpy as np
from scipy.misc import *
import matplotlib.pyplot as plt
import cv2

cursor_pos = np.empty((0,2))

def loadTexture(im):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_LIGHTING)
    glClearColor(0,0,0,1)
    # glEnable(GL_DEPTH_TEST)
    ID = glGenTextures(1)   #generate a texture
    glBindTexture(GL_TEXTURE_2D, ID)    #make our texture ID current 2D texture
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    glTexImage2D(GL_TEXTURE_2D, 0, 3, im.shape[1], im.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, im) #copy texture into current texture ID
    return ID

def setupTexture(ID):
    glEnable(GL_TEXTURE_2D) #Configure texture rendering parameters
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glBindTexture(GL_TEXTURE_2D, ID)


def drawQuad():
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex3f(-1,1,-1)
    glTexCoord2f(0,1); glVertex3f(-1,-1,-1)
    glTexCoord2f(1,1); glVertex3f(1,-1,-1)
    glTexCoord2f(1,0); glVertex3f(1,1,-1)
    glEnd()

def drawPoint():
    global cursor_pos
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glPointSize(10)
    # glColor3f(1,1,1)
    glBegin(GL_POINTS)
    for i in range(cursor_pos.shape[0]):
        glColor4f(1.0,0.0,0.0,1.0); glVertex3f(0, 0, -1)
    glEnd()
    glPopAttrib()

def mouse_button_callback(window, button, action, mods):
    if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS):
        global cursor_pos
        cursor_pos = np.vstack((cursor_pos,glfw.get_cursor_pos(window)))

def main():
    imageName = 'IMG_0081.jpg'
    im = cv2.imread(imageName,cv2.IMREAD_COLOR)
    # cv2.imshow('im',im); cv2.waitKey(0)
    # im = imread('palace.jpg'); plt.imshow(im); plt.show()

    if not glfw.init():
        return
    window = glfw.create_window(640, 480*im.shape[0]/im.shape[1], "Hello World", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_mouse_button_callback(window, mouse_button_callback)

    # Make the window's context current
    glfw.make_context_current(window)

    frame = cv2.imread('IMG_0081.jpg')
    # cap = cv2.VideoCapture('palace.m4v')
    # if(not cap.isOpened()):
    #     raise Exception("Can't Open File")

    while not glfw.window_should_close(window):

        # ret, frame = cap.read()
        # if frame is not None:
        # Render scene geometry here, e.g. using pyOpenGL
        ID = loadTexture(frame)
        setupTexture(ID)

        # Draw or set projection matrix here
        # glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()
        # gluOrtho2D(0.0,640.0,0.0,480.0*im.shape[0]/im.shape[1])

        drawQuad()
        drawPoint()

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()
        # else:
        #     break

    glfw.terminate()

if __name__ == '__main__':
    try:
        main()
    except Exception as inst:
        print inst
