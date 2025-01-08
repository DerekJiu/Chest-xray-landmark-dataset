
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import pathlib
import re

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def draw_organ(ax, array, color = 'b'):
    N = array.shape[0]
    for i in range(0, N):
        x, y = array[i,:]
        circ = plt.Circle((x, y), radius=3, color=color, fill = True)
        ax.add_patch(circ)
    return

def draw_lines(ax, array, color = 'b'):
    N = array.shape[0]
    for i in range(0, N):
        x1, y1 = array[i-1,:]
        x2, y2 = array[i,:]
        ax.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1)
    return

def drawOrgans(RL, LL, H = None, RCLA = None, LCLA = None, img =  None):

    fig, ax = plt.subplots()
    
    if img is not None:
        plt.imshow(img, cmap='gray')
    else:
        img = np.zeros([1024, 1024])
        plt.imshow(img)
    
    plt.axis('off')
    
    draw_lines(ax, RL, 'r')
    draw_lines(ax, LL, 'g')
    
    draw_organ(ax, RL, 'r')
    draw_organ(ax, LL, 'g')
    
    if H is not None:
        draw_lines(ax, H, 'y')
        draw_organ(ax, H, 'y')

    if RCLA is not None:
        draw_lines(ax, RCLA, 'purple')
        draw_organ(ax, RCLA, 'purple')
    
    if RCLA is not None:
        draw_lines(ax, LCLA, 'purple')
        draw_organ(ax, LCLA, 'purple')

    return
data_root = pathlib.Path("Images/")
all_files = list(data_root.glob('*.png'))
all_files = [str(path) for path in all_files]
all_files.sort(key = natural_key)

img1 = all_files[0]
RL = img1.replace('Images','landmarks/RL').replace('.png','.npy')
LL = img1.replace('Images','landmarks/LL').replace('.png','.npy')
H = img1.replace('Images','landmarks/H').replace('.png','.npy')

img = cv2.imread(img1,0)
RL = np.load(RL)
LL = np.load(LL)
H = np.load(H)

drawOrgans(RL,LL,H,img=img)

def getDenseMask(RL, LL, H = None, RCLA = None, LCLA = None, imagesize = 1024):
    img = np.zeros([1024,1024])
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 2, -1)
    
    if H is not None:
        H = H.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [H], -1, 3, -1)

    if RCLA is not None:
        RCLA = RCLA.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [RCLA], -1, 4, -1)
    
    if LCLA is not None:      
        LCLA = LCLA.reshape(-1, 1, 2).astype('int')  
        img = cv2.drawContours(img, [LCLA], -1, 5, -1)

    return img

plt.figure()
aux = getDenseMask(RL,LL,H)
plt.imshow(aux)
