import cv2
import layoutparser as lp
import matplotlib.pyplot as plt
import numpy as np

COLOR_MAP = {
    'Text':   'red',
    'Title':  'blue',
    'List':   'green',
    'Table':  'purple',
    'Figure': 'pink',
}

def draw_layout(img, layout):
    viz = lp.draw_box(img, layout, color_map=COLOR_MAP)
    draw_pil_image(viz)  # show the results


def draw_pil_image(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # If image came from OpenCV, convert BGR → RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]

    plt.imshow(img)
    plt.show()
    plt.close()


def draw_cv2_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.close()
