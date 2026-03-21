"""Visualization utilities for displaying document layouts."""

import cv2
import layoutparser as lp
import matplotlib.pyplot as plt
import numpy as np

# Color maps for layoutparser's draw_box function
COLOR_MAP = {
    "Paragraph": "red",
    "Title": "blue",
    "List": "green",
    "Table": "purple",
    "Figure": "pink",
    "Header": "orange",
}

# Color map matching our annotation categories
GLASANA_COLOR_MAP = {
    "Header": "red",
    "Footer": "grey",
    "PageNum": "green",
    "Section": "purple",
    "Kicker": "pink",
    "Headline": "orange",
    "Deck": "cyan",
    "Subhead": "magenta",
    "Byline": "yellow",
    "Dropcap": "brown",
    "Paragraph": "blue",
    "Quote": "black",
    "Footnote": "lightblue",
    "Figure": "lightcoral",
    "Caption": "lightgreen",
    "Advertisement": "lightyellow",
    "Dateline": "lightpink",
    "EditNote": "lightgrey",
    "MarginNote": "lightcyan",
}


def draw_layout(img, layout, save_path=None, has_score=False):
    """Draw labeled bounding boxes on an image and display it.

    Args:
        img: Image (numpy array, BGR or RGB).
        layout: A layoutparser Layout with TextBlocks.
        save_path: Optional path to save the figure.
    """
    viz = lp.draw_box(
        img,
        [b.set(id=f"{b.score:.2f}/{b.type}" if has_score else f"{b.type}") for b in layout],
        color_map=GLASANA_COLOR_MAP,
        show_element_id=True,
        id_font_size=10,
        id_text_background_color="grey",
        id_text_color="white",
    )
    draw_pil_image(viz, save_path)


def draw_pil_image(img, save_path=None):
    """Display a PIL/numpy image with matplotlib, optionally saving to file.

    Handles BGR-to-RGB conversion for OpenCV images.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Convert BGR to RGB if needed (3-channel images from OpenCV)
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]

    plt.imshow(img)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def draw_cv2_image(img):
    """Display an OpenCV BGR image with matplotlib."""
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.close()
