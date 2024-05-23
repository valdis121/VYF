import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from pyramid import cv_laplacian, cv_pyramid, cv_multiresolution_blend, cv_reconstruct_laplacian
from helper import cv2pil, multiply_nn_mnn
from depth import get_depth_image, get_layers, visualize_labels, close_operation_for_pair
from PIL import Image
import cv2
import numpy as np


def apply_label_blur(labels_image, base_image, weights):
    images = list()
    array = np.array(base_image)
    for i in range(len(weights)):
        mask = np.zeros_like(labels_image, dtype=np.uint8)
        mask[labels_image == i] = 1
        image_label = np.zeros_like(base_image, dtype=np.uint8)
        for x in range(base_image.shape[0]):
            for y in range(base_image.shape[1]):
                image_label[x][y] = array[x][y]
        if weights[i]!=0:
            image_label = cv2.blur(image_label, (weights[i], weights[i]), 0)
        image_label *= cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        images.append(image_label)

    return images


def apply_bokeh(image, kernel_size=15):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), cv2.BORDER_ISOLATED)

    _, mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    bokeh = cv2.bitwise_and(image, image, mask=mask)

    return bokeh

if __name__ == "__main__":
    image = Image.open("pasted image 0.png")

    im = get_depth_image(image)
    labels = get_layers(3, im)
    image = np.array(image)
    visualize_labels(labels)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    weights = input()
    weights = weights.split(",")
    weights = [int(x) for x in weights]
    images = apply_label_blur(labels, image,weights)
    image = np.zeros_like(image, dtype=np.uint8)
    for i in range(0, len(images)):
        image+=images[i]
        # blended_image.show()
    cv2.imwrite("newimage2.png", image)