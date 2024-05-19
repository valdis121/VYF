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


def get_depth_image(image):
    pipe = pipeline(task="depth-estimation", model="Intel/dpt-large")

    depth = pipe(image)["depth"]
    return depth

def create_2d_array(array, x, y):
    result = []
    for i in range(len(array)):
        for j in range(len(array[i])):
            result.append([array[i][j], i/x, j/y])
    return result
def get_layers(n, img, type="fuzzyCmeans"):
    depth_np = np.array(img)
    a = create_2d_array(depth_np, depth_np.shape[0], depth_np.shape[1])
    if type == "kmeans":
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(a)
        labels = kmeans.labels_
    elif type == "fuzzyCmeans":
        data = np.array(a)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data.T, n, 2, error=0.005, maxiter=1000, init=None
        )
        labels = np.argmax(u, axis=0)

    labels = np.reshape(labels, depth_np.shape)
    return labels

def visualize_labels(labels):
    label_colors = {}

    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)

    fixed_palette = np.random.RandomState(42).randint(0, 256, size=(num_labels, 3))

    for i, label in enumerate(unique_labels):
        label_colors[label] = tuple(fixed_palette[i])

    height, width = labels.shape
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            color = label_colors[label]
            pixels[x, y] = color

    legend_image = Image.new("RGB", (100, 30 * num_labels), "white")
    draw = ImageDraw.Draw(legend_image)
    font = ImageFont.load_default()
    for i, label in enumerate(unique_labels):
        draw.rectangle([0, i * 30, 20, (i + 1) * 30], fill=label_colors[label])
        draw.text((30, i * 30 + 10), f"Label {label}", fill="black", font=font)

    combined_image = Image.new("RGB", (image.width + legend_image.width, max(image.height, legend_image.height)))
    combined_image.paste(image, (0, 0))
    combined_image.paste(legend_image, (image.width, 0))

    combined_image.show()


def close_operation_for_pair(labels_image, label1, label2, new_label, kernel_size=3):
    closed_labels_image = np.copy(labels_image)
    mask = ((closed_labels_image == label1) | (closed_labels_image == label2)).astype(np.uint8) * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    label1_mask = (closed_labels_image == label1).astype(np.uint8)
    label2_mask = (closed_labels_image == label2).astype(np.uint8)

    new_label_mask = np.where(closed_mask == 255, 1, 0).astype(np.uint8)

    closed_labels_image = np.where((new_label_mask == 1) & ((label1_mask + label2_mask) == 0), new_label,
                                   closed_labels_image)

    return closed_labels_image




def blend_images(image1, image2, mask, num_levels):
    gauss_pyramid1 = [image1.copy()]
    gauss_pyramid2 = [image2.copy()]
    mask_pyramid = [mask.copy()]

    for _ in range(num_levels - 1):
        image1 = cv2.pyrDown(image1)
        gauss_pyramid1.append(image1)

        image2 = cv2.pyrDown(image2)
        gauss_pyramid2.append(image2)

        mask = cv2.pyrDown(mask.astype(np.float64))
        mask_pyramid.append(mask)

    laplacian_pyramid1 = [gauss_pyramid1[-1]]
    laplacian_pyramid2 = [gauss_pyramid2[-1]]

    for i in range(num_levels - 1, 0, -1):
        img1_up = cv2.pyrUp(gauss_pyramid1[i])
        img2_up = cv2.pyrUp(gauss_pyramid2[i])

        if img1_up.shape[:2] != gauss_pyramid1[i - 1].shape[:2]:
            img1_up = cv2.resize(img1_up, gauss_pyramid1[i - 1].shape[:2][::-1])
        if img2_up.shape[:2] != gauss_pyramid2[i - 1].shape[:2]:
            img2_up = cv2.resize(img2_up, gauss_pyramid2[i - 1].shape[:2][::-1])

        laplacian1 = cv2.subtract(gauss_pyramid1[i - 1], img1_up)
        laplacian2 = cv2.subtract(gauss_pyramid2[i - 1], img2_up)

        laplacian_pyramid1.append(laplacian1)
        laplacian_pyramid2.append(laplacian2)

    blended_pyramid = []
    for laplacian1, laplacian2, msk in zip(laplacian_pyramid1, laplacian_pyramid2, mask_pyramid):
        resized_mask = cv2.resize(msk, laplacian1.shape[:2][::-1])

        expanded_mask = np.expand_dims(resized_mask, axis=-1)
        expanded_mask = np.tile(expanded_mask, (1, 1, 3))

        blended = laplacian1 * (1 - expanded_mask) + laplacian2 * expanded_mask
        blended_pyramid.append(blended)

    result = blended_pyramid[-1]
    for i in range(len(blended_pyramid) - 2, -1, -1):
        result = cv2.pyrUp(result)

        if result.shape[:2] != blended_pyramid[i].shape[:2]:
            result = cv2.resize(result, blended_pyramid[i].shape[:2][::-1])

        result += blended_pyramid[i]

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def create_mask(labels_image, labels_to_remove):
    print(labels_to_remove, labels_image)
    mask = np.zeros_like(labels_image, dtype=np.uint8)
    for label in labels_to_remove:
        mask[labels_image == label] = 255
    return mask
def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image = Image.open("pasted image 0.png")
    im = get_depth_image(image)
    im.save("depth.png")
    labels = get_layers(3, im)
    visualize_labels(labels)
    user_input = input()
    parsed_input = user_input.split(",")
    parsed_input = [int(x) for x in parsed_input]
    labels=close_operation_for_pair(labels, parsed_input[0], parsed_input[1], parsed_input[2])
    visualize_labels(labels)
    img2 = Image.open("beautiful-small-river.jpg")
    img2 = img2.resize((labels.shape[1], labels.shape[0]))
    user_input_labels = input()
    user_input_labels = user_input_labels.split(",")
    user_input_labels = [int(x) for x in user_input_labels]
    mask = create_mask(labels,user_input_labels)
    print(mask)
    visualize_labels(mask)
    image = np.array(image)
    img2 = np.array(img2)

    gpA = cv_pyramid(image.copy(), scale=3)
    gpB = cv_pyramid(img2.copy(), scale=3)
    gpM = cv_pyramid(mask.copy(), scale=3)


    lpA = cv_laplacian(gpA, scale=3)
    lpB = cv_laplacian(gpB, scale=3)
    lpM = cv_laplacian(gpM, scale=3)

    blended_pyramid = cv_multiresolution_blend(gpM, lpA, lpB)
    blended_image = cv_reconstruct_laplacian(blended_pyramid)
    blended_image = cv2pil(blended_image)
    blended_image.save('cv_blended_image.png')
    # blended_image.show()
    plt.imshow(blended_image)
