# Depth-Based Image Processing

This project demonstrates two image processing techniques using depth information: blending images and applying synthetic focus.

## Requirements

- Python 3.x
- `PIL` (Pillow)
- `numpy`
- `opencv-python`

Install the required packages using pip:

```bash
pip install Pillow numpy opencv-python
```

## Usage

Run the script from the command line:

```bash
python main.py
```

You will be prompted to choose an experiment type:

1. Blending two images.
2. Applying synthetic focus using depth information.

### Blending Images

1. When prompted, choose experiment type `1`.
2. Provide the paths to the two images you want to blend.
3. The first image will be depth-analyzed and divided into layers.
4. Visualize and optionally perform morphological operations on these layers.
5. Specify which layers to blend from each image.
6. The resulting blended image will be saved as `cv_blended_image.png` and displayed.

### Synthetic Focus

1. When prompted, choose experiment type `2`.
2. Provide the path to the image for applying synthetic focus.
3. The image will be depth-analyzed and divided into 3 layers.
4. Visualize the layers.
5. Enter blur weights for each layer, separated by commas (e.g., `5,10,15`).
6. The resulting image with synthetic focus will be saved as `newimage.png` and displayed.

## Functions

- **get_depth_image(image)**: Placeholder for extracting depth information from the image.
- **get_layers(n, depth_image)**: Placeholder for segmenting the depth image into `n` layers.
- **visualize_labels(labels)**: Placeholder for visualizing the segmented layers.
- **apply_label_blur(labels, image, weights)**: Applies a Gaussian blur to each layer based on specified weights.
