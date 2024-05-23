from depth import *
from depth_blure import *

if __name__ == "__main__":
    action = input("Choose one experiment type (1 - blending, 2 - synthetic focus): ")
    if action == "1":
        path1 = input("Write path to the image which you want to change: ")
        path2 = input("Write path to the image which you want to append to the goal image: ")
        image1 = Image.open(path1)
        image2 = Image.open(path2)
        image1.show()
        image2.show()
        image1_depth = get_depth_image(image1)
        image1_depth.show()
        n = int(input("How much layers do you want to get?: "))
        labels = get_layers(n, image1_depth)
        visualize_labels(labels)
        a = input("Do you want to do close operation between layers? (1 - yes, 0 - no): ")
        if a == "1":
            while True:
                for x in range(n):
                    for y in range(n):
                        for z in range(n):
                            labels = close_operation_for_pair(labels, x, y, z)
                visualize_labels(labels)
                b = input("One more time? (1 - yes, 0 - not): ")
                if b == "0":
                    break
        image2 = image2.resize((labels.shape[1], labels.shape[0]))
        image1 = np.array(image1)
        image2 = np.array(image2)
        while True:
            must_be_changed = input("What layers number do you want to change?(for example: 1,3,5): ")
            must_be_changed = [int(x) for x in must_be_changed.split(',')]
            mask = create_mask(labels, must_be_changed)
            visualize_labels(mask)
            gpA = cv_pyramid(image1.copy(), scale=3)
            gpB = cv_pyramid(image2.copy(), scale=3)
            gpM = cv_pyramid(mask.copy(), scale=3)
            lpA = cv_laplacian(gpA, scale=3)
            lpB = cv_laplacian(gpB, scale=3)
            lpM = cv_laplacian(gpM, scale=3)
            blended_pyramid = cv_multiresolution_blend(gpM, lpA, lpB)
            blended_image = cv_reconstruct_laplacian(blended_pyramid)
            blended_image = cv2pil(blended_image)
            blended_image.save('cv_blended_image.png')
            blended_image.show()
            b = input("One more time? (1 - yes, 0 - not): ")
            if b == "0":
                break
            visualize_labels(labels)

    elif action == "2":
        path = input("Write path to the image for synthetic focus: ")
        image = Image.open(path)
        image1_depth = get_depth_image(image)
        image1_depth.show()
        n = int(input("How much layers do you want to get?: "))
        labels = get_layers(n, image1_depth)
        visualize_labels(labels)
        a = input("Do you want to do close operation between layers? (1 - yes, 0 - no): ")
        if a == "1":
            while True:
                for x in range(n):
                    for y in range(n):
                        for z in range(n):
                            labels = close_operation_for_pair(labels, x, y, z)
                visualize_labels(labels)
                b = input("One more time? (1 - yes, 0 - not): ")
                if b == "0":
                    break
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        while True:
            weights = input("Enter blur weights for each layer, separated by commas (10,15,20): ")
            weights = weights.split(",")
            weights = [int(x) for x in weights]
            images = apply_label_blur(labels, image, weights)
            final_image = np.zeros_like(image, dtype=np.uint8)
            for img in images:
                final_image += img
            cv2.imwrite("result5.png", final_image)
            final_image_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
            final_image_pil.show()
            b = input("One more time? (1 - yes, 0 - not): ")
            if b == "0":
                break
            visualize_labels(labels)


