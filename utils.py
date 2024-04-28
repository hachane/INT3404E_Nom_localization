import os
import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# Just for testing, if u wanna test -> change the dir to ur repo's location like this.
# os.chdir(r"D:\Documents\GitHub\NomOCR")


# READ IMAGES IN COLORED FORMAT
def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# PLOT BB TO DEBUG
def plot_BB(image_path, coord_path):
    image = read_image(image_path)

    with open(coord_path, "r") as f:
        coordinates = [line.strip().split()[1:] for line in f.readlines()]

    plt.imshow(image)
    for coord in coordinates:
        # Split the coordinates
        x_upper_left, y_upper_left, x_below_right, y_below_right = map(float, coord)
        x = x_upper_left
        y = y_upper_left
        w = x_below_right - x_upper_left
        h = y_below_right - y_upper_left
        # Scale the coordinates to the image dimensions
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        # Convert coordinates to integers
        x, y, w, h = map(int, (x, y, w, h))

        rect_box = plt.Rectangle((x, y), w, h, color='red', fill=False, lw=3)
        plt.gca().add_patch(rect_box)

# CONVERT X_CENTER, Y_CENTER, 
def convert_coord(labels_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            label_file = os.path.join(labels_folder, filename)
            output_file = os.path.join(output_folder, filename)

            with open(label_file, "r") as f, open(output_file, "w") as out:
                for line in f:
                    label, x_center, y_center, width, height = line.strip().split()
                    x_center, y_center, width, height = map(float, [x_center, y_center, width, height])

                    # Convert coordinates
                    x_upper_left = x_center - (width / 2)
                    y_upper_left = y_center - (height / 2)
                    x_lower_right = x_center + (width / 2)
                    y_lower_right = y_center + (height / 2)

                    # Write the new coordinates to the output file
                    out.write(f"{label} {x_upper_left:.6f} {y_upper_left:.6f} {x_lower_right:.6f} {y_lower_right:.6f}\n")



# os.makedirs("labels_new", exist_ok=True)
# os.makedirs("labels_new/train", exist_ok=True)
# os.makedirs("labels_new/val", exist_ok=True)
# convert_coord("wb_localization_dataset/labels/train", "labels_new/train")
# convert_coord("wb_localization_dataset/labels/val", "labels_new/val")


