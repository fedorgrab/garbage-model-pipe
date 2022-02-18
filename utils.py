import os
import tarfile
import subprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import constants
from pathlib import Path
import imghdr


def get_dataset_sample(train_dataset, class_names):
    fig = plt.figure(figsize=(10, 10))
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(13, 13))
    images, labels = next(iter(train_dataset))

    for i in range(5):
        for j in range(5):
            img_i = 5 * i + j
            image, label = images[img_i], labels[img_i]
            ax = axs[i][j]
            ax.imshow(image.numpy().astype("uint8"))
            ax.set_title(class_names[tf.get_static_value(label)])
            ax.set_axis_off()

    return fig


def unzip_dataset_archive():
    with tarfile.open(constants.DATA_ZIP_DIR) as f:
        f.extractall("./")


def find_bad_files(data_dir):
    image_extensions = [".png", ".jpeg"]
    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    bad_images = []
    print("INFO:")
    subprocess.run(["pwd"])
    subprocess.run(["ls"])
    subprocess.run(["ls", data_dir])
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
                bad_images.append(filepath)
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                bad_images.append(filepath)

        return bad_images


def delete_bad_images():
    bad_files = find_bad_files("./data")
    print(bad_files)
    for file_path in bad_files:
        os.remove(file_path)
