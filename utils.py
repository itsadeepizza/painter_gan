import numpy as np
import matplotlib.pyplot as plt


def show_set_images(imgs):
    """ Plot a list of images"""
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n/cols))
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10,7))
    for i in range(n):
        ax[i // cols][i % cols].imshow(imgs[i])
    fig.savefig("image_test.png")