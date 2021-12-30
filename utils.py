import numpy as np
import matplotlib.pyplot as plt
import torch



def show_set_images(imgs):
    """ Plot a list of images"""
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n/cols))
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10,7))
    for i in range(n):
        ax[i // cols][i % cols].imshow(imgs[i])
    fig.savefig("image_test.png")

def plot_color_curve(ax, image, n=75, min=-0.25, max=1.25, **kwargs):
    """Plot the colors histogram curva"""
    y = torch.histc(torch.mean(image,[0]), n, min=min, max=max)
    vals = np.linspace(min, max, n)
    ax.plot(vals, y, **kwargs)
    return ax

def plot_to_image(fig):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Draw figure on canvas
    fig.canvas.draw()
    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8z
    img = np.swapaxes(img, 1, 2)
    return img