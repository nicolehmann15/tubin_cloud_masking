import matplotlib.pyplot as plt


def show_patch(self, patch, color_map): #gray/binary
    """Show an image patch with matplotlib using gray/binary colormap

    Parameter
    patch: image patch to be shown
    color_map: pixel value mode --> gray 0-255, binary 0-1
    """
    plt.imshow(patch, cmap=color_map)
    plt.show()