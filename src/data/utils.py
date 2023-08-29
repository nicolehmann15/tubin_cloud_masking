import matplotlib.pyplot as plt

def show_patch(self, patch, color_map): #gray/binary
    """Show an image patch with matplotlib using gray/binary colormap

    Parameter
    patch: image patch to be shown
    color_map: pixel value mode --> gray 0-255, binary 0-1
    """
    plt.imshow(patch, cmap=color_map)
    plt.show()

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """Call in a loop to create terminal progress bar
    Parameter:
        iteration: current iteration (Int)
        total: total iterations (Int)
        prefix: prefix string (Str)
        suffix: suffix string (Str)
        decimals: positive number of decimals in percent complete (Int)
        length: character length of bar (Int)
        fill: bar fill character (Str)
        printEnd: end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

if __name__ == '__main__':
    items = list(range(0, 64))
    l = len(items)
    printProgressBar(0, l, prefix='Progress:', suffix='Complete', decimals=2, length=50, fill='>')
    for i, item in enumerate(items):
        # Update Progress Bar
        printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', decimals=2, length=50, fill='>')