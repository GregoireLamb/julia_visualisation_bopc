#! /usr/bin/env python

from re import U
import numpy as np
import math
import argparse
import time
from multiprocessing import Pool, TimeoutError
from julia_curve import c_from_group

# Update according to your group size and number (see TUWEL)
GROUP_SIZE   = 2
GROUP_NUMBER = 1

# do not modify BENCHMARK_C
BENCHMARK_C = complex(-0.2, -0.65)

def compute_julia_set_sequential(xmin, xmax, ymin, ymax, im_width, im_height, c):

    zabs_max = 10
    nit_max = 300

    xwidth  = xmax - xmin
    yheight = ymax - ymin

    julia = np.zeros((im_width, im_height))
    for ix in range(im_width):
        for iy in range(im_height):
            nit = 0
            # Map pixel position to a point in the complex plane
            z = complex(ix / im_width * xwidth + xmin,
                        iy / im_height * yheight + ymin)
            # Do the iterations
            while abs(z) <= zabs_max and nit < nit_max:
                z = z**2 + c
                nit += 1
            ratio = nit / nit_max
            julia[ix,iy] = ratio

    return julia

def compute_patch(args):
    """ Compute a patch of the Julia set """
    x, y, xmin, xmax, ymin, ymax, size, xmin_of, xmax_of, ymin_of, ymax_of, c = args.values()

    zabs_max = 10
    nit_max = 300

    xwidth = xmax - xmin
    yheight = ymax - ymin

    xwidth_patch = xmax_of - xmin_of
    yheight_patch = ymax_of - ymin_of

    julia = np.zeros((xwidth_patch, yheight_patch))
    for ix in range(xwidth_patch):
        for iy in range(yheight_patch):
            nit = 0
            # Map pixel position to a point in the complex plane
            z = complex((ix + xmin_of) / size * xwidth + xmin,
                        (iy + ymin_of) / size * yheight + ymin)
            # Do the iterations
            while abs(z) <= zabs_max and nit < nit_max:
                z = z ** 2 + c
                nit += 1
            ratio = nit / nit_max
            julia[ix, iy] = ratio

    return ((x,y),julia)

def arrange_grid(size, patch, patches):
    """ Arrange the patches into a grid """
    julia_img = np.zeros((size, size))
    for x in range(len(patches)):
        (x,y), patch_result = patches[x]
        xmin = x * patch
        ymin = y * patch
        julia_img[xmin:xmin+patch, ymin:ymin+patch] = patch_result
    return julia_img


def compute_julia_in_parallel(size, xmin, xmax, ymin, ymax, patch, nprocs, c):
    """" Compute Julia set in parallel using multiprocessing.Pool """

    """"
    "--patch", help="patch size in pixels (square images)", type=int
    "--nprocs", help="number of workers", type=int
    "size", help="image size in pixels (square images)", type=int
    
    We suppose the final image to be a square image of size: size x size.
    
    """
    # Sequential version
    #return compute_julia_set_sequential(xmin, xmax, ymin, ymax, size, size, c)

    # Parallel version
    task_list = []
    n_patch = int(math.ceil(size/patch))
    patch_grid = np.zeros((n_patch,n_patch)) # 2D array of patches
    pool = Pool(processes=nprocs)

    for x in range(patch_grid.shape[0]):
        xmin_of = x * patch
        xmax_of = (x + 1) * patch
        if xmax_of > size:
            xmax_of = size
        for y in range(patch_grid.shape[1]):
            ymin_of = y * patch
            ymax_of = (y + 1) * patch
            if ymax_of > size:
                ymax_of = size
            task_list.append({'x': x,
                              'y': y,
                              'xmin': xmin,
                              'xmax': xmax,
                              'ymin': ymin,
                              'ymax': ymax,
                              'size': size,
                              'xmin_of': xmin_of,
                              'xmax_of': xmax_of,
                              'ymin_of': ymin_of,
                              'ymax_of': ymax_of,
                              'c': c})

    completed_patches = pool.map(compute_patch, task_list, 1)
    pool.close()
    pool.join()

    julia_img = arrange_grid(size, patch, completed_patches)

    return julia_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="image size in pixels (square images)", type=int, default=500)
    parser.add_argument("--xmin", help="", type=float, default=-1.5)
    parser.add_argument("--xmax", help="", type=float, default=1.5)
    parser.add_argument("--ymin", help="", type=float, default=-1.5)
    parser.add_argument("--ymax", help="", type=float, default=1.5)
    parser.add_argument("--group-size", help="", type=int, default=None)
    parser.add_argument("--group-number", help="", type=int, default=None)
    parser.add_argument("--patch", help="patch size in pixels (square images)", type=int, default=20)
    parser.add_argument("--nprocs", help="number of workers", type=int, default=1)
    parser.add_argument("--draw-axes", help="Whether to draw axes", action="store_true")
    parser.add_argument("-o", help="output file")
    parser.add_argument("--benchmark", help="Whether to execute the script with the benchmark Julia set", action="store_true")
    args = parser.parse_args()

    if args.group_size is not None:
        GROUP_SIZE = args.group_size
    if args.group_number is not None:
        GROUP_NUMBER = args.group_number

    # assign c based on mode
    c = None
    if args.benchmark:
        c = BENCHMARK_C 
    else:
        c = c_from_group(GROUP_SIZE, GROUP_NUMBER)

    stime = time.perf_counter()
    julia_img = compute_julia_in_parallel(
        args.size,
        args.xmin, args.xmax, 
        args.ymin, args.ymax, 
        args.patch,
        args.nprocs,
        c)
    rtime = time.perf_counter() - stime

    print(f"{args.size};{args.patch};{args.nprocs};{rtime}")

    if not args.o is None:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig, ax = plt.subplots()
        ax.imshow(julia_img, interpolation='nearest', cmap=plt.get_cmap("hot"))

        if args.draw_axes:
            # set labels correctly
            im_width = args.size
            im_height = args.size
            xmin = args.xmin
            xmax = args.xmax
            xwidth = args.xmax - args.xmin
            ymin = args.ymin
            ymax = args.ymax
            yheight = args.ymax - args.ymin

            xtick_labels = np.linspace(xmin, xmax, 7)
            ax.set_xticks([(x-xmin) / xwidth * im_width for x in xtick_labels])
            ax.set_xticklabels(['{:.1f}'.format(xtick) for xtick in xtick_labels])
            ytick_labels = np.linspace(ymin, ymax, 7)
            ax.set_yticks([(y-ymin) / yheight * im_height for y in ytick_labels])
            ax.set_yticklabels(['{:.1f}'.format(-ytick) for ytick in ytick_labels])
            ax.set_xlabel("Imag")
            ax.set_ylabel("Real")
        else:
            # disable axes
            ax.axis("off") 

        plt.tight_layout()
        plt.savefig(args.o, bbox_inches='tight')
        #plt.show()