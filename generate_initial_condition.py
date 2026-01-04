#!/usr/bin/env python3
"""
Generate initial conditions for drop/bubble contact line simulations.

Creates smooth interface geometries for a drop/bubble touching a flat wall
with a fillet transition at the contact point.
"""

import argparse
import os
import subprocess as sp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rc('text', usetex=True)


def get_circles(delta):
    """
    Generate interface coordinates for a drop/bubble touching a flat wall.

    The interface consists of:
    1. Circle 1: Main drop, radius 1, centered at (-(1+delta), 0)
    2. Fillet circle: Smooth transition at contact point
    3. Vertical line: Wall contact extending upward

    Parameters
    ----------
    delta : float
        Neck radius / minimum length scale. Separation = 2*delta.

    Returns
    -------
    Interface : pd.DataFrame
        DataFrame with 'x' and 'y' columns for the full interface.
    X1, Y1 : arrays
        Circle 1 (main drop) coordinates.
    Xf, Yf : arrays
        Fillet circle coordinates.
    X2, Y2 : arrays
        Vertical line (wall contact) coordinates.
    Circle1 : pd.DataFrame
        Full circle 1 coordinates (for f2_init.dat).
    X1Full, Y1Full : arrays
        Full circle 1 coordinates.
    """
    X1c = -(1 + delta)

    # phic is the angle where y coordinate is 2*delta
    phic1 = np.arcsin(2 * delta)
    phi1 = np.linspace(np.pi, phic1, int(1e3))
    X1 = X1c + np.cos(phi1)
    Y1 = np.sin(phi1)

    # Full circle 1 for f2_init
    phi = np.linspace(np.pi, 0.0, int(1e3))
    X1Full = X1c + np.cos(phi)
    Y1Full = np.sin(phi)

    # Fillet circle
    Yfc = (1 + delta) * np.tan(phic1)
    Rf = (1 + delta) / np.cos(phic1) - 1

    phifStart = np.pi / 2 - phic1
    phifEnd = np.pi / 2

    phif = np.linspace(phifStart, -phifEnd, int(1e3))
    Xf = -Rf * np.sin(phif)
    Yf = Yfc - Rf * np.cos(phif)

    # Shift coordinates so fillet end is at x=0
    X1 = -(X1 - Xf[-1])
    X1Full = -(X1Full - Xf[-1])
    Xf = -(Xf - Xf[-1])

    # Reflect about x=0 so circle is in -x region
    X1 = -X1
    X1Full = -X1Full
    Xf = -Xf

    # Vertical line (wall contact)
    X2 = Xf[-1] * np.ones(len(X1))
    Y2 = np.linspace(Yf[-1], 16, len(X1))

    # Combine all segments
    X = np.concatenate([X1, Xf, X2])
    Y = np.concatenate([Y1, Yf, Y2])

    Interface = pd.DataFrame({'x': X, 'y': Y})
    Circle1 = pd.DataFrame({'x': X1Full, 'y': Y1Full})

    return Interface, X1, Y1, Xf, Yf, X2, Y2, Circle1, X1Full, Y1Full


def get_facets(L0):
    """
    Run Basilisk to generate facets from the interface data.

    Parameters
    ----------
    L0 : int
        Domain size for Basilisk simulation.

    Returns
    -------
    segs : list
        List of line segments ((z1, r1), (z2, r2)) for plotting.
    """
    exe = ["./InitialCondition", str(L0)]
    p = sp.Popen(exe, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    temp1 = stderr.decode("utf-8")
    temp2 = temp1.split("\n")
    segs = []
    skip = False
    if len(temp2) > 1e2:
        for n1 in range(len(temp2)):
            temp3 = temp2[n1].split(" ")
            if temp3 == ['']:
                skip = False
            else:
                if not skip:
                    temp4 = temp2[n1 + 1].split(" ")
                    r1, z1 = float(temp3[1]), float(temp3[0])
                    r2, z2 = float(temp4[1]), float(temp4[0])
                    segs.append(((z1, r1), (z2, r2)))
                    skip = True
    return segs


def plot_interfaces(Interface, X1, Y1, Xf, Yf, X2, Y2, X1Full, Y1Full,
                    delta, image_name, show=False):
    """
    Plot interface without Basilisk comparison.

    Creates a 2x2 subplot figure with progressively zoomed views.
    """
    plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(24, 12))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # Window definitions for zoom levels
    NWidth = 10
    xW1, xW2, yW1, yW2 = -NWidth*delta, NWidth*delta, 0, NWidth*delta
    NWidth = 5
    xW3, xW4, yW3, yW4 = -NWidth*delta, NWidth*delta, 0, NWidth*delta
    NWidth = 1
    xW5, xW6, yW5, yW6 = -NWidth*delta, NWidth*2*delta, delta, NWidth*2*delta+delta

    # Full view
    ax1.plot(X1, Y1, '-', lw=2, color='#fdae61')
    ax1.plot(Xf, Yf, '-', lw=2, color='#d7191c')
    ax1.plot(X2, Y2, '-', lw=2, color='#2c7bb6')
    ax1.plot(X1Full, Y1Full, '-', lw=2, color='k')
    ax1.plot([xW1, xW2, xW2, xW1, xW1], [yW1, yW1, yW2, yW2, yW1], 'k-', lw=2)
    ax1.axis('square')
    ax1.set_xlim(-(delta + 2 + 0.5), delta)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel(r'$\mathcal{Z}$', fontsize=24)
    ax1.set_ylabel(r'$\mathcal{R}$', fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=24)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    # Zoom 1
    ax2.plot([xW1, xW2, xW2, xW1, xW1], [yW1, yW1, yW2, yW2, yW1], 'k-', lw=4)
    ax2.plot(X1, Y1, '-', lw=4, color='#fdae61')
    ax2.plot(Xf, Yf, '-', lw=4, color='#d7191c')
    ax2.plot(X2, Y2, '-', lw=4, color='#2c7bb6')
    ax2.plot([xW3, xW4, xW4, xW3, xW3], [yW3, yW3, yW4, yW4, yW3], '-', lw=2, color='gray')
    ax2.axis('square')
    ax2.set_xlim(xW1, xW2)
    ax2.set_ylim(yW1, yW2)
    ax2.axis('off')

    # Zoom 2
    ax3.plot([xW3, xW4, xW4, xW3, xW3], [yW3, yW3, yW4, yW4, yW3], '-', lw=4, color='gray')
    ax3.plot(X1, Y1, '-', lw=4, color='#fdae61')
    ax3.plot(Xf, Yf, '-', lw=4, color='#d7191c')
    ax3.plot(X2, Y2, '-', lw=4, color='#2c7bb6')
    ax3.plot([0, 0], [0, yW4], '-', lw=2, color='gray')
    ax3.plot([xW5, xW6, xW6, xW5, xW5], [yW5, yW5, yW6, yW6, yW5], '-', lw=2, color='#abdda4')
    ax3.axis('square')
    ax3.set_xlim(xW3, xW4)
    ax3.set_ylim(yW3, yW4)
    ax3.axis('off')

    # Zoom 3
    ax4.plot([xW5, xW6, xW6, xW5, xW5], [yW5, yW5, yW6, yW6, yW5], '-', lw=4, color='#abdda4')
    ax4.plot(X1, Y1, '-', lw=4, color='#fdae61')
    ax4.plot(Xf, Yf, '-', lw=4, color='#d7191c')
    ax4.plot(X2, Y2, '-', lw=4, color='#2c7bb6')
    ax4.axis('square')
    ax4.set_xlim(xW5, xW6)
    ax4.set_ylim(yW5, yW6)
    ax4.axis('off')

    plt.savefig(image_name, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_interfaces_basilisk(Interface, facets, delta, image_name, show=False):
    """
    Plot interface with Basilisk comparison overlay.

    Orange: Python-generated interface
    Green: Basilisk-computed facets
    """
    plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(24, 12))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # Window definitions for zoom levels
    NWidth = 10
    xW1, xW2, yW1, yW2 = -NWidth*delta, NWidth*delta, 0, NWidth*delta
    NWidth = 5
    xW3, xW4, yW3, yW4 = -NWidth*delta, NWidth*delta, 0, NWidth*delta
    NWidth = 1
    xW5, xW6, yW5, yW6 = -NWidth*delta, NWidth*2*delta, delta, NWidth*2*delta+delta

    # Full view
    ax1.plot(Interface['x'], Interface['y'], '-', lw=4, color='#fdae61')
    line_segments = LineCollection(facets, linewidths=4, colors='#1a9641', linestyle='solid')
    ax1.add_collection(line_segments)
    ax1.plot([xW1, xW2, xW2, xW1, xW1], [yW1, yW1, yW2, yW2, yW1], 'k-', lw=2)
    ax1.axis('square')
    ax1.set_xlim(-(delta + 2 + 0.5), delta)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel(r'$\mathcal{Z}$', fontsize=24)
    ax1.set_ylabel(r'$\mathcal{R}$', fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=24)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    # Zoom 1
    ax2.plot([xW1, xW2, xW2, xW1, xW1], [yW1, yW1, yW2, yW2, yW1], 'k-', lw=4)
    ax2.plot(Interface['x'], Interface['y'], '-', lw=4, color='#fdae61')
    line_segments = LineCollection(facets, linewidths=4, colors='#1a9641', linestyle='solid')
    ax2.add_collection(line_segments)
    ax2.plot([xW3, xW4, xW4, xW3, xW3], [yW3, yW3, yW4, yW4, yW3], '-', lw=2, color='gray')
    ax2.axis('square')
    ax2.set_xlim(xW1, xW2)
    ax2.set_ylim(yW1, yW2)
    ax2.axis('off')

    # Zoom 2
    ax3.plot([xW3, xW4, xW4, xW3, xW3], [yW3, yW3, yW4, yW4, yW3], '-', lw=4, color='gray')
    ax3.plot(Interface['x'], Interface['y'], '-', lw=4, color='#fdae61')
    line_segments = LineCollection(facets, linewidths=4, colors='#1a9641', linestyle='solid')
    ax3.add_collection(line_segments)
    ax3.plot([0, 0], [0, yW4], '-', lw=2, color='gray')
    ax3.plot([xW5, xW6, xW6, xW5, xW5], [yW5, yW5, yW6, yW6, yW5], '-', lw=2, color='#abdda4')
    ax3.axis('square')
    ax3.set_xlim(xW3, xW4)
    ax3.set_ylim(yW3, yW4)
    ax3.axis('off')

    # Zoom 3
    ax4.plot([xW5, xW6, xW6, xW5, xW5], [yW5, yW5, yW6, yW6, yW5], '-', lw=4, color='#abdda4')
    ax4.plot(Interface['x'], Interface['y'], '-', lw=4, color='#fdae61')
    line_segments = LineCollection(facets, linewidths=4, colors='#1a9641', linestyle='solid')
    ax4.add_collection(line_segments)
    ax4.axis('square')
    ax4.set_xlim(xW5, xW6)
    ax4.set_ylim(yW5, yW6)
    ax4.axis('off')

    plt.savefig(image_name, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def generate_initial_conditions(delta, data_folder, image_folder=None,
                                 run_basilisk=False, L0=3, show_plots=False):
    """
    Generate initial conditions for a given delta value.

    Parameters
    ----------
    delta : float
        Neck radius / minimum length scale.
    data_folder : str
        Output folder for data files.
    image_folder : str, optional
        Output folder for images. If None, no images are saved.
    run_basilisk : bool
        Whether to run Basilisk for verification.
    L0 : int
        Domain size for Basilisk (if run_basilisk=True).
    show_plots : bool
        Whether to display plots interactively.
    """
    # Create output folders
    os.makedirs(data_folder, exist_ok=True)
    if image_folder:
        os.makedirs(image_folder, exist_ok=True)

    print(f"Generating delta = {delta}")

    Interface, X1, Y1, Xf, Yf, X2, Y2, Circle1, X1Full, Y1Full = get_circles(delta)

    # Save data files
    f_path = os.path.join(data_folder, "f_init.dat")
    Interface.to_csv(f_path, index=False, header=False, sep=' ')
    print(f"  Saved: {f_path}")

    f_drop_path = os.path.join(data_folder, "f-drop_init.dat")
    Circle1.to_csv(f_drop_path, index=False, header=False, sep=' ')
    print(f"  Saved: {f_drop_path}")

    # Also save to f_Testing.dat for Basilisk
    Interface.to_csv('f_Testing.dat', index=False, header=False, sep=' ')

    if image_folder:
        # Plot without Basilisk
        img_name = os.path.join(image_folder, "TestWithoutBasilisk.pdf")
        plot_interfaces(Interface, X1, Y1, Xf, Yf, X2, Y2, X1Full, Y1Full,
                        delta, img_name, show_plots)
        print(f"  Saved: {img_name}")

        if run_basilisk:
            # Run Basilisk and plot comparison
            facets = get_facets(L0)
            img_name = os.path.join(image_folder, "TestWithBasilisk.pdf")
            plot_interfaces_basilisk(Interface, facets, delta, img_name, show_plots)
            print(f"  Saved: {img_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate initial conditions for drop/bubble contact line simulations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --delta 0.01
  %(prog)s --delta 0.01 --images
  %(prog)s --delta 0.01 --images --basilisk --L0 3
  %(prog)s --delta 0.005 --data-folder MyData --images --show
        """
    )

    parser.add_argument('--delta', type=float, required=True,
                        help='Neck radius / minimum length scale')
    parser.add_argument('--data-folder', type=str, default=None,
                        help='Output folder for data files (default: DataFiles_delta{delta})')
    parser.add_argument('--image-folder', type=str, default=None,
                        help='Output folder for images (default: no images)')
    parser.add_argument('--images', action='store_true',
                        help='Generate images in ImageFiles_delta{delta} folder')
    parser.add_argument('--basilisk', action='store_true',
                        help='Run Basilisk for verification')
    parser.add_argument('--L0', type=int, default=3,
                        help='Domain size for Basilisk (default: 3)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively')

    args = parser.parse_args()

    data_folder = args.data_folder
    if not data_folder:
        data_folder = f"DataFiles_delta{args.delta}"

    image_folder = args.image_folder
    if args.images and not image_folder:
        image_folder = f"ImageFiles_delta{args.delta}"

    generate_initial_conditions(
        delta=args.delta,
        data_folder=data_folder,
        image_folder=image_folder,
        run_basilisk=args.basilisk,
        L0=args.L0,
        show_plots=args.show
    )


if __name__ == '__main__':
    main()
