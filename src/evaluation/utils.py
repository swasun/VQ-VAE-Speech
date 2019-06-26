 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors, colorbar
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import colorsys
import numpy as np


class Utils(object):

    @staticmethod
    def rand_cmap(nlabels, type='bright', first_color_black=False, last_color_black=False, verbose=False):
        """
        Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
        :param nlabels: Number of labels (size of colormap)
        :param type: 'bright' for strong colors, 'soft' for pastel colors
        :param first_color_black: Option to use first color as black, True or False
        :param last_color_black: Option to use last color as black, True or False
        :param verbose: Prints the number of labels and shows the colormap. True or False
        :return: colormap for matplotlib
        :author: https://github.com/delestro/rand_cmap
        """

        if type not in ('bright', 'soft'):
            print ('Please choose "bright" or "soft" for type')
            return

        if verbose:
            print('Number of labels: ' + str(nlabels))

        # Generate color map for bright colors, based on hsv
        if type == 'bright':
            randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                            np.random.uniform(low=0.2, high=1),
                            np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

            # Convert HSV list to RGB
            randRGBcolors = []
            for HSVcolor in randHSVcolors:
                randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]

            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]

            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

        # Generate soft pastel colors, by limiting the RGB spectrum
        if type == 'soft':
            low = 0.6
            high = 0.95
            randRGBcolors = [(np.random.uniform(low=low, high=high),
                            np.random.uniform(low=low, high=high),
                            np.random.uniform(low=low, high=high)) for i in range(nlabels)]

            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]

            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]
            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

        # Display colorbar
        if verbose:
            fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

            bounds = np.linspace(0, nlabels, nlabels + 1)
            norm = colors.BoundaryNorm(bounds, nlabels)

            cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                    boundaries=bounds, format='%1i', orientation=u'horizontal')

        return random_colormap

    @staticmethod
    def build_gif(images, interval=0.1, dpi=72,
        save_gif=True, saveto='animation.gif',
        show_gif=False, cmap=None):
        """
        Take an array or list of images and create a GIF.
        Parameters
        ----------
        images : np.ndarray or list
            List of images to create a GIF of
        interval : float, optional
            Spacing in seconds between successive images.
        dpi : int, optional
            Dots per inch.
        save_gif : bool, optional
            Whether or not to save the GIF.
        saveto : str, optional
            Filename of GIF to save.
        show_gif : bool, optional
            Whether or not to render the GIF using plt.
        cmap : None, optional
            Optional colormap to apply to the images.
        Returns
        -------
        ani : matplotlib.animation.ArtistAnimation
            The artist animation from matplotlib.  Likely not useful.
        Author
        ------
        Creative Applications of Deep Learning w/ Tensorflow.
        Kadenze, Inc.
        Copyright Parag K. Mital, June 2016.
        """

        images = np.asarray(images)
        h, w, *c = images[0].shape
        fig, ax = plt.subplots(figsize=(np.round(w / dpi), np.round(h / dpi)))
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        ax.set_axis_off()

        if cmap is not None:
            axs = list(map(lambda x: [
                ax.imshow(x, cmap=cmap)], images))
        else:
            axs = list(map(lambda x: [
                ax.imshow(x)], images))

        ani = animation.ArtistAnimation(
            fig, axs, interval=interval, repeat_delay=0, blit=False)

        if save_gif:
            ani.save(saveto, writer='imagemagick', dpi=dpi)

        if show_gif:
            plt.show()

        return ani
