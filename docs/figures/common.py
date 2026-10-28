# Shared look of the documentation figures: two themes, one palette, the image writers

# Libraries
import pathlib, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from PIL import Image
from scipy.spatial import cKDTree

# Every image lands here, once per theme, as <name>_light and <name>_dark
IMAGES = pathlib.Path(__file__).resolve().parent / 'images'

# GitHub's two page surfaces, and the inks and series colours chosen against each. 'test' is the
# reference set, 'train' the train split and 'sample' any other sampler a figure scores; every pair
# of the three stays at least 9 OKLab Delta E x 100 apart under simulated colour blindness, in both
# modes
THEMES = {
    'light': {'name': 'light', 'surface': '#ffffff', 'ink': '#1f2328', 'muted': '#59636e',
              'grid': '#d1d9e0', 'test': '#2a78d6', 'train': '#eb6834', 'sample': '#1baf7a'},
    'dark': {'name': 'dark', 'surface': '#0d1117', 'ink': '#e6edf3', 'muted': '#9198a1',
             'grid': '#3d444d', 'test': '#3987e5', 'train': '#d95926', 'sample': '#199e70'},
}

# Resolution of every PNG: 8 inches at 200 dpi is 1600 pixels, twice what a docs page shows
DPI = 200


def use(theme):
    """Sets matplotlib's defaults for one theme"""
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'text.color': theme['ink'], 'axes.labelcolor': theme['ink'],
        'axes.titlecolor': theme['ink'], 'axes.edgecolor': theme['grid'],
        'xtick.color': theme['muted'], 'ytick.color': theme['muted'],
        'xtick.labelcolor': theme['muted'], 'ytick.labelcolor': theme['muted'],
        'axes.facecolor': 'none', 'figure.facecolor': 'none', 'savefig.facecolor': 'none',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': 0.8,
        'ytick.major.width': 0.8, 'legend.frameon': False, 'mathtext.fontset': 'dejavusans',
        'figure.constrained_layout.use': True, 'image.interpolation': 'antialiased',
    })


def ramp(theme, colour):
    """Returns a one-hue sequential colormap from the page surface to a colour"""
    return LinearSegmentedColormap.from_list('ramp', [theme['surface'], colour])


def density_map(theme, depth=1.0, levels=10):
    """Returns the colormap every density is drawn in: GnBu, starting from the page surface

    On the light surface the map runs from white to dark blue. On the dark surface it runs the
    other way, from dark blue to pale green, so a dense region is the bright one on both pages; it
    stops a quarter short of GnBu's near-white end there, so a flat density does not glare.

    NOTE: the map is discrete, so a density reads as filled contours. A smooth ramp bands once
    the PNG is reduced to a palette, and the levels also give the eye something to count.

    Args:
        * theme (dict): Theme
        * depth (float): Fraction of GnBu used; 0.6 stops the light map before the dark blues, so
            blue test samples drawn over a density stay readable (default is 1.0)
        * levels (int): Number of discrete levels (default is 10)

    Returns:
        * cmap (matplotlib.colors.Colormap): The colormap, the surface colour at zero
    """
    gnbu = matplotlib.colormaps['GnBu']
    if theme['name'] == 'light':
        values = np.linspace(0.0, depth, 64)
    else:
        values = np.linspace(1.0, 1.0 - 0.75 * depth, 64)
    colours = [theme['surface']] + [gnbu(value) for value in values[1:]]
    return LinearSegmentedColormap.from_list('density', colours).resampled(levels)


def blank(ax):
    """Removes every axis decoration, for an image or a scatter whose axes carry nothing"""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def note(ax, lines, theme, loc='upper right', size=7.5):
    """Writes metric values in a corner of an axis, one 'name value' line each

    Args:
        * ax (matplotlib.axes.Axes): Axis
        * lines (list of str): Lines of text
        * theme (dict): Theme
        * loc (str): 'upper right', 'upper left', 'lower right' or 'lower left'
            (default is 'upper right')
        * size (float): Font size (default is 7.5)
    """
    assert loc in ['upper right', 'upper left', 'lower right', 'lower left']
    x = 0.98 if 'right' in loc else 0.02
    y = 0.97 if 'upper' in loc else 0.03
    ax.text(x, y, '\n'.join(lines), transform=ax.transAxes, fontsize=size,
            family='DejaVu Sans Mono', color=theme['ink'], ha='right' if 'right' in loc else 'left',
            va='top' if 'upper' in loc else 'bottom', linespacing=1.4)


def legend(fig, theme, entries):
    """Draws one legend across the top of a figure

    Args:
        * fig (matplotlib.figure.Figure): Figure, laid out by constrained layout
        * theme (dict): Theme
        * entries (list of tuple): (label, colour key of the theme, kind), the kind one of
            'dot', 'line', 'patch', 'density' (a patch of density_map, the key unused) or 'fill'
            (the lighter density fill exact marginals are drawn in, the key unused)
    """
    handles = []
    for label, key, kind in entries:
        assert kind in ['dot', 'line', 'patch', 'density', 'fill']
        if kind == 'density':
            handles.append(plt.Rectangle((0, 0), 1, 1, color=density_map(theme)(0.75),
                                         label=label))
        elif kind == 'fill':
            handles.append(plt.Rectangle((0, 0), 1, 1, color=density_map(theme)(0.45), alpha=0.6,
                                         label=label))
        elif kind == 'dot':
            handles.append(plt.Line2D([], [], marker='o', linestyle='none', markersize=6,
                                      color=theme[key], label=label))
        elif kind == 'line':
            handles.append(plt.Line2D([], [], linewidth=1.5, color=theme[key], label=label))
        else:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=theme[key], label=label))
    fig.legend(handles=handles, loc='outside upper center', ncol=len(handles),
               handletextpad=0.4, columnspacing=1.6)


def save(draw, name, colours=256):
    """Draws a figure once per theme and writes it as a transparent, palette PNG

    Args:
        * draw (function): theme (dict) -> matplotlib.figure.Figure
        * name (str): File stem, under IMAGES
        * colours (int or None): Size of the palette the PNG is reduced to; None keeps every
            colour, for ray-traced shading that a palette would band (default is 256)

    Returns:
        * paths (list of pathlib.Path): The two images written
    """
    IMAGES.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in THEMES.values():
        use(theme)
        fig = draw(theme)
        path = IMAGES / (name + '_' + theme['name'] + '.png')
        fig.savefig(path, dpi=DPI, transparent=True)
        plt.close(fig)

        # A palette of 256 colours, alpha included, is a third of the size and looks the same
        if colours is not None:
            image = Image.open(path).convert('RGBA')
            image.quantize(colours, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)
        paths.append(path)
        print('wrote', path, '%.0f kB' % (path.stat().st_size / 1e3))
    return paths


def canvas(fig, theme):
    """Rasterises a figure onto the page surface, for one frame of an animation"""
    fig.patch.set_facecolor(theme['surface'])
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def save_gif(draw_frame, n_frames, name, fps=12, dpi=100, colours=128, palette=None):
    """Draws an animation once per theme and writes it as a looping GIF on the page surface

    NOTE: a GIF has one-bit transparency, so the frames are flattened on GitHub's own surface
    colour instead of being left transparent like the PNGs. That colour is always in the palette
    and every pixel goes to its exact nearest palette colour: Pillow's own lookup is approximate and
    turns the page white into a grey, which shows as a box around the animation.

    Args:
        * draw_frame (function): (theme, index) -> matplotlib.figure.Figure
        * n_frames (int): Number of frames
        * name (str): File stem, under IMAGES
        * fps (int): Frames per second (default is 12)
        * dpi (int): Resolution of the frames (default is 100)
        * colours (int): Size of the palette fitted to the frames (default is 128)
        * palette (function or None): theme -> list of colours, a fixed palette instead of one
            fitted to the frames, for a thin line that a fitted palette would merge into a large
            fill of a nearby colour (default is None)

    Returns:
        * paths (list of pathlib.Path): The two animations written
    """
    IMAGES.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in THEMES.values():
        use(theme)
        plt.rcParams['figure.dpi'] = dpi
        frames = [canvas(draw_frame(theme, index), theme) for index in range(n_frames)]

        # One palette for the whole loop, so colours do not flicker: fixed, or fitted to a strip of
        # frames, and the page surface first either way
        surface = np.round(255 * np.array(to_rgb(theme['surface'])))
        if palette is None:
            strip = Image.fromarray(np.concatenate(frames[::max(n_frames // 8, 1)], axis=0))
            fitted = strip.quantize(colours - 1, method=Image.Quantize.MEDIANCUT).getpalette()
            colours_ = np.array(fitted[:3 * (colours - 1)], dtype=np.float64).reshape(-1, 3)
        else:
            colours_ = np.round(255 * np.array([to_rgb(colour) for colour in palette(theme)]))
        table = np.concatenate([surface[None], colours_])
        tree = cKDTree(table)

        # Every pixel to its nearest palette colour
        images = []
        for frame in frames:
            _, nearest = tree.query(frame.reshape(-1, 3).astype(np.float64))
            image = Image.fromarray(nearest.reshape(frame.shape[:2]).astype(np.uint8), mode='P')
            image.putpalette(table.astype(np.uint8).reshape(-1).tolist())
            images.append(image)
        path = IMAGES / (name + '_' + theme['name'] + '.gif')
        images[0].save(path, save_all=True, append_images=images[1:], loop=0,
                       duration=int(1000 / fps), optimize=True)
        paths.append(path)
        print('wrote', path, '%.0f kB' % (path.stat().st_size / 1e3))
    return paths


# Element colours of the molecule renders, one set that reads on both surfaces
ELEMENT_COLOURS = {'C': '#8d949e', 'N': '#3d6fd6', 'O': '#e0483f', 'H': '#f2f2f2', 'S': '#e8c22c',
                   'Ar': '#8d949e'}


def pymol_render(pdb, width, height, setup, view=None, buffer=2.0, return_view=False):
    """Ray-traces a PDB string with PyMOL on a transparent background

    Args:
        * pdb (str): Contents of a PDB file
        * width (int): Width of the image, in pixels
        * height (int): Height of the image, in pixels
        * setup (function): cmd -> None, applies the representation to the object 'mol'
        * view (tuple or None): A view from a previous render; None orients on the molecule
            (default is None)
        * buffer (float): Margin around the molecule when orienting, in Angstrom
            (default is 2.0)
        * return_view (bool): Whether to also return the view used (default is False)

    Returns:
        * image (numpy.ndarray of shape (height, width, 4)): RGBA image, values in [0, 1]
        if return_view:
            * view (tuple): The view, to pass to the next render of the same molecule
    """
    from pymol import cmd

    # A fresh session per render, so no setting leaks from one molecule to the next
    cmd.reinitialize()
    cmd.read_pdbstr(pdb, 'mol')
    for element, colour in ELEMENT_COLOURS.items():
        cmd.set_color('el_' + element, list(to_rgb(colour)))
        cmd.color('el_' + element, 'mol and elem ' + element)
    cmd.set('ray_opaque_background', 0)
    cmd.set('antialias', 2)
    cmd.set('ray_trace_mode', 0)
    cmd.set('ray_shadows', 0)
    cmd.set('ambient', 0.35)
    cmd.set('specular', 0.25)
    cmd.set('orthoscopic', 1)
    setup(cmd)
    if view is None:
        cmd.orient('mol')
        cmd.zoom('mol', complete=1, buffer=buffer)
    else:
        cmd.set_view(view)

    # Render through a file, the one output PyMOL's headless mode offers
    with tempfile.TemporaryDirectory() as folder:
        path = folder + '/render.png'
        cmd.png(path, width=width, height=height, ray=1, quiet=1)
        image = np.asarray(Image.open(path).convert('RGBA'), dtype=np.float64) / 255.0
    if return_view:
        return image, cmd.get_view()
    return image


def start_pymol():
    """Launches PyMOL without a window, once per process"""
    import pymol
    pymol.finish_launching(['pymol', '-cq'])
