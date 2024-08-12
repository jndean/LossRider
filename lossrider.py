from collections import defaultdict
from dataclasses import dataclass, field
import json
from math import log, sin, cos
from typing import DefaultDict, List, Tuple

import pandas as pd

DEFAULT_PALETTE = (
    "#fd4f38Red",
    "#06a725Green",
    "#3995fdBlue",
    "#ffd54bYellow",
    "#62dad4Cyan",
    "#d171dfMagenta",
)
BASE = "Base Layer"
GRID = "#eeeeeeGrid"
AXES = "Axes"
BLACK_LINES = "#000000BlackLines"


# These match the colours of the rider's scarfs on linerider.com


@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    background: bool = False

    def serialise(self, id_, layer):
        ret = {
            "id": id_,
            "type": 2 if self.background else 0,
            "x1": self.x1, 
            "y1": self.y1,
            "x2": self.x2, 
            "y2": self.y2,
            "flipped": False,
            "leftExtended": False, 
            "rightExtended": False
        }
        if layer:
            ret["layer"] = layer
        return ret

@dataclass
class Rider:
    x: float
    y: float

    def serialise(self):
        return {
            "startPosition": {"x": self.x, "y": self.y},
            "startVelocity": {"x": 0.0, "y": 0},
            "remountable": 1
        } 

@dataclass
class Layer:
    colour: str
    id_: int

    def serialise(self):
        return {
            "name": self.colour, 
            "id": self.id_, 
            "visible": True,
            "editable": True,
        }


@dataclass
class Axes:
    x: float = 0
    y: float = 0
    width: float = 1000
    height: float = 1000
    logx: bool = False
    logy: bool = False
    xlim: Tuple[bool] = (None, None)
    ylim: Tuple[bool] = (None, None)
    xticks: Tuple[float] = ()
    yticks: Tuple[float] = ()
    xticklabels: Tuple[str] = ()
    yticklabels: Tuple[str] = ()
    xlabel: str = ""
    ylabel: str = ""
    tick_fontsize: float = 100
    label_fontsize: float = 100
    grid: bool = True

    lines: DefaultDict[str, List[Segment]] = field(default_factory=lambda: defaultdict(list))
    riders: List[Rider] = field(default_factory=list)
    layer_order: List[str] = field(default_factory=lambda: [BASE, GRID, AXES].copy())

    def convert(self, x, y):
        # Bound check
        (L, R), (B, T) = self.xlim, self.ylim
        if (x < L or x > R or y < B or y > T):
            return None

        if self.logx:
            x, L, R = log(x), log(L), log(R)
        if self.logy:
            y, T, B = log(y), log(T), log(B)
        
        # Transform to view
        x_ = (x - L) * self.width / (R - L)
        y_ = (y - B) * self.height / (T - B)

        return x_, -y_
    
    def draw_curve(self, points, name="Line", colour=BLACK_LINES):
        if colour not in self.layer_order:
            self.layer_order.append(colour)
        prev_xy = None
        spawn = None
        # TODO: could draw partial lines for segments that are only partially on the plot
        for x, y in points:
            xy = self.convert(x, y)
            if None not in (xy, prev_xy):
                if spawn is None:
                    spawn = prev_xy
                self.lines[colour].append(Segment(*prev_xy, *xy))
            prev_xy = xy

        assert spawn is not None, "No line segments to draw within axes bounds"

        # Add a starting ledge
        x, y = spawn
        platform = (
            (x - 40, y - 30),
            (x - 20, y - 21),
            (x, y)
        )
        for (p1, p2) in zip(platform, platform[1:]):
            self.lines[colour].append(Segment(*p1, *p2))
        self.riders.append(Rider(x - 35, y - 35))


    def draw_axes(self):
        # Axes
        self.lines[AXES].append(Segment(self.x, self.y, self.x + self.width, self.y))
        self.lines[AXES].append(Segment(self.x, self.y, self.x, self.y - self.height))

        xticklabels = self.xticklabels if self.xticklabels else [str(x) for x in self.xticks]
        yticklabels = self.yticklabels if self.yticklabels else [str(y) for y in self.yticks]

        # xticks
        tick_size = self.height * 0.015
        for xtick, xlabel in zip(self.xticks, xticklabels):
            xy = self.convert(xtick, self.ylim[0])
            if xy is None:
                raise ValueError(f"xtick ({xtick}) is outside of xlim")
            x, y = xy
            self.lines[AXES].append(Segment(x, y, x, y + tick_size))
            self.text(xlabel, x, y + tick_size * 2, self.tick_fontsize, align='center_x')
            if self.grid:
                self.lines[GRID].append(Segment(x, y, x, y - self.height, background=True))

        # yticks
        for ytick, ylabel in zip(self.yticks, yticklabels):
            xy = self.convert(self.xlim[0], ytick)
            if xy is None:
                raise ValueError(f"ytick ({ytick}) is outside of ylim")
            x, y = xy
            self.lines[AXES].append(Segment(x, y, x - tick_size, y))
            self.text(ylabel, x - tick_size * 2, y, self.tick_fontsize, align='center_y,right')
            if self.grid:
                self.lines[GRID].append(Segment(x, y, x + self.width, y, background=True))

        # Labels
        if self.xlabel:
            self.text(
                self.xlabel, 
                self.x + self.width / 2, 2 * (tick_size + self.tick_fontsize), 
                self.label_fontsize, 
                align="center_x"
            )
        if self.ylabel:
            self.text(
                self.ylabel, 
                self.x - 2 * (tick_size + self.tick_fontsize + self.label_fontsize), 
                self.y - self.height / 2, 
                self.label_fontsize, 
                align="center_x,center_y",
                theta=-3.14149/2,
            )


    def save(self, filename):
        lines, layers = [], []
        for layer_num, layer_name in enumerate(self.layer_order):
            layers.append(Layer(layer_name, layer_num).serialise())
            for seg in self.lines[layer_name]:
                lines.append(seg.serialise(len(lines), layer_num))

        output = {
            "label": "LossRider",
            "creator": "", "description": "", "version": "6.2",
            "duration": 1200,
            "audio": None, "script": "",
            "startPosition": {"x": 0, "y": 0},
            "layers": layers,
            "lines": lines,
            "riders": [rider.serialise() for rider in self.riders],
        }
        
        with open(filename, 'w') as f:
            f.write(json.dumps(output))


    def text(self, text, X, Y, h, align="", theta=0):
        aspect = 0.55
        padding_frac = .4

        t = .4
        m = .65
        FONT = {
            # glyph: width_scale, y_offset, [p1, p2, p3, ...]
            "a": (1,   0, [(0, t), (1, t), (1, 1), (0, 1), (0, m), (1, m)]),
            "b": (1,   0, [(0, 0), (0, 1), (1, 1), (1, t), (0, t)]),
            "c": (1,   0, [(1, t), (0, t), (0, 1), (1, 1)]),
            "d": (1,   0, [(1, 0), (1, 1), (0, 1), (0, t), (1, t)]),
            "e": (1,   0, [(0, m), (1, m), (1, t), (0, t), (0, 1), (1, 1)]),
            "f": (.8,  t, [(1, 0), (0, 0), (0, 1-t), (.8, 1-t), (0, 1-t), (0, 1)]),
            "g": (1,   t, [(1, 1-t), (0, 1-t), (0, 0), (1, 0), (1, 1), (0, 1)]),
            "h": (1,   0, [(0, 0), (0, 1), (0, t), (1, t), (1, 1)]),
            "i": (.2,  0, [(.5, t), (.5, 1)]),
            "j": (.4,  t, [(0, 1), (1, 1), (1, 0)]),
            "k": (1,   0, [(0, 0), (0, 1), (0, m), (.3, m), (1, t), (.3, m), (1, 1)]),
            "l": (.2,  0, [(.5, 0), (.5, 1)]),
            "m": (1.3, 0, [(0, 1), (0, t), (.5, t), (.5, 1), (.5, t), (1, t), (1, 1)]),
            "n": (1,   0, [(0, 1), (0, t), (1, t), (1, 1)]),
            "o": (1,   0, [(0, t), (1, t), (1, 1), (0, 1), (0, t)]),
            "p": (1,   t, [(0, 1), (0, 0), (1, 0), (1, 1-t), (0, 1-t)]),
            "q": (1,   t, [(1, 1), (1, 0), (0, 0), (0, 1-t), (1, 1-t)]),
            "r": (.9,  0, [(0, 1), (0, t), (1, t), (1, m)]),
            "s": (1,   0, [(1, t), (0, t), (0, m), (1, m), (1, 1), (0, 1)]),
            "t": (1,   0, [(1, t), (0, t), (0, 0), (0, 1), (1, 1)]),
            "u": (1,   0, [(0, t), (0, 1), (1, 1), (1, t)]),
            "v": (1,   0, [(0, t), (.5, 1), (1, t)]),
            "w": (1.3, 0, [(0, t), (0, 1), (.5, 1), (.5, t), (.5, 1), (1, 1), (1, t)]),
            "x": (1,   0, [(0, t), (1, 1), (.5, .5+t/2), (0, 1), (1, t)]),
            "y": (1,   t, [(0, 0), (0, 1-t), (1, 1-t), (1, 0), (1, 1), (0, 1)]),
            "z": (1,   0, [(0, t), (1, t), (0, 1), (1, 1)]),
            "0": (1,   0, [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
            "1": (.6,  0, [(.1, .1), (.5, 0), (.5, 1), (.1, 1), (.9, 1)]),
            "2": (1,   0, [(0, 0), (1, 0), (1, .5), (0, .5), (0, 1), (1, 1)]),
            "3": (1,   0, [(0, 0), (1, 0), (1, .5), (0, .5), (1, .5), (1, 1), (0, 1)]),
            "4": (1,   0, [(0, 0), (0, .5), (1, .5), (1, 0), (1, 1)]),
            "5": (1,   0, [(1, 0), (0, 0), (0, .5), (1, .5), (1, 1), (0, 1)]),
            "6": (1,   0, [(1, 0), (0, 0), (0, 1), (1, 1), (1, .5), (0, .5)]),
            "7": (1,   0, [(0, 0), (1, 0), (.2, 1)]),
            "8": (1,   0, [(0, 0), (1, 0), (1, .5), (0, .5), (1, .5), (1, 1), (0, 1), (0, 0)]),
            "9": (1,   0, [(1, .5), (0, .5), (0, 0), (1, 0), (1, 1), (0, 1)]),
            ".": (.25, 0, [(.1, 1), (.9, 1), (.9, .9), (.1, .9), (.1, 1)]),
            "/": (1,   0, [(0, 1), (1, 0)]),
            " ": (1.1, 0, []),
        }
        w = h * aspect
        pad = padding_frac * w
        sinθ, cosθ = sin(theta), cos(theta)

        # Alignment
        x, y = 0, 0
        text_width = sum(w * FONT[char][0] + pad for char in text)
        if "center_x" in align:
            x -= text_width / 2
        elif "right" in align:
            x -= text_width
        if "center_y" in align:
            y -= h / 2
        elif "bottom" in align:
            y -= h

        # Rendering
        for char in text:
            width_scale, y_offset, glyph = FONT[char]
            char_w = w * width_scale
            points = [(x + pad/2 + char_w * dw, y + h * (y_offset + dh)) for dw, dh in glyph]
            points = map(lambda xy: (xy[0] * cosθ - xy[1] * sinθ, xy[0] * sinθ + xy[1] * cosθ), points)  # Rotate
            points = list(map(lambda xy: (xy[0] + X, xy[1] + Y), points))    # Translate
            self.lines[AXES].extend([Segment(*p1, *p2) for (p1, p2) in zip(points[:-1], points[1:])])
            x += char_w + pad

def lossrider(
        df, x, y, 
        hue=None,
        filename="lossrider.save", 
        width=3000, height=1000, 
        xlim=(None, None), ylim=(None, None),
        logy=False, logx=False,
        xlabel="", ylabel="",
        xticks=(), yticks=(),
        xticklabels=(), yticklabels=(),
        tick_fontsize=100,
        label_fontsize=100,
        palette=DEFAULT_PALETTE,
        grid=True,
        legend=None,
    ):

    ax = Axes(
        x=0, y=0, 
        width=width, height=height,
        logx=logx, logy=logy,
        xlim=xlim, ylim=ylim,
        xlabel=xlabel, ylabel=ylabel,
        xticks=xticks, yticks=yticks,
        xticklabels=xticklabels, yticklabels=yticklabels,
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
        grid=grid,
    )

    if hue is not None:
        line_names = df[hue].unique()
        if len(line_names) > 6:
            raise ValueError(
                f"Trying to plot {len(line_names)} lines, but linerider.com only support 6 riders :("
            )
        for line_name, colour in zip(line_names, palette):
            line_df = df[df[hue] == line_name]
            curve = sorted(zip(line_df[x].to_list(), line_df[y].to_list()))
            ax.draw_curve(curve, line_name, f"{colour}Layer")
    else:
        ax.draw_curve(sorted(zip(df[x].to_list(), df[y].to_list())))

    ax.draw_axes()
    ax.save(f"{filename}.json")

    try:
        from IPython.display import IFrame
        return IFrame("https://www.linerider.com/", 900, 600)
    except ImportError:
        pass
