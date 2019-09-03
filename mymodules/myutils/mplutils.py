# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:07:11 2019

@author: 0000145046
"""

from matplotlib.patches import Rectangle
import pathlib


def AddAxesBBoxRect(fig, ax, ec='k'):
    axpos = ax.get_position()
    rect = fig.patches.append(Rectangle((axpos.x0, axpos.y0), axpos.width, axpos.height,
                                        ls='solid', lw=2, ec=ec, fill=False, transform=fig.transFigure))
    return rect

def drawtextInAxesBttomLeft(fig, ax, text, ypos=-0.02, fontsize=10.5):
    pos = ax.get_position()
    fig.text(pos.x0+0.01, pos.y0+ypos, text,
             color="k", fontsize=fontsize)

def hidden_all_axis(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
def drawtextInAxesBttomRight(fig, ax, text, ypos, fontsize=10.5):
    pos = ax.get_position()
    fig.text((pos.x0+pos.x1)/2 - 0.02, pos.y0+ypos, text,
             color="k", fontsize=fontsize)

def my_savefig(fig, filename):
    if pathlib.Path(filename).exists():
        raise FileExistsError
    else:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

