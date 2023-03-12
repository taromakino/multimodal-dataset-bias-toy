import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 16})


def next_color(ax):
    return next(ax._get_lines.prop_cycler)["color"]