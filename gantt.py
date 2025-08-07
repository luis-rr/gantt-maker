"""Let's do the gantt chart again!"""

import numpy as np
import pandas as pd
import json

from matplotlib import pyplot as plt
import matplotlib.patches
import matplotlib.path


def load_style():
    with open('style.json', 'r') as f:
        return json.load(f)


def make_rectangle_marker(width=1.0, height=3):
    verts = [
        (-width / 2, -height / 2),
        ( width / 2, -height / 2),
        ( width / 2,  height / 2),
        (-width / 2,  height / 2),
        (-width / 2, -height / 2)  # close path
    ]
    return matplotlib.path.Path(verts)


STYLE = load_style()
RECT_MARKER = make_rectangle_marker()

TEXT_COL = 'label'


def load_google_sheet(spreadsheet_id_file):
    """
    Loads a publicly shared Google Sheet as a pandas DataFrame.

    Parameters:
        spreadsheet_id_file (str): Path to a file containing
        the Google Spreadsheet ID.

    Returns:
        pd.DataFrame: Loaded DataFrame from the Google Sheet.
    """
    with open(spreadsheet_id_file, 'r') as f:
        spreadsheet_id = f.read().strip()

    url = (
        f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}'
        '/export?format=csv'
    )

    return pd.read_csv(url, header=0)


def load_gantt_df():
    df = load_google_sheet('sheet_id.txt')

    df.index = df['name'].str.strip()
    assert df['name'].is_unique, df['name'][df['name'].duplicated()]
    df.index = df['name']

    assert df['WP'].notna().all()

    df['stop'] = df['stop'] + 1  # make it non-inclusive
    df['duration'] = df['stop'] - df['start']
    assert (df['duration'].dropna() > 0).all()

    return df


# noinspection PyTypeChecker
def place_elements(df):
    """
    Place tasks bars in the y axis.
    If two consecutive tasks don't overlap they'll be place in the same row.
    Non-task items inherit the last task y value.
    """

    tasks = df[df['type'] == 'T']

    # tasks = tasks.sort_values(['WP', 'start', 'stop'], ascending=(True, True, False))

    ys = pd.Series(np.nan, index=df.index)

    ys.loc[tasks.index[0]] = 0

    for i, (t0, t1) in enumerate(zip(tasks.index[:-1], tasks.index[1:])):

        if tasks.loc[t0, 'WP'] != tasks.loc[t1, 'WP']:  # different WP
            ys.loc[t1] = ys.max() + 2.

        elif tasks.loc[t0, 'stop'] > tasks.loc[t1, 'start']:  # overlap
            ys.loc[t1] = ys.max() + 1.

        else:
            ys.loc[t1] = ys.loc[t0]

    ys = ys.ffill().bfill()

    return ys


def layout_tasks(df):
    y = place_elements(df)

    df['x0'] = df['start'] - 0.5  # start of the month
    df['w'] = df['duration']

    df['y0'] = y
    df['h'] = 0.

    df.loc[df['type'] == 'T', 'y0'] -= 0.5
    df.loc[df['type'] == 'T', 'h'] = 1


def layout_wp(df):
    for wp, tasks in df[df['type'] == 'T'].groupby('WP'):
        df.loc[wp, 'start'] = tasks['start'].min()
        df.loc[wp, 'stop'] = tasks['stop'].max()
        df.loc[wp, 'duration'] = df.loc[wp, 'stop'] - df.loc[wp, 'start']

        x0 = tasks['x0'].min()
        df.loc[wp, 'x0'] = x0
        df.loc[wp, 'y0'] = tasks['y0'].min()

        df.loc[wp, 'w'] = (tasks['x0'] + tasks['w']).max() - df.loc[wp, 'x0']
        df.loc[wp, 'h'] = (tasks['y0'] + tasks['h']).max() - df.loc[wp, 'y0']


def shrink_boxes(df, who, length=0.1, thickness=0.2):
    df.loc[who, 'x0'] = df.loc[who, 'x0'] + length * 0.5
    df.loc[who, 'w'] = df.loc[who, 'duration'] - length
    df.loc[who, 'y0'] = df.loc[who, 'y0'] + thickness * 0.5
    df.loc[who, 'h'] = df.loc[who, 'h'] - thickness


def layout_secondment(df):
    extra = 2
    df.loc[df['type'] == 'S', 'y0'] = df['y0'].min()
    df.loc[df['type'] == 'S', 'h'] = (df['y0'] + df['h']).max(0) - df['y0'].min() + extra


def _arc(center, rx, ry, theta1, theta2, num_points):
    if rx == 0 or ry == 0 or num_points <= 1:
        return [center]
    angles = np.linspace(theta1, theta2, num_points)
    return [
        (center[0] + rx * np.cos(t), center[1] + ry * np.sin(t))
        for t in angles
    ]

def rounded_rectangle_patch(
    ax, x, y, width, height,
    radius_bl=0, points_bl=1,
    radius_br=0, points_br=1,
    radius_tr=0, points_tr=1,
    radius_tl=0, points_tl=1,
    aspect_ratio=3./7.,
    **kwargs
):
    """
    Returns a Polygon approximating a rectangle with optionally rounded corners.
    All in data coordinates. Radius is corrected for aspect ratio.
    """
    # Convert scalar radius to (rx, ry) adjusted by aspect ratio
    def radii(r):
        return r, r / aspect_ratio

    # Corner centers (for the arcs)
    rx_bl, ry_bl = radii(radius_bl)
    rx_br, ry_br = radii(radius_br)
    rx_tr, ry_tr = radii(radius_tr)
    rx_tl, ry_tl = radii(radius_tl)

    bl_c = (x + rx_bl, y + ry_bl)
    br_c = (x + width - rx_br, y + ry_br)
    tr_c = (x + width - rx_tr, y + height - ry_tr)
    tl_c = (x + rx_tl, y + height - ry_tl)

    # Build path clockwise
    path = []
    path += _arc(bl_c, rx_bl, ry_bl, np.pi, 1.5 * np.pi, points_bl)
    path += _arc(br_c, rx_br, ry_br, 1.5 * np.pi, 2 * np.pi, points_br)
    path += _arc(tr_c, rx_tr, ry_tr, 0, 0.5 * np.pi, points_tr)
    path += _arc(tl_c, rx_tl, ry_tl, 0.5 * np.pi, np.pi, points_tl)

    return matplotlib.patches.Polygon(path, closed=True, **kwargs)

def softbox_patch(*args, **kwargs):
    return rounded_rectangle_patch(
        *args,
        **{
            **dict(
                radius_br=.15, points_br=10,
                # radius_tl=.15, points_tl=10,
            ),
           ** kwargs
        },
    )


def label_months(ax, months, y1, base_year=None, height=2):
    y0 = y1 - height

    month_idcs = np.arange(len(months)) + 1

    for i, m in zip(month_idcs, months):
        ax.text(
            i,
            y1,
            f'{m}',
            zorder=1e6,
            ha='center',
            va='bottom',
            **STYLE['time']['text'],
        )

    ends = month_idcs[months == 12]
    ends = np.concat([month_idcs[[0]] - 1, ends, month_idcs[[-1]]]) + .5

    for i, (x0, x1) in enumerate(zip(ends[:-1], ends[1:])):
        rect = matplotlib.patches.Rectangle(
            (x0, y0),
            width=x1 - x0,
            height=y1 - y0,
            **{
                **dict(
                    zorder=1e5
                ),
                **STYLE['time']['item'],
            },
        )
        ax.add_patch(rect)

        ax.text(
            (x0 + x1) * .5,
            y0,
            f'year {i + 1}' if base_year is None else base_year + i,
            **{
                **dict(
                    zorder=1e6,
                    ha='center',
                    va='top',
                ),
                **STYLE['time']['text']
            },
        )


def create_fig():
    f, ax = plt.subplots(
        dpi=200,
        figsize=(7, 4),
        constrained_layout=True,
    )
    # ax.set(
    #     aspect='equal'
    # )

    ax.set_xticks(np.arange(1, 24 + 1))

    for m in np.arange(1, 25.5, 1):
        ax.axvline(m - 0.5, color='xkcd:silver', linewidth=.25)

    months = np.arange(1, 25)
    label_months(ax, months, y1=-1)

    first_month = 4  # April
    months = (months - 2 + first_month) % 12 + 1
    label_months(ax, months, y1=-4, base_year=2026)

    ax.tick_params(
        labeltop=False,
        bottom=False, labelbottom=False,
        left=False, labelleft=False,
    )

    ax.set(
        xlim=(0, 25.5)
    )

    ax.relim()  # Recalculate limits based on all artists
    ax.autoscale_view()  # Autoscale view using the new limits

    ax.invert_yaxis()

    return ax


def plot_task(
        ax,
        item,
):
    name = item.loc[TEXT_COL]
    x0 = item.loc['x0']
    y0 = item.loc['y0']
    w = item.loc['w']
    h = item.loc['h']

    rect = softbox_patch(
        ax,
    x0, y0,
        width=w,
        height=h,
        **{
            **dict(zorder=1e5),
            **STYLE[item['style']]['item'],
        },
    )

    ax.text(
        x0,
        y0 + h * 0.5,
        f'{name}',
        **{
            **dict(
                va='center',
                ha='left',
                zorder=1e6,
            ),
            **STYLE[item['style']]['text']
        },
    )

    ax.add_patch(rect)


def plot_wp(ax, item):
    wp_rect = softbox_patch(
        ax,
        item['x0'], item['y0'],
        item['w'],
        item['h'],
        **{
            **STYLE[item['style']]['item'],
        },
    )

    ax.text(
        0,
        item['y0'] + item['h'] * 0.5,
        item[TEXT_COL],
        **{
            **dict(rotation=0, va='center', ha='right'),
            **STYLE[item['style']]['text'],
        }
    )

    ax.add_patch(wp_rect)


def plot_wp_detailed(ax, df):
    tasks = df[df['type'] == 'T']

    tasks = tasks.sort_values(['start', 'stop'], ascending=(True, False))

    for (i, k) in enumerate(tasks.index):
        plot_task(
            ax,
            tasks.loc[k]
        )

    wp = df[df['type'] == 'WP']
    assert len(wp) == 1
    wp = wp.iloc[0]
    plot_wp(ax, wp)


def plot_deliverables(ax, deliverables):
    for k in deliverables.index:
        item = deliverables.loc[k]

        x = item['start'] + 0.5
        y = item['y0'] + item['h'] * .5

        ax.scatter(
            [x],
            [y],
            **{
                **dict(zorder=1e6),
                **STYLE[item['style']]['item'],
            },
        )

        ax.text(
            x, y, f'{item[TEXT_COL]}',
            **{
                **dict(va='center', ha='center', zorder=1e6, ),
                **STYLE[item['style']]['text']
            },
        )


def plot_milestones(ax, df, milestones):
    for k in milestones.index:
        item = milestones.loc[k]

        x = item['start'] + 0.5
        y = item['y0'] + item['h'] * .5

        # w = 0.1
        # h = 1
        #
        # rect = matplotlib.patches.Rectangle(
        #     (x - w / 2, y - h / 2),
        #     w, h,
        #     zorder=1e5,
        #     facecolor=STYLE[item['style']]['item'].get('color', None),
        #     edgecolor='none'
        # )
        # ax.add_patch(rect)
        # ax.scatter(
        #     [x],
        #     [y],
        #     zorder=1e5,
        #     marker=RECT_MARKER,
        #     edgecolor='none',
        #     linewidth=0.25,
        #     s=30,
        # )

        ax.plot(
            [x, x],
            [df.loc[item['WP'], 'y0'], df.loc[item['WP'], 'y0'] + df.loc[item['WP'], 'h']],
            zorder=1e4,
            linewidth=1,
            solid_capstyle='butt',
            color=STYLE[item['style']]['item'].get('color', None),
        )

        ax.axvline(
            [x],
            **{
                **dict(zorder=1e6),
                **STYLE[item['style']]['item'],
            },
        )

        ax.text(
            x,
            0,
            f'{milestones.loc[k, TEXT_COL]}',

            clip_on=False,
            transform=ax.get_xaxis_transform(),

            **{
                **dict(va='center', ha='center', zorder=1e6),
                **STYLE[item['style']]['text'],
            },
        )


def plot_secondments(ax, secondments):
    for k in secondments.index:
        item = secondments.loc[k]

        rect = softbox_patch(
            ax,
            item['x0'], item['y0'],
            item['w'],
            item['h'],
            **{
                **STYLE[item['style']]['item'],
            },
        )

        ax.text(
            # 0,
            item['x0'] + item['w'] * .5,
            item['y0'] + item['h'],
            str(item[TEXT_COL]),
            **{
                **dict(rotation=0, va='bottom', ha='center'),
                **STYLE[item['style']]['text'],
            }
        )

        ax.add_patch(rect)
