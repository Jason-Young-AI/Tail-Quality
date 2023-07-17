import numpy
import pathlib
import argparse

from matplotlib import pyplot

dataset_choices = ['ImageNet', 'COCO']


def draw_coco_count_per_bis(save_dir, bis_to_count):
    fig, ax = pyplot.subplots(1, 1, figsize=(10, 10))

    biss = list()
    counts = list()
    for bis, count in bis_to_count:
        biss.append(str(bis))
        counts.append(count)

    print(f' . Total {len(bis_to_count)} different image sizes.')

    bars = ax.barh(biss[:20], width=counts[:20], color='skyblue')
    ax.bar_label(bars, fontsize=6, rotation=-60)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=8)
    ax.set_ylabel('(H, W)', fontsize=8)
    ax.set_title('Count of each size (H, W) of image (Top 20)', fontsize=8)

    ax.tick_params(axis='x', which='major', pad=3, labelsize=6, labelrotation=0, labelright=True, labelleft=False)
    ax.tick_params(axis='y', which='major', pad=3, labelsize=6, labelrotation=0, labelright=False, labelleft=True)

    figpath = save_dir.joinpath('count_per_bis.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')


def calculate_stat(its):
    if len(its):
        mins = numpy.min(its, axis=-1)
        maxs = numpy.max(its, axis=-1)
        meds = numpy.median(its, axis=-1)
        avgs = numpy.average(its, axis=-1)
        vars = numpy.var(its, axis=-1, ddof=1)
        stds = numpy.std(its, axis=-1, ddof=1)
        q1s = numpy.quantile(its, 0.25, axis=-1)
        q3s = numpy.quantile(its, 0.75, axis=-1)
        iqrs = q3s - q1s
        lower_whiskers = q1s - 1.5 * iqrs
        upper_whiskers = q3s + 1.5 * iqrs
    else:
        mins = numpy.array([])
        maxs = numpy.array([])
        meds = numpy.array([])
        avgs = numpy.array([])
        vars = numpy.array([])
        stds = numpy.array([])
        q1s = numpy.array([])
        q3s = numpy.array([])
        iqrs = numpy.array([])
        lower_whiskers = numpy.array([])
        upper_whiskers = numpy.array([])

    return dict(
        mins = mins,
        maxs = maxs,
        meds = meds,
        avgs = avgs,
        vars = vars,
        stds = stds,
        q1s = q1s,
        q3s = q3s,
        iqrs = iqrs,
        lower_whiskers = lower_whiskers,
        upper_whiskers = upper_whiskers
    )


def draw_coco_stat(save_dir, bis_to_its, stat_name='avgs'):
    # 1. Draw 3D Scatter
    #    A. H > W, (x, y, z) = (H, W, stat(Time))
    #    B. H > W, (x, y, z) = (W, H, stat(Time))
    xy_xs = list()
    xy_ys = list()
    xy_zs = list()
    yx_xs = list()
    yx_ys = list()
    yx_zs = list()
    for bis, its in bis_to_its.items():
        x, y = bis
        stat = calculate_stat(numpy.array(its))
        if x <= y:
            for s in stat[stat_name]:
                xy_xs.append(x)
                xy_ys.append(y)
                xy_zs.append(s)
        else:
            for s in stat[stat_name]:
                yx_xs.append(y)
                yx_ys.append(x)
                yx_zs.append(s)

    fig, axes = pyplot.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})
    # 1-A. Un-swapped
    ax = axes[0]
    ax.scatter(xy_xs, xy_ys, xy_zs, s=5, c='xkcd:lavender', marker="<", label="Short Height")
    ax.scatter(yx_ys, yx_xs, yx_zs, s=5, c='xkcd:gold', marker=">", label="Long Height")
    ax.invert_xaxis()
    ax.view_init(elev=35, azim=-45)

    ax.set_xlabel('Height', fontsize=8)
    ax.set_ylabel('Width', fontsize=8)
    ax.set_zlabel('Average Time', fontsize=8)
    ax.set_title('Average time of each image', fontsize=8)

    ax.tick_params(axis='x', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='y', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='z', which='major', pad=1, labelsize=6)
    ax.legend()

    # 1-B. Swapped
    ax = axes[1]
    ax.scatter(xy_xs, xy_ys, xy_zs, s=5, c='xkcd:lavender', marker="<", label="Short Height")
    ax.scatter(yx_xs, yx_ys, yx_zs, s=5, c='xkcd:gold', marker=">", label="Long Height")
    ax.invert_xaxis()
    ax.view_init(elev=35, azim=-45)

    ax.set_xlabel('Height', fontsize=8)
    ax.set_ylabel('Width', fontsize=8)
    ax.set_zlabel('Average Time', fontsize=8)
    ax.set_title('Average time of each image (Swapped)', fontsize=8)

    ax.tick_params(axis='x', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='y', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='z', which='major', pad=1, labelsize=6)
    ax.legend()

    figpath = save_dir.joinpath(f'3d_scatter_{stat_name}.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')

    # 2. Draw 2D Scatter
    #    A. (x, y) = (H * W, stat(Time))
    #    B. (x, y) = (H / W, stat(Time))
    xy = list()
    for x, y, z in zip(xy_xs, xy_ys, xy_zs):
        xy.append((x/y, (x, y), z))
    xy = sorted(xy, key=lambda item: item[0])

    div_xy_ls = list()
    div_xy_xs = list()
    div_xy_ys = list()
    for div_xy, l, z in xy:
        div_xy_ls.append(str(l))
        div_xy_xs.append(div_xy)
        div_xy_ys.append(z)

    xy = list()
    for x, y, z in zip(xy_xs, xy_ys, xy_zs):
        xy.append((x*y, (x, y), z))
    xy = sorted(xy, key=lambda item: item[0])

    mul_xy_ls = list()
    mul_xy_xs = list()
    mul_xy_ys = list()
    for mul_xy, l, z in xy:
        mul_xy_ls.append(str(l))
        mul_xy_xs.append(mul_xy)
        mul_xy_ys.append(z)

    yx = list()
    for x, y, z in zip(yx_xs, yx_ys, yx_zs):
        yx.append((x/y, (x, y), z))
    yx = sorted(yx, key=lambda item: item[0])

    div_yx_ls = list()
    div_yx_xs = list()
    div_yx_ys = list()
    for div_yx, l, z in yx:
        div_yx_ls.append(str(l))
        div_yx_xs.append(div_yx)
        div_yx_ys.append(z)

    yx = list()
    for x, y, z in zip(yx_xs, yx_ys, yx_zs):
        yx.append((x*y, (x, y), z))
    yx = sorted(yx, key=lambda item: item[0])

    mul_yx_ls = list()
    mul_yx_xs = list()
    mul_yx_ys = list()
    for mul_yx, l, z in yx:
        mul_yx_ls.append(str(l))
        mul_yx_xs.append(mul_yx)
        mul_yx_ys.append(z)

    fig, axes = pyplot.subplots(1, 2, figsize=(20, 10))
    # 2-A. Div
    ax = axes[0]
    ax.scatter(div_xy_xs, div_xy_ys, s=5, c='xkcd:lavender', marker="<", label="Short Height")
    ax.scatter(div_yx_xs, div_yx_ys, s=5, c='xkcd:gold', marker=">", label="Long Height")

    ax.set_xlabel('Height / Width', fontsize=8)
    ax.set_ylabel('Average Time', fontsize=8)
    ax.set_title('Average time of each image', fontsize=8)

    step = (div_xy_xs[-1] - div_xy_xs[0]) / 20
    xticks = list()
    xticklabels = list()
    index = 0
    position = div_xy_xs[0] - 0.00000001
    while len(xticks) < 21:
        if position < div_xy_xs[index]:
            xticks.append(div_xy_xs[index])
            xticklabels.append(div_xy_ls[index])
            position = position + step
        else:
            index = index + 1

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis='x', which='major', pad=1, labelsize=6, rotation=45)
    ax.tick_params(axis='y', which='major', pad=1, labelsize=6)
    ax.legend()

    # 2-B. Mul
    ax = axes[1]
    ax.scatter(mul_xy_xs, mul_xy_ys, s=5, c='xkcd:lavender', marker="<", label="Short Height")
    ax.scatter(mul_yx_xs, mul_yx_ys, s=5, c='xkcd:gold', marker=">", label="Long Height")

    ax.set_xlabel('Height * Width', fontsize=8)
    ax.set_ylabel('Average Time', fontsize=8)
    ax.set_title('Average time of each image', fontsize=8)

    step = (mul_xy_xs[-1] - mul_xy_xs[0]) / 20
    xticks = list()
    xticklabels = list()
    index = 0
    position = mul_xy_xs[0] - 1
    while len(xticks) < 21:
        if position < mul_xy_xs[index]:
            xticks.append(mul_xy_xs[index])
            xticklabels.append(mul_xy_ls[index])
            position = position + step
        else:
            index = index + 1

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis='x', which='major', pad=1, labelsize=6, rotation=45)
    ax.tick_params(axis='y', which='major', pad=1, labelsize=6)
    ax.legend()

    figpath = save_dir.joinpath(f'2d_scatter_{stat_name}.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')


def draw_coco_specific_stat(save_dir, bis_to_its, bis_to_count, top=5):

    def draw_essential_stat(ax, hw_e_stat, wh_e_stat, x_label):
        hw_e_i = list(range(0, len(hw_e_stat)))
        wh_e_i = list(range(len(hw_e_i), len(hw_e_i) + len(wh_e_stat)))

        if len(hw_e_i) != 0:
            ax.scatter(hw_e_i, hw_e_stat, s=6, c='xkcd:chocolate', marker="<", label="Short Height")
        if len(wh_e_i) != 0:
            ax.scatter(wh_e_i, wh_e_stat, s=6, c='xkcd:orangered', marker=">", label="Long Height")

        ax.set_xlabel(f'{x_label} Time', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.set_title(f'{x_label} Time V.S. Count', fontsize=8)
        ax.legend()

    total_draw = 0
    for index, ((h, w), _) in enumerate(bis_to_count):
        if total_draw < top:
            if (h, w) in bis_to_its.keys() and (w, h) in bis_to_its.keys():
                total_draw = total_draw + 1

                hw_stat = calculate_stat(numpy.array(bis_to_its[(h, w)]))
                wh_stat = calculate_stat(numpy.array(bis_to_its[(w, h)]))

                fig, axes = pyplot.subplots(2, 2, figsize=(20, 20))

                draw_essential_stat(axes[0, 0], hw_stat['meds'], wh_stat['meds'], 'Median')
                draw_essential_stat(axes[0, 1], hw_stat['avgs'], wh_stat['avgs'], 'Average')
                draw_essential_stat(axes[1, 0], hw_stat['vars'], wh_stat['vars'], 'Variance')
                draw_essential_stat(axes[1, 1], hw_stat['stds'], wh_stat['stds'], 'Standard Deviation')

                figpath = save_dir.joinpath(f'specific_stat_{total_draw}_notop{index+1}_{h}-{w}_{h/w:.5f}.pdf')
                fig.savefig(figpath)
                print(f' - Fig Exported: {figpath}')
        else:
            break


def draw_coco(data, save_dir):
    image_ids = data['image_ids']
    batch_image_sizes = data['batch_image_sizes']
    inference_times = data['inference_times']

    bis_to_its = dict()
    for batch_image_size, inference_time in zip(batch_image_sizes, inference_times):
        height, width = batch_image_size

        bis = (height, width)
        it = bis_to_its.get(bis, list())
        it.append(inference_time)
        bis_to_its[bis] = it

    bis_to_count = dict()
    for bis in  bis_to_its.keys():
        its = bis_to_its.get(bis, list())
        bis = (bis[0], bis[1]) if bis[0] <= bis[1] else (bis[1], bis[0])
        count = bis_to_count.get(bis, 0)
        count = count + len(its)
        bis_to_count[bis] = count
    bis_to_count = list(bis_to_count.items())
    bis_to_count = sorted(bis_to_count, key=lambda x: x[1])[::-1]

    print(f'[Begin] Drawing ...')

    print(f' v Drawing ...')
    draw_coco_count_per_bis(save_dir, bis_to_count)
    print(f' ^ Draw Count V.S. Image Size Finished.\n')

    print(f' v Drawing ...')
    draw_coco_stat(save_dir, bis_to_its, 'avgs')
    print(f' ^ Draw Statistics \'avg\' Finished.\n')

    print(f' v Drawing ...')
    draw_coco_stat(save_dir, bis_to_its, 'mins')
    print(f' ^ Draw Statistics \'min\' Finished.\n')

    print(f' v Drawing ...')
    draw_coco_stat(save_dir, bis_to_its, 'maxs')
    print(f' ^ Draw Statistics \'max\' Finished.\n')

    print(f' v Drawing ...')
    draw_coco_stat(save_dir, bis_to_its, 'vars')
    print(f' ^ Draw Statistics \'var\' Finished.\n')

    print(f' v Drawing ...')
    draw_coco_specific_stat(save_dir, bis_to_its, bis_to_count, top=10)
    print(f' ^ Draw Specific Statistics Finished.\n')

    print(f'[End] All Finished.')


def draw_imagenet(data, save_dir):
    pass


def draw(npz_path, save_dir, dataset_type='ImageNet'):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"

    data = numpy.load(npz_path)

    if dataset_type == 'ImageNet':
        draw_imagenet(data, save_dir)

    if dataset_type == 'COCO':
        draw_coco(data, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-p', '--npz-path', type=str, required=True)
    parser.add_argument('-d', '--save-dir', type=str, required=True)
    parser.add_argument('-t', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    arguments = parser.parse_args()

    npz_path = pathlib.Path(arguments.npz_path)
    save_dir = pathlib.Path(arguments.save_dir)
    dataset_type = arguments.dataset_type

    assert npz_path.is_file(), f"No Such NPZ File: {npz_path}"
    assert dataset_type in dataset_choices, f"No Such Dataset Type: {dataset_type}"

    if not save_dir.is_dir():
        print(f"No Save Dir Exists: {save_dir}, now creating it.")
        save_dir.mkdir(parents=True, exist_ok=True)

    draw(npz_path, save_dir, dataset_type)