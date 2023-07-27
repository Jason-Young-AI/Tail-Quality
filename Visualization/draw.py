import numpy
import pathlib
import argparse

from matplotlib import pyplot

from constant import dataset_choices, combine_choices, stat_map
from calculate_quality import calculate_stat, combine_ImageNet_times, combine_MMLU_times, combine_COCO_times, calculate_acc, calculate_map, calculate_wf1


def draw_imagenet_count_per_ois(imgs_save_dir, ois_to_count):
    # origin image size
    fig, ax = pyplot.subplots(1, 1, figsize=(10, 10))

    oiss = list()
    counts = list()
    for ois, count in ois_to_count:
        oiss.append(str(ois))
        counts.append(count)

    print(f' . Total {len(ois_to_count)} different image sizes.')

    bars = ax.barh(oiss[:20], width=counts[:20], color='skyblue')
    ax.bar_label(bars, fontsize=6, rotation=-60)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=8)
    ax.set_ylabel('(H, W)', fontsize=8)
    ax.set_title('Count of each size (H, W) of image (Top 20)', fontsize=8)

    ax.tick_params(axis='x', which='major', pad=3, labelsize=6, labelrotation=0, labelright=True, labelleft=False)
    ax.tick_params(axis='y', which='major', pad=3, labelsize=6, labelrotation=0, labelright=False, labelleft=True)

    figpath = imgs_save_dir.joinpath('count_per_ois.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')


def draw_coco_count_per_bis(imgs_save_dir, bis_to_count):
    # batch image size
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

    figpath = imgs_save_dir.joinpath('count_per_bis.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')


def draw_imagenet_stat(imgs_save_dir, ois_to_cts, stat_name='avgs'):
    # 1. Draw 3D Scatter
    #    A. H > W, (x, y, z) = (H, W, stat(Time))
    #    B. H > W, (x, y, z) = (W, H, stat(Time))
    xy_xs = list()
    xy_ys = list()
    xy_zs = list()
    yx_xs = list()
    yx_ys = list()
    yx_zs = list()
    for ois, cts in ois_to_cts.items():
        x, y = ois
        stat = calculate_stat(numpy.array(cts))
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
    ax.view_init(elev=35, azim=0)
    #ax.view_init(elev=35, azim=-45)

    ax.set_xlabel('Height', fontsize=8)
    ax.set_ylabel('Width', fontsize=8)
    ax.set_zlabel(f'{stat_map[stat_name]} Time', fontsize=8)
    ax.set_title(f'{stat_map[stat_name]} time of each image', fontsize=8)

    ax.tick_params(axis='x', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='y', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='z', which='major', pad=1, labelsize=6)
    ax.legend()

    # 1-B. Swapped
    ax = axes[1]
    ax.scatter(xy_xs, xy_ys, xy_zs, s=5, c='xkcd:lavender', marker="<", label="Short Height")
    ax.scatter(yx_xs, yx_ys, yx_zs, s=5, c='xkcd:gold', marker=">", label="Long Height")
    ax.invert_xaxis()
    ax.view_init(elev=35, azim=0)
    #ax.view_init(elev=35, azim=-45)

    ax.set_xlabel('Height', fontsize=8)
    ax.set_ylabel('Width', fontsize=8)
    ax.set_zlabel(f'{stat_map[stat_name]} Time', fontsize=8)
    ax.set_title(f'{stat_map[stat_name]} time of each image (Swapped)', fontsize=8)

    ax.tick_params(axis='x', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='y', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='z', which='major', pad=1, labelsize=6)
    ax.legend()

    figpath = imgs_save_dir.joinpath(f'3d_scatter_{stat_name}.pdf')
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
    ax.set_ylabel(f'{stat_map[stat_name]} Time', fontsize=8)
    ax.set_title(f'{stat_map[stat_name]} time of each image', fontsize=8)

    step = (div_xy_xs[-1] - div_xy_xs[0]) / 20 # ticks' label step
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
    ax.set_ylabel(f'{stat_map[stat_name]} Time', fontsize=8)
    ax.set_title(f'{stat_map[stat_name]} time of each image', fontsize=8)

    step = (mul_xy_xs[-1] - mul_xy_xs[0]) / 20 # ticks' label step
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

    figpath = imgs_save_dir.joinpath(f'2d_scatter_{stat_name}.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')


def draw_coco_stat(imgs_save_dir, bis_to_cts, stat_name='avgs'):
    # 1. Draw 3D Scatter
    #    A. H > W, (x, y, z) = (H, W, stat(Time))
    #    B. H > W, (x, y, z) = (W, H, stat(Time))
    xy_xs = list()
    xy_ys = list()
    xy_zs = list()
    yx_xs = list()
    yx_ys = list()
    yx_zs = list()
    for bis, cts in bis_to_cts.items():
        x, y = bis
        stat = calculate_stat(numpy.array(cts))
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
    ax.set_zlabel(f'{stat_map[stat_name]} Time', fontsize=8)
    ax.set_title(f'{stat_map[stat_name]} time of each image', fontsize=8)

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
    ax.set_zlabel(f'{stat_map[stat_name]} Time', fontsize=8)
    ax.set_title(f'{stat_map[stat_name]} time of each image (Swapped)', fontsize=8)

    ax.tick_params(axis='x', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='y', which='major', pad=1, labelsize=6)
    ax.tick_params(axis='z', which='major', pad=1, labelsize=6)
    ax.legend()

    figpath = imgs_save_dir.joinpath(f'3d_scatter_{stat_name}.pdf')
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
    ax.set_ylabel(f'{stat_map[stat_name]} Time', fontsize=8)
    ax.set_title(f'{stat_map[stat_name]} time of each image', fontsize=8)

    step = (div_xy_xs[-1] - div_xy_xs[0]) / 20 # ticks' label step
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
    ax.set_ylabel(f'{stat_map[stat_name]} Time', fontsize=8)
    ax.set_title(f'{stat_map[stat_name]} time of each image', fontsize=8)

    step = (mul_xy_xs[-1] - mul_xy_xs[0]) / 20 # ticks' label step
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

    figpath = imgs_save_dir.joinpath(f'2d_scatter_{stat_name}.pdf')
    fig.savefig(figpath)
    print(f' - Fig Exported: {figpath}')


def draw_imagenet_specific_stat(imgs_save_dir, ois_to_cts, ois_to_count, top=5):

    def draw_essential_stat(ax, hw_e_stat, wh_e_stat, x_label):
        hw_e_i = list(range(0, len(hw_e_stat)))
        wh_e_i = list(range(len(hw_e_i), len(hw_e_i) + len(wh_e_stat)))

        if len(hw_e_i) != 0:
            ax.scatter(hw_e_i, hw_e_stat, s=6, c='xkcd:chocolate', marker="<", label="Short Height")
        if len(wh_e_i) != 0:
            ax.scatter(wh_e_i, wh_e_stat, s=6, c='xkcd:orangered', marker=">", label="Long Height")

        ax.set_xlabel('Images ID', fontsize=8)
        ax.set_ylabel(f'{x_label} Time', fontsize=8)
        ax.set_title(f'{x_label} Time Details', fontsize=8)
        ax.legend()

    total_draw = 0
    for index, ((h, w), _) in enumerate(ois_to_count):
        if total_draw < top:
            if (h, w) in ois_to_cts.keys() and (w, h) in ois_to_cts.keys():
                total_draw = total_draw + 1

                hw_stat = calculate_stat(numpy.array(ois_to_cts[(h, w)]))
                wh_stat = calculate_stat(numpy.array(ois_to_cts[(w, h)]))

                fig, axes = pyplot.subplots(2, 2, figsize=(20, 20))

                draw_essential_stat(axes[0, 0], hw_stat['meds'], wh_stat['meds'], 'Median')
                draw_essential_stat(axes[0, 1], hw_stat['avgs'], wh_stat['avgs'], 'Average')
                draw_essential_stat(axes[1, 0], hw_stat['vars'], wh_stat['vars'], 'Variance')
                draw_essential_stat(axes[1, 1], hw_stat['stds'], wh_stat['stds'], 'Standard Deviation')

                figpath = imgs_save_dir.joinpath(f'specific_stat_{total_draw}_notop{index+1}_{h}-{w}_{h/w:.5f}.pdf')
                fig.savefig(figpath)
                print(f' - Fig Exported: {figpath}')
        else:
            break


def draw_coco_specific_stat(imgs_save_dir, bis_to_cts, bis_to_count, top=5):

    def draw_essential_stat(ax, hw_e_stat, wh_e_stat, x_label):
        hw_e_i = list(range(0, len(hw_e_stat)))
        wh_e_i = list(range(len(hw_e_i), len(hw_e_i) + len(wh_e_stat)))

        if len(hw_e_i) != 0:
            ax.scatter(hw_e_i, hw_e_stat, s=6, c='xkcd:chocolate', marker="<", label="Short Height")
        if len(wh_e_i) != 0:
            ax.scatter(wh_e_i, wh_e_stat, s=6, c='xkcd:orangered', marker=">", label="Long Height")

        ax.set_xlabel('Images ID', fontsize=8)
        ax.set_ylabel(f'{x_label} Time', fontsize=8)
        ax.set_title(f'{x_label} Time Details', fontsize=8)
        ax.legend()

    total_draw = 0
    for index, ((h, w), _) in enumerate(bis_to_count):
        if total_draw < top:
            if (h, w) in bis_to_cts.keys() and (w, h) in bis_to_cts.keys():
                total_draw = total_draw + 1

                hw_stat = calculate_stat(numpy.array(bis_to_cts[(h, w)]))
                wh_stat = calculate_stat(numpy.array(bis_to_cts[(w, h)]))

                fig, axes = pyplot.subplots(2, 2, figsize=(20, 20))

                draw_essential_stat(axes[0, 0], hw_stat['meds'], wh_stat['meds'], 'Median')
                draw_essential_stat(axes[0, 1], hw_stat['avgs'], wh_stat['avgs'], 'Average')
                draw_essential_stat(axes[1, 0], hw_stat['vars'], wh_stat['vars'], 'Variance')
                draw_essential_stat(axes[1, 1], hw_stat['stds'], wh_stat['stds'], 'Standard Deviation')

                figpath = imgs_save_dir.joinpath(f'specific_stat_{total_draw}_notop{index+1}_{h}-{w}_{h/w:.5f}.pdf')
                fig.savefig(figpath)
                print(f' - Fig Exported: {figpath}')
        else:
            break


# def draw_mmlu_specific_stat(imgs_save_dir, bis_to_cts, bis_to_count, top=5):

#     def draw_essential_stat(ax, hw_e_stat, wh_e_stat, x_label):
#         hw_e_i = list(range(0, len(hw_e_stat)))
#         wh_e_i = list(range(len(hw_e_i), len(hw_e_i) + len(wh_e_stat)))

#         if len(hw_e_i) != 0:
#             ax.scatter(hw_e_i, hw_e_stat, s=6, c='xkcd:chocolate', marker="<", label="Short Height")
#         if len(wh_e_i) != 0:
#             ax.scatter(wh_e_i, wh_e_stat, s=6, c='xkcd:orangered', marker=">", label="Long Height")

#         ax.set_xlabel('Images ID', fontsize=8)
#         ax.set_ylabel(f'{x_label} Time', fontsize=8)
#         ax.set_title(f'{x_label} Time Details', fontsize=8)
#         ax.legend()

#     total_draw = 0
#     for index, ((h, w), _) in enumerate(bis_to_count):
#         if total_draw < top:
#             if (h, w) in bis_to_cts.keys() and (w, h) in bis_to_cts.keys():
#                 total_draw = total_draw + 1

#                 hw_stat = calculate_stat(numpy.array(bis_to_cts[(h, w)]))
#                 wh_stat = calculate_stat(numpy.array(bis_to_cts[(w, h)]))

#                 fig, axes = pyplot.subplots(2, 2, figsize=(20, 20))

#                 draw_essential_stat(axes[0, 0], hw_stat['meds'], wh_stat['meds'], 'Median')
#                 draw_essential_stat(axes[0, 1], hw_stat['avgs'], wh_stat['avgs'], 'Average')
#                 draw_essential_stat(axes[1, 0], hw_stat['vars'], wh_stat['vars'], 'Variance')
#                 draw_essential_stat(axes[1, 1], hw_stat['stds'], wh_stat['stds'], 'Standard Deviation')

#                 figpath = imgs_save_dir.joinpath(f'specific_stat_{total_draw}_notop{index+1}_{h}-{w}_{h/w:.5f}.pdf')
#                 fig.savefig(figpath)
#                 print(f' - Fig Exported: {figpath}')
#         else:
#             break


def draw_ImageNet(extracted_data, combine_type, imgs_save_dir):
    main_results = extracted_data['main_results']
    origin_image_sizes = extracted_data['other_results']['origin_image_sizes']
    #inference_times = extracted_data['other_results']['inference_times']
    #preprocess_times = extracted_data['other_results']['preprocess_times']
    #postprocess_times = extracted_data['other_results']['postprocess_times']

    combined_times = combine_ImageNet_times(extracted_data['other_results'], combine_type)

    ois_to_cts = dict()
    for origin_image_size, combined_time in zip(origin_image_sizes, combined_times):
        height, width = origin_image_size

        ois = (height, width)
        ct = ois_to_cts.get(ois, list())
        ct.append(combined_time)
        ois_to_cts[ois] = ct

    ois_to_count = dict()
    for ois in  ois_to_cts.keys():
        cts = ois_to_cts.get(ois, list())
        ois = (ois[0], ois[1]) if ois[0] <= ois[1] else (ois[1], ois[0])
        count = ois_to_count.get(ois, 0)
        count = count + len(cts)
        ois_to_count[ois] = count
    ois_to_count = list(ois_to_count.items())
    ois_to_count = sorted(ois_to_count, key=lambda x: x[1])[::-1]

    print(f'[Begin] Drawing ...')

    #print(f' v Drawing ...')
    #draw_imagenet_count_per_ois(imgs_save_dir, ois_to_count)
    #print(f' ^ Draw Count V.S. Origin Image Size Finished.\n')

    #print(f' v Drawing ...')
    #draw_imagenet_stat(imgs_save_dir, ois_to_cts, 'avgs')
    #print(f' ^ Draw Statistics \'avg\' Finished.\n')

    #print(f' v Drawing ...')
    #draw_imagenet_stat(imgs_save_dir, ois_to_cts, 'mins')
    #print(f' ^ Draw Statistics \'min\' Finished.\n')

    #print(f' v Drawing ...')
    #draw_imagenet_stat(imgs_save_dir, ois_to_cts, 'maxs')
    #print(f' ^ Draw Statistics \'max\' Finished.\n')

    #print(f' v Drawing ...')
    #draw_imagenet_stat(imgs_save_dir, ois_to_cts, 'vars')
    #print(f' ^ Draw Statistics \'var\' Finished.\n')

    print(f' v Drawing ...')
    draw_imagenet_specific_stat(imgs_save_dir, ois_to_cts, ois_to_count, top=10)
    print(f' ^ Draw Specific Statistics Finished.\n')

    print(f'[End] All Finished.')


def draw_COCO(extracted_data, combine_type, imgs_save_dir):
    main_results = extracted_data['main_results']
    batch_image_sizes = extracted_data['other_results']['batch_image_sizes']
    #inference_times = extracted_data['other_results']['inference_times']
    #preprocess_times = extracted_data['other_results']['preprocess_times']
    #postprocess_times = extracted_data['other_results']['postprocess_times']

    combined_times = combine_COCO_times(extracted_data['other_results'], combine_type)

    bis_to_cts = dict()
    for batch_image_size, combined_time in zip(batch_image_sizes, combined_times):
        height, width = batch_image_size

        bis = (height, width)
        ct = bis_to_cts.get(bis, list())
        ct.append(combined_time)
        bis_to_cts[bis] = ct

    bis_to_count = dict()
    for bis in  bis_to_cts.keys():
        cts = bis_to_cts.get(bis, list())
        bis = (bis[0], bis[1]) if bis[0] <= bis[1] else (bis[1], bis[0])
        count = bis_to_count.get(bis, 0)
        count = count + len(cts)
        bis_to_count[bis] = count
    bis_to_count = list(bis_to_count.items())
    bis_to_count = sorted(bis_to_count, key=lambda x: x[1])[::-1]

    print(f'[Begin] Drawing ...')

    print(f' v Drawing ...')
    draw_coco_count_per_bis(imgs_save_dir, bis_to_count)
    print(f' ^ Draw Count V.S. Image Size Finished.\n')

    # print(f' v Drawing ...')
    # draw_coco_count_per_nop(imgs_save_dir, bis_to_count)
    # print(f' ^ Draw Count V.S. Number of Pixels Finished.\n')

    print(f' v Drawing ...')
    draw_coco_stat(imgs_save_dir, bis_to_cts, 'avgs')
    print(f' ^ Draw Statistics \'avg\' Finished.\n')

    print(f' v Drawing ...')
    draw_coco_stat(imgs_save_dir, bis_to_cts, 'mins')
    print(f' ^ Draw Statistics \'min\' Finished.\n')

    print(f' v Drawing ...')
    draw_coco_stat(imgs_save_dir, bis_to_cts, 'maxs')
    print(f' ^ Draw Statistics \'max\' Finished.\n')

    print(f' v Drawing ...')
    draw_coco_stat(imgs_save_dir, bis_to_cts, 'vars')
    print(f' ^ Draw Statistics \'var\' Finished.\n')

    print(f' v Drawing ...')
    draw_coco_specific_stat(imgs_save_dir, bis_to_cts, bis_to_count, top=10)
    print(f' ^ Draw Specific Statistics Finished.\n')

    print(f'[End] All Finished.')


def draw(extracted_data, dataset_type, combine_type, imgs_save_dir):
    assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"
    assert combine_type in combine_choices, f"Wrong Type of Combine: {combine_type}"

    draw_by_dst = globals()['draw_' + dataset_type]
    draw_by_dst(extracted_data, combine_type, imgs_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw Figs for Datasets')

    parser.add_argument('-i', '--imgs-save-dir', type=str, required=True)

    parser.add_argument('-d', '--data-dir', type=str, default=None)
    parser.add_argument('-n', '--data-filename', type=str, default=None)
    parser.add_argument('-s', '--save-dir', type=str, default=None)
    parser.add_argument('-f', '--save-filename', type=str, default=None)
    parser.add_argument('-p', '--npz-path', type=str, default=None)
    parser.add_argument('-t', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)
    arguments = parser.parse_args()

    combine_type = arguments.combine_type
    assert combine_type in combine_choices, f"No Such Combine Type: {combine_type}"

    dataset_type = arguments.dataset_type
    assert dataset_type in dataset_choices, f"No Such Dataset Type: {dataset_type}"

    imgs_save_dir = pathlib.Path(arguments.imgs_save_dir)

    if not imgs_save_dir.is_dir():
        print(f"No Imgs Save Dir Exists: {imgs_save_dir}, now creating it.")
        imgs_save_dir.mkdir(parents=True, exist_ok=True)

    if arguments.data_dir is None:
        # Direct Load From NPZ
        if arguments.npz_path is None:
            raise AttributeError("At least one argument of \{--data-dir or --npz-path\} must be specified.")
        else:
            npz_path = pathlib.Path(arguments.npz_path)
            assert npz_path.is_file(), f"No Such NPZ File: {npz_path}"
            extracted_data = numpy.load(npz_path)
    else:
        # Load From Raw Data
        data_dir = pathlib.Path(arguments.data_dir)
        data_filename = arguments.data_filename
        assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"
        assert data_filename is not None, f"While using argument \'--data-dir\', one must specify \'--data-filename\'"

        from extract_data import extract_data
        extracted_data = extract_data(data_dir, data_filename, dataset_type)

        if arguments.save_dir is None:
            print(f"The extracted data will not be saved!\n")
            print(f"If one want to save the extracted data, please specify arguments \'--save-dir\' and \'--save-filename\'")
        else:
            save_dir = pathlib.Path(arguments.save_dir)
            save_filename = arguments.save_filename
            assert save_dir.is_dir(), f"No Such Save Dir: {save_dir}"
            assert save_filename is not None, f"While using argument \'--save-dir\', one must specify \'--save-filename\'"
            save_filepath = save_dir.joinpath(save_filename)
            print(f" + Saving data into \'{save_filepath}.npz\' ...")
            numpy.savez(save_filepath, **extracted_data)
            print(f" - Saved.")

    draw(extracted_data, dataset_type, combine_type, imgs_save_dir)