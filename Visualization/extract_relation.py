import os
import re
import json
import numpy
import pathlib
import argparse

import matplotlib.pyplot as plt

relation_choices = ['ST', 'ML', 'DL']
dataset_choices = ['ImageNet', 'MS-COCO']


def filename_match_pattern(filename, pattern):
    regex = re.compile(pattern)
    return bool(regex.match(filename))


def extract_ST(results_dir, output_dir, dataset_type='ImageNet'):
    results_files = list()
    perform_files = list()
    results_pattern = r'\w+\.main\.\d+'
    perform_pattern = r'\w+\.time\.\d+'
    for filename in os.listdir(results_dir):
        path = os.path.join(results_dir, filename)
        if os.path.isfile(path):
            if filename_match_pattern(filename, results_pattern):
                results_files.append(path)
            if filename_match_pattern(filename, perform_pattern):
                perform_files.append(path)

    if len(results_files) == 0 or len(perform_files) == 0:
        return None

    if dataset_type in {'ImageNet', 'MS-COCO'}:
        image_ids = list()
        origin_image_sizes = list()
        batch_image_sizes = list()
        inference_times = list()
        pre_times = list()
        post_times = list()
        assert dataset_type in dataset_choices, f"Wrong Type of Dataset: {dataset_type}"
        results = json.load(open(results_files[0]))
        for instance in results:
            if dataset_type == 'MS-COCO':
                image_ids.append(instance['result'][0]['image_id'])
                batch_image_sizes.append(instance['batch_image_size'])
            else:
                image_ids.append(instance['image_id'])
            origin_image_sizes.append(instance['origin_image_size'])

        for perform_file in perform_files:
            perform = json.load(open(perform_file))
            inference_times_per = list()
            pre_times_per = list()
            post_times_per = list()
            for instance in perform:
                inference_times_per.append(instance['inference_time'])
                pre_times_per.append(instance['preprocess_time'])
                post_times_per.append(instance['postprocess_time'])
            inference_times.append(inference_times_per)
            pre_times.append(pre_times_per)
            post_times.append(post_times_per)

        inference_times = numpy.array(inference_times).transpose()
        pre_times = numpy.array(pre_times).transpose()
        post_times = numpy.array(post_times).transpose()

        # 1. For each kind of image shape (e.g., [H, W]) that contains N different images, plot the box plot.
        #  - Sorted by image shape.

        origin_sorted_shapes = sorted(enumerate(origin_image_sizes), key=lambda x: (x[1][0], x[1][1]) if x[1][0] < x[1][1] else (x[1][1], x[1][0]))
        heights = list()
        widths = list()
        avg_times = list()
        max_times = list()
        min_times = list()
        for index, shape in origin_sorted_shapes:
            h, w = shape
            if h > w:
                h, w = w, h
            avg_inf_t = numpy.average(inference_times[index])
            max_inf_t = numpy.max(inference_times[index])
            min_inf_t = numpy.min(inference_times[index])
            heights.append(h)
            widths.append(w)
            avg_times.append(avg_inf_t)
            max_times.append(max_inf_t)
            min_times.append(min_inf_t)

        numpy.savez(os.path.join(output_dir, 'origin_shape_perf.npz'), heights=numpy.array(heights), widths=numpy.array(widths), avg_times=numpy.array(avg_times), max_times=numpy.array(max_times), min_times=numpy.array(min_times))

        if dataset_type == 'MS-COCO':
            batch_sorted_shapes = sorted(enumerate(batch_image_sizes), key=lambda x: (x[1][0], x[1][1]) if x[1][0] < x[1][1] else (x[1][1], x[1][0]))
            heights = list()
            widths = list()
            avg_times = list()
            max_times = list()
            min_times = list()
            for index, shape in batch_sorted_shapes:
                h, w = shape
                if h > w:
                    h, w = w, h
                avg_inf_t = numpy.average(inference_times[index])
                max_inf_t = numpy.max(inference_times[index])
                min_inf_t = numpy.min(inference_times[index])
                heights.append(h)
                widths.append(w)
                avg_times.append(avg_inf_t)
                max_times.append(max_inf_t)
                min_times.append(min_inf_t)

            numpy.savez(os.path.join(output_dir, 'batch_shape_perf.npz'), heights=numpy.array(heights), widths=numpy.array(widths), avg_times=numpy.array(avg_times), max_times=numpy.array(max_times), min_times=numpy.array(min_times))

        #xs = list()
        #ys = list()
        #zs = list()
        #with open(os.path.join(output_dir, 'shape_perf'), 'w') as f:
        #    for shape, avg_inf_t in shape_perf:
        #        x, y = shape
        #        xs.append(x)
        #        ys.append(y)
        #        zs.append(avg_inf_t)
        #        f.write(f"{shape}: {avg_inf_t:.5f}\n")
        
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

        #xs = numpy.array(xs)
        #ys = numpy.array(ys)
        #zs = numpy.array(zs)
        ## 绘制散点图
        #ax.scatter(xs, ys, zs, c='r', marker='o')

        ## 设置坐标轴标签
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        #plt.savefig('temp.pdf')


    return None


def extract_ML(results_dir):
    pass


def extract_DL(results_dir):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-d', '--results-dir', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-l', '--relation-type', type=str, default='ST', choices=relation_choices)
    parser.add_argument('-s', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    arguments = parser.parse_args()

    results_dir = arguments.results_dir
    output_dir = arguments.output_dir
    relation_type = arguments.relation_type
    dataset_type = arguments.dataset_type

    assert pathlib.Path(results_dir).is_dir(), f"No Such Results Dir: {results_dir}"
    assert relation_type in relation_choices, f"No Such Results File: {relation_type}"
    assert dataset_type in dataset_choices, f"No Such Results File: {dataset_type}"

    if not pathlib.Path(output_dir).is_dir():
        print(f"No Output Dir Exists: {output_dir}, now creating it.")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    relation_extract = globals()['extract_' + arguments.relation_type]

    relation = relation_extract(results_dir, output_dir, dataset_type)

    if relation_type == 'ST' and dataset_type == 'ImageNet':
        pass