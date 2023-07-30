import numpy
import pathlib
import argparse

from calculate_quality import combine_times, calculate_stat
from extract_data import extract_data
from constant import dataset_choices, combine_choices, rm_outs_choices, detect_choices


def remove_outliers(new_times, new_stats, outliers_mode):
    if outliers_mode == 'quantile':
        q1s = new_stats['q1s'].reshape(-1, 1)
        q3s = new_stats['q3s'].reshape(-1, 1)
        iqr = q3s - q1s
        lower_outlier_flags = new_times < (q1s - 1.5 * iqr)
        upper_outlier_flags = (q3s + 1.5 * iqr < new_times)
        outlier_flags = lower_outlier_flags | upper_outlier_flags
    if outliers_mode == 'guassian':
        outlier_flags = numpy.absolute(new_times - new_stats['avgs'].reshape(-1, 1)) > 3 * new_stats['stds'].reshape(-1, 1)
    if outliers_mode == 'median':
        meds = new_stats['meds'].reshape(-1, 1)
        mads = numpy.median(numpy.absolute(new_times - meds), axis=-1)
        outlier_flags = numpy.absolute(new_times - meds) > 3 * mads

    return outlier_flags


def check_minimum_n(times, thresholds=[0.01, 0.01, 0.01, 0.01, 0.01], outliers_mode='quantile', tolerance=10, detect_type='slide', init_num=30):
    instance_number, run_number = times.shape
    total_suf = numpy.array([0 for _ in range(instance_number)])
    fin = numpy.array([False for _ in range(instance_number)])
    minimum_n = numpy.array([run_number for _ in range(instance_number)])
    stay_n = init_num
    stay_times = times[:, :stay_n]
    stay_stats = calculate_stat(stay_times)
    diff_stats = dict()
    fin_nums = list()
    for try_i, new_n in enumerate(range(stay_n+1, run_number+1)):
        if detect_type == 'slide':
            new_times = times[~fin, new_n-stay_n:new_n]
        else:
            new_times = times[~fin, :new_n]
        new_stats = calculate_stat(new_times)
        outlier_flags = remove_outliers(new_times, new_stats, outliers_mode)
        new_stats = dict(
            q1s = list(),
            q3s = list(),
            meds = list(),
            avgs = list(),
            vars = list(),
            #kurts = list(),
            #skews = list(),
        )
        for outlier_flag, new_time in zip(outlier_flags, new_times):
            clean_time = new_time[~outlier_flag]
            new_stat = calculate_stat(clean_time)
            new_stats['q1s'].append(new_stat['q1s'])
            new_stats['q3s'].append(new_stat['q3s'])
            new_stats['meds'].append(new_stat['meds'])
            new_stats['avgs'].append(new_stat['avgs'])
            new_stats['vars'].append(new_stat['vars'])
            #new_stats['kurts'].append(new_stat['kurts'])
            #new_stats['skews'].append(new_stat['skews'])
        new_stats['q1s'] = numpy.array(new_stats['q1s'])
        new_stats['q3s'] = numpy.array(new_stats['q3s'])
        new_stats['meds'] = numpy.array(new_stats['meds'])
        new_stats['avgs'] = numpy.array(new_stats['avgs'])
        new_stats['vars'] = numpy.array(new_stats['vars'])
        #new_stats['kurts'] = numpy.array(new_stats['kurts'])
        #new_stats['skews'] = numpy.array(new_stats['skews'])

        diff_stats['q1s'] = numpy.absolute(new_stats['q1s'] - stay_stats['q1s'][~fin]) / numpy.absolute(stay_stats['q1s'][~fin])
        q1s_suf_flag = diff_stats['q1s'] < thresholds[0]
        q1s_suf_num = numpy.sum(q1s_suf_flag)
        stay_stats['q1s'][~fin] = new_stats['q1s']

        diff_stats['q3s'] = numpy.absolute(new_stats['q3s'] - stay_stats['q3s'][~fin]) / numpy.absolute(stay_stats['q3s'][~fin])
        q3s_suf_flag = diff_stats['q3s'] < thresholds[1]
        q3s_suf_num = numpy.sum(q3s_suf_flag)
        stay_stats['q3s'][~fin] = new_stats['q3s']

        diff_stats['meds'] = numpy.absolute(new_stats['meds'] - stay_stats['meds'][~fin]) / numpy.absolute(stay_stats['meds'][~fin])
        meds_suf_flag = diff_stats['meds'] < thresholds[2]
        meds_suf_num = numpy.sum(meds_suf_flag)
        stay_stats['meds'][~fin] = new_stats['meds']

        diff_stats['avgs'] = numpy.absolute(new_stats['avgs'] - stay_stats['avgs'][~fin]) / numpy.absolute(stay_stats['avgs'][~fin])
        avgs_suf_flag = diff_stats['avgs'] < thresholds[3]
        avgs_suf_num = numpy.sum(avgs_suf_flag)
        stay_stats['avgs'][~fin] = new_stats['avgs']

        diff_stats['vars'] = numpy.absolute(new_stats['vars'] - stay_stats['vars'][~fin]) / numpy.absolute(stay_stats['vars'][~fin])
        vars_suf_flag = diff_stats['vars'] < thresholds[4]
        vars_suf_num = numpy.sum(vars_suf_flag)
        stay_stats['vars'][~fin] = new_stats['vars']
        #diff_stats['kurts'] = numpy.absolute(new_stats['kurts'] - stay_stats['kurts']) / numpy.absolute(stay_stats['kurts'])
        #kurts_suf = numpy.sum(diff_stats['kurts'] < threshold) >= instance_tolerance
        #diff_stats['skews'] = numpy.absolute(new_stats['skews'] - stay_stats['skews']) / numpy.absolute(stay_stats['skews'])
        #skews_suf = numpy.sum(diff_stats['skews'] < threshold) >= instance_tolerance


        suf_flag = q1s_suf_flag & q3s_suf_flag & meds_suf_flag & avgs_suf_flag & vars_suf_flag
        f_flag = numpy.array([False for _ in range(instance_number)])
        f_flag[~fin] = suf_flag
        suf_flag = f_flag

        total_suf = numpy.where(suf_flag & ~fin, total_suf+1, total_suf) # if suf and not fin, then total suf plus 1
        total_suf_flag = total_suf > tolerance
        total_suf_num = numpy.sum(total_suf_flag)
        if total_suf_num != 0:
            minimum_n = numpy.where(total_suf_flag, new_n, minimum_n)
            fin = numpy.where((total_suf_flag | fin)==True, True, False) # if finish or total_suf, then fin set to True

        total_suf = numpy.where(suf_flag & ~fin, total_suf, 0) # if not suf, total_suf set to 0, else keep

        print(f"No.{try_i+1}/{run_number - init_num}. Finish: {numpy.sum(fin)}; This Finish: {total_suf_num}; q1s: {q1s_suf_num}; q3s: {q3s_suf_num}; meds: {meds_suf_num}; avgs: {avgs_suf_num}; vars: {vars_suf_num}.")
        fin_nums.append(numpy.sum(fin))
        #if q1s_suf and q3s_suf and meds_suf and avgs_suf and vars_suf:
        #    total_suf += 1

        #stat_str = f"total_suf: {total_suf} .. "
        #stat_str = ""
        #for k, v in diff_stats.items():
        #    stat_str += f" {k}: {numpy.around(v[fin][:1]*100, 2)}%;"
        #print(stat_str)

        if numpy.sum(~fin) == 0:
            print("All Finished, Stop!")
            break
        #if total_suf > tolerance:
        #    minimum_n = new_n
        #    break

    return minimum_n, numpy.array(fin_nums)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)
    parser.add_argument('-t', '--thresholds', type=float, nargs='+', default=[0.01, 0.01, 0.01, 0.01, 0.01])
    parser.add_argument('-l', '--tolerance', type=int, default=10)
    parser.add_argument('-i', '--init-num', type=int, default=30)

    parser.add_argument('-m', '--detect-type', type=str, default='cumulate', choices=detect_choices)
    parser.add_argument('-s', '--dataset-type', type=str, default='ImageNet', choices=dataset_choices)
    parser.add_argument('-c', '--combine-type', type=str, default='i', choices=combine_choices)
    parser.add_argument('-r', '--rm-outs-type', type=str, default='none', choices=rm_outs_choices)
    arguments = parser.parse_args()

    detect_type = arguments.detect_type
    dataset_type = arguments.dataset_type
    combine_type = arguments.combine_type
    rm_outs_type = arguments.rm_outs_type

    init_num = arguments.init_num
    assert init_num > 0

    thresholds = arguments.thresholds
    assert len(thresholds) == 5

    tolerance = arguments.tolerance

    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"

    extracted_data = extract_data(data_dir, data_filename, dataset_type)
    combined_times = combine_times(extracted_data['other_results'], combine_type)

    minimum_n, fin_nums = check_minimum_n(combined_times, thresholds=thresholds, tolerance=tolerance, detect_type=detect_type, init_num=init_num)
    i95 = numpy.sum((fin_nums < numpy.floor(0.95 * combined_times.shape[0]))) + init_num
    i90 = numpy.sum((fin_nums < numpy.floor(0.90 * combined_times.shape[0]))) + init_num
    i85 = numpy.sum((fin_nums < numpy.floor(0.85 * combined_times.shape[0]))) + init_num
    i80 = numpy.sum((fin_nums < numpy.floor(0.80 * combined_times.shape[0]))) + init_num
    print(f"Need total run(instance): {numpy.sum(minimum_n)}.")
    print(f"Need total run(dataset): {i95} (95%); {i90} (90%); {i85} (85%); {i80} (80%).")