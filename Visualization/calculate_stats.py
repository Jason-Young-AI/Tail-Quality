import scipy
import numpy
import pathlib
import argparse

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import jensenshannon

from calculate_quality import combine_times, calculate_stat
from extract_data import extract_data, get_main_record
from constant import dataset_choices, combine_choices, rm_outs_choices, detect_choices, level_choices, fit_choices


def remove_outliers(new_times, new_stats, outliers_mode):
    instance_number, run_number = new_times.shape
    outlier_flags = numpy.array([[False for _ in range(run_number)] for i in range(instance_number)])
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


def gmm_aic(n_components, ins_times):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(ins_times)
    return gmm.aic(ins_times)


def kde_aic(bandwidth, ins_times):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(ins_times)
    log_likelihood = kde.score(ins_times)
    num_params = 2  # KDE has two parameters: bandwidth and kernel
    num_samples = ins_times.shape[0]
    return -2 * log_likelihood + 2 * num_params + (2 * num_params * (num_params + 1)) / (num_samples - num_params - 1)


def fit(ins_times, fit_type='kde'):
    ins_times = ins_times.reshape(-1, 1)
    if fit_type == 'kde':
        bandwidth_grid = [0.005, 0.01, 0.03, 0.07, 0.1]
        best_bandwidth  = min(bandwidth_grid, key=lambda x: kde_aic(x, ins_times))
        model = KernelDensity(bandwidth=best_bandwidth).fit(ins_times)
    if fit_type == 'gmm':
        n_components_grid = [2, 3, 4, 5, 6]
        best_n_components = min(n_components_grid, key=lambda x: gmm_aic(x, ins_times))
        model = GaussianMixture(n_components=best_n_components).fit(ins_times)
    return model


def check_all_jsdis(check_model, models, ins_times):
    js_dis = list()
    x = numpy.linspace(ins_times.min(), ins_times.max(), 1000).reshape(-1, 1)
    for model in models:
        js_dis.append(jensenshannon(numpy.exp(check_model.score_samples(x)), numpy.exp(model.score_samples(x))))

    if len(js_dis) == 0:
        js_dis = [1,]

    return js_dis


def check_minimum_n(times, thresholds=[0.01, 0.01, 0.01, 0.01, 0.01], outliers_mode='quantile', tolerance=10, detect_type='slide', init_num=30, using_stat=False, fit_type='kde'):
    instance_number, run_number = times.shape
    total_suf = numpy.array([0 for _ in range(instance_number)])
    fin = numpy.array([False for _ in range(instance_number)])
    minimum_n = numpy.array([run_number for _ in range(instance_number)])
    stay_n = init_num
    stay_times = times[:, :stay_n]
    if using_stat:
        stay_stats = calculate_stat(stay_times)
        diff_stats = dict()
    else:
        all_models = [list() for _ in range(instance_number)]
        jsdiss = list()
    fin_nums = list()
    for try_i, new_n in enumerate(range(stay_n+1, run_number+1)):
        if detect_type == 'slide':
            new_times = times[~fin, new_n-stay_n:new_n]
        else:
            new_times = times[~fin, :new_n]
        
        outlier_flags = remove_outliers(new_times, calculate_stat(new_times), outliers_mode)
        if using_stat:
            new_stats = dict(
                q1s = list(),
                q3s = list(),
                meds = list(),
                avgs = list(),
                vars = list(),
                #kurts = list(),
                #skews = list(),
            )
        else:
            temp_models = list()
            for f_id, f in enumerate(fin):
                if not f:
                    temp_models.append(all_models[f_id])
            assert len(temp_models) == new_times.shape[0]
        for id, (outlier_flag, new_time, models) in enumerate(zip(outlier_flags, new_times, temp_models)):
            clean_time = new_time[~outlier_flag]
            if using_stat:
                new_stat = calculate_stat(clean_time)
                new_stats['q1s'].append(new_stat['q1s'])
                new_stats['q3s'].append(new_stat['q3s'])
                new_stats['meds'].append(new_stat['meds'])
                new_stats['avgs'].append(new_stat['avgs'])
                new_stats['vars'].append(new_stat['vars'])
                #new_stats['kurts'].append(new_stat['kurts'])
                #new_stats['skews'].append(new_stat['skews'])
            else:
                new_model = fit(clean_time, fit_type=fit_type)
                if len(models) == tolerance:
                    jsdis = numpy.array(check_all_jsdis(new_model, models, clean_time))
                    jsdiss.append(jsdis)
                    models.append(new_model)
                    #if id % 50 == 0:
                    #    print(f"   Fitted {id}")
                    #    #print(f"   Fitted {id} {jsdis}")
                else:
                    jsdis = numpy.array([1, ])
                    jsdiss.append(jsdis)
                    models.append(new_model)
                    #if id % 50 == 0:
                    #    print(f"   Added {id}")

        if using_stat:
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
        else:
            for models in temp_models:
                if len(models) > tolerance:
                    models.pop(0)

            jsdiss = numpy.array(jsdiss)
            js_suf_flag = numpy.sum(jsdiss > thresholds[0], axis=-1) == 0
            js_suf_num = numpy.sum(js_suf_flag)
            suf_flag = js_suf_flag

            temp_i = 0
            for f_id, f in enumerate(fin):
                if not f:
                    all_models[f_id] = temp_models[temp_i]
                    temp_i += 1
            assert temp_i == len(temp_models)

        f_flag = numpy.array([False for _ in range(instance_number)])
        f_flag[~fin] = suf_flag
        suf_flag = f_flag

        total_suf = numpy.where(suf_flag & ~fin, total_suf+1, total_suf) # if suf and not fin, then total suf plus 1
        if using_stat:
            total_suf_flag = total_suf > tolerance
        else:
            total_suf_flag = total_suf > 0
        total_suf_num = numpy.sum(total_suf_flag)
        if total_suf_num != 0:
            minimum_n = numpy.where(total_suf_flag, new_n, minimum_n)
            fin = numpy.where((total_suf_flag | fin)==True, True, False) # if finish or total_suf, then fin set to True

        total_suf = numpy.where(suf_flag & ~fin, total_suf, 0) # if not suf, total_suf set to 0, else keep

        if using_stat:
            print(f" - No.{try_i+1}/{run_number - init_num}. Finish: {numpy.sum(fin)}/{len(fin)}; This Finish: {total_suf_num}; q1s: {q1s_suf_num}; q3s: {q3s_suf_num}; meds: {meds_suf_num}; avgs: {avgs_suf_num}; vars: {vars_suf_num}.")
        else:
            print(f" - No.{try_i+1}/{run_number - init_num}. Finish: {numpy.sum(fin)}/{len(fin)}; This Finish: {total_suf_num}; jsdis: {js_suf_num}.")
            jsdiss = list()
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


def check_common_dist(times, minimum_n, outliers_mode='quantile'):
    np_fit = list()
    print(f"Checking for distribution: 'norm' ...")
    for index, (ins_times, ins_n) in enumerate(zip(times, minimum_n)):
        # ins_time: [run_number]
        r_times = ins_times[:ins_n].reshape(1, -1) # r_times: [1, run_number]
        outlier_flags = remove_outliers(r_times, calculate_stat(r_times), outliers_mode)

        # p - 0.05
        sw_result = scipy.stats.shapiro(r_times[~outlier_flags])
        sw_reject = sw_result.pvalue < 0.05

        nt_result = scipy.stats.normaltest(r_times[~outlier_flags])
        nt_reject = nt_result.pvalue < 0.05

        ad_result = scipy.stats.anderson(r_times[~outlier_flags])
        sig = ad_result.significance_level[2] 
        cri = ad_result.critical_values[2]
        ad_reject = sig > cri
        # Why choose ad_test or sw_test? 
        # The Anderson-Darling (A-D) test is a variation on the K-S test, but gives more weight to the tails of the distribution.
        # The K-S test is more sensitive to differences that may occur closer to the center of the distribution, while the A-D test is more sensitive to variations observed in the tails.
        # the Shapiro Wilk generally performs very well in a wide variety of situations, including some that "sensitivity to tails".
        # Apart perhaps from the Anderson-Darling, which might beat it on tail sensitivity in some cases.
        # see the discussion in the book "Goodness of fit techniques" by D'Agostino and Stephens.
        # 1. https://stats.stackexchange.com/questions/77855/is-shapiro-wilk-test-insensitive-on-the-tails
        # 2. Goodness of fit techniques - D'Agostino and Stephens
        # Maybe: D'Agostino's K^2 test is less sensitive to the tails of the distribution.
        if (not sw_reject) or (not ad_reject):
            print(f" - No.{index+1}/{len(minimum_n)} instance (within {ins_n} run) can not reject H0:")
            print(f"      Shapiro-Wilk: {'No' if sw_reject else 'Yes'} ({sw_result.pvalue})")
            print(f"      D'Agostino's: {'No' if nt_reject else 'Yes'} ({nt_result.pvalue})")
            print(f"      Anderson-Dar: {'No' if ad_reject else 'Yes'} (Sig - Cri: {sig} - {cri})")
            if sw_reject or ad_reject:
                print(f"   But may still use np method.")
        if sw_reject or ad_reject:
            np_fit.append(True)
        else:
            print(f" - [Use NP] No.{index+1}/{len(minimum_n)} instance (within {ins_n} run)")

    return np_fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Quality')
    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-n', '--data-filename', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, required=True)

    parser.add_argument('-a', '--using-stat', action='store_true')

    parser.add_argument('-t', '--thresholds', type=float, nargs='+', default=[0.01, 0.01, 0.01, 0.01, 0.05])
    parser.add_argument('-l', '--tolerance', type=int, default=4)
    parser.add_argument('-i', '--init-num', type=int, default=30)
    parser.add_argument('-u', '--check-npz-path', type=str, default=None)

    parser.add_argument('-k', '--check-min-n', action='store_true')
    parser.add_argument('-e', '--n-level', type=str, default='specific', choices=level_choices)

    parser.add_argument('-f', '--fit-type', type=str, default='kde', choices=fit_choices)

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
    assert len(thresholds) == 5 or len(thresholds) == 1

    tolerance = arguments.tolerance

    data_dir = pathlib.Path(arguments.data_dir)
    data_filename = arguments.data_filename
    assert data_dir.is_dir(), f"No Such Data Dir: {data_dir}"

    extracted_data = extract_data(data_dir, data_filename, dataset_type)
    combined_times = combine_times(extracted_data['other_results'], combine_type)

    combined_times = get_main_record(combined_times, arguments.batch_size)

    #for ct in combined_times[4823]:
    #    print(f"{ct}\t")

    instance_number, run_number = combined_times.shape
    minimum_n = numpy.array([run_number for _ in range(instance_number)])
    dataset_minimum_n = dict(
        i100 = run_number,
        i95 = run_number,
        i90 = run_number,
        i85 = run_number,
        i80 = run_number,
    )
    if arguments.check_min_n:
        minimum_n, fin_nums = check_minimum_n(combined_times, thresholds=thresholds, tolerance=tolerance, detect_type=detect_type, init_num=init_num, using_stat=arguments.using_stat, fit_type=arguments.fit_type)
        i95 = numpy.sum((fin_nums < numpy.floor(0.95 * combined_times.shape[0]))) + init_num
        i90 = numpy.sum((fin_nums < numpy.floor(0.90 * combined_times.shape[0]))) + init_num
        i85 = numpy.sum((fin_nums < numpy.floor(0.85 * combined_times.shape[0]))) + init_num
        i80 = numpy.sum((fin_nums < numpy.floor(0.80 * combined_times.shape[0]))) + init_num
        dataset_minimum_n['i95'] = i95
        dataset_minimum_n['i90'] = i90
        dataset_minimum_n['i85'] = i85
        dataset_minimum_n['i80'] = i80
        print(f"Need total run(instance): {numpy.sum(minimum_n)}.")
        print(f"Need total run(dataset): {i95} (95%); {i90} (90%); {i85} (85%); {i80} (80%).")

    if arguments.n_level == 'specific':
        min_ns = minimum_n
    else:
        min_ns = numpy.array([dataset_minimum_n[arguments.n_level] for _ in range(combined_times.shape[0])])

    np_fit = check_common_dist(combined_times, min_ns, outliers_mode='none')

    check_data = dict(
        min_ns = minimum_n,
        np_fit = np_fit,
        **dataset_minimum_n,
    )

    if arguments.check_npz_path is not None:
        check_npz_path = pathlib.Path(arguments.check_npz_path)
        check_npz_path = check_npz_path.with_suffix('.npz')
        print(f" + Saving check results into \'{check_npz_path}' ...")
        numpy.savez(check_npz_path, **check_data)
        print(f" - Saved.")