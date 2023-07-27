quality_choices = ['acc', 'map', 'wf1']
dataset_choices = ['ImageNet', 'COCO', 'MMLU', 'MELD']
combine_choices = ['i', 'ip', 'pi', 'pip'] # inf, inf->post, pre->inf, pre->inf->post
rm_outs_choices = ['quantile', 'gaussian']
stat_map = dict(
    mins = 'Minimum',
    maxs = 'Maximum',
    meds = 'Median',
    avgs = 'Average',
    vars = 'Variation',
    stds = 'Standard Deviation',
)