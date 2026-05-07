# Results Summary

## Seed-Level Results

| mode | seed | status | final_accuracy | final_loss | max_raw_gain | max_applied_gain | min_scale | runtime_seconds | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hc | 0 | completed | 0.078125 | 4.6268 | 62.8828 | 62.8828 | 1 | 16664.9 | runs/pub15k_hc_seed0 |
| hc | 1 | completed | 0.125 | 4.27979 | 3764.66 | 3764.66 | 1 | 15210.8 | runs/pub15k_hc_seed1 |
| hc | 2 | completed | 0.125 | 3.87998 | 740.982 | 740.982 | 1 | 16490.6 | runs/pub15k_hc_seed2 |
| mhc | 0 | completed | 0.109375 | 4.57741 | 605.269 | 1.03069 | 1 | 8620.45 | runs/pub15k_mhc_seed0 |
| mhc | 1 | completed | 0.0625 | 4.62607 | 37.2023 | 1.03003 | 1 | 8618.64 | runs/pub15k_mhc_seed1 |
| mhc | 2 | completed | 0.046875 | 4.69368 | 415.685 | 1.03083 | 1 | 8596.73 | runs/pub15k_mhc_seed2 |
| harm | 0 | completed | 0.046875 | 4.64018 | 653.642 | 12.4039 | 0.330017 | 7841.99 | runs/pub15k_harm_seed0 |
| harm | 1 | completed | 0.125 | 4.44406 | 1.29856e+07 | 12.6605 | 0.105535 | 16713.1 | runs/pub15k_harm_seed1 |
| harm | 2 | completed | 0.046875 | 4.7312 | 1.23776e+08 | 12.3913 | 0.10712 | 7859.22 | runs/pub15k_harm_seed2 |

## Aggregate Task Metrics

| mode | n | final_accuracy_mean | final_accuracy_std | final_accuracy_min | final_accuracy_max | final_loss_mean | final_loss_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hc | 3 | 0.109375 | 0.0220971 | 0.078125 | 0.125 | 4.26219 | 0.30514 |
| mhc | 3 | 0.0729167 | 0.0265574 | 0.046875 | 0.109375 | 4.63239 | 0.0476775 |
| harm | 3 | 0.0729167 | 0.0368285 | 0.046875 | 0.125 | 4.60515 | 0.119811 |

## Aggregate Gain Metrics

| mode | n | max_raw_gain_mean | max_raw_gain_std | max_raw_gain_min | max_raw_gain_max | max_applied_gain_mean | max_applied_gain_std | max_applied_gain_min | max_applied_gain_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hc | 3 | 1522.84 | 1609.2 | 62.8828 | 3764.66 | 1522.84 | 1609.2 | 62.8828 | 3764.66 |
| mhc | 3 | 352.719 | 236.148 | 37.2023 | 605.269 | 1.03051 | 0.000345935 | 1.03003 | 1.03083 |
| harm | 3 | 4.55874e+07 | 5.55412e+07 | 653.642 | 1.23776e+08 | 12.4852 | 0.124034 | 12.3913 | 12.6605 |

## Aggregate Scale and Runtime Metrics

| mode | n | min_scale_mean | min_scale_std | min_scale_min | min_scale_max | mean_scale_mean | floor_hits_mean | floor_hits_max | runtime_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hc | 3 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 16122.1 |
| mhc | 3 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 8611.94 |
| harm | 3 | 0.18089 | 0.10545 | 0.105535 | 0.330017 | 0.429609 | 0 | 0 | 10804.8 |
