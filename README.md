# Predictive HDD Failure


Use Backblaze HDD SMART data and XGBoost to create a model that can predict (2 days in advance) HDD failure using past 7 days record of SMART data.

Raw Dataset: https://www.backblaze.com/b2/hard-drive-test-data.html

**Pre-processed Data**

The training set consists of all the drive failures from 2015Q1-2018Q3, and a subset (40 each day) of the data for working drives, sampled at intervals (effectively random since drive counts change every day).

The evalutation set consists of all the drive failures in 2018Q4, and a subset (100 each day) of the data for working drives.

* Training Set: https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/train.csv
* Evaluation Set: https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/eval.csv