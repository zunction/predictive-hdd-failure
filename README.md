# Predictive HDD Failure

Use Backblaze HDD SMART data and XGBoost to create a model that can predict (2 days in advance) HDD failure using past 7 days record of SMART data.

* Raw Dataset: https://www.backblaze.com/b2/hard-drive-test-data.html
* [What is SMART data?](https://www.hdsentinel.com/smart/index.php)

We have two main categories of data in our pre-processed dataset:

* "normal": 7 days worth of SMART data, HDD is functioning normally two days from now
* "failure": 7 days worth of SMART data, HDD fails two days from now

Both categories also include the HDD model and capacity as additional features.

## Results

The evaluation set consists of 4208 "normal" drives and 319 "failure" drives. These data are from **October to December 2018**, and takes place directly after the training set which contains data from **January 2015 to September 2018**.

* 61% precision and recall in predicting drives that are about to fail in the evaluation set
* 3% false positive (drive is normal but we predict failure)
* 39% false negative (drive is going to fail but we predict normal operation)

With more hyper-parameter tuning and feature engineering, it is likely we can do better.

## Files

**Notebooks**

* `drive_data_xgboost` contains the model with some elaboration and evaluation
* `processing_okfail` generates the combined train/eval CSV files from the preprocessed data
* `preprocess_data` performs data cleaning and dropping of some columns from the raw data from BackBlaze

**Pre-processed Data**

You need to download the `train.csv` and `eval.csv` files in order to run the main `drive_data_xgboost` notebook.

The training set `train.csv` consists of all the drive failures from 2015Q1-2018Q3, and a subset (40 each day) of the data for working drives, sampled at intervals (effectively random since drive counts change every day).

The evaluation set `eval.csv` consists of all the drive failures in 2018Q4, and a subset (100 each day) of the data for working drives.

* Training Set: https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/train.csv
* Evaluation Set: https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/eval.csv