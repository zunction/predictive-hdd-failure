# Predictive HDD Failure

We use [Backblaze](https://www.backblaze.com/) HDD data dataset and XGBoost to create a model that can predict (2 days in advance) HDD failure using past `n` days record of SMART data. For demonstration purposes, we take `n=7`.

More information:

* Dataset Documentation: https://www.backblaze.com/b2/hard-drive-test-data.html
* [What is SMART data?](https://www.hdsentinel.com/smart/index.php)

We have two main categories of data in our processed dataset:

* "normal": `n` days worth of SMART data for a HDD functioning normally two days later
* "failure": `n` days worth of SMART data for a HDD which fails two days later

Both categories also include the HDD model and capacity as additional features.

## Results

We obtain reasonable results on the evaluation set, consisting of 4208 "normal" drives and 319 "failure" drives.

* 61% precision and recall in predicting drives that are about to fail in the evaluation set
* 3% false positive (drive is normal but we predict failure)
* 39% false negative (drive is going to fail but we predict normal operation)

These data are from **October to December 2018**, and takes place directly after the training set which contains data from **January 2015 to September 2018**.

![](images/cnf_matrix.png)

With hyper-parameter tuning and feature engineering, it is likely we can do better.

## Running the Code

**Notebooks**

You need to download the `train.csv` and `eval.csv` files in order to run the main `drive_data_xgboost` notebook. Please view the next section for more details.

* `drive_data_xgboost` contains the model training with some elaboration and evaluation
* `processing_okfail` generates the combined train/eval CSV files from the preprocessed data
* `preprocess_data` performs data cleaning and dropping of some columns from the raw data from BackBlaze

To run `drive_data_xgboost`, you will need to have cuDF and RAPIDS installed. The easiest way to ensure this is to use the `nvaitc/ai-lab` container.

```
docker pull nvaitc/ai-lab:latest
nvidia-docker run --rm -p 8888:8888 -v /home/$USER/predictive-hdd-failure:/home/jovyan nvaitc/ai-lab
```

Please note that you will require the nvidia-driver>=396 and nvidia-docker2 runtime on the host machine. [Additional Instructions](https://github.com/NVAITC/ai-lab/blob/master/INSTRUCTIONS.md)

**Processed Data**

The training set `train.csv` consists of all the drive failures from 2015Q1-2018Q3, and a subset (40 each day) of the data for working drives, sampled at intervals (effectively random since drive counts change every day).

The evaluation set `eval.csv` consists of all the drive failures in 2018Q4, and a subset (100 each day) of the data for working drives.

* Training Set: https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/train.csv
* Evaluation Set: https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/eval.csv