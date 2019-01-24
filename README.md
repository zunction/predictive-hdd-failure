# Predictive HDD Failure

We use [Backblaze](https://www.backblaze.com/) HDD (hard disk drive) data dataset and XGBoost to create a model that can predict (two days in advance) HDD failure using past `n` days record of [SMART](https://en.wikipedia.org/wiki/S.M.A.R.T.) data. For demonstration purposes, we take `n=7`.

We have two main categories of data in our processed dataset:

* "normal": `n` days worth of SMART data for a HDD functioning normally two days later
* "failure": `n` days worth of SMART data for a HDD which fails two days later

Both categories also include the HDD model and capacity as additional features.

To find out more about the dataset, please visit the [dataset webpage](https://www.backblaze.com/b2/hard-drive-test-data.html).

## Results

We obtain reasonable results on the evaluation set, consisting of 4208 cases of "normal" drives and 319 cases of "failure" drives. The evaluation data is dated from **October to December 2018**, and takes place directly after the training set which contains data from **January 2015 to September 2018**. We achieve **61% recall** in correctly predicting drives that are about to fail in the evaluation set. This means that:

* 39% false negative (drive is going to fail but we predict normal operation)
* 3% false positive (drive is normal but we predict failure)

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
# download repository
git clone https://github.com/NVAITC/predictive-hdd-failure
cd predictive-hdd-failure

# download processed data
wget https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/train.csv
wget https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/eval.csv

# download and run container
docker pull nvaitc/ai-lab:latest
nvidia-docker run --rm -p 8888:8888 -v /home/$USER/predictive-hdd-failure:/home/jovyan nvaitc/ai-lab
```

Please note that you will require the nvidia-driver>=396 and nvidia-docker2 runtime on the host machine. [Additional Instructions](https://github.com/NVAITC/ai-lab/blob/master/INSTRUCTIONS.md)

**Processed Data**

The training set `train.csv` consists of all the drive failures from 2015Q1-2018Q3, and a subset (about 40 each day) of the data for working drives, sampled at intervals (effectively random since drive counts change every day).

The evaluation set `eval.csv` consists of all the drive failures in 2018Q4, and a subset (about 100 each day) of the data for working drives.

* Training Set: https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/train.csv
* Evaluation Set: https://s3-ap-southeast-1.amazonaws.com/deeplearning-iap-material/hdd_test_data/eval.csv
