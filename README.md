# Classification of Long Sequential Data using Circular Dilated Convolutional Neural Networks

arXiv preprint: [https://arxiv.org/abs/2201.02143](https://arxiv.org/abs/2201.02143). 

## **Architecture**
CDIL-CNN is a novel convolutional model for sequence classification. We use symmetric dilated convolutions, a circular mixing protocol, and an average ensemble learning.

#### Symmetric Dilated Convolutions
<p align="left">
<img src="figures/dil.png" width="600">
</p>


#### Circular Mixing
<p align="left">
<img src="figures/cir1.png" width="150">
<img src="figures/cir2.png" width="150">
<img src="figures/cir3.png" width="150">
</p>


#### CDIL-CNN
<p align="left">
<img src="figures/cdil.png" width="300">
</p>




## **Experiments**

### Synthetic Task
To reproduce the synthetic data experiment results, you should:
1. Run ***syn_data_generation.py***;
2. Run ***syn_main.py*** for one experiment or run ***syn_all.sh*** for all experiments.

The generator will create 6 files for each sequence length and store them in the **./syn_datasets/** folder in the following format:
`adding2000_{length}_train.pt`
`adding2000_{length}_train_target.pt`
`adding2000_{length}_test.pt`
`adding2000_{length}_test_target.pt`
`adding2000_{length}_val.pt`
`adding2000_{length}_val_target.pt`

By default, it iterates over 8 sequence lengths: `[2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14]`.

You can run different models for different lengths. The **./syn_log/** folder will save all results.

We provide our used configurations in ***syn_config.py***.


### Long Range Arena
Long Range Arena (LRA) is a public benchmark suite. The datasets and the download link can be found in [the official GitHub repository](https://github.com/google-research/long-range-arena). 

To reproduce the LRA experiment results, you should:
1. Download `lra_release.gz` (~7.7 GB), extract it, move the folder `./lra_release/lra_release` into our **./create_datasets/** folder, and run ***all_create_datasets.sh***. 
2. Run ***lra_main.py*** for one experiment or run ***lra_all.sh*** for all experiments.

The dataset creators will create 3 files for each task and store them in the **./lra_datasets/** folder in the following format:
`{task}.train.pickle`
`{task}.test.pickle`
`{task}.dev.pickle`

You can run different models on different tasks. The **./lra_log/** folder will save all results.

We provide our used configurations in ***lra_config.py***.


### Time Series
The [UEA & UCR Repository](http://www.timeseriesclassification.com/) consists of various time series classification datasets. We use three audio datasets: [FruitFlies](http://www.timeseriesclassification.com/description.php?Dataset=FruitFlies), [RightWhaleCalls](http://www.timeseriesclassification.com/description.php?Dataset=RightWhaleCalls), and [MosquitoSound](http://www.timeseriesclassification.com/description.php?Dataset=MosquitoSound).

To reproduce the time series results, you should:
1. Download the datasets, extract them, move the extracted folders into our **./time_datasets/** folder, and run ***time_arff_generation.py***. 
2. Run ***time_main.py*** for one experiment or run ***time_all.sh*** for all experiments.

The generator will create 2 files for each dataset and store them in the **./time_datasets/** folder in the following format:
`{dataset}_train.csv`
`{dataset}_test.csv`

You can run different models on different datasets. The **./time_log/** folder will save all results.

We provide our used configurations in ***time_main.py***.

