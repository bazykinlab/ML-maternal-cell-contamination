# ML-maternal-cell-contamination

This repository contains accompanying code for the paper [Accurate Fetal Variant Calling in the Presence of Maternal Cell Contamination](https://www.biorxiv.org/content/10.1101/552414v1). It consists of three parts:

1. Utilities for:
	* Working with VCF files
	* Using a pretrained model to recalibrate the genotype (GATK output) of a contaminated sample
	* Training your own model on synthetic data

2. Notebooks to reproduce our results from the corresponding paper, as well as an example notebook with a demonstration of the usage of our scripts. See `notebooks/example.ipynb`.

3. A CLI script for genotype recalibration. See `python/main.py`.

```
usage: main.py [-h] [-v] [-r] [-c] [--model_path MODEL_PATH]
               [--method {xgboost,logistic-regression,confidence-intervals,meta-classifier}]
               [--contamination CONTAMINATION] [--output OUTPUT]
               [--GQ_sa GQ_SA] [--GQ_mo GQ_MO] [--GQ_fa GQ_FA] [--DP_sa DP_SA]
               [--DP_mo DP_MO] [--DP_fa DP_FA] [--mode {micro,macro}]
               path sample_name mother_name father_name

positional arguments:
  path                  path to input VCF file
  sample_name           name of sample column in VCF file
  mother_name           name of mother column in VCF file
  father_name           name of father column in VCF file

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         enable verbose output
  -r, --recalibrate     recalibrate a VCF file
  -c, --estimate_contamination
                        estimate and output the fraction of maternal cell contamination
                        given a VCF file
  --model_path MODEL_PATH
                        path to saved recalibrator model
  --method {xgboost,logistic-regression,confidence-intervals,meta-classifier}
                        recalibration method
  --contamination CONTAMINATION
                        user provided contamination estimate (calculated otherwise)
  --output OUTPUT       path to output VCF file
  --GQ_sa GQ_SA         Genotype quality cutoff to use when estimating
                        contamination (sample column)
  --GQ_mo GQ_MO         Genotype quality cutoff to use when estimating
                        contamination (mother column)
  --GQ_fa GQ_FA         Genotype quality cutoff to use when estimating
                        contamination (father column)
  --DP_sa DP_SA         Read depth cutoff to use when estimating contamination
                        (sample column)
  --DP_mo DP_MO         Read depth cutoff to use when estimating contamination
                        (mother column)
  --DP_fa DP_FA         Read depth cutoff to use when estimating contamination
                        (father column)
  --mode {micro,macro}  Averaging mode for contamination estimation

```