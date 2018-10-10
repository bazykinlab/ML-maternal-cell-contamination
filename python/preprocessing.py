import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

class VCF:
    def __init__(self, filename, save_vcf=False):
        self.df, self.header = read_vcf(filename, return_header=True)
        
    def process(self, contamination_factor):
        self.df_processed, self.row_idx = process_vcf(self.df, contamination_factor, return_idx=True)

def get_suffixes(data_dir):
    """
    Scan a dir for pairs of vcf files with the naming scheme
    `abortus/justchild.{suffix}.trio.vcf` and lists all suffixes
    """
    suffixes_ab = []
    suffixes_gt = []

    for filename in os.listdir(data_dir):
        if filename.endswith("trio.vcf"):
            if filename.startswith("abortus"):
                suffixes_ab.append(".".join(filename.split(".")[1:]))
            elif filename.startswith("justchild"):
                suffixes_gt.append(".".join(filename.split(".")[1:]))

    assert sorted(suffixes_ab) == sorted(suffixes_gt)
    return suffixes_ab

def read_vcf(vcf, return_header=False):
    """
    Parse a vcf file ignoring the headers and load it into a pandas
    dataframe.
    """
    def get_header_size(vcf):
        header_size = 0
        with open (vcf, 'r') as f:
            while True:
                line = f.readline()
                header_size += 1
                if line.startswith('#CHROM'):
                    break
        return header_size
    
    header_size = get_header_size(vcf)
    
    df = pd.read_csv(vcf, sep='\t', skiprows=header_size-1)

    if return_header:
        with open (vcf, 'r') as f:
            header_lines = f.readlines()[:header_size]

        return df, header_lines

    return df

def index_by_chrom_and_pos(df):
    df.index = pd.MultiIndex.from_arrays(df[['#CHROM', 'POS']].values.T)
    return df

def process_vcf(df, contamination_factor, return_idx=False):
    df = index_by_chrom_and_pos(df)

    def split_info(s): # For splitting 'INFO' field
        info_list = s.split(";")
        info_dict = {pair.split("=")[0]: pair.split("=")[1] for pair in info_list}
        return info_dict
    
    gt_cols = ["GT", "DP", "PL", "AD"]
    gt_dict = {
        '0/0': 0,
        '0/1': 1,
        '1/1': 2
    }

    def split_gt(df, col_names): # Split genotype information
        fmts = df["FORMAT"].unique()
        dfs = []

        for fmt in fmts:
            df0 = df[df['FORMAT'] == fmt].copy()
            fmt_list = fmt.split(':')

            for col_name in col_names:
                for var in gt_cols:
                    assert var in fmt_list
                    var_index = fmt_list.index(var)
                    df0[col_name + "^" + var] = df0[col_name].str.split(":").str[var_index]

            dfs.append(df0)

        df1= pd.concat(dfs, axis=0)
        return df1  

    to_drop = ["INFO", "ID", "QUAL", "FILTER", "FORMAT"]
    field_names = ["#CHROM", "POS", "INFO", "INFO", "ID", "QUAL", "FILTER", "FORMAT", "REF", "ALT"]
    sample_columns = [col_name for col_name in df.columns.values if col_name not in field_names]
    
    info_dicts = df["INFO"].apply(split_info).values

    for field in ['AC', 'AF']:
        df[field] = np.vectorize(lambda x:x[field])(info_dicts)

    df = split_gt(df, sample_columns)
    df.drop(sample_columns + to_drop,
            axis=1, inplace=True)

    df.dropna(axis=0, inplace=True)

    keep_rows = np.ones(df.shape[0])
    # Encoding genotypes as a label (labelling scheme provided by a dictionary)
    for col_name in filter(lambda col: col.endswith("GT"), df.columns.values): # Iterate on genotype columns
        keep_rows = np.logical_and(df[col_name].isin(['0/0', '0/1', '1/1']).values, keep_rows)
        df.replace({col_name: gt_dict}, inplace=True)

    df = df[keep_rows]
        
    # Separating PL (posterior likelihood) scores into their own columns
    for col_name in filter(lambda col: col.endswith("PL"), df.columns.values):
        split_cols = df[col_name].str.split(',', expand=True)
        for i in range(min(split_cols.shape[1], 3)):
            df[col_name + str(i)] = split_cols[i]

        df.drop([col_name], axis=1, inplace=True)

    # Separate AD (allele depth)
    for col_name in filter(lambda col: col.endswith("AD"), df.columns.values):
        split_cols = df[col_name].str.split(',', expand=True)
        for i in range(2):
            df[col_name + str(i)] = split_cols[i]

        df.drop([col_name], axis=1, inplace=True)

    allele_dict = {
        "A": 1,
        "C": 2,
        "G": 3,
        "T": 4
    }

    def label_allele(allele):
        if allele in allele_dict:
            return allele_dict[allele]

        return 0

    for col_name in ["REF", "ALT"]:
        df[col_name] = np.vectorize(label_allele)(df[col_name].values)

    # One-hot encode genotypes
    for col_name in filter(lambda x: x[-2:] == "GT", df.columns.values):
        dummies = pd.get_dummies(df[col_name])
        dummy_col_names = dummies.columns.values
        df[[col_name + "^" + str(x) + "^1H" for x in dummy_col_names]] = dummies
        
    for col_name in filter(lambda col: col[-3:-1] == "PL", df.columns.values):
        df[col_name] = np.log10(df[col_name].values.astype(np.float)+1)

    # df.drop(["#CHROM", "POS"], axis=1, inplace=True)
    df = df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)
    df['contamination'] = np.ones((df.shape[0],))*contamination_factor
    

    if return_idx:
        return df, keep_rows

    return df

def load_suffix(suffix, data_dir):
    contamination_factor = float("0." + suffix.split(".")[1])
    ab = VCF(data_dir + "abortus." + suffix)
    ab.process(contamination_factor)
    gt = VCF(data_dir + "justchild." + suffix)
    gt.process(contamination_factor)
    idx = ab.df_processed.index.intersection(gt.df_processed.index)
    ab.df_processed = ab.df_processed.loc[idx]
    gt.df_processed = gt.df_processed.loc[idx]

    ab.df_processed['justchild^GT'] = gt.df_processed['justchild^GT']
    
    return ab.df_processed

def load_suffixes(data_dir):
    df_cum = pd.DataFrame()

    for suffix in tqdm(get_suffixes(data_dir)):
        df = load_suffix(suffix, data_dir)
        df_cum = df_cum.append(df)

    return df_cum