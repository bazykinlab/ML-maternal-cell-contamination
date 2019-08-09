def calculate_contamination(df,
                 sample_name,
                 mother_name,
                 father_name,
                 sample_gq_threshold=30,
                 mother_gq_threshold=30, 
                 father_gq_threshold=30,
                 sample_dp_threshold=10,
                 mother_dp_threshold=10,
                 father_dp_threshold=10,
                 mode='micro',
                 adjust=False,
                 verbose=False):
    
    assert mode in ['micro', 'macro']
    
    counts = []
    mccs = []
    
    for gt in [0, 2]:
        mo_allele = 0 if gt == 0 else 1
        fa_allele = 1 if gt == 0 else 0

        cond = ( (df[mother_name + "^GT"] == gt) &
                 (df[father_name + "^GT"] == 2 - gt) &
                 (df[sample_name + "^GQ"] > sample_gq_threshold) &
                 (df[mother_name + "^GQ"] > mother_gq_threshold) &
                 (df[father_name + "^GQ"] > father_gq_threshold) &
                 (df[sample_name + "^DP"] > sample_dp_threshold) &
                 (df[mother_name + "^DP"] > mother_dp_threshold) &
                 (df[father_name + "^DP"] > father_dp_threshold)
               )
        
        df1 = df[cond].copy()
        counts.append(len(df1))
        
        if adjust:
            motherImpurity = float(df1[mother_name +'^AD'+str(fa_allele)].sum())/df1[mother_name + '^DP'].sum()

        if mode == 'macro':
            mean_mother_allele_share = (df1[sample_name+'^AD{}'.format(mo_allele)]/df1[sample_name+'^DP']).mean()
            
        elif mode == 'micro':
            mean_mother_allele_share = float(df1[sample_name+'^AD{}'.format(mo_allele)].sum())/df1[sample_name + '^DP'].sum()
            
        if adjust:
            mean_mother_allele_share /= 1-2*motherImpurity

        mccs.append(2*mean_mother_allele_share-1)
        
    contamination = (mccs[0]*counts[0] + mccs[1]*counts[1])/sum(counts)
    return contamination
        
    
        