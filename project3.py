# -*- coding: utf-8 -*-
# Example main.py
from smart_fun import *
import argparse                                   
import json
import sys
import pandas as pd


def main(args):

    if(len(sys.argv) > 1):                   # Identifies the file name from arguments
        file_name = sys.argv[2]
    #print(n)

    f_name, content_list = extract_input(file_name)

    normalized_corpus = normalize_corpus(content_list)

    cluster_id = predicting_cluster(normalized_corpus)

    result = OrderedDict()
    result['city'] = ",".join(f_name.split(' '))
    result['raw_text'] = content_list[0]
    result['clean_text'] = normalized_corpus[0]
    result['cluster_id'] = cluster_id[0]

    df = pd.DataFrame(result, index=range(1))

    df.to_csv('smartcity_predict.tsv', sep='\t')

    print(f_name+" clusterid: "+str(cluster_id[0]))
    
   
if __name__ == '__main__':                  
    
    parser = argparse.ArgumentParser(description='Provide Document Details!')   # Argument Parser
    parser.add_argument('--document', type=str, help='Document to be predicted!', required=True)     # Specifying filters to be detected in arguments
    args = parser.parse_args()
    
    if args.document:                  # Validating if --input exists
        main(args)
    else:
        print("Please specify document to be predicted --document flag!")         #Error message on missing --document filter




    