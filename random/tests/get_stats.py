#!/usr/bin/env python

'''
This script extracts information from the output printed in a file by main_explain.py. It converts that information into a pandas dataframe and writes it to a csv file.
'''

import sys
if len(sys.argv) != 3:
    raise("Usage: ./get_stats.py  path_to_input_file  path_to_output_file")
path = sys.argv[1]
out = sys.argv[2]

data = list()
with open(path, 'r') as file:
    lines = file.readlines()
    for i in range(len(lines)):
        if 'Epoch: 0500' in lines[i]:
            data.append((int(lines[i+3].strip()[11]), int(lines[i+3].strip()[-1])))
            if int(lines[i+3].strip()[11]) == 1:
                print((int(lines[i+3].strip()[11]), int(lines[i+3].strip()[-1])))

# from pprint import pprint
# pprint(data)

import pandas as pd
df = pd.DataFrame(data, columns=['original', 'new'])
# print(df.head())
df.to_csv(out)