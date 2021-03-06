# -*- coding: utf-8 -*-
"""
Data Imports for the Report Generator

Create DataFrames from Model Run results

Created on Fri Mar 11 15:15:15 2016

@author: sjsmith

(c) The Haskell Company
"""
# Module imports
import pandas as pd
import numpy as np
import os
import pickle


def find_scenario_and_replication(x):
    '''
    Extract t he scenario and replication number from the filename string
    '''
    s = x[x.find("_s") + 2:x.find("_r")]
    r = x[x.find("_r") + 2:x.find(".")]

    #Check if what was found is a valid integer, if not, set equal to 1
    if not s.isdigit():
        s = 1
    if not r.isdigit():
        r = 1
    s = int(s)
    r = int(r)
    return s, r

def d3d_settings(output_id,
                  output_path):
    '''
    a
    '''

    # Initialize output_data dataframe
    output_data = pd.DataFrame({'':[]})


    print(f'    Loading {output_id} from xls...')
    # Read and process the model output
    for fn in os.listdir(output_path):

        fn_beginning = fn.split('_')[0] # Grabs the scenario name

        if fn_beginning.find(output_id) != -1:

            temp = pd.read_csv(''.join([output_path, '\\', fn]))


            if output_data.empty:
                output_data = temp.copy()
            else:
                pass#output_data = output_data.append(temp)

    return output_data


def simple_csv(output_id, output_path, delimiter=None):
    '''
    Import a series of data files where the first column is Time (in seconds)
    and each additional column contains numbers describing a quantity in the
    model
    '''
    if delimiter is None:
        delimiter = ','
    output_data = pd.DataFrame({'':[]})


    print(f'    Loading {output_id} from xls')
    # Read and process the model output
    for fn in os.listdir(output_path):
        if fn.find(output_id) != -1:

            # index_col=False to fix trailing comma in csv file
            temp = pd.read_csv(''.join([output_path, '\\', fn]), delimiter=delimiter)

            s, r = find_scenario_and_replication(fn)

            # Create a series for the scenario and replication number
            s_series = pd.Series(s, index=np.arange(len(temp.index)))
            r_series = pd.Series(r, index=np.arange(len(temp.index)))

            # Insert the series into the DataFrame
            temp.insert(0, 'Replication', r_series)
            temp.insert(0, 'Scenario', s_series)

            if output_data.empty:
                    output_data = temp.copy()
            else:
                output_data = output_data.append(temp)

    output_data = output_data.reset_index()
    del output_data['index']
    return output_data


if __name__ == '__main__':
    pass