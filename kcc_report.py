# -*- coding: utf-8 -*-
"""
Report Generator

Create HTML report from Model Run results

Created on Fri Mar 11 15:15:15 2016

@author: sjsmith

(c) The Haskell Company
"""
# Module imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pathlib import Path
import webbrowser
from datetime import datetime


# Haskell Co Library1
from mrg import report_gen
from mrg import figures
from mrg import readcsv
from mrg import readxls



# Set styles
#plt.style.use('seaborn-darkgrid')


widget = '<?xml version="1.0" encoding="UTF-8" ?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"><svg width="40px" height="40px" viewBox="0 0 250 250" version="1.1" xmlns="http://www.w3.org/2000/svg"><path fill="#ffffff" d=" M 0.00 0.00 L 124.94 0.00 C 83.34 41.64 41.54 83.08 0.00 124.78 L 0.00 0.00 Z"/><path fill="#075e93" d=" M 124.94 0.00 L 125.17 0.00 C 146.09 20.78 166.94 41.64 187.72 62.55 C 177.43 62.51 167.13 62.53 156.84 62.53 C 156.81 80.36 156.85 98.19 156.82 116.02 C 146.42 116.09 136.03 116.04 125.63 116.05 C 125.65 121.90 125.63 127.75 125.66 133.61 C 104.94 133.64 84.22 133.59 63.51 133.62 C 63.48 151.38 63.58 169.14 63.46 186.89 C 84.24 187.02 105.03 186.90 125.82 186.95 C 125.75 169.18 125.78 151.41 125.81 133.64 C 136.16 133.59 146.51 133.62 156.85 133.62 C 156.78 151.38 156.86 169.14 156.82 186.90 C 164.89 187.01 172.97 186.91 181.04 187.01 C 183.30 187.02 185.72 186.44 187.72 187.80 C 166.64 208.27 146.13 229.34 125.22 250.00 L 124.94 250.00 C 83.26 208.41 41.68 166.71 0.00 125.12 L 0.00 124.78 C 41.54 83.08 83.34 41.64 124.94 0.00 Z"/><path fill="#ffffff" d=" M 125.17 0.00 L 250.00 0.00 L 250.00 124.95 C 229.38 104.15 208.50 83.60 187.96 62.73 C 187.90 103.49 187.93 144.25 187.94 185.01 C 187.90 185.74 187.80 187.21 187.76 187.95 C 189.86 185.02 192.63 182.70 195.13 180.14 C 213.40 161.81 231.78 143.58 250.00 125.20 L 250.00 250.00 L 125.22 250.00 C 146.13 229.34 166.64 208.27 187.72 187.80 C 185.72 186.44 183.30 187.02 181.04 187.01 C 172.97 186.91 164.89 187.01 156.82 186.90 C 156.86 169.14 156.78 151.38 156.85 133.62 C 146.51 133.62 136.16 133.59 125.81 133.64 C 125.78 151.41 125.75 169.18 125.82 186.95 C 105.03 186.90 84.24 187.02 63.46 186.89 C 63.58 169.14 63.48 151.38 63.51 133.62 C 84.22 133.59 104.94 133.64 125.66 133.61 C 125.63 127.75 125.65 121.90 125.63 116.05 C 136.03 116.04 146.42 116.09 156.82 116.02 C 156.85 98.19 156.81 80.36 156.84 62.53 C 167.13 62.53 177.43 62.51 187.72 62.55 C 166.94 41.64 146.09 20.78 125.17 0.00 Z"/><path fill="#ffffff" d=" M 63.55 62.54 C 84.30 62.51 105.04 62.53 125.79 62.53 C 125.68 80.36 125.96 98.20 125.66 116.03 C 104.93 115.82 84.20 116.04 63.47 115.91 C 63.59 98.12 63.43 80.33 63.55 62.54 Z"/><path fill="#075e93" d=" M 187.96 62.73 C 208.50 83.60 229.38 104.15 250.00 124.95 L 250.00 125.20 C 231.78 143.58 213.40 161.81 195.13 180.14 C 192.63 182.70 189.86 185.02 187.76 187.95 C 187.80 187.21 187.90 185.74 187.94 185.01 C 187.93 144.25 187.90 103.49 187.96 62.73 Z"/><path fill="#ffffff" d=" M 0.00 125.12 C 41.68 166.71 83.26 208.41 124.94 250.00 L 0.00 250.00 L 0.00 125.12 Z"/></svg>'
kc_logo = r'<svg xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://www.w3.org/2000/svg" xml:space="preserve" height="40" width="300" version="1.1" xmlns:cc="http://creativecommons.org/ns#" xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><rdf:RDF><cc:Work rdf:about=""><dc:format>image/svg+xml</dc:format><dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/><dc:title/></cc:Work></rdf:RDF></metadata><g transform="matrix(1.25,0,0,-1.25,-200.25,719.48025)"><g fill-rule="nonzero" transform="matrix(0.82318642,0,0,0.82318642,28.325536,96.189241)"><path fill="#414749" d="m240.82,576.96,6.142,0-0.311-4.442-6.143,0,0.312,4.442"/><path fill="#414749" d="m407.53,555.56c3.037,0,3.23,2.821,3.392,5.112-1.369,0-2.666,0.069-3.586-0.375-0.99-0.48-1.73-1.302-1.814-2.499-0.086-1.231,0.777-2.238,2.008-2.238zm2.822-1.167-0.068,0c-1.395-1.294-2.713-2.236-5.223-2.236-4.071,0-5.782,2.316-5.782,5.352,0,3.421,3.506,5.618,9.735,5.618h2.055c0.105,1.504,0.629,3.986-1.951,3.986-1.831,0-2.276-1.739-2.313-2.577h-5.917c0.135,2.24,1.103,3.736,2.903,4.523,1.838,0.819,4.075,1.025,5.993,1.025,3.49,0,7.772-1.736,7.483-5.993l-0.486-8.257c-0.049-0.718-0.059-1.47,0.037-3.522h-6.134l-0.332,2.081"/><path fill="#414749" d="m438.21,561.87,6.152,7.65,7.384,0-6.625-7.274,6.28-9.95-8.002,0-5.189,9.574zm-7.174-9.574,1.729,24.669,6.504,0-1.729-24.669-6.504,0"/><path fill="#414749" d="m325.61,566.96,0.068,0c1.669,2.147,3.591,2.504,5.708,2.504l-0.385-5.505c-0.438,0.033-0.882,0.033-1.328,0.033-4.286,0-4.418-2.453-4.577-5.296l-0.386-6.402h-6.672l0.892,12.754c0.108,1.54,0.146,2.565,0.104,4.459h6.359l0.217-2.547"/><path fill="#414749" d="m330.9,552.3,1.728,24.669,6.294,0-1.727-24.669-6.295,0"/><path fill="#414749" d="m285.36,560.66c-0.1-2.098,0.428-5.06,2.783-5.06,2.994,0,3.731,2.98,3.842,5.366,0.098,2.109-0.319,5.237-2.86,5.229-3.244-0.009-3.656-3.256-3.765-5.535zm-5.157,16.307,6.446,0-0.691-9.834,0.125,0c1.304,1.539,3.115,2.824,5.436,2.824,5.271,0,7.301-4.636,7.014-8.821-0.291-4.25-2.513-9.088-8.4-9.088-1.984,0-3.912,1.091-4.804,2.365h-0.137l-0.386-2.097h-6.331l1.728,24.651"/><path fill="#414749" d="m311.77,563.02c0.064,0.924-0.145,1.847-0.608,2.565s-1.165,1.162-2.163,1.093c-2.266-0.102-2.87-1.709-3.176-3.658h5.947zm-6.253-3.243c-0.19-2.374,0.353-4.288,2.782-4.288,1.815,0,2.958,1.207,2.958,2.329h6.025c-0.135-2.09-1.594-3.716-3.304-4.673-1.573-0.958-3.377-1.127-5.309-1.127-7.018,0-9.731,2.909-9.297,9.432,0.361,5.438,4.744,8.653,9.912,8.571,6.914-0.11,9.137-4.176,8.478-10.244h-12.245"/><path fill="#414749" d="m215.23,576.96,7.235,0-1.733-24.662-7.235,0,1.733,24.662"/><path fill="#414749" d="m230.47,576.96-8.649-10.957,7.707-13.705,8.667,0-8.639,14.063,9.091,10.599-8.177,0"/><path fill="#414749" d="m240.22,569.51,6.142,0-1.205-17.207-6.142,0,1.205,17.207"/><path fill="#414749" d="m254.14,569.51-0.002-2.325,0.068,0c1.917,2.506,4.555,2.731,6.176,2.731,2.347,0,4.748-1.298,5.232-3.039,1.328,2.058,3.698,3.074,6.31,3.142,3.757,0,6.369-2.4,6.099-6.207l-0.817-11.511h-6.608l0.718,10.104c0.087,1.639-0.045,3.499-2.202,3.499-3.165,0-3.041-3.533-3.074-3.739l-0.678-9.868h-6.627l0.733,10.108c0.031,1.68-0.188,3.499-2.283,3.499-3.34,0-3.269-3.533-3.266-3.739l-0.686-9.874h-6.814l1.197,17.217,6.524,0.002"/><path fill="#414749" d="m347.27,544.02-5.979,0,3.854,8.001-6.313,17.489,7.33,0,2.846-10.679,4.521,10.679,6.048,0-12.307-25.49"/><path fill="#414749" d="m359.77,566.05,8.327,0-2.662-5.384-8.327,0,2.662,5.384"/><path fill="#414749" d="m381.47,577.62c5.088,0,10.288-2.221,10.062-8.396h-7.009c0,1.619-0.944,3.463-3.317,3.463-2.375,0-5.164-1.768-5.651-6.89-0.49-5.12,0.149-8.997,4.294-8.997,3.58,0,4.297,3.087,4.371,3.689h6.954c0-2.749-2.281-8.434-11.702-8.434-9.948,0-11.72,7.229-10.892,13.969,0.696,5.643,4.713,11.596,12.89,11.596"/><path fill="#414749" d="m391.66,552.3,1.729,24.669,6.295,0-1.729-24.669-6.295,0"/><path fill="#414749" d="m425.48,566.96,0.068,0c1.67,2.147,3.591,2.504,5.747,2.504l-0.386-5.505c-0.477,0.033-0.922,0.033-1.366,0.033-4.286,0-4.417-2.453-4.577-5.296l-0.387-6.402h-6.671l0.892,12.754c0.108,1.54,0.145,2.565,0.102,4.459h6.361l0.217-2.547"/><path fill="#3c93cb" d="m180.82,548.15c-10.497,0-18.967,7.274-18.967,16.283,0,9.011,8.47,16.285,18.967,16.285,10.496,0,18.967-7.274,18.967-16.285,0-9.009-8.471-16.283-18.967-16.283zm-20.62,16.28c0-9.903,9.228-17.93,20.612-17.93s20.613,8.027,20.613,17.93-9.229,17.931-20.613,17.931c-11.408,0-20.612-8.028-20.612-17.931"/><path fill="#3c93cb" d="m183.48,553.98,0,7.918c1.48-3.311,4.097-4.041,6.218-4.041h5.886v3.548h-4.12c-2.124,0-2.789,1.576-2.789,2.482,0,2.194,1.78,3.864,4.432,6.209h-4.696c-2.031-1.331-4.015-3.44-4.926-5.807v10.963h-5.333v-10.963c-0.911,2.367-2.895,4.476-4.926,5.807h-4.696c2.653-2.345,4.432-4.015,4.432-6.209,0-0.905-0.665-2.481-2.789-2.481h-4.119v-3.548h5.886c2.12,0,4.735,0.729,6.217,4.04v-7.918h5.323zm-7.034-1.705,0,4.99c-0.421-0.284-1.467-1.107-3.623-1.107h-8.469v6.958h5.226c1.308,0,1.391,0.779,1.391,1.015,0,2.01-3.454,4.791-7.287,7.676l10.272,0.019c0.628-0.462,1.616-1.162,2.496-2.152v7.308h8.733v-7.308c0.88,0.99,1.868,1.69,2.496,2.152l10.273-0.019c-3.835-2.886-7.288-5.666-7.288-7.676,0-0.236,0.083-1.015,1.391-1.015h5.226v-6.958h-8.469c-2.156,0-3.202,0.823-3.623,1.106v-4.989h-8.745"/></g></g></svg>'
machine_colors = [ '#98fb98',  # Pale Green
                   '#000080',  # Blue
                   '#87cefa',  # Light Sky Blue
                   '#ff0000']  # Red

custom_colors = ['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231',
                 '#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe',
                 '#008080','#e6beff','#aa6e28','#fffac8','#800000',
                 '#aaffc3','#808000','#ffd8b1','#000080','#000000']

custom_colors = 3 * custom_colors


def add_to_html(fig_type, fig, title, description=None, wide=None):
    if description is None:
        description = ''

    # Convert all mpl figures to svg
    if fig_type == 'svg':
        fig = figures.fig_to_svg(fig)


    if fig_type == 'tabs':
        for k, v in fig.items():
            fig[k] = figures.fig_to_svg(v)

    # Bootstrap column width
    if wide is True:
        wide = str(12)
    else:
        wide = str(6)

    params['figs'].append(tuple([fig_type,
                                 fig,
                                 title,
                                 title.replace(" ", ""),
                                 description,
                                 wide]))


if __name__ == '__main__':

    print('\n\n')
    print('************************************************')
    print('     Haskell Model Results Report Generator')
    print('************************************************\n\n')

    plt.rcParams.update({'figure.max_open_warning': 0})
    ###########################################################################
    # Setup Report Filenames and Destinations
    ###########################################################################

    report_name = 'K-C_Mobile_'
    report_title = 'K-C Mobile Simulation Results'

    # Gets path to reports directory
    path = str(Path(os.getcwd()).parent)
    report_path = path + r'\Reports'

    # Generated report file path and name
    report_time = datetime.now()
    fnstr = '%y%m%d_%H%M'
    report_fn_template = report_path + '\\' + report_name + r'20{}.html'
    report_file = report_fn_template.format(report_time.strftime(fnstr))

    # Gets path to Model Output files
    output_path = path + r'\Output'

    params = report_gen.define_params(report_title=report_title)



    ###########################################################################
    # import data files by a unique identifier in the filename
    ###########################################################################
    print('Importing data...')

#    ############ State Logs ############
#    state_converting1 = readxls.d3d_state_log('State-SCADAConverting1stFloor', output_path)
#    state_converting2 = readxls.d3d_state_log('State-SCADAConverting2ndFloor', output_path)
#    state_tm = readxls.d3d_state_log('State-SCADATMs', output_path)
#    state_trucks = readxls.d3d_state_log('State-TruckManager', output_path)
#    state_lifts = readxls.d3d_state_log('State-LiftManager', output_path)
#
#
#
#    ############ Throughputs ############
#    tp_converting1 = readxls.d3d_throughput('Throughput-SCADAConverting1stFloor', output_path)
#    tp_converting2 = readxls.d3d_throughput('Throughput-SCADAConverting2ndFloor', output_path)
#    tp_tm = readxls.d3d_throughput('Throughput-SCADATMs', output_path)
#    tp_trucks = readxls.d3d_throughput('Throughput-TruckManager', output_path)
#    tp_lifts = readxls.d3d_throughput('Throughput-LiftManager', output_path)
#
#
#    ############  WIP ############
#    wip = readxls.d3d_throughput('WIP', output_path)
#    misc = readxls.d3d_throughput('Miscellaneous', output_path)





    ###########################################################################
    # Process Data
    ###########################################################################
    print('Processing data...')

    ########### Summary Table ############
    misc_grouped = misc.groupby(['Scenario', 'Title']).mean()
    del misc_grouped['Replication']
    misc_grouped['Scenario'] = [x[0] for x in misc_grouped.index]
    misc_grouped['Title'] = [x[1] for x in misc_grouped.index]


    t_max = wip['Time Stamp'].max()
    t_max_wip = wip[wip['Time Stamp'] == t_max].copy()

    num_scenarios = len(t_max_wip.Replication.unique())

    exported_mt = list(misc_grouped[misc_grouped.Title == 'Tons Exported'].Value)
    exported_mt = [int(x) for x in exported_mt]
    exported_rolls = [int(t_max_wip[t_max_wip.Scenario == s].Export.sum() / num_scenarios)
                      for s in sorted(t_max_wip.Scenario.unique())]

    overflow_mt = list(misc_grouped[misc_grouped.Title == 'Tons Overflowed'].Value)
    overflow_mt = [round(x, 1) for x in overflow_mt]
    overflow_rolls = [round(t_max_wip[t_max_wip.Scenario == s].Overflow.sum() /num_scenarios, 1)
                      for s in sorted(t_max_wip.Scenario.unique())]

    tm_mt = [int(tp_tm[tp_tm.Scenario == s].LoadsCreated.sum() / num_scenarios)
             for s in sorted(tp_tm.Scenario.unique())]

    converting_mt = [int((tp_converting1[tp_converting1.Scenario == s].LoadsConsumed.sum() +
                      tp_converting2[tp_converting2.Scenario == s].LoadsConsumed.sum()) / num_scenarios)
                      for s in sorted(tp_tm.Scenario.unique())]

    summary = pd.DataFrame(index=sorted(wip.Scenario.unique()),
                    columns=['TMs', 'Converting', 'Export', 'Overflow'])

    summary.TMs = [f'{t} MT' for t in tm_mt]
    summary.Converting = [f'{t} MT' for t in converting_mt]
    summary.Export = [f'{t} MT ({r} rolls)'  for t, r in zip(exported_mt, exported_rolls)]
    summary.Overflow = [f'{t} MT ({r} rolls)'  for t, r in zip(overflow_mt, overflow_rolls)]
    summary.index.rename('Scenario', inplace=True)


    ########### WIP CHARTS ############
    # WIP processing
    actual_wip = wip.copy()

    # Remove columns to not be summed
    del actual_wip['Export']
    del actual_wip['Overflow']
    del actual_wip['Scenario']
    del actual_wip['Replication']
    del actual_wip['Time Stamp']

    # Sum Columns
    wip_sum = pd.concat([wip.Scenario, wip.Replication,
                         wip['Time Stamp'].copy() / 3600,
                         actual_wip.sum(axis=1)], axis=1, )

    wip_sum.rename(columns={0:'WIP', 'Time Stamp' : 'Time'}, inplace=True)




    ########### ELEVATOR THROUGHPUT TABLE ############

    elevator_summary = pd.DataFrame(index=sorted(tp_lifts.Scenario.unique()),
                    columns=['Per Week (avg)', 'Per Day (avg)', 'Per Hour (avg)'])

    elevator_perweek = list(tp_lifts.groupby(['Scenario']).sum()['LoadsCreated'] / num_scenarios)
    elevator_perweek = [round(x, 1) for x in elevator_perweek]


    elevator_summary['Per Week (avg)'] = elevator_perweek
    elevator_summary['Per Day (avg)'] = [round(x / 7, 1) for x in elevator_perweek]
    elevator_summary['Per Hour (avg)'] = [round(x / (24 * 7), 1) for x in elevator_perweek]
    elevator_summary.index.rename('Scenario', inplace=True)

    ########### TOUCHES PER ROLL TABLE ############

    touch_summary = pd.DataFrame(index=sorted(misc_grouped.Scenario.unique()),
                    columns=['Total Touches', 'Total Rolls', 'Touches Per Roll'])

    rolls = list(misc_grouped[misc_grouped.Title == 'Total Rolls (Without Export)'].Value)
    touches = list(misc_grouped[misc_grouped.Title == 'Total Touches (Without Export)'].Value)
    avg_touches = list(misc_grouped[misc_grouped.Title == 'Average Touches Per Roll (Without Export)'].Value)

    touch_summary['Total Touches'] = [round(x, 1) for x in touches]
    touch_summary['Total Rolls'] = [round(x, 1) for x in rolls]
    touch_summary['Touches Per Roll'] = [round(x, 3) for x in avg_touches]
    touch_summary.index.rename('Scenario', inplace=True)


    ###########################################################################
    # Create Charts
    ###########################################################################
    xaxis_length = 24 * 14
    machine_colors = [ '#98fb98',  # Pale Green
                       '#87cefa', # Light Sky Blue
                       '#ff0000',  # Red
                       '#000080',# Blue
					   '#D3D3D3' #Gray
                       ]

    fig1_svg = {}
    unit_ops = state_converting1['Machine Name'].unique()

    states = ['Producing', 'Starved', 'Faulted', 'Changeover', 'Standby']

    for s in sorted(state_converting1.Scenario.unique()):


        fig1, ax1 = figures.d3d_time_in_state(state_converting1[state_converting1.Scenario == s],
                                            unit_ops,
                                            states,
                                            colors=machine_colors)
        fig1.set_size_inches(6, len(unit_ops))
        ax1.set_xlabel('Hours')
        ax1.set_xlim(0, xaxis_length)
        key = 'S' + str(s)
        fig1_svg.update({key : fig1})


    fig2_svg = {}
    unit_ops = state_converting2['Machine Name'].unique()


    for s in sorted(state_converting2.Scenario.unique()):
        fig2, ax2 = figures.d3d_time_in_state(state_converting2[state_converting2.Scenario == s],
                                            unit_ops,
                                            states,
                                            colors=machine_colors)
        fig2.set_size_inches(6, len(unit_ops))

        ax2.set_xlabel('Hours')
        ax2.set_xlim(0, xaxis_length)
        key = 'S' + str(s)
        fig2_svg.update({key : fig2})

    fig3_svg = {}
    unit_ops = state_tm['Machine Name'].unique()

    for s in sorted(state_tm.Scenario.unique()):
        fig3, ax3 = figures.d3d_time_in_state(state_tm[state_tm.Scenario == s],
                                            unit_ops,
                                            states,
                                            colors=machine_colors)
        fig3.set_size_inches(6, len(unit_ops))
        ax3.set_xlabel('Hours')
        ax3.set_xlim(0, xaxis_length)
        key = 'S' + str(s)
        fig3_svg.update({key : fig3})


    fig4_svg = {}
    unit_ops = state_trucks['Machine Name'].unique()

    for s in sorted(state_trucks.Scenario.unique()):
        fig4, ax4 = figures.d3d_time_in_state(state_trucks[state_trucks.Scenario == s],
                                            unit_ops,
                                            states,
                                            colors=machine_colors)
        fig4.set_size_inches(6, len(unit_ops))
        ax4.set_xlabel('Hours')
        ax4.set_xlim(0, xaxis_length)
        key = 'S' + str(s)
        fig4_svg.update({key : fig4})




    ########### WIP CHARTS ############


    fig5_svg = {}

    fig5, ax = figures.qvt(wip_sum,
                             stacked=False,
                             plot_by_cols=True,
                             col_names=['WIP'],
                             x_label='Time (hr)',
                             y_label='WIP',
                             minmax=False)
    ax.set_xlim(0, xaxis_length)
    fig5.set_size_inches(10, 6)

    fig6_svg = {}
    wip['Time'] = wip['Time Stamp'].copy() / 3600

    for s in sorted(wip.Scenario.unique()):
        fig6, ax = figures.qvt(wip[wip.Scenario == s],
                                 stacked=True,
                                 plot_by_cols=False,
                                 col_names=wip.filter(regex=r'TM5/TM7').columns,
                                 x_label='Time (hr)',
                                 y_label='WIP',
                                 minmax=False,
                                 num_cols=4)
        #ax.set_xlim(0, xaxis_length)
        fig6.set_size_inches(10, 6)
        key = 'S' + str(s)
        fig6_svg.update({key : fig6})

    fig7_svg = {}
    for s in sorted(wip.Scenario.unique()):
        fig7, ax = figures.qvt(wip[wip.Scenario == s],
                                 stacked=True,
                                 plot_by_cols=False,
                                 col_names=wip.filter(regex='Floor 3 Bay').columns,
                                 x_label='Time (hr)',
                                 y_label='WIP',
                                 minmax=False,
                                 num_cols=5)
        #ax.set_xlim(0, xaxis_length)
        fig7.set_size_inches(10, 6)
        key = 'S' + str(s)
        fig7_svg.update({key : fig7})

    fig8_svg = {}
    for s in sorted(wip.Scenario.unique()):
        fig8, ax = figures.qvt(wip[wip.Scenario == s],
                                 stacked=True,
                                 plot_by_cols=False,
                                 col_names=wip.filter(regex='TM8 Storage').columns,
                                 x_label='Time (hr)',
                                 y_label='WIP',
                                 minmax=False,
                                 num_cols=2)
        #ax.set_xlim(0, xaxis_length)
        fig8.set_size_inches(10, 6)
        key = 'S' + str(s)
        fig8_svg.update({key : fig8})

    fig9_svg = {}
    for s in sorted(wip.Scenario.unique()):
        fig9, ax = figures.qvt(wip[wip.Scenario == s],
                                 stacked=True,
                                 plot_by_cols=False,
                                 col_names=wip.filter(regex='1st Floor Bay').columns,
                                 x_label='Time (hr)',
                                 y_label='WIP',
                                 minmax=False,
                                 num_cols=2)
        #ax.set_xlim(0, xaxis_length)
        fig9.set_size_inches(10, 6)
        key = 'S' + str(s)
        fig9_svg.update({key : fig9})

    fig10_svg = {}
    for s in sorted(wip.Scenario.unique()):
        fig10, ax = figures.qvt(wip[wip.Scenario == s],
                                 stacked=True,
                                 plot_by_cols=False,
                                 col_names=wip.filter(regex='Expansion').columns,
                                 x_label='Time (hr)',
                                 y_label='WIP',
                                 minmax=False,
                                 num_cols=2)
        #ax.set_xlim(0, xaxis_length)
        fig10.set_size_inches(10, 6)
        key = 'S' + str(s)
        fig10_svg.update({key : fig10})




    custom_cols = ['TM11 Longs', 'TM6 Longs', 'TM5 (HRT5 Only)',
                   'TM5/HRT5', 'CFT Storage', 'TM7 Brown', 'TM7 White',
                   'TM6 Short Storage']


    fig11_svg = {}
    for s in sorted(wip.Scenario.unique()):
        fig11, ax = figures.qvt(wip[wip.Scenario == s],
                                 stacked=True,
                                 plot_by_cols=False,
                                 col_names=custom_cols,
                                 x_label='Time (hr)',
                                 y_label='WIP',
                                 minmax=False,
                                 num_cols=4)
        #ax.set_xlim(0, xaxis_length)
        fig11.set_size_inches(10, 6)
        key = 'S' + str(s)
        fig11_svg.update({key : fig11})

    ###########################################################################
    # Adding to Reports
    ###########################################################################
    print('Generating plots & tables...')



    #-----------------------------------------------------------------------
    add_to_html('div', '', 'Summary', description='')
    #-----------------------------------------------------------------------

    title = 'Summary'
    desc = ''

    add_to_html('df', summary, title, description=desc, wide=True)

    add_to_html('df', elevator_summary, 'Rolls Moved By Elevator')

    add_to_html('df', touch_summary, 'Touches Per Roll')

    #-----------------------------------------------------------------------
    add_to_html('div', '', 'Time In State', description='')
    #-----------------------------------------------------------------------

    title = 'Converting 1st Floor'
    desc = ''
    add_to_html('tabs', fig1_svg, title, description=desc)

    title = 'Converting 2nd Floor'
    desc = ''
    add_to_html('tabs', fig2_svg, title, description=desc)

    add_to_html('invis_div', '', '', description='')

    title = 'TMs'
    desc = ''
    add_to_html('tabs', fig3_svg, title, description=desc)

    title = 'Trucks'
    desc = ''
    add_to_html('tabs', fig4_svg, title, description=desc)




    #-----------------------------------------------------------------------
    add_to_html('div', '', 'Work In Progress', description='')
    #-----------------------------------------------------------------------

    title = 'WIP'
    desc = ''

    add_to_html('svg', fig5, title, description=desc, wide=True)
    add_to_html('tabs', fig6_svg, 'Stadium WIP', description=desc, wide=True)
    add_to_html('tabs', fig7_svg, '3rd Floor WIP', description=desc, wide=True)
    add_to_html('tabs', fig8_svg, 'Gymnasium WIP', description=desc, wide=True)
    add_to_html('tabs', fig9_svg, '1st Floor Bays WIP', description=desc, wide=True)
    add_to_html('tabs', fig10_svg, 'Expansion WIP', description=desc, wide=True)
    add_to_html('tabs', fig11_svg, 'Other WIP', description=desc, wide=True)
    add_to_html('invis_div', '', '', description='')




    ########################################################################
    # Generate report
    ########################################################################
    print('Generating report...')

    print('\n\n')
    print('************************************************\n\n')
    exp_title = 'Simulation Results'
    exp_desc = ''

    exp_desc = '<h2>' + exp_title + '</h2><p>' + exp_desc + '</p>'

    scenario_desc = ['Baseline (41 Bays)',
                     'Building Expansion - Additional Roll Storage: 200 Rolls (+4 Bays)',
                     'Stadium - Lost Roll Storage: 550 Rolls (-10 Bays)',
                     'Building Expansion - Additional Roll Storage: 190 Rolls (+4 Bays)',
                     'Building Expansion - Additional Roll Storage: 180 Rolls (+4 Bays)',
                     'Building Expansion - Additional Roll Storage: 170 Rolls (+4 Bays)']
    scenario_names = []



    for s in sorted(tp_tm.Scenario.unique()):

        scenario_names.append(('Scenario {0}'.format(s)))
        #scenario_desc.append(input('Day {0} Description: '.format(s)))
        #scenario_desc.append('Description {0}'.format(s))

    # Add the scenario dictionary to the params
    params['scenario_names'] = scenario_names
    params['scenario_desc'] = scenario_desc
    params['zipped_scenario_list'] = zip(scenario_names, scenario_desc)
    params['client_logo'] = kc_logo

    # Add the explanation to the params
    params['experiment_explanation'] = exp_desc
    report_gen.generate(params, report_fn=report_file)

    webbrowser.open(report_file)
