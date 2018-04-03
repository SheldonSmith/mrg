# -*- coding: utf-8 -*-
"""
Figure Generator

Create Matplotlib figures from Model Run results

Created on Fri Mar 11 15:15:15 2016

@author: sjsmith

(c) The Haskell Company
"""
# Module imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

# Member imports
from io import StringIO


color_cycle = ['#0082C2',  # Blue
               '#EB1C23',  # Red
               '#00A84C',  # Green
               '#004C7D',  # Dark Blue
               '#F57F1E',  # Orange
               '#73BF42',  # Lt Green
               '#8C1719',  # Dark Red
               '#FAA619',  # Lt Orange
               '#D7A86E']  # Brown

custom_colors = ['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231',
                 '#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe',
                 '#008080','#e6beff','#aa6e28','#fffac8','#800000',
                 '#aaffc3','#808000','#ffd8b1','#000080','#000000']


def fig_to_svg(fig):
    '''
    Accepts a matplotlib figure as import and returns SVG as a string.
    '''
    # TODO: Remove the text up to <svg>, get the sizing correct
    str_svg = StringIO()
    fig.savefig(str_svg, format='svg',bbox_inches='tight')
    svg = str_svg.getvalue()
    svg = svg.replace('\u2212', '-')  # jinja2 will trip up on \u2212
    return svg


def p2f(x):
    '''
    If utilization data is provided as a percentage, strip the % sign and
    convert to a float
    '''
    return float(str(x).strip('%'))


def clean_time_in_state_data(df, state_names, normalize=True):

    data = []
    # Remove any % signs
#    for state in states:
##        df[state_name] = df[state_name].apply(p2f)
#        df[state] = df[state].mean()
#        data.append(df[state_name].mean())
#    data = [float(i)/sum(data) for i in data]
    return data


def check_scenario_replication(data):

    if not 'Scenario' in data.columns:
        # Create a series for the scenario and replication number
        s, r = 1, 1

        s_series = pd.Series(s, index=np.arange(len(data.index)))
        r_series = pd.Series(r, index=np.arange(len(data.index)))

        # Insert the series into the DataFrame
        data.insert(0, 'Replication', r_series)
        data.insert(0, 'Scenario', s_series)

    return data

def d3d_time_in_state(data,
                     unit_ops,
                     states,
                     normalize=False,
                     colors=None):
    '''
    Tests
    '''
    if colors is None:
        colors = color_cycle

    if normalize is None:
        normalize = False

    data = check_scenario_replication(data).copy()

    data['Duration'] = (data['Length (seconds)'] / 3600).copy()

    scenarios = data.Scenario.unique()
    num_scenarios = len(scenarios)

    replications = data.Replication.unique()
    num_replications = len(replications)

    nrows, ncols = 1, num_scenarios
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    plt.tight_layout(h_pad=7, rect=[0.05, 0.10, 1, 0.98])

    # If a single unit op, convert to list of length 1
    if type(unit_ops) == str:
        unit_ops = [unit_ops]

    for i, s in zip(range(0, ncols), scenarios):
    # Select ax for plotting
        if ncols > 1:
            this_ax = ax[i]
        else:
            this_ax = ax

        # Select the data for this scenario only
        data_s = data[data.Scenario == s].copy()

        data_s = data_s.groupby(['State','Machine Name']).sum()
        data_s.reset_index(inplace=True)
        data_s.Duration = data_s.Duration / num_replications


        data_for_plot = np.zeros([len(unit_ops),len(states)])
        for k, u in enumerate(unit_ops):
            for j, s in enumerate(states):
                val = data_s.Duration[(data_s['Machine Name'] == u) &(data_s['State'] == s)]
                if len(val) == 0:
                    val = 0.0
                data_for_plot[k, j] = val

        patch_handles = []
        left = np.zeros(len(unit_ops))
        y_pos = np.arange(0, -1 * len(unit_ops), -1)
        data_for_plot = np.array(data_for_plot)
        data_for_plot = data_for_plot.transpose()

        for index, d in enumerate(data_for_plot):
            bar = this_ax.barh(y_pos,
                               d,
                               color=colors[index % len(colors)],
                               align='center',
                               alpha=0.5,
                               height=0.9,
                               left=left,
                               linewidth=0)
            patch_handles.append(bar)
            left += d
        for j in range(len(patch_handles)):
            for k, patch in enumerate(patch_handles[j].get_children()):
                bl = patch.get_xy()
                x = 0.5*patch.get_width() + bl[0]
                y = 0.5*patch.get_height() + bl[1]
                if data_for_plot[j,k]*100 > 5:
                    if normalize is True:
                        this_ax.text(x,
                                     y,
                                     "%d%%" % round(data_for_plot[j,k]*100),
                                     size='medium',
                                     ha='center',
                                     va='center',
                                     weight='extra bold',
                                     color='black')
                    else:
                        pct = "%d%%" % round(data_for_plot[j,k]*100/sum(data_for_plot[:,k]),1)
                        text = str(round(data_for_plot[j,k],2))
                        this_ax.text(x,
                                     y,
                                     text + '\n' + pct,
                                     size='medium',
                                     ha='center',
                                     va='center',
                                     weight='extra bold',
                                     color='black')
        this_ax.get_yaxis().set_ticks([])

        if i == 0:
            this_ax.set_yticks(y_pos)
            this_ax.set_yticklabels(unit_ops)
        this_ax.set_ylim(min(y_pos)-0.4,0.4)

        this_ax.set_title('Scenario: ' + str(scenarios[i]))
        this_ax.set_facecolor('white')
        if normalize is True:
            this_ax.set_xticks([0,.25,.50,.75,1.0])
            vals = this_ax.get_xticks()
            this_ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals])
        for tick in this_ax.get_xticklabels():
            tick.set_rotation(0)


    legend_items = states
    patches = []

    for i in range(0, len(legend_items)):
        patch = matplotlib.patches.Patch(color=colors[i % len(colors)])
        patches.append(patch)


    # Set the legend
    leg = fig.legend(patches,
                    legend_items,
                    bbox_to_anchor=(0.5, 0.12),
                    loc='upper center',
                    ncol=(len(legend_items) // 3 + 2),
                    fancybox=True,
                    fontsize=12)
    # Set the linewidth of each legend object
    leg.get_frame().set_facecolor('#FFFFFF')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)


    return fig, ax


def d3d_state_log(data,
                 unit_ops,
                 states,
                 colors=None):
    '''
    Tests
    '''
    if colors is None:
        colors = color_cycle


    data = check_scenario_replication(data).copy()


    data['Duration'] = data['Length (seconds)'] / 3600

    # If a single unit op, convert to list of length 1
    if type(unit_ops) == str:
        unit_ops = [unit_ops]

    num_unit_ops = len(unit_ops)

    scenarios = data.Scenario.unique()
    num_scenarios = len(scenarios)

    replications = data.Replication.unique()
    num_replications = len(replications)

    nrows, ncols = num_unit_ops, num_scenarios
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    fig.set_size_inches(6, 3 * num_unit_ops * num_replications / 5)
    plt.tight_layout(h_pad=2, rect=[0.1, 0.1, 0.9, 0.9])

    for i, s in enumerate(scenarios):

        data_scenario = data[data.Scenario == s].copy()
        for j, uo in enumerate(unit_ops):
            # Select ax for plotting

            if ncols > 1 & nrows > 1:
                this_ax = ax[j, i]
            elif ncols > 1:
                this_ax = ax[i]
            elif nrows > 1:
                this_ax = ax[j]
            else:
                this_ax = ax

            # Select the data for this scenario only
            data_s = data_scenario[data_scenario['Machine Name'] == uo].copy()

            uniques = data_s.Replication.unique()

            patch_handles = []

            for k, u in enumerate(uniques):
                left = 0

                this_data = data_s[data_s.Replication == u].copy()
                for d, s in zip(this_data.Duration, this_data.State):
                    try:
                        col_index = states.index(s)
                    except:
                        col_index = len(colors) - 1
                    bar = this_ax.barh(k,
                                       d,
                                       color=colors[col_index],
                                       align='center',
                                       alpha=1,
                                       height=0.9,
                                       left=left,
                                       linewidth=0)
                    patch_handles.append(bar)
                    left += d

#            if i == 0:
            this_ax.set_yticks([x for x in range(num_replications)])
            this_ax.set_yticklabels(range(1, num_replications+1))
            this_ax.set_ylim(-0.6, num_replications - 0.6)

            #this_ax.set_title('Scenario: ' + str(scenarios[i]))
            this_ax.set_facecolor('white')

            for tick in this_ax.get_xticklabels():
                tick.set_rotation(0)

    return fig, ax


def qvt( data,
         col_names=None,
         stacked=False,
         plot_by_cols=False,
         x_label=None,
         y_label=None,
         scenario_names=None,
         colorcycle=None,
         minmax=False,
         num_cols=None):
    '''
    Creates a Nx1 grid of quantity vs. time charts during the model runs.
    '''

    if colorcycle is None:
        colorcycle = (color_cycle + custom_colors) * 2


    # Check data has right format
    data = check_scenario_replication(data)

    # Initialize figure
    num_subplots = 1
    subplots = []
    if (plot_by_cols):
        num_subplots = len(col_names)
        subplots = col_names
    else:
        num_subplots = len(data.Scenario.unique())
        subplots = data.Scenario.unique()

    fig, ax = plt.subplots(nrows=num_subplots, ncols=1)

    # Default figure size
    fig.set_size_inches(8, 4 + 2.2 * num_subplots)
    plt.tight_layout(h_pad=3, rect=[0.02, 0.20, 1.00, 0.98])

    # Main loop
    i_color = 0
    for s, r in zip(subplots, range(0, num_subplots)):
        # Select subplot for each scenario
        if num_subplots > 1:
            this_ax = ax[r]
        else:
            this_ax = ax

        if (plot_by_cols):
            temp = data[['Scenario','Time', s]].copy()
            scenarios = data.Scenario.unique()
            for scenario, i in zip(scenarios, range(0,len(scenarios))):

                data_in = temp[temp.Scenario == scenario]

                data_mean, data_min, data_max = qvt_data_mean_min_max(data_in)

                color = colorcycle[i_color]


                this_ax = qvt_line_plot(this_ax,
                                        data_mean,
                                        data_min,
                                        data_max,
                                        s,
                                        color,
                                        minmax)

                i_color += 1
        else:
            # Calculate min, max, mean for all replications for a given sceanrio
            temp = data[data.Scenario == s].copy()

            data_mean, data_min, data_max = qvt_data_mean_min_max(temp)

            # Filters out Time, Scenario, Replication columns
            if col_names is None:
                cols = data.columns[3:]
            else:
                cols = col_names

            # Fills Between the Lines
            if stacked:
                # Mean value plot
                this_ax.stackplot(data_mean.Time,
                                 np.array(data_mean[cols]).T,
                                 alpha=0.75,
                                 linewidth=0.5,
                                 colors=colorcycle[:len(cols)])
            # Regular Line Plot
            else:
                for col, i in zip(cols, range(0, len(cols))):
                    color = colorcycle[i]
                    qvt_line_plot(this_ax,
                                  data_mean,
                                  data_min,
                                  data_max,
                                  col,
                                  color,
                                  minmax)


        chart_title = str(s)

        if not plot_by_cols:
            chart_title = chart_title

        this_ax.annotate(chart_title, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=20, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))

        if y_label is None:
            y_label = ''
        if x_label is None:
            x_label = 'Time'

        this_ax.set_ylabel(y_label, fontsize=15)
        this_ax.set_xlabel(x_label, fontsize=15)
#
        # Set x, y axis limits & formatting
        _, ub = this_ax.get_ylim()
        this_ax.set_xlim(data.Time.min(), data.Time.max())
        this_ax.set_ylim(0, ub)  # always set lb to 0
        this_ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Always select the first plot for formatting the legend
        if num_subplots > 1:
            this_ax = ax[0]
        else:
            this_ax = ax

        legend_items = []

        if plot_by_cols:
            if scenario_names is None:
                legend_items = data.Scenario.unique()
            else:
                legend_items = scenario_names
            legend_items = [str(l) for l in legend_items]
        else:
            legend_items = cols

        if num_cols is None:
            num_cols = (len(legend_items) // 3 + 2)

        # Stacked chart legend gets boxes
        patches = []
        if True:
            for i in range(0, len(legend_items)):
                patch = matplotlib.patches.Patch(color=colorcycle[i % len(colorcycle)])
                patches.append(patch)

        # Regular chart legend gets lines
        else:
            patches, _ = this_ax.get_legend_handles_labels()

        # Set the legend
        leg = fig.legend(patches,
                        legend_items,
                        bbox_to_anchor=(0.5, 0.16),
                        loc='upper center',
                        ncol=num_cols,
                        fancybox=True,
                        fontsize=12)
        # Set the linewidth of each legend object
        leg.get_frame().set_facecolor('#FFFFFF')
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

    return fig, ax

def qvt_line_plot(this_ax,
                  data_mean,
                  data_min,
                  data_max,
                  scenario,
                  color,
                  minmax):

    # Mean value plot
    this_ax.plot(data_mean.Time,
                 data_mean[scenario],
                 color=color,
                 linewidth=2)
    # Min/Max value plot
    if minmax:
        this_ax.fill_between(data_max.Time,
                             data_min[scenario],
                             data_max[scenario],
                             alpha=0.3,
                             color=color)
    return this_ax

def qvt_data_mean_min_max(input_in):
    data_mean = input_in.groupby(input_in.Time).mean()
    data_max = input_in.groupby(input_in.Time).max()
    data_min = input_in.groupby(input_in.Time).min()

    data_mean['Time'] = data_mean.index
    data_max['Time'] = data_max.index
    data_min['Time'] = data_min.index

    return data_mean, data_min, data_max




































def time_in_state(data,
                 machines,
                 states,
                 chart_title,
                 normalize=False,
                 colors=None):

    if colors is None:
        colors = color_cycle

    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    data_for_plot = []

    for m in machines:
        data_for_plot.append(data[m])

    patch_handles = []
    left = np.zeros(len(machines))
    y_pos = np.arange(0,-1*len(machines),-1)
    data_for_plot = np.array(data_for_plot)
    data_for_plot = data_for_plot.transpose()
    for index, d in enumerate(data_for_plot):
        bar = ax.barh( y_pos,
                       d,
                       color=colors[index % len(colors)],
                       align='center',
                       alpha=0.5,
                       height=0.9,
                       left=left,
                       linewidth=0)
        patch_handles.append(bar)
        left += d
    for j in range(len(patch_handles)):
        for k, patch in enumerate(patch_handles[j].get_children()):
            bl = patch.get_xy()
            x = 0.5*patch.get_width() + bl[0]
            y = 0.5*patch.get_height() + bl[1]
            if data_for_plot[j,k]*100 > 5:
                ax.text( x,
                         y,
                         "%d%%" % round(data_for_plot[j,k]*100,2),
                         size=5,
                         ha='center',
                         va='center',
                         weight='extra bold',
                         color='black')
    ax.get_yaxis().set_ticks([])


    ax.set_yticks(y_pos)
    ax.set_yticklabels(machines, size=5)
    ax.set_ylim(min(y_pos)-0.4,0.4)
    # Sets the x limit and doesn't squish the graphs horizontally!!!!!!
    ax.set_xlim(0,1)
    ax.set_title(chart_title)
    ax.xmargin = 0
    ax.ymargin = 0
    #this_ax.set_xlim(0,1)
    ax.set_facecolor('white')
    ax.set_xticks([0,.25,.50,.75,1.0])
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals], size=5)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)



    return fig, ax, data_for_plot


def bar_chart(data,
              col_name=None,
              y_axis_label=None,
              x_axis_label=None,
              x_axis_labels=None,
              colorcycle=None):
    '''
    - Pass a dataframe and make a simple bar chart
    - Plots the index as each bar, and the value in column name passed of the df
    - No need to plot by scenario as you will likely be comparing scenarios on
    the same bar chart
    '''
    if col_name != None:
        # Extract only the data we need
        this_data = data.loc[:, [col_name]]
    else:
        this_data = data

    if colorcycle is None:
        colorcycle = color_cycle

    # Create bar chart
    # linewidth adds a line around the bars
    # width changes the width of the bars
    ax = this_data.plot(kind='bar', rot=45, fontsize=6, color=colorcycle,
                        legend=False, linewidth=0, width=0.775)

    # Adds a label to the y axis
    if y_axis_label is not None:
        ax.set_ylabel(y_axis_label,fontsize=8)

    # Rezies the x axis label
    if x_axis_label is not None:
        x_axis_label = this_data.index.name
        ax.set_xlabel(x_axis_label,fontsize=8)

    # Adds commas to the y axis
    ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    if x_axis_labels is not None:
        plt.xticks(np.arange(0,len(x_axis_labels),1), x_axis_labels)
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, ha='center')


    # Save the plot to output
    fig = ax.get_figure()

    # Adjusts the padding on the sides of the plot
    fig.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.975)

    return fig, ax

def compatibility_matrix(data, cols, rows, col_names=None, row_names=None, marker=None, color='#0069AA'):

    # Use all columns if subset is not selected
    if col_names is None:
        col_names = cols
    if row_names is None:
        row_names = rows

    # Default to a circle
    if marker is None:
        marker=(1,3)

    # Append datapoints to the scatter plot data according to row/col location
    data = data.loc[rows,cols]
    data = pd.DataFrame(np.array(data))
    x, y = [], []
    for r in data.index:
        for c in data.columns:
            if data.iloc[r, c] == 1:
                y.append(r)
                x.append(c)

    # Set figure size so the x/y axis are equal in scale
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10 * len(rows)/len(cols))
    plt.tight_layout(h_pad=2, rect=[0.1, 0.1, 0.9, 0.9])
    ax.scatter(x,y, marker=marker, c=color, s=300)

    plt.xticks(np.arange(0,len(cols),1), col_names)
    plt.yticks(np.arange(0,len(rows),1), row_names)

    buffer = 0.5
    ax.set_xlim(min(x)-buffer,max(x)+buffer)
    ax.set_ylim(min(y)-buffer,max(y)+buffer)

    ax.xaxis.tick_top()
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, ha='left')

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_ylim(ax.get_ylim()[::-1])

    return fig, ax




def boxplot(data,
            ylabel=None,
            xlabel=None,
            scenario_names=None):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(10, 7)
    ax.boxplot(data)

    if ylabel is None:
        ylabel = ''
    if xlabel is None:
        xlabel = ''
    scenario_names = [x.replace(' + ', '\n') for x in scenario_names]
    scenario_names = [x.replace(' ', '\n', 1) for x in scenario_names]

    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)

    if scenario_names is not None:
        ax.set_xticklabels([x for x in scenario_names])

    for tick in ax.get_xticklabels():
        tick.set_rotation(0)

    return fig, ax

def tis(data,
        machines=None,
        states=None,
        colors=None,
        plotByScenario=None,
        scenarios=None,
        scenario_names=None,
        threshold=5):


    if colors is None:
        colors = color_cycle

    #data = check_scenario_replication(data)

    if machines is None:
        machines = data.select_dtypes(include=[object]).columns

    if plotByScenario is None:
        plotByScenario = False


    if states is None:
        states = data.State.unique()

    if scenarios is None:
        scenarios = data.Scenario.unique()

    if scenario_names is None:
        scenario_names = scenarios

    scenario_names = [x.replace(' + ', '\n') for x in scenario_names]
    #scenario_names = [x.replace(' ', '\n', 1) for x in scenario_names]

    nrows, ncols  = 1, 1
    rows, cols = None, None
    y_labels= None

    if plotByScenario:
        nrows, ncols = len(scenarios), len(machines)
        rows, cols = scenarios, machines
        y_labels = scenario_names
    else:
        nrows, ncols = len(machines), len(scenarios)
        rows, cols = machines, scenarios
        y_labels = rows

    fig, ax = plt.subplots(nrows=1, ncols=ncols)
    fig.set_size_inches(8, 3 + 1 * ncols)
    plt.tight_layout(h_pad=7, rect=[0.1, 0.15, 1, 0.98])




    for ic, c in zip(range(0, ncols), cols): #cols = scenarios
        # Select ax for plotting
        if ncols > 1:
            this_ax = ax[ic]
        else:
            this_ax = ax

        patch_handles = []
        this_data = []
        left = np.zeros(nrows)
        y_pos = np.arange(0,-1*nrows,-1)
        for istate, state in zip(range(0, len(states)), states): #rows = state

            d = []
            if plotByScenario:
                d = list(data[data.State == state][c])
            else:
                temp_data = data[data.Scenario == c]
                temp_data = temp_data[temp_data.State == state][rows]
                d = temp_data.values
                d = list(d[0])


            this_data.append(d)

            bar = this_ax.barh(y_pos,
                               d,
                               color=colors[istate % len(colors)],
                               align='center',
                               alpha=0.5,
                               height=0.9,
                               left=left,
                               linewidth=0)
            patch_handles.append(bar)
            left += d

        for j in range(len(patch_handles)):
                for k, patch in enumerate(patch_handles[j].get_children()):
                    bl = patch.get_xy()
                    x = 0.5*patch.get_width() + bl[0]
                    y = 0.5*patch.get_height() + bl[1]
                    if this_data[j][k]*100 > threshold:
                        this_ax.text(x,
                                     y,
                                     "%d%%" % round(this_data[j][k]*100),
                                     size='small',
                                     ha='center',
                                     va='center',
                                     weight='extra bold',
                                     color='black')

        this_ax.get_yaxis().set_ticks([])
        if ic == 0:
            this_ax.set_yticks(y_pos)
            this_ax.set_yticklabels(y_labels)
        this_ax.set_ylim(min(y_pos)-0.4,0.4)
        this_ax.set_title(str(cols[ic]))
        #this_ax.set_xlim(0,1)
        this_ax.set_facecolor('white')
        this_ax.set_xticks(np.arange(0,1.1,0.2))
        vals = this_ax.get_xticks()
        this_ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals])
        for tick in this_ax.get_xticklabels():
            tick.set_rotation(90)


        if ncols > 1:
            this_ax = ax[0]
        else:
            this_ax = ax


        legend_items = states

        # Stacked chart legend gets boxes
        patches = []
        if True:
            for i in range(0, len(legend_items)):
                patch = matplotlib.patches.Patch(color=colors[i % len(colors)])
                patches.append(patch)

        # Set the legend
        leg = fig.legend(patches,
                        legend_items,
                        bbox_to_anchor=(0.5, 0.12),
                        loc='upper center',
                        ncol=(len(legend_items) // 3 + 2),
                        fancybox=True,
                        fontsize=11)
        # Set the linewidth of each legend object
        leg.get_frame().set_facecolor('#FFFFFF')
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

    return fig, ax



def reconfigure_dataframe_by_scenario(data, col_name):
    '''
    Takes in wip data and reconfigures the col_name values to have columns by scenario
    '''

    # Select only the columns to rearrange dataframe by
    data = data[['Scenario', 'Replication', 'Time', col_name]]

    # Initialize output_data dataframe
    output_data = pd.DataFrame([])

    # Loop through the scenarios and build the dataframe by scenario
    for s in data.Scenario.unique():
        temp = data[data.Scenario == s].copy()

        # Name the columns properly
        scenario_col_name = 'Scenario ' + str(s)
        temp.rename(columns={col_name: scenario_col_name}, inplace=True)

        # Create and append the output dataframe
        if output_data.empty:  # If the dataframe is empty
            output_data = temp.copy()
        else:
            output_data[scenario_col_name] = temp[scenario_col_name]

    return output_data

def quantity_vs_time(data,
                     col_names=None,
                     stacked=False,
                     colorcycle=None):
    '''
    Creates a Nx1 grid of quantity vs. time charts during the model runs.
    '''

    if colorcycle is None:
        colorcycle = color_cycle
    # Check data has right format
    data = check_scenario_replication(data)

    # Initialize figure
    nrows = len(data.Scenario.unique())
    fig, ax = plt.subplots(nrows=nrows, ncols=1)

    # Default figure size
    fig.set_size_inches(10, 3 + 2.2 * nrows)
    plt.tight_layout(h_pad=2, rect=[0.05, 0.12, 0.95, 0.98])

    # Main loop
    for s, r in zip(data.Scenario.unique(), range(0, nrows)):
        # Select subplot for each scenario
        if nrows > 1:
            this_ax = ax[r]
        else:
            this_ax = ax

        # Calculate min, max, mean for all replications for a given sceanrio
        temp = data[data.Scenario == s].copy()
        data_mean = temp.groupby(temp.Time).mean()
        data_max = temp.groupby(temp.Time).max()
        data_min = temp.groupby(temp.Time).min()
        #data_max = temp.groupby(temp.Time).quantile(0.85)
        #data_min = temp.groupby(temp.Time).quantile(0.15)

        data_mean['Time'] = data_mean.index
        data_max['Time'] = data_max.index
        data_min['Time'] = data_min.index

        # Filters out Time, Scenario, Replication columns
        if col_names is None:
            cols = data.columns[3:]
        else:
            cols = col_names

        # Fills Between the Lines
        if stacked:
            # Mean value plot
            this_ax.stackplot(data_mean.Time,
                             np.array(data_mean[cols]).T,
                             alpha=0.75,
                             linewidth=0.5,
                             colors=colorcycle[:len(cols)])
        # Regular Line Plot
        else:
            for col, i in zip(cols, range(0, len(cols))):
                # Mean value plot
                this_ax.plot(data_mean.Time,
                                 data_mean[col],
                                 color=colorcycle[i % len(colorcycle)])
                # Min/Max value plot
                this_ax.fill_between(data_max.Time,
                                     data_min[col],
                                     data_max[col],
                                     alpha=0.3,
                                     color=colorcycle[i % len(colorcycle)])


        # Set axis labels
        this_ax.set_ylabel('Scenario: ' + str(s))
        this_ax.set_xlabel('Time')

        # Set x, y axis limits & formatting
        _, ub = this_ax.get_ylim()
        this_ax.set_xlim(data.Time.min(), data.Time.max())
        this_ax.set_ylim(0, ub)  # always set lb to 0
        this_ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Always select the first plot for formatting the legend
        if nrows > 1:
            this_ax = ax[0]
        else:
            this_ax = ax

        # Stacked chart legend gets boxes
        patches = []
        if stacked:
            for i in range(0, len(cols)):
                patch = matplotlib.patches.Patch(color=colorcycle[:len(cols)])
                patches.append(patch)

        # Regular chart legend gets lines
        else:
            patches, _ = this_ax.get_legend_handles_labels()

        # Set the legend
        leg = fig.legend(patches,
                        cols,
                        bbox_to_anchor=(0.5, 0.1),
                        loc='upper center',
                        ncol=(len(cols) // 2 + 1),
                        fancybox=True)

        # Set the linewidth of each legend object
        leg.get_frame().set_facecolor('#FFFFFF')
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

    return fig, ax

def histogram(data,
              col_name,
              bins=None,
              num_bins=None,
              group_by_col=None,
              percentage=False,
              x_lim=None,
              y_lim=None):
    '''
    Create histogram plots
    '''
    # Check data has right format
    data = check_scenario_replication(data)

    scenario_names = sorted(data.Scenario.unique())

    # Extract only the data we need
    this_data = data.loc[:, ['Scenario', col_name]]


    # Set the number of bins
    if bins is None:
        data_min = this_data[col_name].min()
        data_max = this_data[col_name].max()
        if group_by_col is None:
            n_bins = 100
        else:
            n_bins = 25

        if data_max < 1:
            data_max = 1
        if ((data_min > 0) & (data_min < 0.1)):
            data_min = 0
        bins = np.arange(data_min, data_max, (data_max - data_min) / n_bins)

    if num_bins is not None:
        data_min = this_data[col_name].min()
        data_max = this_data[col_name].max()

        if data_max < 1:
            data_max = 1
        if ((data_min > 0) & (data_min < 0.1)):
            data_min = 0
        bins = np.arange(data_min, data_max, (data_max - data_min) / num_bins)

    bin_width = bins[1] - bins[0]

    scenarios = this_data.Scenario.unique()
    numScenarios = len(scenarios)

    # Determine if we need more than one chart per scenario
    if group_by_col is not None:
        groups = sorted(data[group_by_col].unique())
        numGroups = len(groups) - 1
    else:
        numGroups = 0

    # Initialize figure
    fig, ax = plt.subplots(numScenarios, numGroups + 1)
    # Default figure size
    fig.set_tight_layout(False)
    fig.set_size_inches(12, 2.5 + 2 * numScenarios)

    i = 0 #iterator to go with s
    for s in sorted(this_data.Scenario.unique()):

        # Select subplot for each scenario
        if ((numScenarios == 1) & (numGroups == 0)):
            this_ax = ax
        elif (numScenarios > 1) & (numGroups == 0):
            this_ax = ax[i]

        # Plotting multiple histograms per scenario
        if numGroups != 0:
            j = 0 #iterator to go with g
            for g in groups:
                # Select subplot for each scenario & group
                if numScenarios > 1:
                    this_ax = ax[i, j]
                else:
                    this_ax = ax[j]

                # Extract only the data we need
                this_data = data[(data.Scenario == s) & (data[group_by_col] == g)][col_name]

                # Calculate bins
                model_data, _ = np.histogram(this_data,
                                             bins,
                                             normed=True)
                # Plot bins
                this_ax.bar(bins[1:],
                            model_data,
                            facecolor=color_cycle[j % len(color_cycle)],
                            alpha=1,
                            width=bin_width)

                # Set x axis limits
                if x_lim is None:
                    this_ax.set_xlim([data_min, data_max])
                else:
                    this_ax.set_xlim(x_lim)

                # Remove y axis numbers
                this_ax.set_yticklabels([])


                # Set histogram titles and y labels for top/left row/column plots
                if i == 0:
                    this_ax.set_title(g)
                if j == 0:
                    this_ax.set_ylabel('Scenario: ' + str(scenario_names[i]))


                vals = this_ax.get_xticks()
                # Format x axis labels as percent
                if percentage is True:
                    this_ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals])
                    for tick in this_ax.get_xticklabels():
                        tick.set_rotation(90)

                # Format x axis labels with thousands place comma
                else:
                    this_ax.get_xaxis().set_major_formatter(
                        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                    for tick in this_ax.get_xticklabels():
                        tick.set_rotation(45)
                j += 1

        # Plotting a single histogram per scenario
        else:
            # Extract only the data we need
            this_data = data[data.Scenario == s][col_name]

            # Calculate bins
            model_data, edges = np.histogram(this_data,
                                             bins,
                                             normed=True)
            # Plot bins
            this_ax.bar(bins[1:],
                        model_data,
                        facecolor=color_cycle[0 % len(color_cycle)],
                        alpha=1,
                        width=bin_width)

            # Remove y axis numbers
            this_ax.set_yticklabels([])

            # Set x axis limits
            if x_lim is None:
                this_ax.set_xlim([data_min, data_max])
            else:
                this_ax.set_xlim(x_lim)

            this_ax.set_ylabel('Scenario: ' + str(scenario_names[i]))

            # Format x axis labels as percent
            if percentage is True:
                vals = this_ax.get_xticks()
                this_ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals])
        i += 1


    return fig, ax


if __name__ == '__main__':
    pass