import numpy as np
import pandas as pd

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')


def _calculate_hist(probabilities,
                    bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]):
    '''Calculates the counts and mean of each bin for a histogram using \
    defined bin intervals.

    Parameters:
        probabilities (numpy.array): Array of probabilites
        bins (list): Defined bins to calculate. Default [0,.1,.2,.3,.4,
                     .5,.6,.7,.8,.9,1]
    Returns:
        (counts, means): Lists of the counts and means of each bin
    '''
    counts = list()
    means = list()
    for i in range(0, len(bins) - 1):
        counts.append(len(probabilities[(probabilities > bins[i])
                                        & (probabilities <= bins[i + 1])]))
        means.append(np.mean(probabilities[(probabilities > bins[i])
                                           & (probabilities <= bins[i + 1])]))
    return np.array(counts), np.array(means)


def create_reliability_diagram(probability_array, Y, columns, highlight_color,
                               title):
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    prob_df = pd.DataFrame(probability_array, columns=columns)

    label_index = pd.DataFrame(Y, columns=columns, dtype=bool)

    positive_probabilities = prob_df[label_index].unstack().dropna().values

    positive_counts, positive_means = _calculate_hist(positive_probabilities,
                                                      bins)

    negative_probabilities = prob_df[~label_index].unstack().dropna()

    negative_counts, negative_means = _calculate_hist(negative_probabilities,
                                                      bins)

    pct_positive = positive_counts / (positive_counts + negative_counts)

    # calculate confidence interval
    # see https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha = 0.05
    z = 1 - ((1/2) * alpha)
    pct_negative = 1 - pct_positive
    inverse_n = 1/(positive_counts+negative_counts)
    con_ints = z * np.sqrt(inverse_n * pct_positive * pct_negative)
    print('confidence intervals +-: ', con_ints)

    width = 0.04  # the width of the bars

    fig, ax1 = plt.subplots()
    rects1 = ax1.bar(positive_means - width / 2, negative_counts, width,
                     color='black', edgecolor='black')
    rects2 = ax1.bar(positive_means + width / 2, positive_counts, width,
                     color='white', edgecolor='black')

    # add some text for labels, title and axes ticks

    ax1.set_title(title)
    ax1.set_ylabel('Count')
    ax1.legend((rects1[0], rects2[0]), ('Prediction disagrees with call',
                                        'Prediction agrees with call'),
               loc='upper left', bbox_to_anchor=(.1, 1))

    ax2 = ax1.twinx()
    ax2.plot([0, 1], [0, 1], 'k--', color='grey')
    r2 = metrics.r2_score(positive_means, pct_positive)
    (_, caps, _) = ax2.errorbar(positive_means, pct_positive, yerr=con_ints,
                                fmt="-o", color=highlight_color, markersize=8,
                                capsize=8)
    for cap in caps:
        cap.set_markeredgewidth(1)
    ax2.text(.6, .85, '$R^2$: {0:0.2f}'.format(r2), color=highlight_color,
             fontsize=15)
    ax2.tick_params(axis='y', colors=highlight_color)
    ax2.set_ylabel("Percent call agreement")

    plt.show()
