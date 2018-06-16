import numpy as np
import pandas as pd

from sklearn import metrics
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from itertools import cycle

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
        count = len(probabilities[(probabilities > bins[i])
                                  & (probabilities <= bins[i + 1])])
        if count == 0:
            counts.append(0)
            bin_mean = np.mean([bins[i], bins[i+1]])
            means.append(bin_mean)
        else:
            counts.append(count)
            means.append(np.mean(probabilities[(probabilities > bins[i]) &
                                               (probabilities <= bins[i + 1])])
                         )
    return np.array(counts), np.array(means)


def create_reliability_diagram(probability_array, Y, columns, highlight_color,
                               title, ax1, y2_label, y_label, legend):
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

    pct_positive = np.nan_to_num(pct_positive)

    # calculate confidence interval
    # see https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha = 0.05
    z = 1 - ((1/2) * alpha)
    pct_negative = 1 - pct_positive
    inverse_n = 1/(positive_counts+negative_counts)
    con_ints = z * np.sqrt(inverse_n * pct_positive * pct_negative)
    print('confidence intervals +-: ', con_ints)

    width = 0.04  # the width of the bars

    # fig, ax1 = plt.subplots()
    rects1 = ax1.bar(positive_means - width / 2, negative_counts, width,
                     color='black', edgecolor='black')
    rects2 = ax1.bar(positive_means + width / 2, positive_counts, width,
                     color='white', edgecolor='black')

    # add some text for labels, title and axes ticks

    ax1.set_title(title, fontsize=8)
    if y_label:
        ax1.set_ylabel('Count (1000s)', fontsize=8)
    ax1.set_xlabel('Model output', fontsize=8)
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
    ax1.yaxis.set_major_formatter(ticks_y)
    ax1.tick_params(labelsize=8, pad=1.5)
    ax1.yaxis.labelpad = 1
    ax1.xaxis.labelpad = 1

    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=8, pad=1.5)
    ax2.plot([0, 1], [0, 1], 'k--', color='grey', linewidth=1)
    r = pearsonr(positive_means, pct_positive)[0]
    (_, caps, _) = ax2.errorbar(positive_means, pct_positive, yerr=con_ints,
                                fmt="-o", color=highlight_color, markersize=2,
                                capsize=1, linewidth=.5)
    for cap in caps:
        cap.set_markeredgewidth(1)
    ax2.tick_params(axis='y', colors=highlight_color)
    if y2_label:
        ax2.set_ylabel("Percent call agreement", fontsize=8)
    ax2.yaxis.labelpad = 1

    ax2.text(.15, .5, 'r={0:0.2f}'.format(r), color=highlight_color,
             fontsize=8)
    if legend:
        return ax1.legend((rects1[0], rects2[0]),
                          ('Prediction disagrees with call',
                           'Prediction agrees with call'),
                          loc='lower left', bbox_to_anchor=(0, -0.4),
                          fontsize=8)


def create_roc_curve(Y, probabilities, class_lookup, title, ax):
    '''Create ROC curve to compare multiclass model performance.

    Parameters:
        Y (numpy.array): Truth labels
        probabilities (numpy.array): Output of model for each class
        class_lookup (dict): lookup hash of truth labels
        title (str): Plot title
    '''
    n_classes = Y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    ax.set_title(title)
    if n_classes == 3:
        colors = cycle(['orange', 'red', 'black'])
    else:
        colors = cycle(['orange', 'red', 'aqua', 'black'])
    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = metrics.roc_curve(Y[:, i], probabilities[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], color=color,
                label='ROC curve of class {0} (area = {1:0.2f})'.format(
                    class_lookup[i], roc_auc[i]))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    # plt.show()


def create_feature_importance_plot(feature_importance_metrics, title):
    '''Create bar plot illustrating importance of each feature.

    Parameters:
        feature_importance_metrics (pandas.DataFrame): DataFrame with features
                                                       on index and a column
                                                       named delta_auc
                                                       containing the change
                                                       in roc auc values
        title (str): Title of plot
    '''
    feature_importance_metrics.replace(
        {'feature': {'var': 'variant', 'ref': 'reference',
                     'avg': 'average',
                     '_se_': '_single_end_',
                     '3p': '3_prime', '_': ' '}},
        regex=True, inplace=True)
    sns.barplot(y='feature', x='delta_auc',
                data=feature_importance_metrics.head(30),
                color='cornflowerblue')
    plt.xlabel('Delta average AUC')
    plt.ylabel('Feature')
    plt.title(title)


def make_model_output_plot(probabilities, title):
    """Make plot that show the distribution of model output

        Parameters:
            probabilities (numpy.array): array of model output with samples
                                         on row and classes in columns
            title (str): Plot title
    """
    ax = sns.distplot(probabilities[:, 0:1])
    sns.distplot(probabilities[:, 1:2], ax=ax)
    sns.distplot(probabilities[:, 2:3], ax=ax)
    ax.legend(['Ambiguous', 'Fail', 'Somatic'])
    ax.set_xlabel('Model output')
    ax.set_ylabel('Density')
    ax.set_title(title)
