import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing


def determine_feature_importance(model, X, Y, remove_reviewer):
    """Fits model on entire training dataset then shuffles one feature at a
       time and evaluated the performace of each feature individually.

        Args:
            model (KerasClassifier): Trained model
            X (numpy.array): Features
            Y (numpy.array): Classes
            remove_reviewer (bool): True removes reviewer feature

        Returns (pandas.Dataframe) Dataframe of accuracy measures"""

    unshuffled_auc = get_roc_auc(model.predict_proba(X), Y)

    # dictionary of slice objects to select each feature individually
    features_to_shuffle = {'disease': np.s_[:, 0:8],
                           'reviewer': np.s_[:, 9:12],
                           'normal_VAF': np.s_[:, 13],
                           'normal_depth': np.s_[:, 14],
                           'normal_other_bases_count': np.s_[:, 15],
                           'normal_ref_avg_basequality': np.s_[:, 16],
                           'normal_ref_avg_clipped_length': np.s_[:, 17],
                           'normal_ref_avg_distance_to_effective_3p_end':
                               np.s_[:, 18],
                           'normal_ref_avg_distance_to_q2_start_in_q2_reads':
                               np.s_[:, 19],
                           'normal_ref_avg_mapping_quality': np.s_[:, 20],
                           'normal_ref_avg_num_mismaches_as_fraction':
                               np.s_[:, 21],
                           'normal_ref_avg_pos_as_fraction': np.s_[:, 22],
                           'normal_ref_avg_se_mapping_quality': np.s_[:, 23],
                           'normal_ref_avg_sum_mismatch_qualities':
                               np.s_[:, 24],
                           'normal_ref_count': np.s_[:, 25],
                           'normal_ref_num_minus_strand': np.s_[:, 26],
                           'normal_ref_num_plus_strand': np.s_[:, 27],
                           'normal_ref_num_q2_containing_reads': np.s_[:, 28],
                           'normal_var_avg_basequality': np.s_[:, 29],
                           'normal_var_avg_clipped_length': np.s_[:, 30],
                           'normal_var_avg_distance_to_effective_3p_end':
                               np.s_[:, 31],
                           'normal_var_avg_distance_to_q2_start_in_q2_reads':
                               np.s_[:, 32],
                           'normal_var_avg_mapping_quality': np.s_[:, 33],
                           'normal_var_avg_num_mismaches_as_fraction':
                               np.s_[:, 34],
                           'normal_var_avg_pos_as_fraction': np.s_[:, 35],
                           'normal_var_avg_se_mapping_quality': np.s_[:, 36],
                           'normal_var_avg_sum_mismatch_qualities':
                               np.s_[:, 37],
                           'normal_var_count': np.s_[:, 38],
                           'normal_var_num_minus_strand': np.s_[:, 39],
                           'normal_var_num_plus_strand': np.s_[:, 40],
                           'normal_var_num_q2_containing_reads': np.s_[:, 41],
                           'tumor_VAF': np.s_[:, 42],
                           'tumor_depth': np.s_[:, 43],
                           'tumor_other_bases_count': np.s_[:, 44],
                           'tumor_ref_avg_basequality': np.s_[:, 45],
                           'tumor_ref_avg_clipped_length': np.s_[:, 46],
                           'tumor_ref_avg_distance_to_effective_3p_end':
                               np.s_[:, 47],
                           'tumor_ref_avg_distance_to_q2_start_in_q2_reads':
                               np.s_[:, 48],
                           'tumor_ref_avg_mapping_quality': np.s_[:, 49],
                           'tumor_ref_avg_num_mismaches_as_fraction':
                               np.s_[:, 50],
                           'tumor_ref_avg_pos_as_fraction': np.s_[:, 51],
                           'tumor_ref_avg_se_mapping_quality': np.s_[:, 52],
                           'tumor_ref_avg_sum_mismatch_qualities':
                               np.s_[:, 53],
                           'tumor_ref_count': np.s_[:, 54],
                           'tumor_ref_num_minus_strand': np.s_[:, 55],
                           'tumor_ref_num_plus_strand': np.s_[:, 56],
                           'tumor_ref_num_q2_containing_reads': np.s_[:, 57],
                           'tumor_var_avg_basequality': np.s_[:, 58],
                           'tumor_var_avg_clipped_length': np.s_[:, 59],
                           'tumor_var_avg_distance_to_effective_3p_end':
                               np.s_[:, 60],
                           'tumor_var_avg_distance_to_q2_start_in_q2_reads':
                               np.s_[:, 61],
                           'tumor_var_avg_mapping_quality': np.s_[:, 62],
                           'tumor_var_avg_num_mismaches_as_fraction':
                               np.s_[:, 63],
                           'tumor_var_avg_pos_as_fraction': np.s_[:, 64],
                           'tumor_var_avg_se_mapping_quality': np.s_[:, 65],
                           'tumor_var_avg_sum_mismatch_qualities':
                               np.s_[:, 66],
                           'tumor_var_count': np.s_[:, 67],
                           'tumor_var_num_minus_strand': np.s_[:, 68],
                           'tumor_var_num_plus_strand': np.s_[:, 69],
                           'tumor_var_num_q2_containing_reads': np.s_[:, 70]}
    if remove_reviewer:
        features_to_shuffle.pop('reviewer')
        for feature in features_to_shuffle:
            if feature is not 'disease':
                features_to_shuffle[feature] = np.s_[:, features_to_shuffle[
                                                        feature][1] - 4]

    shuffled_aucs = list()
    for feature in features_to_shuffle:
        shuffled_X = np.copy(X)
        np.random.shuffle(shuffled_X[features_to_shuffle[feature]])
        shuffled_auc = get_roc_auc(model.predict_proba(shuffled_X), Y)
        shuffled_aucs.append([feature, shuffled_auc])

    feature_aucs = pd.DataFrame(shuffled_aucs,
                                columns=['feature', 'shuffled_auc'])
    feature_aucs['delta_auc'] = unshuffled_auc - feature_aucs.shuffled_auc
    feature_aucs.sort_values('delta_auc', ascending=False, inplace=True)
    return feature_aucs


def get_roc_auc(probabilities, Y):
    # for binary classifier add labels for fail category
    if len(Y.shape) == 1:
        Y = np.array([np.logical_not(Y), Y]).T
    n_classes = Y.shape[1]
    fpr = [0] * n_classes
    tpr = [0] * n_classes
    roc_auc = [0] * n_classes
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(Y[:, i], probabilities[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    return np.mean(roc_auc)


def print_accuracy_and_classification_report(labels, prediction):
    """Print model accuracy and classification report.

    Args:
        labels (numpy.array): Truth lables
        prediction (numpy.array): Model predictions
    """
    print('Cross validation accuracy:')
    print('\t', metrics.accuracy_score(labels, prediction))
    print('\nCross validation classification report\n')
    print(metrics.classification_report(labels, prediction))


def predict_classes(probabilities):
    """Predict class labels from probabilities

    Args:
        probabilities (numpy.array): model output for each class
    Returns:
        (numpy.array): Model predicted class labels
    """
    predicted = np.array(
        [list(a).index(max(list(a))) for a in list(probabilities)])
    label_binarizer = preprocessing.LabelBinarizer()

    label_binarizer.fit(range(max(predicted) + 1))
    return label_binarizer.transform(predicted)


def get_somatic_error_type(truth, prediction):
    """Return a dataframe that outlines the somaric error type i.e. False pos

    Args:
        truth (np.array): one hot encoded array with truth labels columns=
                          ['a', 'f', 's']
        prediction (np.array): one column array of predictions. 0=a, 1=f, 2=s
    Returns:
        (pandas.DataFrame): dataframe with columns truth label, predicted
                            label, and somatic error type
    """
    somatic_error_type = []
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(prediction)+1))
    predicted_transformed = label_binarizer.transform(prediction)
    somatic_prediction = predicted_transformed[:, 2:]
    for call in range(len(somatic_prediction)):
        if somatic_prediction[call] and truth[:, 2:][call]:
            somatic_error_type.append('True Positive')
        elif somatic_prediction[call] and not truth[:, 2:][call]:
            somatic_error_type.append('False Negative')
        elif not somatic_prediction[call] and truth[:, 2:][call]:
            somatic_error_type.append('False Positive')
        elif not somatic_prediction[call] and not truth[:, 2:][call]:
            somatic_error_type.append('True Negative')
    somatic_error = pd.DataFrame([np.argmax(truth, axis=1), prediction,
                                  np.array(somatic_error_type)]).T
    somatic_error.columns = ['truth', 'prediction', 'error']
    somatic_error.replace({0: 'a', 1: 'f', 2: 's'}, inplace=True)
    return somatic_error


def calculate_kappa(table):
    """https://en.wikipedia.org/wiki/Fleiss%27_kappa"""
    N, k = table.shape

    n = table.sum(axis=1)[0]

    p = table.sum(axis=0) / (N * n)

    P = (1 / (n * (n - 1))) * ((table ** 2).sum(axis=1) - n)

    p_bar = P.sum() / N

    P_e = (p ** 2).sum()

    kappa = (p_bar - P_e) / (1 - P_e)

    return kappa
