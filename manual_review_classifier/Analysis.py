import numpy as np
import pandas as pd

def determine_feature_importance(model, X, Y):
    """Fits model on entire training dataset then shuffles one feature at a 
       time and evaluated the performace of each feature individually.

        Returns (pandas.Dataframe) Dataframe of accuracy measures"""
    model.fit(X, Y, epochs=10, batch_size=5)

    unshuffled_metric = model.evaluate(X, Y, batch_size=5)

    # dictionary of slice objects to select each feature individually
    features_to_shuffle = {'disease': np.s_[:, 0:8],
                           'reviewer': np.s_[:, 9:12],
                           'normal_VAF': np.s_[:, 13],
                           'normal_depth': np.s_[:, 14],
                           'normal_other_bases_count': np.s_[:, 15],
                           'normal_ref_avg_basequality': np.s_[:, 16],
                           'normal_ref_avg_clipped_length': np.s_[:, 17],
                           'normal_ref_avg_distance_to_effective_3p_end': np.s_[
                                                                          :,
                                                                          18],
                           'normal_ref_avg_distance_to_q2_start_in_q2_reads': np.s_[
                                                                              :,
                                                                              19],
                           'normal_ref_avg_mapping_quality': np.s_[:, 20],
                           'normal_ref_avg_num_mismaches_as_fraction': np.s_[:,
                                                                       21],
                           'normal_ref_avg_pos_as_fraction': np.s_[:, 22],
                           'normal_ref_avg_se_mapping_quality': np.s_[:, 23],
                           'normal_ref_avg_sum_mismatch_qualities': np.s_[:,
                                                                    24],
                           'normal_ref_count': np.s_[:, 25],
                           'normal_ref_num_minus_strand': np.s_[:, 26],
                           'normal_ref_num_plus_strand': np.s_[:, 27],
                           'normal_ref_num_q2_containing_reads': np.s_[:, 28],
                           'normal_var_avg_basequality': np.s_[:, 29],
                           'normal_var_avg_clipped_length': np.s_[:, 30],
                           'normal_var_avg_distance_to_effective_3p_end': np.s_[
                                                                          :,
                                                                          31],
                           'normal_var_avg_distance_to_q2_start_in_q2_reads': np.s_[
                                                                              :,
                                                                              32],
                           'normal_var_avg_mapping_quality': np.s_[:, 33],
                           'normal_var_avg_num_mismaches_as_fraction': np.s_[:,
                                                                       34],
                           'normal_var_avg_pos_as_fraction': np.s_[:, 35],
                           'normal_var_avg_se_mapping_quality': np.s_[:, 36],
                           'normal_var_avg_sum_mismatch_qualities': np.s_[:,
                                                                    37],
                           'normal_var_count': np.s_[:, 38],
                           'normal_var_num_minus_strand': np.s_[:, 39],
                           'normal_var_num_plus_strand': np.s_[:, 40],
                           'normal_var_num_q2_containing_reads': np.s_[:, 41],
                           'tumor_VAF': np.s_[:, 42],
                           'tumor_depth': np.s_[:, 43],
                           'tumor_other_bases_count': np.s_[:, 44],
                           'tumor_ref_avg_basequality': np.s_[:, 45],
                           'tumor_ref_avg_clipped_length': np.s_[:, 46],
                           'tumor_ref_avg_distance_to_effective_3p_end': np.s_[
                                                                         :,
                                                                         47],
                           'tumor_ref_avg_distance_to_q2_start_in_q2_reads': np.s_[
                                                                             :,
                                                                             48],
                           'tumor_ref_avg_mapping_quality': np.s_[:, 49],
                           'tumor_ref_avg_num_mismaches_as_fraction': np.s_[:,
                                                                      50],
                           'tumor_ref_avg_pos_as_fraction': np.s_[:, 51],
                           'tumor_ref_avg_se_mapping_quality': np.s_[:, 52],
                           'tumor_ref_avg_sum_mismatch_qualities': np.s_[:,
                                                                   53],
                           'tumor_ref_count': np.s_[:, 54],
                           'tumor_ref_num_minus_strand': np.s_[:, 55],
                           'tumor_ref_num_plus_strand': np.s_[:, 56],
                           'tumor_ref_num_q2_containing_reads': np.s_[:, 57],
                           'tumor_var_avg_basequality': np.s_[:, 58],
                           'tumor_var_avg_clipped_length': np.s_[:, 59],
                           'tumor_var_avg_distance_to_effective_3p_end': np.s_[
                                                                         :,
                                                                         60],
                           'tumor_var_avg_distance_to_q2_start_in_q2_reads': np.s_[
                                                                             :,
                                                                             61],
                           'tumor_var_avg_mapping_quality': np.s_[:, 62],
                           'tumor_var_avg_num_mismaches_as_fraction': np.s_[:,
                                                                      63],
                           'tumor_var_avg_pos_as_fraction': np.s_[:, 64],
                           'tumor_var_avg_se_mapping_quality': np.s_[:, 65],
                           'tumor_var_avg_sum_mismatch_qualities': np.s_[:,
                                                                   66],
                           'tumor_var_count': np.s_[:, 67],
                           'tumor_var_num_minus_strand': np.s_[:, 68],
                           'tumor_var_num_plus_strand': np.s_[:, 69],
                           'tumor_var_num_q2_containing_reads': np.s_[:, 70]}
    shuffled_metrics = list()
    for feature in features_to_shuffle:
        shuffled_X = np.copy(X)
        np.random.shuffle(shuffled_X[features_to_shuffle[feature]])
        shuffled_metric = model.evaluate(shuffled_X, Y, batch_size=5)
        shuffled_metrics.append([feature] + shuffled_metric)

    feature_metrics = pd.DataFrame(shuffled_metrics,
                                   columns=['feature', 'shuffled_loss',
                                            'shuffled_accuracy'])

    feature_metrics['delta_loss'] = unshuffled_metric[
                                        0] - feature_metrics.shuffled_loss
    feature_metrics['delta_accuracy'] = unshuffled_metric[
                                            1] - feature_metrics.shuffled_accuracy
    feature_metrics.sort_values('delta_accuracy', ascending=False,
                                inplace=True)
    return feature_metrics
