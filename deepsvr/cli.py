import pkg_resources
import click

from deepsvr.PrepareData import PrepareData
from deepsvr.ClassifyData import classify_samples
from deepsvr.CreateClassifier import create_classifier


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    version = pkg_resources.\
        get_distribution('deepsvr').version
    click.echo(version)
    ctx.exit()


@click.group()
@click.option('-v', '--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True)
@click.help_option('-h', '--help')
def main():
    click.echo('DeepSVR')


@main.command()
@click.help_option('-h', '--help')
@click.option('--header/--no-header',
              default=False,
              help='Specify whether header is present in sample file')
@click.option('--skip_bam_readcount/--no-skip_bam_readcount',
              default=False,
              help='If bam readcount files already exist in output directory '
                   'as a result of a prior run of the prepare_data command, '
                   'skip the bam-readcount step')
@click.option('--samples-file-path', '-sfp',
              help='File path of tsv file with sample information. File should'
                   'have the following columns in order: sample_name, '
                   'tumor_bam_path, normal_bam_path, manual_review_file_path'
                   ', reviewer, solid_tumor, reference_genome_fasta_file_path.'
                   'If the reviewer is specified in the manual review file or '
                   'not available for the dataset leave the column blank.'
                   'The manual review file should have the following columns'
                   'in order: chromosome, start, stop, reference, variant,'
                   'call, tag, notes, reviewer. For the manual review file'
                   'the reviewer column is optional. The solid tumor column '
                   'should be 0 or 1, 0 denotes a blood based tumor and 1 a '
                   'solid tumor. Labels need to matchabove.')
@click.option('--output-dir-path', '-odp', default='~/training_data',
              help='Specify output directory: Readcount files and compressed '
                   'pandas dataframe will be output here '
                   '(default:~/training_data)')
def prepare_data(header, skip_bam_readcount,
                 samples_file_path, output_dir_path):
    """
    Prepare data for training or classification.
    """
    PrepareData(samples_file_path, header, output_dir_path, skip_bam_readcount)


@main.command()
@click.help_option('-h', '--help')
@click.option('--prepared-data-path', '-pdp',
              help='Specify the \'train.pkl\' file produced by the '
                   '\'prepare_data\' to perform inference on. Ignore the '
                   'call.pkl used in training classifiers')
@click.option('--model-file-path', '-mfp',
              help='Specify the file path for the model json file. Created by '
                   'the train_classifier command.')
@click.option('--model-weights-path', '-mwp',
              help='Specify the file path for the model weights file. Created '
                   'by the train_classifier command.')
@click.option('--predictions-out-path', '-pop',
              help='Specify the file path for the predictions tab separated '
                   'file.')
def classify_data(prepared_data_path, model_file_path, model_weights_path,
                  predictions_out_path):
    """Preform automated somatic variant refinement on mutations."""
    classify_samples(prepared_data_path, model_file_path, model_weights_path,
                     predictions_out_path)
    print('Classification complete. '
          'Predictions in {}'.format(predictions_out_path))


@main.command()
@click.help_option('-h', '--help')
@click.option('--training-file-path', '-tfp',
              help='Specify the pickle file produced by the \'prepare_data\' '
                   'command to be used to train a new classifier.')
@click.option('--label-file-path', '-lfp',
              help='Specify the label (manual review call) pickle file '
                   'produced by the \'prepare_data\' command to be used to '
                   'train a new classifier.')
@click.option('--model-out-file-path', '-mop',
              default='./deepsvr_model.json',
              help='Specify output file path for model json file'
                   '(default:./deepsvr_model.json)')
@click.option('--weights-out-file-path', '-wop',
              default='./deepsvr_model_weights.h5',
              help='Specify output file path for model weights file'
                   '(default:data/deepsvr_model_weights.h5)')
def train_classifier(training_file_path, label_file_path, model_out_file_path,
                     weights_out_file_path):
    """Train a new classifier for somatic variant refinement."""
    create_classifier(training_file_path, label_file_path, model_out_file_path,
                      weights_out_file_path)


if __name__ == '__main__':
    main()
