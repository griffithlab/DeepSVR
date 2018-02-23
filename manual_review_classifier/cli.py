import pkg_resources
import click

from manual_review_classifier.PrepareData import PrepareData
from ClassifyData import ClassifyData
from ReadCount import ReadCount


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    version = pkg_resources.\
        get_distribution('manual_review_classifier').version
    click.echo(version)
    ctx.exit()


@click.command()
@click.option('-v', '--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True)
@click.help_option('-h', '--help')
@click.option('--header/--no-header',
              default=False,
              help='Specify whether header is present in sample file')
@click.option('--skip_bam_readcount/--no-skip_bam_readcount',
              default=False,
              help='If bam readcount files already exist in output directory, '
                   'skip the bam-readcount step')
@click.option('--samples-file-path', '-sfp',
              help='File path of tsv file with sample information. File should'
                   'have the following columns in order: sample_name, '
                   'tumor_bam_path, normal_bam_path, manual_review_file_path'
                   ', reviewer, disease, reference_genome_fasta_file_path.'
                   'If the reviewer is specified in the manual review file or '
                   'not available for the dataset leave the column blank.'
                   'The manual review file should have the following columns'
                   'in order: chromosome, start, stop, reference, variant,'
                   'call, tag, notes, reviewer. For the manual review file'
                   'the reviewer column is optional. Labels need to match'
                   'above.')
@click.option('--output-dir-path', '-odp', default='~/training_data',
              help='Specify output directory: Readcount files and compressed '
                   'pandas dataframe will be output here '
                   '(default:~/training_data)')
@click.option('--classifier', '-cl', default='~/deep_learning_classifier.json',
              help='Specify classifier file: compressed .json '
                   '(default: ../data/deep_learning_classifier.json)')
@click.option('--solid_tumor/--no-solid_tumor',
              default=True,
              help='Designate if tumor is solid tumor or hematologic tumor')



def main(header, skip_bam_readcount, samples_file_path, output_dir_path, classifier, solid_tumor):
    """
    Prepare data for training or classification in manual review model.
    """

    
    PrepareData(samples_file_path, header, output_dir_path, skip_bam_readcount)
    ClassifyData(solid_tumor, classifier, samples_file_path, header, output_dir_path)


if __name__ == '__main__':
    main()
