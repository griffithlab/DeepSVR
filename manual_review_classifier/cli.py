import pkg_resources
import click

from manual_review_classifier.PrepareData import PrepareData


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
@click.option('--sample-file-path', '-sfp',
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
def main(header, sample_file_path,  output_dir_path):
    """
    Prepare data for training or classification in manual review model.
    """

    PrepareData(sample_file_path, header, output_dir_path)


if __name__ == '__main__':
    main()
