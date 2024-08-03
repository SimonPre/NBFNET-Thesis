from util.helper_classes.padding_holder import ToPad


def compute_paddings_sizes_for_batching(
    batch_size: int, n_triples_train: int, n_triples_valid: int, n_triples_test: int
) -> ToPad:
    """
    Helper function to compute how many triples need to be added to a test, train and validation to make total length
    evenly divisible by the target batch_size.
    :param batch_size: target batch size
    :param n_triples_train: number of triples in train
    :param n_triples_valid: number of triples in validation
    :param n_triples_test: number of triples in test
    :return:
        Helper object holding the three computed numbers of how many triples need to be added.
    """
    return ToPad(
        train=(batch_size - (n_triples_train % batch_size)) % batch_size,
        valid=int(batch_size / 2 - (n_triples_valid % (batch_size / 2))),
        test=int(batch_size / 2 - (n_triples_test % (batch_size / 2))),
    )
