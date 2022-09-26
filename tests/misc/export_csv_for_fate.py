import numpy as np
import pandas as pd


def export_fate_csv(role, dataset_name, trainset, testset):
    print('Exporting LinkeFL dataset to FATE-compatible CSV files...')
    if role == 'active_party':
        header = ['id', 'y']
    else:
        header = ['id']

    header.extend(['x' + str(i) for i in range(trainset.n_features)])
    df = pd.DataFrame(
        np.concatenate((trainset.get_dataset(), testset.get_dataset()), axis=0),
        columns=header
    )

    if role == 'active_party':
        df['id'] = df['id'].astype('Int64')
        df['y'] = df['y'].astype('Int64')
        df.to_csv('{}_active_full'.format(dataset_name), index=False)
    else:
        df['id'] = df['id'].astype('Int64')
        df.to_csv('{}_passive_full'.format(dataset_name), index=False)
    print('Done.')