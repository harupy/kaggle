

def assert_columns_equal(left, right):
    ncols_l, ncols_r = len(left.columns), len(right.columns)
    assert ncols_l == ncols_r, 'columns length must be equal: (left: {}, right: {}).'.format(ncols_l, ncols_r)

    cols_diff = [col_pair for col_pair in zip(left.columns, right.columns) if len(set(col_pair)) != 1]
    assert len(cols_diff) == 0, 'columns order must be equal: {}.'.format(cols_diff)


def assert_dtypes_equal(left, right):
    dtypes_diff = [(col, dt_l, dt_r) for col, dt_l, dt_r in zip(left.columns, left.dtypes, right.dtypes) if dt_l != dt_r]
    assert len(dtypes_diff) == 0, 'dtypes must be equal: {}'.format(dtypes_diff)
