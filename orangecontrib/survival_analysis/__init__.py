from Orange.data import Table

from .evaluation import scoring  # noqa: F401

# Remove this when we require Orange 3.34
if not hasattr(Table, "get_column"):
    import scipy.sparse as sp
    import numpy as np

    def get_column(self, column):
        col, _ = self.get_column_view(column)
        if sp.issparse(col):
            col = col.toarray().reshape(-1)
        if self.domain[column].is_primitive():
            col = col.astype(np.float64)
        return col

    Table.get_column = get_column
