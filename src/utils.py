from sklearn.model_selection import BaseCrossValidator

class TimeStepSplit(BaseCrossValidator):
    def __init__(self, df, n_splits=4):
        self.df = df
        self.n_splits = n_splits
        self.time_steps = sorted(df['timeStep'].unique())

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        n_time_steps = len(self.time_steps)
        fold_size = n_time_steps // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size

            train_steps = self.time_steps[:train_end]
            test_steps = self.time_steps[test_start:test_end]

            train_idx = self.df[self.df['timeStep'].isin(train_steps)].index.to_numpy()
            test_idx = self.df[self.df['timeStep'].isin(test_steps)].index.to_numpy()

            yield train_idx, test_idx