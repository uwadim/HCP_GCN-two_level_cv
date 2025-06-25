from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import numpy as np


class StratifiedSampler(object):
    """Stratified Sampling

    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size, random_state):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector
        self.random_state = random_state

    def gen_sample_array(self):
        skf = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5, random_state=self.random_state)
        x_random = np.random.rand(self.class_vector.size(0))
        y_real = self.class_vector.numpy()

        train_index, test_index = next(skf.split(x_random, y_real))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


class StratifiedKFoldSampler(object):
    """
    Get stratified indices for n folds
    """

    def __init__(self, class_vector, n_splits, random_state, is_shuffle=False):
        self.n_splits = n_splits
        self.class_vector = class_vector
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=is_shuffle)

    def get_samplers(self):
        # Fake features for making split
        x_random = np.random.rand(self.class_vector.size(0))
        y_real = self.class_vector.numpy()
        result_dict = {}
        for idx, (train_indices, test_indices) in enumerate(self.skf.split(x_random, y_real)):
            result_dict[idx] = {'train_indices': train_indices,
                                'test_indices': test_indices}
        return result_dict
