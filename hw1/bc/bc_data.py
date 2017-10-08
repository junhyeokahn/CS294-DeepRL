import pickle
import numpy as np

class Data(object):

    def __init__(self, data_file, train_ratio, val_ratio):
        data = pickle.load((open(data_file, "rb")))

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.train, self.val, self.test = self.split(data)
        print('[Data Distrubuted]')

        # self.print_stat(self.train, "Training")
        # self.print_stat(self.val, "Validation")
        # self.print_stat(self.test, "Test")

        obs_mean = np.mean(self.train["observations"], axis=0)
        obs_std = np.std(self.train["observations"], axis=0)

        self.train, self.val, self.test = self.split(data)
        # self.normalize(self.train, obs_mean, obs_std)
        # self.normalize(self.val, obs_mean, obs_std)
        # self.normalize(self.test, obs_mean, obs_std)
        print('[Observation Normalized]')


    def split(self, data):
        """
        split dataset into trainig set, validate set and test set
        """
        obs, actions = data["observations"], data["actions"]
        actions = np.squeeze(actions)
        assert len(obs) == len(actions)

        n_total = len(obs)
        n_train, n_val = int(n_total * self.train_ratio), int(n_total * self.val_ratio)

        train_set = {"observations": obs[:n_train], "actions": actions[:n_train]}
        val_set = {"observations": obs[n_train:n_train + n_val], "actions": actions[n_train:n_train + n_val]}
        test_set = {"observations": obs[n_train + n_val:], "actions": actions[n_train + n_val:]}

        return train_set, val_set, test_set


    @staticmethod
    def print_stat(data, title):
        obs, actions = data["observations"], data["actions"]
        print("%s Observations %s, mean: %s" % (title, str(obs.shape),
            str(np.mean(obs, axis=0))))

    @staticmethod
    def normalize(data, mean, std):
        """Normalize observations"""
        obs = data["observations"]
        data["observations"] = (obs - mean) / (std + 1e-6)  # See load_policy.py

    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        num_data = len(data["observations"])
        num_batch_per_epoch = int( (num_data-1) / batch_size) + 1

        for epoch in range(num_epochs):
            obs, actions = data["observations"], data["actions"]
            if shuffle:
                idx = np.random.permutation(np.arange(num_data))
                obs = obs[idx]
                actions = actions[idx]
            for i in range(num_batch_per_epoch):
                start_idx = i*batch_size
                end_idx = min((i+1)*batch_size, num_data)
                yield obs[start_idx:end_idx], actions[start_idx:end_idx]
