from torch.utils.data import Dataset
import pickle
import os


class VCTKFeaturesDataset(Dataset):

    def __init__(self, vctk_path, subdirectory):
        self._vctk_path = vctk_path
        self._subdirectory = subdirectory
        features_path = self._vctk_path + os.sep + 'features'
        self._sub_features_path = features_path + os.sep + self._subdirectory
        self._files_number = len(os.listdir(self._sub_features_path))

    def __getitem__(self, index):
        dic = None
        with open(self._sub_features_path + os.sep + str(index) + '.pickle', 'rb') as file:
            dic = pickle.load(file)

        return dic['raw_features'], dic['one_hot'], dic['speaker_id'], dic['quantized_features']

    def __len__(self):
        return self._files_number
