from dataset.vctk import VCTK
from vq_vae_speech.utils import load_wav, mu_law_encode

from torch.utils.data import Dataset
import numpy as np
import random
import pathlib


class VCTKDataset(Dataset):
    def __init__(self,audios,speaker_dic, params):

        self.audios = audios
        self.speaker_dic = speaker_dic

        self.params = params
        if params['length'] is None:
            self.length = None
        else:
            self.length = params['length'] + 1

        #self.length = params['length']+1
        self.quantize = params['quantize']

    def preprocessing(self,raw, quantized):

        if self.length is not None:
            if len(raw) <=self.length :
                # padding
                pad = self.length  - len(raw)
                raw = np.concatenate(
                    (raw, np.zeros(pad, dtype=np.float32)))
                quantized = np.concatenate(
                    (quantized, self.quantize // 2 * np.ones(pad)))
                quantized = quantized.astype(np.long)
            else:
                # triming
                start = random.randint(0, len(raw) -self.length  - 1)
                raw = raw[start:start + self.length ]
                quantized = quantized[start:start + self.length ]

        #ont_hot for input of wavenet
        one_hot = np.identity(
            self.quantize, dtype=np.float32)[quantized]
        one_hot = np.expand_dims(one_hot.T, 2)

        raw = np.expand_dims(raw, 0)  # expand channel
        raw = np.expand_dims(raw, -1)  # expand height

        #target for wavenet
        quantized = np.expand_dims(quantized, 1)

        return raw, one_hot[:, :-1],quantized[1:]

    def __getitem__(self, index):
        wav_filename = self.audios[index]
        raw = load_wav(wav_filename,self.params)

        quantized = mu_law_encode(raw)

        speaker = pathlib.Path(wav_filename).parent.name

        speaker_id = np.array(
            self.speaker_dic[speaker], dtype=np.long)

        raw, one_hot, quantized = self.preprocessing(raw,quantized)

        return raw, one_hot, speaker_id,quantized

    def __len__(self):
        return len(self.audios)


if __name__ =='__main__':

    from torch.utils.data import DataLoader
    vctk = VCTK('./')
    params={'length':7680, 'quantize':256, 'sr':16000, 'res_type':'kaiser_fast','top_db':20}
    train_dataset = VCTKDataset(vctk.audios_train,vctk.speaker_dic,params)
    val_dataset = VCTKDataset(vctk.audios_val,vctk.speaker_dic, params)

    train_loader = DataLoader(train_dataset, batch_size=4,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    raw, one_hot, speaker_id, quantized = next(iter(train_loader))
    raw_val, one_hot_val, speaker_id_val, quantized_val = next(iter(val_loader))

    print(raw)
