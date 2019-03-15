 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-WaveNet.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from vq_vae_wavenet.wavenet_auto_encoder import WaveNetAutoEncoder
from vq_vae_wavenet.trainer import Trainer
from vq_vae_wavenet.evaluator import Evaluator
from vq_vae_wavenet.wavenet_type import WaveNetType
from vq_vae_speech.mu_law import MuLaw
from dataset.speech_dataset import SpeechDataset

import os
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import librosa
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def train(configuration, model, use_cuda, train_loader, val_loader, device):
    parameters = list(model.parameters())
    opt = torch.optim.Adam([p for p in parameters if p.requires_grad], lr=configuration['learning_rate'])

    for epoch in range(configuration['start_epoch'], configuration['num_epochs']):
        train_bar = tqdm(train_loader)
        model.train()
        for data in train_bar:
            x_enc, x_dec, speaker_id, quantized = data
            if use_cuda:
                x_enc, x_dec, speaker_id, quantized = x_enc.to(device), x_dec.to(device), speaker_id.to(device), quantized.to(device)

            opt.zero_grad()
            loss, _, _ = model(x_enc, x_dec, speaker_id, quantized)
            loss.mean().backward()
            opt.step()

            train_bar.set_description('Epoch {}: loss {:.4f}'.format(epoch + 1, loss.mean().item()))

        model.eval()
        data_val = next(iter(val_loader))
        with torch.no_grad():
            x_enc_val, x_dec_val, speaker_id_val, quantized_val = data_val
            if use_cuda:
                x_enc_val, x_dec_val, speaker_id_val, quantized_val = x_enc_val.to(device), x_dec_val.to(device), speaker_id_val.to(device), quantized_val.to(device)
            _, out = model(x_enc_val, x_dec_val, speaker_id_val, quantized_val)

            output = out.argmax(dim=1).detach().cpu().numpy().squeeze()
            input_mu = x_dec_val.argmax(dim=1).detach().cpu().numpy().squeeze()
            input = x_enc_val.detach().cpu().numpy().squeeze()

            output = MuLaw.decode(output)
            input_mu = MuLaw.decode(input_mu)

            #librosa.output.write_wav(os.path.join(save_path, '{}_output.wav'.format(epoch)), output, configuration['sampling_rate'])
            #librosa.output.write_wav(os.path.join(save_path, '{}_input_mu.wav'.format(epoch)), input_mu, configuration['sampling_rate'])
            #librosa.output.write_wav(os.path.join(save_path, '{}_input.wav'.format(epoch)), input, configuration['sampling_rate'])



        """torch.save({'epoch': epoch,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'vq': vq.state_dict()
                    }, os.path.join(save_path, '{}_checkpoint.pth'.format(epoch)))"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', nargs='?', default='data', type=str, help='The path of the data directory')
    parser.add_argument('--results_path', nargs='?', default='results', type=str, help='The path of the results directory')
    parser.add_argument('--loss_plot_name', nargs='?', default='loss.png', type=str, help='The file name of the training loss plot')
    parser.add_argument('--model_name', nargs='?', default='model.pth', type=str, help='The file name of trained model')
    args = parser.parse_args()

    # Dataset and model hyperparameters
    configuration = get_config('../configurations/vctk.yaml')

    #use_cuda = torch.cuda.is_available()
    use_cuda = False
    #device = torch.device('cuda' if use_cuda else 'cpu') # Use GPU if cuda is available
    #gpu_ids = [i for i in range(torch.cuda.device_count())]
    #device = 'cuda:1'
    device = 'cpu'
    gpu_ids = [1]

    # Set the result path and create the directory if it doesn't exist
    results_path = '..' + os.sep + args.results_path
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    
    dataset_path = '..' + os.sep + args.data_path

    dataset = SpeechDataset(configuration, gpu_ids, use_cuda)

    auto_encoder = WaveNetAutoEncoder(configuration, dataset.speaker_dic, device).to(device) # Create an AutoEncoder model using our GPU device
    auto_encoder = auto_encoder.double()
    #auto_encoder = nn.DataParallel(auto_encoder.to(device), device_ids=gpu_ids) if use_cuda else auto_encoder

    optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration['learning_rate'], amsgrad=True) # Create an Adam optimizer instance
    #trainer = Trainer(device, auto_encoder, optimizer, dataset) # Create a trainer instance
    #trainer.train(configuration.num_training_updates) # Train our model
    train(configuration, auto_encoder, use_cuda, dataset.training_loader, dataset.validation_loader, device)
    #auto_encoder.save(results_path + os.sep + args.model_name) # Save our trained model
    #trainer.save_loss_plot(results_path + os.sep + args.loss_plot_name) # Save the loss plot

    #evaluator = Evaluator(device, auto_encoder, dataset) # Create en Evaluator instance to evaluate our trained model
    