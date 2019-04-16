from torch import nn


class GlobalConditioning(object):

    @staticmethod
    def compute(speaker_dic, speaker_ids, x_one_hot, gin_channels=128, expand=True):
        speakers_embedding = GlobalConditioning._Embedding(len(speaker_dic), gin_channels, padding_idx=None, std=0.1)    

        # Extract the batch size and the signal length
        B, _, T = x_one_hot.size()

        # (B x 1) -> (B x 1 x gin_channels)
        global_conditioning = speakers_embedding(speaker_ids.view(B, -1).long())

        # (B x gin_channels x 1)
        global_conditioning = global_conditioning.transpose(1, 2)

        # Check if the result have the right dimension
        assert global_conditioning.dim() == 3

        """
        Return the global conditioning if the expand
        option is set to False
        """
        if not expand:
            return global_conditioning

        # Expand global conditioning features to all time steps
        expanded_global_conditioning = GlobalConditioning._expand_global_features(B, T, global_conditioning, bct=True)

        return expanded_global_conditioning

    @staticmethod
    def _Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        m.weight.data.normal_(0, std)
        return m

    @staticmethod
    def _expand_global_features(B, T, g, bct=True):
        """
        Expand global conditioning features to all time steps

        Args:
            B (int): Batch size.
            T (int): Time length.
            g (Tensor): Global features, (B x C) or (B x C x 1).
            bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

        Returns:
            Tensor: B x C x T or B x T x C or None
        """
        if g is None:
            return None
        g = g.unsqueeze(-1) if g.dim() == 2 else g
        if bct:
            g_bct = g.expand(B, -1, T)
            return g_bct.contiguous()
        else:
            g_btc = g.expand(B, -1, T).transpose(1, 2)
            return g_btc.contiguous()


from experiments.device_configuration import DeviceConfiguration
from dataset.vctk_speech_stream import VCTKSpeechStream
from error_handling.console_logger import ConsoleLogger
import yaml

if __name__ == "__main__":
    configuration = None
    with open('../../configurations/vctk_features.yaml', 'r') as configuration_file:
        configuration = yaml.load(configuration_file)
    device_configuration = DeviceConfiguration.load_from_configuration(configuration)
    data_stream = VCTKSpeechStream(configuration, device_configuration.gpu_ids, device_configuration.use_cuda)
    (_, x_dec_val, speaker_id, _, wav_filename) = next(iter(data_stream.training_loader))

    x = x_dec_val.squeeze(-1)    
    global_conditioning = GlobalConditioning.compute(
        speaker_dic=data_stream.speaker_dic,
        speaker_ids=speaker_id,
        x_one_hot=x,
        expand=False
    )
    ConsoleLogger.success('global_conditioning.size(): {}'.format(global_conditioning.size()))

    expanded_global_conditioning = GlobalConditioning.compute(
        speaker_dic=data_stream.speaker_dic,
        speaker_ids=speaker_id,
        x_one_hot=x,
        expand=True
    )
    ConsoleLogger.success('expanded_global_conditioning.size(): {}'.format(expanded_global_conditioning.size()))
