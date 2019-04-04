 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
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

class Configuration(object):
    """
    The configuration instance list the hyperparameters of
    the model, inspired from [deepmind/sonnet].

    References:
        [deepmind/sonnet] https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb.

        [van den Oord et al., 2017] van den Oord A., and Oriol Vinyals. "Neural discrete representation
        learning." Advances in Neural Information Processing Systems(NIPS). 2017.

        [Roy et al., 2018] A. Roy, A. Vaswani, A. Neelakantan, and N. Parmar. Theory and experiments on vector
        quantized autoencoders.arXiv preprint arXiv:1805.11063, 2018.

        [He, K et al., 2015] He, K., Zhang, X., Ren, S and Sun, J. Deep Residual Learning for Image Recognition. arXiv e-prints arXiv:1502.01852.

        [ksw0306/ClariNet] https://github.com/ksw0306/ClariNet.
    """

    default_batch_size = 32 # Just to test
    default_num_training_updates = 10000 # Just to test
    default_encoder_num_hiddens = 768
    default_encoder_num_residual_hiddens = 768
    default_encoder_num_residual_layers = 4
    default_decoder_num_hiddens = 256
    default_decoder_num_residual_hiddens = 256
    default_decoder_num_residual_layers = 2

    """
    This value is not that important, usually 64 works.
    This will not change the capacity in the information-bottleneck.
    """
    default_embedding_dim = 64 # Same as specified in the paper

    default_num_embeddings = 512 # The higher this value, the higher the capacity in the information bottleneck.

    """
    Commitment cost should be set appropriately. It's often useful to try a couple
    of values. It mostly depends on the scale of the reconstruction cost
    (log p(x|z)). So if the reconstruction cost is 100x higher, the
    commitment_cost should also be multiplied with the same amount.
    """
    default_commitment_cost = 0.25 # Same as specified in the paper

    """
    Only uses for the EMA updates (instead of the Adam optimizer).
    This typically converges faster, and makes the model less dependent on choice
    of the optimizer. In the original VQ-VAE paper [van den Oord et al., 2017],
    EMA updates were not used (but suggested in appendix) and compared in
    [Roy et al., 2018].
    """
    default_decay = 0.99 # TODO

    default_learning_rate = 4e-4 # Same as specified in the paper

    """
    Weight initialization proposed by [He, K et al., 2015].
    PyTorch doc: https://pytorch.org/docs/stable/nn.html#torch.nn.init.kaiming_normal_.
    The model seems to converge faster using it.
    In addition to that, I used nn.utils.weight_norm() before each use of kaiming_normal(),
    as they do in [ksw0306/ClariNet], because it works better.
    """
    default_use_kaiming_normal = True

    default_shuffle_dataset = True

    def __init__(self, batch_size=default_batch_size, num_training_updates=default_num_training_updates, \
        encoder_num_hiddens=default_encoder_num_hiddens, encoder_num_residual_hiddens=default_encoder_num_residual_hiddens, \
        encoder_num_residual_layers=default_encoder_num_residual_layers, decoder_num_hiddens=default_decoder_num_hiddens, \
        decoder_num_residual_hiddens=default_decoder_num_residual_hiddens, \
        decoder_num_residual_layers=default_decoder_num_residual_layers, embedding_dim=default_embedding_dim, \
        num_embeddings=default_num_embeddings, commitment_cost=default_commitment_cost, \
        decay=default_decay, learning_rate=default_learning_rate, use_kaiming_normal=default_use_kaiming_normal, \
        shuffle_dataset=default_shuffle_dataset):

        self._batch_size = batch_size
        self._num_training_updates = num_training_updates
        self._encoder_num_hiddens = encoder_num_hiddens
        self._encoder_num_residual_hiddens = encoder_num_residual_hiddens
        self._encoder_num_residual_layers = encoder_num_residual_layers
        self._decoder_num_hiddens = decoder_num_hiddens
        self._decoder_num_residual_hiddens = decoder_num_residual_hiddens
        self._decoder_num_residual_layers = decoder_num_residual_layers
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._learning_rate = learning_rate 
        self._use_kaiming_normal = use_kaiming_normal
        self._shuffle_dataset = shuffle_dataset

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_training_updates(self):
        return self._num_training_updates

    @property
    def encoder_num_hiddens(self):
        return self._encoder_num_hiddens

    @property
    def encoder_num_residual_hiddens(self):
        return self._encoder_num_residual_hiddens

    @property
    def encoder_num_residual_layers(self):
        return self._encoder_num_residual_layers

    @property
    def decoder_num_hiddens(self):
        return self._decoder_num_hiddens

    @property
    def decoder_num_residual_hiddens(self):
        return self._decoder_num_residual_hiddens

    @property
    def decoder_num_residual_layers(self):
        return self._decoder_num_residual_layers

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def commitment_cost(self):
        return self._commitment_cost

    @property
    def decay(self):
        return self._decay

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def use_kaiming_normal(self):
        return self._use_kaiming_normal

    @property
    def shuffle_dataset(self):
        return self._shuffle_dataset

    @staticmethod
    def build_from_args(args):
        return Configuration(
            batch_size=args.batch_size,
            num_training_updates=args.num_training_updates,
            encoder_num_hiddens=args.encoder_num_hiddens,
            encoder_num_residual_hiddens=args.encoder_num_residual_hiddens,
            encoder_num_residual_layers=args.encoder_num_residual_hiddens,
            decoder_num_hiddens=args.decoder_num_hiddens,
            decoder_num_residual_hiddens=args.decoder_num_residual_hiddens,
            decoder_num_residual_layers=args.decoder_num_residual_hiddens,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            decay=args.decay,
            learning_rate=args.learning_rate,
            use_kaiming_normal=args.use_kaiming_normal,
            shuffle_dataset=not args.unshuffle_dataset
        )
