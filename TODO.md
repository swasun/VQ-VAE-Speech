* Compute more embedding stats (using stats_to_compute.md)
* Compute phoneme stats (using stats_to_compute.md)
* Plot an example of alignement using the groundtruth mapping
* Make an animation of the second embedding projection
* Compute the receptive field of the model
* Refractor jitter computation with the index version and compare it with the sequential one (e.g. [0, 1, -1, 0] with 0.12 to not be 0 and 0.5 to be 1 or -1)
* Add options in configuration to record quantized and one hot in audio preprocessing
* Handle LibriSpeech dataset
* Handle the flow_wavenet and clarinet as possible decoders in vq_vae_wavenet decoder
* Add a description of each sub module in the README?
* Replace logging with loguru