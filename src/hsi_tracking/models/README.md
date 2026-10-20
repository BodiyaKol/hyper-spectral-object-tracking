# HSI Model Components

`SpectralProjector` uses a sensor-specific learnable 1x1 convolution: VIS `16→3`, NIR `25→3`, RedNIR `15→3`. Its initial weights select low, middle, and high bands; training can learn combinations from all bands.

`CNNPromptAdapter` consumes the complete HSI crop as `B×C×H×W`. It returns one prompt tensor per transformer block, each shaped `B×N×768`, where `N` is the number of template/search patch tokens. `HSIAdaptedViPT` sends the projector output through ViPT's RGB patch embedding and adds each adapter prompt immediately before its matching transformer block. Candidate elimination is disabled for this dense-prompt path so token grids stay aligned.

The upstream ViPT backbone and BBox head are frozen. Only the spectral projector and CNN prompt adapter are trainable. The wrapper supports saving/loading adapter-only checkpoints with sensor metadata. This is an experimental CPU-capable integration, not an official HSI-pretrained ViPT model; RGB-T pretrained weights initialize the backbone, and adapters need labeled HOT training sequences for meaningful results.

See [`scripts/cnn-adapter/README.md`](../../../scripts/cnn-adapter/README.md) for patch, training, and inference commands.