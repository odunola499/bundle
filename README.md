# Bundle: An LLM Workbench

Bundle is a modular and extensible PyTorch-based workbench for building and experimenting with transformer-style neural networks and their building blocks. It's designed to be easy to use and modify, allowing you to quickly prototype and test new ideas.

## Core Features

- **Modular Design:** Easily swap out components like attention mechanisms and positional encodings.
- **Flexible Architectures:** Configure encoder-only, decoder-only, or encoder-decoder models.
- **Configurable:** Define your model architecture using a single `config.yaml` file.
- **Extensible:** Add new components with minimal boilerplate.

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your model:**
   Edit `config.yaml` to define your desired architecture. You can choose between `encoder-decoder`, `encoder`, and `decoder` for the `model_type`.

3. **Train your model:**
   ```python
   import yaml
   import torch
   from model import Transformer
   from types import SimpleNamespace

   with open("config.yaml", "r") as f:
       config_dict = yaml.safe_load(f)
   
   def dict_to_namespace(d):
       for k, v in d.items():
           if isinstance(v, dict):
               d[k] = dict_to_namespace(v)
       return SimpleNamespace(**d)

   config = dict_to_namespace(config_dict)
   
   model = Transformer(config.model)

   # Example usage for an encoder-decoder model
   if config.model.model_type == "encoder-decoder":
       src = torch.randint(0, config.model.vocab_size, (1, 10))
       tgt = torch.randint(0, config.model.vocab_size, (1, 20))
       output = model(src=src, tgt=tgt)
       print(output.shape)
   ```

## How to Add New Components

### 1. Attention Mechanisms

1.  Open `attention/attention.py`.
2.  Create a new class that inherits from `Attention`.
3.  Implement your attention logic in the `forward` method.
4.  In `blocks.py`, add your new attention class to the `_get_attention_layer` method in `EncoderBlock` and `DecoderBlock`.

### 2. Positional Encodings

1.  Open `encoding/encoding.py`.
2.  Create a new class that inherits from `PositionalEncoding`.
3.  Implement your positional encoding logic in the `forward` method.
4.  In `model.py`, add your new positional encoding class to the `_get_pos_encoding` method in the `Transformer` class.

### 3. Feed-Forward Networks

1.  Open `feedforward/feedforward.py`.
2.  Create a new class that inherits from `FeedForward`.
3.  Implement your feed-forward logic in the `forward` method.
4.  In `blocks.py`, add your new feed-forward class to the `_get_ff_layer` method in `EncoderBlock` and `DecoderBlock`.

### 4. Norm
1. Open `norm/norm.py`
2. Create a new class that inherits from `Normalisation`
3. Implement the normalisation logic in the 'forward' method.
4. In the `__init__.py`, add your new normalisation class

### 5. Activation
1. Open `activation/activation.py`
2. Follow the same steps as `Norm`


## TODO
- Add popular norm algorithms as abstractions
  - [ ] BatchNorm
  - [ ] LayerNorm
  - [ ] RMSNorm
  - [ ] GroupNorm

- Add popular activation algorithms as abstractions
  - [ ] Relu
  - [ ] Swish
- Add MOE to feed-forward logic as reference for MOE blog


