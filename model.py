import torch
from torch import nn
from blocks import EncoderBlock, DecoderBlock
from encoding.encoding import AbsolutePositionalEncoding, RoPE

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim_model)
        self.pos_encoding = self._get_pos_encoding(config.positional_encoding)
        
        if config.model_type == "encoder-decoder":
            self.encoder = nn.ModuleList([EncoderBlock(config.encoder) for _ in range(config.encoder.num_layers)])
            self.decoder = nn.ModuleList([DecoderBlock(config.decoder) for _ in range(config.decoder.num_layers)])
        elif config.model_type == "encoder":
            self.encoder = nn.ModuleList([EncoderBlock(config.encoder) for _ in range(config.encoder.num_layers)])
            self.decoder = None
        elif config.model_type == "decoder":
            self.decoder = nn.ModuleList([DecoderBlock(config.decoder) for _ in range(config.decoder.num_layers)])
            self.encoder = None
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        self.fc_out = nn.Linear(config.dim_model, config.vocab_size)

    def _get_pos_encoding(self, pos_config):
        if pos_config.name == "absolute":
            return AbsolutePositionalEncoding(pos_config)
        elif pos_config.name == "rope":
            return RoPE(pos_config)
        else:
            raise ValueError(f"Unknown positional encoding: {pos_config.name}")

    def forward(self, src=None, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None):
        if self.config.model_type == "encoder-decoder":
            assert src is not None and tgt is not None, "Source and target inputs are required for encoder-decoder models."
            src_emb = self.embedding(src)
            tgt_emb = self.embedding(tgt)

            src_emb = self.pos_encoding(src_emb)
            tgt_emb = self.pos_encoding(tgt_emb)

            enc_output = src_emb
            for layer in self.encoder:
                enc_output, _ = layer(enc_output, src_mask)

            dec_output = tgt_emb
            for layer in self.decoder:
                dec_output, _ = layer(dec_output, enc_output, tgt_mask, memory_mask)
            
            return self.fc_out(dec_output)

        elif self.config.model_type == "encoder":
            assert src is not None, "Source input is required for encoder-only models."
            src_emb = self.embedding(src)
            src_emb = self.pos_encoding(src_emb)
            
            enc_output = src_emb
            for layer in self.encoder:
                enc_output, _ = layer(enc_output, src_mask)
            
            return self.fc_out(enc_output)

        elif self.config.model_type == "decoder":
            assert tgt is not None, "Target input is required for decoder-only models."
            tgt_emb = self.embedding(tgt)
            tgt_emb = self.pos_encoding(tgt_emb)

            dec_output = tgt_emb
            for layer in self.decoder:
                dec_output, _ = layer(dec_output, None, tgt_mask, memory_mask)
            
            return self.fc_out(dec_output)
