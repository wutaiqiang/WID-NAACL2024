import torch.nn as nn


class Model(nn.Module):
    """
    Pretraining models consist of three (five) parts:
        - embedding
        - encoder
        - tgt_embedding (optional)
        - decoder (optional)
        - target
    """

    def __init__(self, args, embedding, encoder, tgt_embedding, decoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.tgt_embedding = tgt_embedding
        self.decoder = decoder
        self.target = target

        if "mlm" in args.target and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight
        elif "lm" in args.target and args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight

        if self.decoder is not None and args.share_embedding:
            self.tgt_embedding.word_embedding.weight = self.embedding.word_embedding.weight

    def forward(self, src, tgt, seg,mask_dict,current_step,soft_tgt, tgt_in=None, tgt_seg=None,if_dev=False):
        emb = self.embedding(src, seg,mask_dict,current_step)
        memory_bank = self.encoder(emb, seg,mask_dict,current_step)
        if self.decoder:
            tgt_emb = self.tgt_embedding(tgt_in, tgt_seg)
            memory_bank = self.decoder(memory_bank, tgt_emb, (seg, tgt_seg))
        if if_dev:
            loss_info = self.target(memory_bank, tgt, seg, None)
        else:
            loss_info = self.target(memory_bank, tgt, seg,mask_dict,current_step, soft_tgt)

        return loss_info
