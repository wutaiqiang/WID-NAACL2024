import torch
import sys
import torch.nn as nn

from uer.embeddings.wordpos_embedding import WordPosEmbedding


class WordPosSegEmbedding(WordPosEmbedding):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(WordPosSegEmbedding, self).__init__(args, vocab_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.weight_squeeze = args.weight_squeeze
        if args.weight_squeeze:
            self.emb_compactor = nn.Linear(args.emb_size, args.emb_size, bias=False)

    def forward(self, src, seg,mask_dict,current_step):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            emb: [batch_size x seq_length x hidden_size]
        """
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        if self.weight_squeeze:
            emb = self.emb_compactor(emb)

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb,"module.embedding.emb_compactor.weight",mask_dict,current_step)
        emb = self.dropout(emb)

        return emb
