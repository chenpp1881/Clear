import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, args, hidden_size=512, num_classes=2):
        super().__init__()
        self.args = args
        self.token_emb = nn.Embedding(50265, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.Trans = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        embeddings = self.token_emb(input_ids)
        src_mask = input_ids != 1
        output = self.Trans(embeddings, src_key_padding_mask=~src_mask)
        pooled_output = output.mean(dim=1)
        logits = self.fc1(pooled_output)
        return logits
