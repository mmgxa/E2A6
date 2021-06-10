# E2A6

## Model
```
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.encoder = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           dropout=dropout,
                           batch_first=True)        

        self.decoder = nn.LSTM(hidden_dim, 
                       hidden_dim, 
                       num_layers=n_layers, 
                       batch_first=True)

        self.fc= nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, text_lengths):

       
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        
        packed_output, (hidden_enc, cell_enc) = self.encoder(packed_embedded)
        output_enc, output_enc_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)   
        print(output_enc)
        output_dec, (hidden_dec, cell_dec) = self.decoder(output_enc)
        print(output_dec)
        output_dense = self.fc(hidden_dec)   

        output = F.softmax(output_dense[0], dim=1)

        return output
```

After 10 Epochs:

Train Loss: 0.868 | Train Acc: 68.36%
Val. Loss: 0.802 |  Val. Acc: 75.00% 
