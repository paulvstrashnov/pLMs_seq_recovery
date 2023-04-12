from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
import torch
from .protein_model import ProteinModel
from typing import Optional

class ESM2(ProteinModel):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
    ):
        self.device = device

        # Load the ESM transformer model and tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.encoder = EsmModel.from_pretrained(model_name, add_cross_attention=False, is_decoder=False).to(self.device)
        self.encoder.eval()
        self.decoder = EsmForMaskedLM.from_pretrained(model_name).to(self.device)
        self.decoder.eval()
    def encode(self, sequence):
        # Tokenize the input sequence
        input_ids = self.tokenizer.encode(sequence, return_tensors="pt").to(self.device)

        # Pass the input through the encoder
        with torch.no_grad():
            embedding = self.encoder(input_ids).last_hidden_state

        return embedding # torch.Size([1, seq_len+2, emb_dim])

    def decode(self, embedding):
        # Pass the embeddings through the decoder
        with torch.no_grad():
            output_logits = self.decoder.lm_head(features = embedding).to(self.device)
            predicted_token_ids = output_logits[0, :].argmax(axis=-1)

        # Convert token ids back to amino acid sequence
        decoded_sequence = self.tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
        decoded_sequence = ''.join(decoded_sequence.split())

        return decoded_sequence
    
    def batch_encode(self, sequences, max_length=256):
        # Tokenize the input sequence
        # Equal length sequences:)
        tokenized = self.tokenizer(sequences, return_attention_mask=False, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        # Variable length sequences: ()
        #tokenized = self.tokenizer(sequences, return_attention_mask=False, return_tensors="pt", padding=True).to(self.device)
        input_ids = tokenized['input_ids']

        # Pass the input through the encoder
        with torch.no_grad():
            embeddings = self.encoder(input_ids).last_hidden_state

        return embeddings # torch.Size([batch_size, seq_len+2, emb_dim])
    
    
    def batch_decode(self, embeddings):
        # Pass the embeddings through the decoder
        with torch.no_grad():
            output_logits = self.decoder.lm_head(features = embeddings).to(self.device)
            predicted_token_ids = output_logits.argmax(axis=-1)

        # Convert token ids back to amino acid sequence
        decoded_sequences = self.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=False)
        
        # for each seq in decoded_sequences:
        # 1. remove all tokens after the first <eos> token
        # 2. remove special tokens usning list comprehension
        # 3. remove all whitespaces
        # this does not work since the <eos> token is often not in the decoded sequence
        #decoded_sequences = [seq.split('<eos>')[0] for seq in decoded_sequences]
        #decoded_sequences = [''.join([token for token in seq.split() if '<' not in token]) for seq in decoded_sequences]

        decoded_sequences = [''.join(seq.split()) for seq in decoded_sequences]

        return decoded_sequences # list of seqs
 