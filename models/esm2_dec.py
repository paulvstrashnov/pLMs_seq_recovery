from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
import torch
from .protein_model import ProteinModel
from typing import Optional

class ESM2dec(ProteinModel):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
    ):
        self.device = device

        # Load the ESM transformer model and tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.encoder = EsmModel.from_pretrained(model_name, add_cross_attention=False, is_decoder=True).to(self.device)
        self.encoder.eval()
        self.decoder = EsmForMaskedLM.from_pretrained(model_name).to(self.device)
        self.decoder.eval()
    def encode(self, sequence):
        # Tokenize the input sequence
        input_ids = self.tokenizer.encode(sequence, return_tensors="pt").to(self.device)

        # Pass the input through the encoder
        with torch.no_grad():
            embedding = self.encoder(input_ids).last_hidden_state

        return embedding

    def decode(self, embedding):
        # Pass the embeddings through the decoder
        with torch.no_grad():
            output_logits = self.decoder.lm_head(features = embedding).to(self.device)
            predicted_token_ids = output_logits[0, :].argmax(axis=-1)

        # Convert token ids back to amino acid sequence
        decoded_sequence = self.tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
        decoded_sequence = ''.join(decoded_sequence.split())

        return decoded_sequence
