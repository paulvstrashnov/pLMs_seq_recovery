from transformers import ElectraTokenizer, ElectraModel, ElectraForMaskedLM
import torch
from .protein_model import ProteinModel
from typing import Optional
import re

class Electra(ProteinModel):
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
    ):
        self.device = device

        # Load the Electra transformer model and tokenizer
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.encoder = ElectraModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()
        self.decoder = ElectraForMaskedLM.from_pretrained(model_name).to(self.device)
        self.decoder.eval()
    def encode(self, sequence):
        processed_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        input_ids = self.tokenizer.encode(processed_sequence, return_tensors="pt", add_special_tokens = True).to(self.device)

        # Pass the input through the encoder
        with torch.no_grad():
            embedding = self.encoder(input_ids).last_hidden_state[0]

        return embedding

    def decode(self, embedding):
        # Pass the embeddings through the decoder
        with torch.no_grad():
            output_logits = self.decoder.generator_lm_head(embedding).to(self.device)
            predicted_token_ids = output_logits.argmax(axis=-1)

        # Convert token ids back to amino acid sequence
        decoded_sequence = self.tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
        decoded_sequence = ''.join(decoded_sequence.split())

        return decoded_sequence[1:-1]
