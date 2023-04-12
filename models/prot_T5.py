# Requires sentencepiece and accelerate in the environment

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5EncoderModel
import torch
from .protein_model import ProteinModel
from typing import Optional
import re

class ProtT5Model(ProteinModel):
    def __init__(
        self,
        model_name: str, # 'Rostlab/prot_t5_xl_half_uniref50-enc'
        device: Optional[str] = None,
    ):
        self.device = device

        # Load the T5 transformer model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.encoder = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.encoder.full() if device=='cpu' else self.encoder.half()
        self.encoder.eval()
        self.decoder = T5ForConditionalGeneration.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16).to(self.device)
        self.decoder.eval()


    def encode(self, sequence):

        processed_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        inputs = self.tokenizer.encode(processed_sequence, return_tensors="pt").to(self.device)

        # Pass the input through the encoder
        with torch.no_grad():
            embedding = self.encoder(inputs)

        return embedding

    def decode(self, embedding):
        # Pass the embeddings through the decoder
        with torch.no_grad():
            input_ids = torch.tensor(self.tokenizer.encode("lm_head: ", add_special_tokens=True)).unsqueeze(0).to(self.device)
            output_logits = self.decoder.generate(
                input_ids=input_ids, 
                encoder_outputs=embedding, 
                max_length=embedding.last_hidden_state.shape[1]+2, 
                num_beams=5, 
                early_stopping=True
                )[0]
            predicted_token_ids = output_logits

        # Convert token ids back to amino acid sequence
        decoded_sequence = self.tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
        decoded_sequence = ''.join(decoded_sequence.split())

        return decoded_sequence
