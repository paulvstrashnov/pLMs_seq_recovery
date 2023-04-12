# evaluate_batch.py

import numpy as np
import re
from .distance import levenshtein_distance, needleman_wunsch_score

def evaluate_batch(batch, model, device):
    """
    Evaluates a single batch of protein sequences using the given protein language model.

    """
    
    # single sequence encoding (takes any length sequence, but slow)
    encoded = [model.encode(sequence) for sequence in batch]
    decoded = [model.decode(embedding) for embedding in encoded]
    
    # batch encoding (requires equal length sequences since decoders often miss <eos> token)
    # does not work or not implemented for all models
    #encoded = model.batch_encode(batch)
    #decoded = model.batch_decode(encoded)


    distances = [levenshtein_distance(seq, dec) for seq, dec in zip(batch, decoded)]
    mean_distance = np.mean(distances)

    normalized_distances = [levenshtein_distance(seq, dec) / len(seq) for seq, dec in zip(batch, decoded)]
    mean_normalized_distance = np.mean(normalized_distances)

    # some decoders produce sequences with U's, which are not in the standard alphabet
    # remove U's from decoded sequences
    for ind, dec in enumerate(decoded):
        if 'U' in dec:
            print(f'U found in dec: {dec}')
            decoded[ind] = "".join(list(re.sub(r"[UZOB]", "", dec)))
        
    scores = [needleman_wunsch_score(seq, dec) for seq, dec in zip(batch, decoded)]
    self_scores = [needleman_wunsch_score(seq, seq) for seq in batch]
    nw_results = [score - self_score for score, self_score in zip(scores, self_scores)]
    #print(scores, self_scores, nw_results)

    mean_score = np.mean(nw_results)
    return mean_distance, mean_normalized_distance, mean_score
