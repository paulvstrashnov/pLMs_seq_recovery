from Levenshtein import distance
import warnings
from Bio import BiopythonDeprecationWarning
warnings.simplefilter('ignore', BiopythonDeprecationWarning)
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo

def levenshtein_distance(seq1, seq2):
    return distance(seq1, seq2)

def needleman_wunsch_score(seq1, seq2, matrix=MatrixInfo.blosum62, gap_open=-10, gap_extend=-0.5):
    alignments = pairwise2.align.globalds(seq1, seq2, matrix, gap_open, gap_extend)
    return alignments[0].score