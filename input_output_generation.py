# Generate FCGR (input) and HotEncoding of sequence (output)
from pathlib import Path
from src.encoder_output import EncoderOutput
from src.generate_io import GenerateFCGRHotEncodeSeq
from parameters import PARAMETERS
from Bio import SeqIO

KMER = PARAMETERS["KMER"]
ORDER_NUCLEOTIDES = PARAMETERS["ORDER_NUCLEOTIDES"]
MAX_NUM_NUCLEOTIDES = PARAMETERS["MAX_NUM_NUCLEOTIDES"]
FOLDER_FASTA = PARAMETERS["FOLDER_FASTA"]

# list of fasta files
list_fasta = list(Path(FOLDER_FASTA).rglob("*fasta"))[:100]

# FCGR and seq enconding 
gen_io = GenerateFCGRHotEncodeSeq(
            destination_folder = "data/io", 
            kmer = KMER, 
            order_output_model=ORDER_NUCLEOTIDES,
            max_len_seq = MAX_NUM_NUCLEOTIDES
            )


gen_io(list_fasta = list_fasta)
