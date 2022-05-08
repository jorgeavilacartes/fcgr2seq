"""
This script assumes that each Fasta file contains only one sequence
"""
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
import numpy as np 
import pandas as pd
from .fcgr import FCGR
from .encoder_output import EncoderOutput
from collections import namedtuple

InputOutputPaths = namedtuple("IOPaths",["input_path","output_path"])

class GenerateFCGRHotEncodeSeq: 

    def __init__(self, destination_folder: Path = "img", kmer: int = 8, 
            order_output_model=["A","C","G","T","N"], max_len_seq = None): 
        self.destination_folder = Path(destination_folder)
        self.kmer = kmer
        self.fcgr = FCGR(kmer)
        self.counter = 0
        self.max_len_seq = max_len_seq
        # Create destination folder if needed
        self.destination_folder.mkdir(parents=True, exist_ok=True)

        # encoder output
        self.encoder_output = EncoderOutput(order_output_model)

        # info paths
        self.list_paths = []

    def __call__(self, list_fasta,):
         
        for fasta in tqdm(list_fasta, desc="Generating FCGR"):
            self.from_fasta(fasta)

        pd.DataFrame(self.list_paths).to_csv(Path(self.destination_folder).joinpath("list_paths_io.csv"),sep=",")

    def from_fasta(self, path: Path,):
        """FCGR for a sequence in a fasta file.
        The FCGR image will be save in 'destination_folder/specie/label/id_fasta.jpg'
        """
        # load fasta file
        path = Path(path)
        fasta  = self.load_fasta(path)
        record = next(fasta)
                
        # Generate and save FCGR for the current sequence
        *_, specie, label  = str(path.parents[0]).split("/")
        id_fasta = path.stem
        path_save_fcgr = self.destination_folder.joinpath("fcgr/{}/{}/{}.npy".format(specie, label, id_fasta))
        path_save_fcgr.parents[0].mkdir(parents=True, exist_ok=True)
        self.fcgr_from_seq(str(record.seq), path_save_fcgr)

        # Generate output
        path_save_label = self.destination_folder.joinpath("label/{}/{}/{}.npy".format(specie, label, id_fasta))
        path_save_label.parents[0].mkdir(parents=True, exist_ok=True)
        self.hotencoding_from_seq(str(record.seq), path_save_label) 

        self.list_paths.append(InputOutputPaths(path_save_fcgr,path_save_label))
        self.counter +=1
        
    def hotencoding_from_seq(self, seq: str, path_save):
        "Get hot-encoding from a sequence"
        if not Path(path_save).is_file():
            seq = self.preprocessing(seq)

            # ensure lenght of the hot-encoding 
            len_seq = len(seq)
            if len_seq < self.max_len_seq:
                seq = seq + "N"*(self.max_len_seq - len_seq)
            elif len_seq > self.max_len_seq:
                seq = seq[:self.max_len_seq]
            else:
                pass

            label = []
            for n in seq:
                label.append(np.array(self.encoder_output([n])).T)
            np.save(path_save, np.array(label))

    def fcgr_from_seq(self, seq:str, path_save):
        "Get FCGR from a sequence"
        if not Path(path_save).is_file():
            seq = self.preprocessing(seq)
            chaos = self.fcgr(seq)
            np.save(path_save, chaos)

    def reset_counter(self,):
        self.counter=0
        
    @staticmethod
    def preprocessing(seq):
        seq = seq.upper()
        for letter in "BDEFHIJKLMOPQRSUVWXYZ":
            seq = seq.replace(letter,"N")
        return seq

    @staticmethod
    def load_fasta(path: Path):
        return SeqIO.parse(path, "fasta")