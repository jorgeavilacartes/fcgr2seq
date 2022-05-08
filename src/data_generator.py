"Load data to the model (Same as dataset.py but with keras)"

from typing import List, Union, Callable, Optional
from pathlib import Path
import numpy as np
import tensorflow as tf
from .npy_loader import InputOutputLoader
from collections import defaultdict

class DataGenerator(tf.keras.utils.Sequence):
    """Data Generator  for keras from a list of paths to files""" 

    def __init__(self, 
                list_paths_fcgr: List[Union[str, Path]],
                list_paths_labels: List[Union[str, Path]], 
                batch_size: int = 8,
                shuffle: bool = True,
                kmer: int = 8,          
                preprocessing: Optional[Callable] = None 
                ):
        self.list_paths_fcgr = list_paths_fcgr
        self.list_paths_labels = list_paths_labels  
        self.batch_size = batch_size 
        self.shuffle = shuffle
        self.kmer = kmer
        self.preprocessing = preprocessing if callable(preprocessing) else lambda x: x
        self.input_output_loader = InputOutputLoader()
        
        assert len(list_paths_fcgr) == len(list_paths_labels), "number of FCGR and labels is different"
        # initialize first batch
        self.on_epoch_end()

    def on_epoch_end(self,):
        """Updates indexes after each epoch (starting for the epoch '0')"""
        self.indexes = np.arange(len(self.list_paths_fcgr))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # shuffle indexes in place

    def __len__(self):
        # Must be implemented
        """Denotes the number of batches per epoch"""
        delta = 1 if len(self.list_paths_fcgr) % self.batch_size else 0 
        return len(self.list_paths_fcgr) // self.batch_size + delta

    def __getitem__(self, index):
        # Must be implemented
        """To feed the model with data in training
        It generates one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of paths to ecg
        list_paths_temp_fcgr = [self.list_paths_fcgr[k] for k in indexes]

        list_paths_temp_label = [self.list_paths_labels[k] for k in indexes]

        # Generate data
        X, y = self.input_output_generation(list_paths_temp_fcgr, list_paths_temp_label)
        return X, y
    
    def input_output_generation(self, list_path_temp_fcgr: List[str], list_path_temp_label: List[str]): 
        """Generates and augment data containing batch_size samples
        Args:
            list_path_temp (List[str]): sublist of list_path
        Returns:
            X : numpy.array
            y : numpy.array hot-encoding
        """ 
        X_batch = []
        y_batch = []

        y_aux = defaultdict(list)

        for path_fcgr, path_label in zip(list_path_temp_fcgr, list_path_temp_label): 
            
            # load fcgr and label
            fcgr, label = self.input_output_loader(path_fcgr, path_label)
            
            # build input batch
            fcgr = self.preprocessing(fcgr)
            X_batch.append(np.expand_dims(np.expand_dims(fcgr,axis=0),axis=-1)) # add to list with batch dims
            
            # build output list of batches | label size (n_outputs, 5)
            for i in range(label.shape[0]):
                y_aux[i].append(label[i,:])

            #label = [label[:,i] for i in range(label.shape[1])]
            #y_batch.append(label)#np.expand_dims(label,axis=0))

        # convert to batch
        y_batch = [np.array(y_aux[n_out]) for n_out in range(len(y_aux))]
        #y_batch = [np.expand_dims(y,axis=0) for y in y_batch]

        return np.concatenate(X_batch, axis=0), y_batch