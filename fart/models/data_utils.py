"""
Data loading and preprocessing utilities for FART model.
"""
import math
import itertools
from collections import Counter
from copy import deepcopy
from typing import List, Optional, Tuple

import pandas as pd
from datasets import Dataset
from rdkit import Chem


def control_smiles_duplication(random_smiles, duplicate_control=lambda x: 1):
    """
    Returns augmented SMILES with the number of duplicates controlled by the function duplicate_control.

    Taken from https://github.com/volkamerlab/maxsmi/blob/main/maxsmi/utils/utils_smiles.py

    Parameters
    ----------
    random_smiles : list
        A list of random SMILES, can be obtained by `smiles_to_random()`.
    duplicate_control : func, Optional, default: 1
        The number of times a SMILES will be duplicated, as function of the number of times
        it was included in `random_smiles`.
        This number is rounded up to the nearest integer.

    Returns
    -------
    list
        A list of random SMILES with duplicates.

    Notes
    -----
    When `duplicate_control=lambda x: 1`, then the returned list contains only unique SMILES.
    """
    counted_smiles = Counter(random_smiles)
    smiles_duplication = {
        smiles: math.ceil(duplicate_control(counted_smiles[smiles]))
        for smiles in counted_smiles
    }
    return list(
        itertools.chain.from_iterable(
            [[smiles] * smiles_duplication[smiles] for smiles in smiles_duplication]
        )
    )


def smiles_to_random(smiles: str, int_aug: int = 50) -> Optional[List[str]]:
    """
    Takes a SMILES (not necessarily canonical) and returns `int_aug` random variations of this SMILES.

    Taken from https://github.com/volkamerlab/maxsmi/blob/main/maxsmi/utils/utils_smiles.py

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    int_aug : int, Optional, default: 50
        The number of random SMILES generated.

    Returns
    -------
    list or None
        A list of `int_aug` random (may not be unique) SMILES or None if the initial SMILES is not valid.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        if int_aug > 0:
            return [
                Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                for _ in range(int_aug)
            ]
        elif int_aug == 0:
            return [smiles]
        else:
            raise ValueError("int_aug must be greater or equal to zero.")


def augmentation_without_duplication(smiles: str, augmentation_number: int) -> List[str]:
    """
    Takes a SMILES and returns a list of unique random SMILES.

    Taken from https://github.com/volkamerlab/maxsmi/blob/main/maxsmi/utils/utils_smiles.py

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    augmentation_number : int
        The integer to generate the number of random SMILES.

    Returns
    -------
    list
        A list of unique random SMILES (no duplicates).
    """
    smiles_list = smiles_to_random(smiles, augmentation_number)
    return control_smiles_duplication(smiles_list, lambda x: 1)


def load_datasets(train_path: str, val_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test datasets from CSV files.

    Parameters
    ----------
    train_path : str
        Path to training CSV file.
    val_path : str
        Path to validation CSV file.
    test_path : str
        Path to test CSV file.

    Returns
    -------
    tuple
        Three pandas DataFrames: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Reset index to avoid "__index_level_0__" column
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


def augment_dataset(dataset: Dataset, augmentation_numbers: List[int], tastes: List[str], label_column: str, smiles_column: str) -> Dataset:
    """
    Augments the dataset by generating new SMILES strings for specified tastes.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    augmentation_numbers : list
        Numbers of new SMILES to generate for each taste.
    tastes : list
        Taste categories to augment.
    label_column : str
        Name of the label column in the dataset.
    smiles_column : str
        Name of the SMILES column in the dataset.

    Returns
    -------
    Dataset
        Augmented dataset with new SMILES strings.
    """
    augmented_data = []

    for i, taste in enumerate(tastes):
        for entry in dataset:
            if entry[label_column] == taste:
                original_smiles = entry[smiles_column]
                new_smiles_list = augmentation_without_duplication(original_smiles, augmentation_numbers[i])

                for new_smiles in new_smiles_list:
                    new_entry = deepcopy(entry)
                    new_entry[smiles_column] = new_smiles
                    augmented_data.append(new_entry)
            else:
                augmented_data.append(entry)

    # Convert augmented_data list to Dataset object
    augmented_dataset = Dataset.from_dict(
        {key: [entry[key] for entry in augmented_data] for key in augmented_data[0]}
    )

    return augmented_dataset

