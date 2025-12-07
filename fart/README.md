## Flavor Analysis and Recognizion Transformer (FART)

## Description
The Flavor Analysis and Recognition Transformer (FART) is a state-of-the-art machine learning model designed to predict molecular taste from chemical structures encoded as SMILES. Developed using the pre-trained foundation model ChemBERTa, FART leverages a transformer architecture to classify molecules across four key taste categories—sweet, bitter, sour, and umami—while also accommodating tasteless or undefined compounds. Unlike previous approaches that relied on binary classification, FART performs parallel multi-class predictions with an accuracy exceeding 91%, offering interpretability through gradient-based visualizations of molecular features. This novel approach facilitates the identification of key structural elements influencing taste, enabling applications in both flavor compound discovery and rational food design.

## Installation
All .ipynb notebooks can be run on GoogleColab without any further modifications. 

## Overview of files

* The entire raw, curated and enriched dataset is found in /dataset
* The source databases are found in dataset/individual-datasets
* Files for the extraction, curation and enrichment of the dataset are found in /dataset/scripts
* Files for the training of the tree-based and transformer models are found in /models
* Script for the generation of the interpretability heatmaps are found in /plots

## Dataset

The FART dataset is the largest publicly available collection of molecular tastants to date, comprising 15,025 curated entries derived from six independent sources. Each molecule is annotated with one or more taste labels (sweet, bitter, sour, umami, or undefined) following rigorous curation protocols. Duplicates were removed based on canonicalized SMILES, reducing overlap among entries. Additional enrichment with metadata such as PubChem ID, IUPAC name, molecular formula, and molecular weight was performed using the PubChem API. The dataset adheres to the FAIR principles, ensuring accessibility and reusability, and is hosted publicly to support further research. Its chemical diversity spans a molecular weight range centered at 374 Da ± 228, making it suitable for small molecule taste prediction.

`FART_Data_Extraction.ipynb` extracts data from five different online sources and produces the dataset `fart_uncurated.csv`. 

`FART_Data_curation.ipynb` curates the extracted data by for example removing duplicates through standardized SMILES. This scripts produces the dataset `fart_curated.csv` which was used in the training of the machine learning models. 

`FART_dataset_enrichment.ipynb` can be optionally used to retrieve more features for molecules which are also listed on PubChem. This script produces the `fart_enriched.csv` dataset which additionally includes the columns `PubChemID`, `IUPAC Name`, `Molecular Formula`, `Molecular Weight`, `InChI` and `InChiKey`. 

## Isomeric SMILES Standardization

To ensure consistent molecular representation across the dataset, the following scripts standardize all SMILES to isomeric format, which preserves stereochemical information (unlike canonical SMILES). This is important because the original dataset contained a mix of isomeric and non-isomeric SMILES representations.

### Workflow:

1. **`get_pubchem_cid_from_phytocompounds.py`**: Extracts PubChem Compound IDs (CIDs) for compounds in the phytocompounds database by scraping their web pages. This is a prerequisite step to enable isomeric SMILES lookup for phytocompound entries.

2. **`data_extraction.py`**: Creates `fart_uncurated_with_ids.csv` by combining data from individual datasets and enriching entries with PubChem identifiers (PubChemID, InChI, and INCHIKEY). This file serves as a lookup table for mapping canonical SMILES to PubChem identifiers needed for isomeric SMILES retrieval.

3. **`add_isomeric_smiles.py`**: Processes the split datasets (`fart_train.csv`, `fart_val.csv`, `fart_test.csv`) to replace canonical SMILES with isomeric SMILES from PubChem. This script:
   - Matches entries by canonical SMILES to `fart_uncurated_with_ids.csv`
   - Uses PubChem IDs, InChI, or INCHIKEY to fetch isomeric SMILES via the PubChem API
   - Replaces the "Canonicalized SMILES" column with isomeric SMILES where available
   - Removes the "is_multiclass" column
   - Uses optimized batch processing (up to 100 compounds per API request) to comply with PubChem rate limits (5 requests/second)
   - Outputs standardized split files in `dataset/isomeric-splits/` with the same filenames

The resulting isomeric-splits ensure all molecular structures are represented consistently with stereochemistry preserved, which is crucial for accurate taste prediction models.

## Random Forest Models 

All three tree-based classifiers were trained in `model/Tree-Baseline-Models.ipynb`.

## Transformer Models

The transformer models were trained in `model/FART_Models.ipynb`. The data is loaded using the hugging face api. For different pretrained models one needs to adjust the `model_checkpoint` parameter. To use a weighted loss fontion, one needs to use `trainer = CustomTrainer` instead of `trainer = Trainer`. To use augmentation on needs to set `augmentation = True`.

