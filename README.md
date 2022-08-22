# CMF

This is the source code for paper [Understanding Event Predictions via Contextualized Multilevel Feature Learning](https://yue-ning.github.io/docs/CIKM21_cmf.pdf) appeared in CIKM21


## Data
We processed some country based datasets from the ICEWS data. Please find example datasets in this [Google Drive Link](https://drive.google.com/drive/folders/1WKdTOE5tGSBUHE5xAY4dwImQ3uU2N_7R?usp=sharing). A brief introduction of the data file is as follows:
- `loc2id.txt` location to id mapping (several locations are considered for each country).
- `loc_entity2id.txt` entity to id mapping.
- `data_count.pkl` event count data
- `data_label.pkl` location, time, etc.
- `data_graph.bin` DGL graphs.
- `loc_text_emb.pkl` document embeddings.

## Prerequisites
The code has been successfully tested in the following environment. (For older dgl versions, you may need to modify the code)
- Python 3.7.9
- PyTorch 1.7.0+cu92
- dgl 0.5.2
- Sklearn 0.23.2 

## Getting Started
### Prepare your code
Clone this repo.
```bash
git clone https://github.com/amy-deng/CMF
cd CMF
```
### Prepare your data
Download the dataset (e.g., `EG`) from the given link and store them in `data` filder. Or prepare your own dataset in a similar format. The folder structure is as follows:
```sh
- CMF
	- data
		- EG
		- your own dataset
	- src
```

### Training and testing
Please run following commands for training and testing under the `src` folder. We take the dataset `EG` as the example.

**Evaluate the event prediction model**
```python
python train_pred.py -sl 7 -s -m cmf -ho 1 --gpu 1 -d EG -hd 32 -nl 2 -td 64 --eid 13 --lr 0.003 -w -l 5
```
**Evaluate the event prediction and explanation model**
```python
python train_pred_exp.py -sl 7 -s -m cmf -ho 1 --gpu 1 -d EG -hd 32 -nl 2 -td 64 --eid 13 --lr 0.003 -w -l 1
```

## Cite

Please cite our paper if you find this code useful for your research:

```
@inbook{10.1145/3459637.3482309,
author = {Deng, Songgaojun and Rangwala, Huzefa and Ning, Yue},
title = {Understanding Event Predictions via Contextualized Multilevel Feature Learning},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3482309},
booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
pages = {342–351},
numpages = {10}
}
```
