# Combinatorial optimization with physics-inspired graph neural networks

This is my implementation of the [PI-GNN paper](https://arxiv.org/abs/2107.01188), using [`torch_geometric`](https://pytorch-geometric.readthedocs.io/).

Data here is from [this publicly available datset](https://web.stanford.edu/~yyye/yyye/Gset/), the same as that being used in the numerical benchmark of the original paper. You can download it using the helper script:

```bash
python3 data-helper.py
```

The default directory hierarchy is like below:

```
.
├── data/
│   ├── dataset/
│   ├── links.txt
│   └── raw/
├── log/
├── model/
├── model.py
├── params.py
└── utils.py
```

- `model.py` is the main file for the model.
- `params.py` contains the parameters for the model that one can tweak.
- `utils.py` contains the manipulation of data and a logger class.
- Processed data are stored in `data/dataset/` in the form of `nx.Graph` objects, and the Hamiltonian matrix is only calculated when the data is loaded for the sake of memory efficiency (This could be improved if replaced with a sparse matrix along with sparse multiplication in place of corresponding standard matrix multiplication).
