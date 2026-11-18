# Cupid Graphing

Cupid Graphing is a Python course team project that recommends indirect matches from survey profiles. It takes responses in a terminal survey and opens an interactive Plotly graph showing the new profile and its recommended matches.

## How it works

1. Clean CSV profiles, encode categories, and scale numeric values.
2. Use NetworkX to create an undirected graph. Generate edge weights from Euclidean similarity and apply a minimum threshold.
3. Add the new profile. Explore two-hop paths and rank indirect matches using the product of the edge weights.
4. Display up to five recommendations and show the relevant portion of the graph in a browser.

| File | Role |
|---|---|
| `data_loader.py` | CSV cleaning and vector encoding |
| `graph_model.py` | Node and edge construction |
| `recommendation.py` | Similarity and two-hop ranking |
| `visualizer.py` | Plotly and Matplotlib rendering |
| `main.py` | Survey and workflow entry point |

## Setup and usage

From the developed version of the repository, activate a Python virtual environment. Place an authorized CSV in the project root as `cupid_algorithm_dataset.csv`.

```sh
python -m pip install -r requirements.txt
python main.py
```

Complete the survey and view the ranked matches in the terminal. The interactive plot opens in a browser.

## Tests and limitations

Run the included algorithm doctests with `python -m doctest data_loader.py graph_model.py recommendation.py`. Graph construction compares profile pairs, so the work grows quadratically with the number of profiles. Sparse graphs can yield fewer recommendations. Scores depend on the chosen features and threshold and describe graph similarity.
