# Loss Rider
Finally, a Python plotting library that can (only) output __Line Rider__ maps!

<p align="center" width="100%">
    <br>
    <img  src="https://github.com/user-attachments/assets/219ac1fa-57a5-4c7f-bab3-7bbb5a78a70b">
    <br>
</p>

ML practitioners can experience gradient descent like never before!
<p align="center" width="100%">
    <br>
    <img  src="https://github.com/user-attachments/assets/025fb50e-7b03-452e-8b45-a15e258012db">
    <br>
</p>


With support for all important features of a line graph.
<p align="center" width="100%">
    <br>
    <img  src="https://github.com/user-attachments/assets/da32dd51-ba91-4d3d-9bff-30c5f6c051d8">
    <br>
</p>


And don't forget interactive plotting in Jupyter Notebooks!
<p align="center" width="100%">
    <br>
    <img src="https://github.com/user-attachments/assets/12bed788-a3a3-441c-a991-a6565b526e00">
    <br>
</p>

The above plots all use data from the [Unit-Scaled Maximal Update Parameterization](https://arxiv.org/abs/2407.17465) paper which proposes a more usable version of μP.

# Installation
```bash
pip install lossrider
```

# Usage

```python
import pandas as pd
from lossrider import lossrider

# Load a csv that contains columns named "Validation Loss", "Run Count" and "model_type"
data = pd.read_csv("./_data/sweep_df.csv")

# Plot it!
lossrider(
    data,
    x="Run Count", 
    y="Validation Loss",
    hue="model_type",
    xlim=(0.6, 340),
    ylim=(3.2, 3.8),
    xticks=(1, 10, 100), 
    yticks=[x/10 for x in range(32, 39)],
    width=1000, height=500, fontsize=30,
    logx=True, grid=False,
    legend=True, legend_loc=(.65, 1),
    outfile='maps/sweep_strategies',
)
```
The above produces the below plot

![lossridergif_sweep](https://github.com/user-attachments/assets/84cc70ff-a28c-4bc6-9ddc-00dbec9e5063)

