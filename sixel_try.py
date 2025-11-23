# 3. Your code (as short as possible)
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

# example data
s = pl.Series("values", [3, 7, 8, 5, 12, 14, 21, 13, 11, 9, 20])

sns.set(style="darkgrid")
plt.figure(figsize=(8, 4))
sns.histplot(s.to_pandas(), kde=True, bins=10)  # or lineplot, etc.
plt.title("Polars Series → Seel plot in terminal")
plt.tight_layout()
plt.show()          # ← renders instantly as sixel in your SSH terminal
