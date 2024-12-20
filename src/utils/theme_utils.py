import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Color palette
colors = [
  "#e50000", # red
  "#165B33", # green
  "#187F42", # light green
  "#f8b229", # yellow
  "#91060d" # dark red
]

colors_pastel = [
    "#558a6a"
    "#97bd88",
    "#e8e791",
    "#e0878a",
]
# Create a custom colormap
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_palette", colors)