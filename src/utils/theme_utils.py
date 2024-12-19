import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Color palette
colors = [
  "#e50000", 
  "#165B33", 
  "#187F42", 
  "#f8b229", 
  "#91060d"
]

# Create a custom colormap
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_palette", colors)