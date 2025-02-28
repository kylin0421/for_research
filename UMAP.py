import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import pandas as pd


with open("sum_of_6.json", "r", encoding="utf-8") as file:
    difference_tensor=torch.tensor(json.load(file), dtype=torch.float32)

print(difference_tensor.shape)

num_samples=difference_tensor.shape[0]
num_features=difference_tensor.shape[-1]

numpy_data=difference_tensor.cpu().numpy().reshape(num_samples,num_features)

print(numpy_data.shape)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(numpy_data)