<a href="https://colab.research.google.com/github/ronva-h/technology_fundamentals/blob/main/C3%20Machine%20Learning%20I/Tech%20Fun%20C3%20E2%20Support%20Vector%20Machines%20and%20Boosting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Technology Fundamentals, Course 3 Enrichment 1: Support Vector Machines and Boosting

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

**Teaching Assistants**: Varsha Bang, Harsha Vardhan

**Contact**: vbang@uw.edu, harshav@uw.edu
<br>

---

<br>

In this session, we're continuing on the topic of supervised learning models with Support Vector Machines or SVMs.

<br>

---

<br>

<a name='top'></a>

# Contents

* 5.0 [Preparing Environment and Importing Data](#x.0)
  * 5.0.1 [Import Packages](#x.0.1)
  * 5.0.2 [Load Dataset](#x.0.2)
* 5.1 [Support Vector Machines](#x.1)
  * 5.1.1 [Margin Maximization](#x.1.1)
  * 5.1.2 [Kernel SVM](#x.1.2)
  * 5.1.3 [Tuning SVMs](#x.1.3)
* 5.2 [Boosting](#x.2)
  * 5.2.1 [AdaBoost](#x.2.1)

<br>

---

<a name='x.0'></a>

## 5.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 5.0.1 Import Packages

[back to top](#top)


```python
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from ipywidgets import interact, FloatSlider, interactive
```

<a name='x.0.2'></a>

### 5.0.2 Load Dataset

[back to top](#top)

For this session, we will use dummy datasets from sklearn.

<a name='x.1'></a>

## 5.1 Support Vector Machines

[back to top](#top)

I want us to think back on Session 3 during our discussion on GMMs. GMMs, we will remember, are what we call a *generative* method. It is because of this, that GMMs are able to generate new data based on the underlying distribution it assumes. SVMs, on the other hand, are a *discriminative* model. It's perogative is to find the best way to delineate between data clusters. It does this by concerning itself with *margin*, something we will explore in the following cells.


```python
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=42, centers=2, cluster_std=2.5)
y[25] = 1
y[33] = 1
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
```




    <matplotlib.collections.PathCollection at 0x7f91f266b310>




    
![png](S7_Boosting_files/S7_Boosting_7_1.png)
    


We can also view this with an interactive plotly graph:


```python
# making the data labels
labels = []
i = 0
for row in X:
  labels.append("X: {}\nindex: {}".format(np.array2string(row.round()),i))
  i += 1

# the scatter plot
fig = px.scatter(x=X[:,0], y=X[:,1], hover_name=labels, color=y,
                 labels={
                     "y": "Feat 1",
                     "x": "Feat 2",
                     "color": "Label",
                 })
# updating fig attributes
fig.update_layout(
  autosize=False,
  width=800,
  height=500,
  title='A Smattering of Points with Outlier (index 45)'
  )
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="784e2dc5-27cb-4435-8776-94ae36d297bf" class="plotly-graph-div" style="height:500px; width:800px;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("784e2dc5-27cb-4435-8776-94ae36d297bf")) {
                    Plotly.newPlot(
                        '784e2dc5-27cb-4435-8776-94ae36d297bf',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>Feat 2=%{x}<br>Feat 1=%{y}<br>Label=%{marker.color}", "hovertext": ["X: [-4.  9.]\nindex: 0", "X: [7. 5.]\nindex: 1", "X: [-4.  9.]\nindex: 2", "X: [7. 7.]\nindex: 3", "X: [5. 5.]\nindex: 4", "X: [-2. 13.]\nindex: 5", "X: [5. 4.]\nindex: 6", "X: [1. 2.]\nindex: 7", "X: [-3.  5.]\nindex: 8", "X: [-5.  8.]\nindex: 9", "X: [-2.  5.]\nindex: 10", "X: [ 2. -1.]\nindex: 11", "X: [-4. 10.]\nindex: 12", "X: [-6. 10.]\nindex: 13", "X: [3. 3.]\nindex: 14", "X: [3. 1.]\nindex: 15", "X: [5. 8.]\nindex: 16", "X: [2. 3.]\nindex: 17", "X: [ 5. -1.]\nindex: 18", "X: [-2.  4.]\nindex: 19", "X: [ 5. -1.]\nindex: 20", "X: [4. 3.]\nindex: 21", "X: [ 7. -0.]\nindex: 22", "X: [7. 3.]\nindex: 23", "X: [5. 4.]\nindex: 24", "X: [ 1. 11.]\nindex: 25", "X: [4. 3.]\nindex: 26", "X: [-0. 12.]\nindex: 27", "X: [6. 3.]\nindex: 28", "X: [4. 9.]\nindex: 29", "X: [-3.  8.]\nindex: 30", "X: [-6.  7.]\nindex: 31", "X: [-5. 10.]\nindex: 32", "X: [1. 8.]\nindex: 33", "X: [-2.  5.]\nindex: 34", "X: [6. 7.]\nindex: 35", "X: [5. 3.]\nindex: 36", "X: [-2. 11.]\nindex: 37", "X: [-5.  8.]\nindex: 38", "X: [-2.  4.]\nindex: 39", "X: [-1.  9.]\nindex: 40", "X: [ 7. -1.]\nindex: 41", "X: [2. 4.]\nindex: 42", "X: [6. 2.]\nindex: 43", "X: [4. 0.]\nindex: 44", "X: [1. 8.]\nindex: 45", "X: [-3. 13.]\nindex: 46", "X: [-2. 10.]\nindex: 47", "X: [-5.  6.]\nindex: 48", "X: [2. 4.]\nindex: 49", "X: [5. 6.]\nindex: 50", "X: [-7.  8.]\nindex: 51", "X: [-4.  8.]\nindex: 52", "X: [6. 3.]\nindex: 53", "X: [-2.  4.]\nindex: 54", "X: [ 6. -0.]\nindex: 55", "X: [-2. 11.]\nindex: 56", "X: [5. 2.]\nindex: 57", "X: [ 2. -0.]\nindex: 58", "X: [-2.  7.]\nindex: 59", "X: [-2.  8.]\nindex: 60", "X: [4. 4.]\nindex: 61", "X: [-4. 10.]\nindex: 62", "X: [-0.  2.]\nindex: 63", "X: [ 8. -0.]\nindex: 64", "X: [2. 2.]\nindex: 65", "X: [9. 2.]\nindex: 66", "X: [-6.  8.]\nindex: 67", "X: [1. 2.]\nindex: 68", "X: [5. 3.]\nindex: 69", "X: [-4. 12.]\nindex: 70", "X: [2. 1.]\nindex: 71", "X: [9. 3.]\nindex: 72", "X: [7. 4.]\nindex: 73", "X: [4. 4.]\nindex: 74", "X: [4. 3.]\nindex: 75", "X: [5. 1.]\nindex: 76", "X: [-3. 12.]\nindex: 77", "X: [2. 6.]\nindex: 78", "X: [-3.  6.]\nindex: 79", "X: [5. 0.]\nindex: 80", "X: [-4. 14.]\nindex: 81", "X: [-0.  6.]\nindex: 82", "X: [-5.  5.]\nindex: 83", "X: [2. 3.]\nindex: 84", "X: [-3.  8.]\nindex: 85", "X: [-4.  8.]\nindex: 86", "X: [-9. 11.]\nindex: 87", "X: [ 5. -1.]\nindex: 88", "X: [-5. 10.]\nindex: 89", "X: [ 8. -2.]\nindex: 90", "X: [-4.  8.]\nindex: 91", "X: [-2.  8.]\nindex: 92", "X: [-4. 11.]\nindex: 93", "X: [-3.  7.]\nindex: 94", "X: [ 0. 11.]\nindex: 95", "X: [-0. 10.]\nindex: 96", "X: [ 3. -2.]\nindex: 97", "X: [-3. 10.]\nindex: 98", "X: [-2.  8.]\nindex: 99"], "legendgroup": "", "marker": {"color": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0], "coloraxis": "coloraxis", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scatter", "x": [-3.707133217665975, 7.347506944166294, -3.870154434365707, 6.695029236214326, 5.208528672738425, -1.6057086092817152, 5.331605834553149, 0.7632202585627699, -3.489468005883144, -4.607241431109347, -2.3403771113329404, 1.5475020590328965, -3.6828835878901307, -5.829662745298826, 2.9398170322818746, 2.6817456053875084, 4.790454361080667, 1.9626475910753207, 4.7854006323431015, -2.2797956817139946, 5.175113196553611, 4.158976424275295, 6.617458703835719, 7.048319159338906, 5.28958582184916, 1.4388344157157285, 4.447124562692841, -0.47788306706725514, 5.9447327502703455, 3.9782367531332112, -3.09458105986109, -6.205502598971319, -5.386681566608507, 1.1549242992511348, -1.6501518991315967, 6.107021570728778, 5.072831150856056, -1.6810390445438403, -4.530431630285719, -1.9870386355408618, -0.663031173064224, 6.673671879652275, 2.2031746606597977, 5.8229728976619635, 4.026408546220925, 1.1855374888010402, -2.598762720827629, -1.76889693039131, -5.275035058067821, 1.66162009322148, 4.6723835659228685, -6.821492204335332, -3.667741855083906, 5.534347237098809, -1.904291944137665, 6.204047205640617, -2.2665037496826494, 5.283754813035012, 2.4160927621642934, -1.6051075604336649, -2.2915799524573224, 4.081221872913474, -3.833598132470347, -0.15704920201950223, 7.536367783746611, 2.4302352457252687, 9.355343589254428, -6.047624478178786, 0.6211707498250325, 5.266110962092792, -3.660794550452219, 2.1635380234013804, 9.304315114089992, 7.496935872515653, 3.5235914560605486, 4.2366645570630785, 4.8890072489472045, -2.689222927003585, 2.3413182506435937, -2.542940684897585, 5.372560019474804, -4.013464153626242, -0.4528353427947778, -4.779257811855778, 1.9841195519128392, -2.7983183290233513, -4.264330357746131, -9.058560383277111, 4.553099411964993, -5.041275423888809, 8.146864613568349, -4.010794347849762, -1.6989876995657627, -4.201502623817647, -3.365983914369674, 0.06830118318712719, -0.22069232879756484, 2.522894541057089, -3.05837734264653, -2.4964139814465978], "xaxis": "x", "y": [8.550138686538778, 4.607674814027989, 9.291592602472987, 6.715152140575601, 5.241026569646802, 12.859377544363245, 4.041127806530792, 2.1445771209558004, 5.355498757868025, 8.241255188570285, 5.45241566266468, -1.327971848769959, 10.370686237163234, 9.50643921787113, 2.5538039268432406, 1.168015893426543, 8.131274965153947, 3.179350722048695, -0.8842560606358258, 4.04536384169609, -1.1411772628392383, 2.7270380397747633, -0.30029895304611554, 3.0051220012819777, 3.9277268633840077, 10.932872951080594, 2.826049620982342, 12.404886199625379, 2.715631367023697, 8.77359260041478, 8.42894373582537, 7.21467560721155, 9.953531174062501, 8.449845376981983, 4.606685739791486, 7.449308748465678, 2.9364631332628237, 11.45314894600422, 7.7598935192369805, 4.1151108184988825, 9.442706831173247, -1.103991107144156, 3.940881193296862, 1.7910974022985502, 0.08882927304700816, 7.718610582514203, 12.925895267733337, 9.666924308648046, 6.023769567996645, 3.6145537055253065, 5.607004876834024, 7.60856730509589, 7.8499617442726795, 3.3751309998613177, 4.231085516553827, -0.1697242070999745, 11.435898604530545, 1.787054894525314, -0.06635602847286393, 7.401486741685511, 8.266767752033653, 3.7581709191709622, 10.297454710981711, 1.9068849953176898, -0.07853611193854437, 2.357482448804552, 2.4096142160203295, 7.962672821284924, 2.4347543302714927, 2.839290207683171, 11.65709169374561, 0.5574253599338022, 3.157751986220201, 3.8530022656576675, 4.114166669749412, 2.983296825977078, 0.7144805486502339, 11.523118372928382, 5.848005696484581, 6.37000880580857, 0.1872911388748124, 13.644981589470666, 5.9621770032707655, 5.483526874860093, 3.157150760528686, 8.261526889225099, 8.1951307617039, 11.069042389136381, -0.9485254101080982, 9.799904459686505, -1.53145797303997, 8.28505175371513, 8.05158042715753, 10.543476850300491, 7.008592955144274, 11.342486425988819, 9.836163902347533, -1.8139483777739294, 9.907067556977687, 8.427818294760455], "yaxis": "y"}],
                        {"autosize": false, "coloraxis": {"colorbar": {"title": {"text": "Label"}}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "height": 500, "legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "A Smattering of Points with Outlier (index 45)"}, "width": 800, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Feat 2"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Feat 1"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('784e2dc5-27cb-4435-8776-94ae36d297bf');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>



```python
clf = svm.SVC(kernel='linear', C=.01)
clf.fit(X, y)
```




    SVC(C=0.01, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)




```python
def plot_boundaries(X, clf, ax=False):
  plot_step = 0.02
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                        np.arange(y_min, y_max, plot_step))
  
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  
  if ax:
    cs = ax.contourf(xx, yy, Z, cmap='viridis', alpha=0.2)
    ax.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='grey', alpha=0.9)
    return ax
  else:
    cs = plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.2)
    plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='grey', alpha=0.9)
  
```


```python
plot_boundaries(X, clf)
```


    
![png](S7_Boosting_files/S7_Boosting_12_0.png)
    


<a name='x.1.1'></a>

### 5.1.1 Margin Maximization

[back to top](#top)

We can change how the learner defines the boarders by the 'C' parameter, what is happening here?


```python
fig, ax = plt.subplots(1, 2, figsize=(10,5))
clf = svm.SVC(kernel='linear', C=.05)
clf.fit(X, y)
plot_boundaries(X, clf, ax[0])

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)
plot_boundaries(X, clf, ax[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f91f0ccf110>




    
![png](S7_Boosting_files/S7_Boosting_14_1.png)
    


We can see more clearly what is happening when we view the *margin* around the decision boundary, where point labels are used to determine the label space.


```python
fig, ax = plt.subplots(1, 2, figsize=(10,5))
def plot_svm(penalty, subplot, X, y):
  clf = svm.SVC(kernel='linear', C=penalty)
  clf.fit(X, y)

  # get the separating hyperplane
  w = clf.coef_[0]
  a = -w[0] / w[1]
  xx = np.linspace(-5, 5)
  yy = a * xx - (clf.intercept_[0]) / w[1]

  # plot the parallels to the separating hyperplane that pass through the
  # support vectors (margin away from hyperplane in direction
  # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
  # 2-d.
  margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
  yy_down = yy - np.sqrt(1 + a ** 2) * margin
  yy_up = yy + np.sqrt(1 + a ** 2) * margin

  # plot the line, the points, and the nearest vectors to the plane
  # plt.clf()
  subplot.plot(xx, yy, 'k-')
  subplot.plot(xx, yy_down, 'k--')
  subplot.plot(xx, yy_up, 'k--')

  subplot.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
              facecolors='none', zorder=10, edgecolors='k')
  subplot.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap='viridis',
              edgecolors='grey')

  subplot.axis('tight')
  plot_step = 0.02
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                        np.arange(y_min, y_max, plot_step))
  
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  cs = subplot.contourf(xx, yy, Z, cmap='viridis', alpha=0.2)

  subplot.set_xlim(x_min, x_max)
  subplot.set_ylim(y_min, y_max)
  subplot.set_title('SVC, C={}'.format(penalty))

for penalty, subplot in ((1, ax[0]), (0.05, ax[1])):
  plot_svm(penalty, subplot, X, y)
    

plt.show()
```


    
![png](S7_Boosting_files/S7_Boosting_16_0.png)
    


In the above diagrams, the circled datapoints affect the *margin* between the two class labels. The SVM algorithm is trying to maximize the margin. The dotted lines are called the *support vectors* and are defined by the outermost points from the decision boundary that fall within the margin.

Effectively, the higher we set the penalty, C, the more we are saying "Don't trust the overall distribution so much, just do your best to segment the local points. A lower C takes into account many more points in the sampled data near the boundary.

> Do the points falling outside either of the support vectors affect the decision boundary?


```python
def changing_margins(penalty=0.1):
  fig, subplot = plt.subplots(1, 1, figsize=(7,7))
  plot_svm(penalty, subplot, X, y)
  fig.show()
```


```python
myslider = FloatSlider(
    value=0.1,
    min=0.1,
    max=3.01,
    step=0.1,
    description='C:',
    continuous_update=True,
    orientation='horizontal',
    readout=True,
)
```

At C=0.1 5 purple and 5 yellow points fall within the decision margin, and are taken into consideration when setting the decision boundary. At C=3.0 this decreases to the nearest 2 purple and 3 yellow. 


```python
int_plot = interactive(changing_margins,penalty=myslider)
output = int_plot.children[-1]
output.layout.height = '350px'
int_plot
```


    interactive(children=(FloatSlider(value=0.1, description='C:', max=3.01, min=0.1), Output(layout=Layout(height…


<a name='x.1.2'></a>

### 5.1.2 Kernel SVM

[back to top](#top)

When we were looking at KNN and GMM, we saw that GMM had the added advantage of non-linear decision boundaries. We can achieve a similar advantage with kernel based SVMs.

First, I want us to revisit some ideas from the session on feature engineering.

How do we leverage linear based classifiers with non-linear real boundaries between clusters? We create new features.


```python
from sklearn.datasets import make_circles
X, y = make_circles(random_state=42, noise=.01)
relabel = dict(zip([0,1,2,3],[0,1,0,1]))
y = np.vectorize(relabel.get)(y)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
```




    <matplotlib.collections.PathCollection at 0x7f91f2672310>




    
![png](S7_Boosting_files/S7_Boosting_24_1.png)
    


We run into hardtimes with the raw linear SVM:


```python
fig, ax = plt.subplots(1, 1, figsize=(5,5))
plot_svm(penalty=0.1, subplot=ax, X=X, y=y)
```


    
![png](S7_Boosting_files/S7_Boosting_26_0.png)
    


#### 5.1.2.1 Exercise: Engineer Features for Linear SVM

Create a new X3 feature that is a linear combination of X1 and X2. Separate this new represenation using Linear SVM


```python
# Code Cell for Exercise 5.1.2.1
```

Plot along this 3rd dimension, to show we can make an easy distinction between the two classes with a linear hyperplane:


```python
# Code Cell for Exercise 5.1.2.1
```

But the guesswork involved in what features to engineer, as well as the computational cost to determine them, can be a real deal breaker. Kernel SVMs introduce the so-called *kernel trick*, where we learn the classification in a higher-dimensional space without actually computing the representation. Instead, it works by directly computing the distance of the datapoints in the new space.


```python
clf = svm.SVC(kernel='rbf', C=0.1)
clf.fit(X, y)
plot_boundaries(X, clf)
```


    
![png](S7_Boosting_files/S7_Boosting_32_0.png)
    


<a name='x.2'></a>

## 5.2 Boosting

[back to top](#top)

The last supervised learning algorithms we will cover, are the boosting learners. Similar to Bagging, Boosting algorithms leverage the idea of training on variations of the available data, only this time they do so in serial rather than parallel.

What do I mean by this?

It's a little nuanced, but the idea is straight forward. The first model trains on the dataset, it generates some error. The datapoints creating the greatest amount of error are emphasized in the second round of training, and so on, as the sequence of models proceeds, ever troublesome datapoints receive ever increasing influence. 



<a name='x.2.1'></a>

### 5.2.1 AdaBoost

[Back to Top](#top)

AdaBoost is the first boosting learner. It's _weak learners_ the things that are stitched together in serial, are typically _stumps_ or really shallow decision trees


```python
X, y = make_circles(random_state=42, noise=.01)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
clf.fit(X,y)
plot_boundaries(X, clf)
```


    
![png](S7_Boosting_files/S7_Boosting_35_0.png)
    



```python
X, y = make_blobs(random_state=42, centers=2, cluster_std=2.5)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5))
clf.fit(X,y)
plot_boundaries(X, clf)
```


    
![png](S7_Boosting_files/S7_Boosting_36_0.png)
    



```python
from sklearn.datasets import make_moons
X, y = make_moons(random_state=42, noise=.05)
clf = AdaBoostClassifier()
clf.fit(X,y)
plot_boundaries(X, clf)
```


    
![png](S7_Boosting_files/S7_Boosting_37_0.png)
    


<a name='x.2.1'></a>

### 5.2.1 Gradient Boosting

[Back to Top](#top)

_Gradient Boosting_ builds on the idea of _AdaBoost_. The term _gradient_ implies that 2 or more derivatives are being taken somewhere. What this is referring to, is while AdaBoost is subject to a predefined loss function, Gradient Boosting can take in any arbitrary (as long as it is differentiable) loss function, to coordinate the training of its weak learners.


```python
X, y = make_circles(random_state=42, noise=.01)
clf = GradientBoostingClassifier(loss='deviance')
clf.fit(X,y)
plot_boundaries(X, clf)
```


    
![png](S7_Boosting_files/S7_Boosting_39_0.png)
    



```python
X, y = make_blobs(random_state=42, centers=2, cluster_std=2.5)
clf = GradientBoostingClassifier()
clf.fit(X,y)
plot_boundaries(X, clf)
```


    
![png](S7_Boosting_files/S7_Boosting_40_0.png)
    



```python
from sklearn.datasets import make_moons
X, y = make_moons(random_state=42, noise=.05)
clf.fit(X,y)
plot_boundaries(X, clf)
```


    
![png](S7_Boosting_files/S7_Boosting_41_0.png)
    


# References

[back to top](#top)

* [Generative vs Discriminative Models](https://medium.com/swlh/machine-learning-generative-vs-discriminative-models-9d0fdd156296#:~:text=Commonly%20used%20discriminative%20learning%20algorithms,random%20forest%20and%20gradient%20boosting.&text=The%20most%20commonly%20used%20generative%20algorithm%20is%20the%20naive%20Bayes%20classifier.)


```python

```
