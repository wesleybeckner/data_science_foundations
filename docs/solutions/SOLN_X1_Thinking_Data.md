# Data Science Foundations <br> Extras 1: Thinking Data

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

---

<br>

Today we are going to take our newfound knowledge and tackle some data problems

<br>

---

<br>

<a name='x.0'></a>

## Prepare Environment and Import Data

[back to top](#top)


```python
# basic packages
import pandas as pd
import numpy as np
import random
import copy

# visualization packages
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns; sns.set()
import graphviz 

# stats packages
import scipy.stats as stats
from scipy.spatial.distance import cdist
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# sklearn preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# sklearn modeling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.mixture import GaussianMixture

# sklearn evaluation
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
```

## Warm Up

Add aditional feature(s) to X to predict y with a model limited to a linear classification boundary


```python
from sklearn.datasets import make_circles
X, y = make_circles(random_state=42, noise=.01)
relabel = dict(zip([0,1,2,3],[0,1,0,1]))
y = np.vectorize(relabel.get)(y)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
```




    <matplotlib.collections.PathCollection at 0x7f20ec66d280>




    
![png](SOLN_X1_Thinking_Data_files/SOLN_X1_Thinking_Data_5_1.png)
    



```python
X2 = (X**2).sum(axis=1)
X_ = np.hstack((X,X2.reshape(-1,1)))
```

We can separate:


```python
px.scatter_3d(x=X_[:,0], y=X_[:,1], z=X_[:,2], color=y)
```


<div>                            <div id="23c76349-0c4c-433e-bc34-57f6a8211455" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("23c76349-0c4c-433e-bc34-57f6a8211455")) {                    Plotly.newPlot(                        "23c76349-0c4c-433e-bc34-57f6a8211455",                        [{"hovertemplate":"x=%{x}<br>y=%{y}<br>z=%{z}<br>color=%{marker.color}<extra></extra>","legendgroup":"","marker":{"color":[1,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,0,1,1,0,1,1],"coloraxis":"coloraxis","symbol":"circle"},"mode":"markers","name":"","scene":"scene","showlegend":false,"x":[-0.42229838489974536,0.7485816145848708,-0.6450977253766977,0.8000536425436547,0.7436580402757358,0.19688136854079694,-0.9251976351182513,-0.6439451432602576,0.31369494190757896,0.9922029811934012,-0.6389293283312876,-0.8075121035938413,-0.7485312466967836,-0.5436472732419947,0.255940646479863,0.8925358850372773,-0.7907611685951266,-0.7809697490763782,0.04796627706072417,-0.7193422145214146,0.6394731661924931,0.05747324813870975,-0.9836154892738543,0.527123745433386,-0.5962744088092541,-0.3199796523277611,0.30235638817264293,0.7040585720445872,0.4191519084312905,-0.7522813588751582,0.19824858481611726,0.9253588256559914,-0.23484343477059344,-0.9263962592994999,0.5295895394382066,0.8013907247327959,-0.35384220275839257,-0.24858650553675077,-0.3094710209278159,-0.4206550542413567,0.9923956959479598,-0.4178825635175654,0.6264564667629178,0.656088711644121,-0.9757861410572192,-0.7304032497177185,-0.5706854949457302,-0.9883079235304738,0.052288916963517354,-0.0730676281701983,-1.0092199970997282,0.9397813419658724,-0.5418407674347141,0.07095204675812743,0.5329719019691332,-0.735281210845652,0.7346375681292824,-0.15133496101565513,-0.18587472415089182,0.13660431101343226,0.596606584060097,0.7840928147304432,0.7073877469977338,0.6242382881411088,0.8031589090715232,-0.4195563137982508,0.8569429179440176,-0.4962948573257487,0.146173859772076,0.7716605691754996,-0.7011611772329107,0.43184904866075885,0.4347227296141881,0.9719471487664937,0.7726919591338647,0.5038454580401076,-0.8231806824818784,-0.6264606918532787,0.5895260274461493,0.3275664577116472,-0.060487414320652286,-0.3361256928671704,-0.06945995327887833,-0.8697387067322236,0.9802820450999337,0.7732134327573692,-0.8756231675328253,0.9874786376898969,-0.9777172865018552,-0.05436003175535164,0.32070805538674396,-0.7991046154679821,-0.15038148878015015,-0.5138532099197699,-0.7923792964804648,0.2461416507483467,-0.6851155260924963,-0.19470823659876327,0.4273042621600151,0.7900072611607154],"y":[-0.6845295470874115,0.3075362548317282,0.47619864829576936,-0.588905130497542,-0.6957860892740887,-0.9650220860429349,0.3512816789010189,-0.4710393913755656,0.9584177398018461,-0.008438963603439258,0.7608576660568395,-0.5889194648059373,0.18870100084377522,-0.840088494804106,-0.73797295320655,0.48999100487146674,-0.09130028301433397,-0.20211356869998176,0.9957382532658814,-0.6866440302970587,0.4666304203810586,-0.8009790291122068,-0.13844647582181538,-0.8493943576983302,0.5770743189077636,0.9602053595144137,-0.9561943912102361,-0.39998810849123984,0.9033623243986034,0.2862747484599416,0.992337119535715,-0.3689662035364896,-0.7751100018833784,-0.34837884071514347,-0.6358097064066637,0.5894718461644061,0.7171791749814915,0.7748584286830591,-0.9488871490335301,0.9102619393784438,-0.13730410419768682,-0.9005121578333893,0.7636902271365865,-0.46417337052550134,-0.25138922485561194,0.6907171436858398,-0.5469811931618388,0.008733169690893945,0.7839865492256549,1.0054391347787237,0.12652301848326394,0.3696487613072706,0.835861781933367,-0.9933096555130018,0.8549808501904745,-0.27661370357448806,0.6829021015870697,0.7664840986279404,-0.9824444001272132,-0.7860826768386622,0.5402922082434758,-0.08921730466694154,0.4160138914251549,-0.763309370355407,0.0005928128405319588,0.6751650868247381,-0.49022724938284173,0.6281308832586755,0.7908657712243993,-0.29481151012941076,-0.38349695223022706,-0.9083160642937471,0.6560200444641161,-0.2481835890557028,-0.20370448321588835,0.6082753832510313,-0.0026801835184289816,-0.7738673530164978,-0.5584396934433554,0.716354023542139,0.7865909866292667,-0.7060723985339563,-1.0149449109907747,-0.4961126822954557,0.13028319010622122,0.19208015797855874,0.47765958579599965,0.24193183982264083,0.24413955542497917,-0.7924649462245741,-0.7196013833044757,0.10803481689569931,-0.7806428560057732,-0.6120303682464148,0.5803928879779222,0.7559830401785916,0.3810635232912717,0.9765044815890472,-0.6833973905783381,0.10183860540241418],"z":[0.6469166267246302,0.6549529817304178,0.6429162279249072,0.9868950836736969,1.037145562894078,0.9700298998291549,0.9793894819599234,0.636543455755937,1.0169690805452785,0.9845379719957731,0.9871350745993313,0.9989019334778623,0.5959070950008836,1.0013010368056845,0.610109694184927,1.0367114909342257,0.6336389674364277,0.6507636436250633,0.9937954327520547,0.9889332459351607,0.6266698795052553,0.6448705793291472,0.9866668574069244,0.9993302178894801,0.6885579401436848,1.0243813103439992,1.00572709925072,0.6556889598038631,0.9917518114844406,0.6478804745168574,1.0240354601900332,0.9924250155715657,0.6559469538745009,0.9795778459021335,0.6847190630432998,0.9897041511082291,0.6395502734800442,0.6622008352361525,0.9961591343950855,1.0055274729398045,1.003701634365565,0.9855479832977592,0.9756704677730097,0.6459093154518468,1.0153551354528447,1.010579079779729,0.6248703598142018,0.9768288199659667,0.6173690402039679,1.016246732031007,1.0345330767520917,1.0198291774431754,0.9922563357510634,0.9916982646745257,1.015051302481022,0.6171536000278429,1.0060476368589424,0.6104001438750962,0.99974641241949,0.6365867126132911,0.647855086432469,0.6227612695639431,0.6734649824612303,0.9723146352537099,0.6450645846480232,0.6318753949150324,0.9746739206518741,0.6408569919113091,0.6468354653750303,0.6823738605250397,0.6386969088285143,1.011531673483285,0.6193461503819113,1.0062763538718866,0.6385483801923821,0.6238591874568349,0.6776336193950235,0.9913236785020494,0.6593958282495468,0.6204628712627677,0.6223841075376015,0.6115187133769294,1.0349378574555754,1.0025732115226327,0.9779265975695622,0.6347537996855028,0.9948746114230242,1.0336450750138624,1.0155352148476573,0.63095570404716,0.6206798076436617,0.6502397081239187,0.6320178608005848,0.6386262930006932,0.9647208539062293,0.6320960692707871,0.6145926928761555,0.9914722999628939,0.6496209259093966,0.6344825742369634],"type":"scatter3d"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"scene":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"xaxis":{"title":{"text":"x"}},"yaxis":{"title":{"text":"y"}},"zaxis":{"title":{"text":"z"}}},"coloraxis":{"colorbar":{"title":{"text":"color"}},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"legend":{"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('23c76349-0c4c-433e-bc34-57f6a8211455');
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

                        })                };                });            </script>        </div>


and now predict


```python
model = LogisticRegression()
model.fit(X_, y)
y_pred = model.predict(X_)
r2_score(y, y_pred)
```




    1.0



## Exploratory Data Analysis

which columns are numerical, string; which contain nans/nulls; what is the VIF between features


```python
airbnb = pd.read_csv("https://raw.githubusercontent.com/wesleybeckner/datasets/main/datasets/airbnb/AB_NYC_2019.csv")
```


```python
airbnb.shape
```




    (48895, 16)




```python
airbnb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>2787</td>
      <td>John</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>Private room</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2595</td>
      <td>Skylit Midtown Castle</td>
      <td>2845</td>
      <td>Jennifer</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3647</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>4632</td>
      <td>Elisabeth</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>Private room</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3831</td>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>1</td>
      <td>270</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5022</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>7192</td>
      <td>Laura</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
airbnb.dtypes
```




    id                                  int64
    name                               object
    host_id                             int64
    host_name                          object
    neighbourhood_group                object
    neighbourhood                      object
    latitude                          float64
    longitude                         float64
    room_type                          object
    price                               int64
    minimum_nights                      int64
    number_of_reviews                   int64
    last_review                        object
    reviews_per_month                 float64
    calculated_host_listings_count      int64
    availability_365                    int64
    dtype: object




```python
airbnb.isnull().sum(axis=0)
```




    id                                    0
    name                                 16
    host_id                               0
    host_name                            21
    neighbourhood_group                   0
    neighbourhood                         0
    latitude                              0
    longitude                             0
    room_type                             0
    price                                 0
    minimum_nights                        0
    number_of_reviews                     0
    last_review                       10052
    reviews_per_month                 10052
    calculated_host_listings_count        0
    availability_365                      0
    dtype: int64




```python
airbnb.nunique()
```




    id                                48895
    name                              47905
    host_id                           37457
    host_name                         11452
    neighbourhood_group                   5
    neighbourhood                       221
    latitude                          19048
    longitude                         14718
    room_type                             3
    price                               674
    minimum_nights                      109
    number_of_reviews                   394
    last_review                        1764
    reviews_per_month                   937
    calculated_host_listings_count       47
    availability_365                    366
    dtype: int64




```python
plt.figure(figsize=(10,6))
sns.scatterplot(x=airbnb.longitude,y=airbnb.latitude,hue=airbnb.neighbourhood_group)
plt.ioff()
```




    <matplotlib.pyplot._IoffContext at 0x7f20937cf0d0>




    
![png](SOLN_X1_Thinking_Data_files/SOLN_X1_Thinking_Data_19_1.png)
    



```python
X = airbnb.copy()
```

reviews_per_month has some 'nans'


```python
X_num = X.select_dtypes(exclude='object')
X_num.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48890</th>
      <td>36484665</td>
      <td>8232441</td>
      <td>40.67853</td>
      <td>-73.94995</td>
      <td>70</td>
      <td>2</td>
      <td>0</td>
      <td>NaN</td>
      <td>2</td>
      <td>9</td>
    </tr>
    <tr>
      <th>48891</th>
      <td>36485057</td>
      <td>6570630</td>
      <td>40.70184</td>
      <td>-73.93317</td>
      <td>40</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>2</td>
      <td>36</td>
    </tr>
    <tr>
      <th>48892</th>
      <td>36485431</td>
      <td>23492952</td>
      <td>40.81475</td>
      <td>-73.94867</td>
      <td>115</td>
      <td>10</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>27</td>
    </tr>
    <tr>
      <th>48893</th>
      <td>36485609</td>
      <td>30985759</td>
      <td>40.75751</td>
      <td>-73.99112</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>48894</th>
      <td>36487245</td>
      <td>68119814</td>
      <td>40.76404</td>
      <td>-73.98933</td>
      <td>90</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_num.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539</td>
      <td>2787</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>0.21</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2595</td>
      <td>2845</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>0.38</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3647</td>
      <td>4632</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3831</td>
      <td>4869</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>89</td>
      <td>1</td>
      <td>270</td>
      <td>4.64</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5022</td>
      <td>7192</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_num.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.889500e+04</td>
      <td>4.889500e+04</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>38843.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.901714e+07</td>
      <td>6.762001e+07</td>
      <td>40.728949</td>
      <td>-73.952170</td>
      <td>152.720687</td>
      <td>7.029962</td>
      <td>23.274466</td>
      <td>1.373221</td>
      <td>7.143982</td>
      <td>112.781327</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.098311e+07</td>
      <td>7.861097e+07</td>
      <td>0.054530</td>
      <td>0.046157</td>
      <td>240.154170</td>
      <td>20.510550</td>
      <td>44.550582</td>
      <td>1.680442</td>
      <td>32.952519</td>
      <td>131.622289</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.539000e+03</td>
      <td>2.438000e+03</td>
      <td>40.499790</td>
      <td>-74.244420</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.471945e+06</td>
      <td>7.822033e+06</td>
      <td>40.690100</td>
      <td>-73.983070</td>
      <td>69.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.190000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.967728e+07</td>
      <td>3.079382e+07</td>
      <td>40.723070</td>
      <td>-73.955680</td>
      <td>106.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.720000</td>
      <td>1.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.915218e+07</td>
      <td>1.074344e+08</td>
      <td>40.763115</td>
      <td>-73.936275</td>
      <td>175.000000</td>
      <td>5.000000</td>
      <td>24.000000</td>
      <td>2.020000</td>
      <td>2.000000</td>
      <td>227.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.648724e+07</td>
      <td>2.743213e+08</td>
      <td>40.913060</td>
      <td>-73.712990</td>
      <td>10000.000000</td>
      <td>1250.000000</td>
      <td>629.000000</td>
      <td>58.500000</td>
      <td>327.000000</td>
      <td>365.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.dropna(inplace=True)
X_num = X.select_dtypes(exclude='object')
```


```python
vif = [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
pd.DataFrame(vif, index=X_num.columns)
```


```python
X_num.drop('longitude', axis=1, inplace=True)
```

    /home/wbeckner/anaconda3/envs/py39/lib/python3.9/site-packages/pandas/core/frame.py:4906: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    



```python
X_num
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>latitude</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539</td>
      <td>2787</td>
      <td>40.64749</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>0.21</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2595</td>
      <td>2845</td>
      <td>40.75362</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>0.38</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3831</td>
      <td>4869</td>
      <td>40.68514</td>
      <td>89</td>
      <td>1</td>
      <td>270</td>
      <td>4.64</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5022</td>
      <td>7192</td>
      <td>40.79851</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5099</td>
      <td>7322</td>
      <td>40.74767</td>
      <td>200</td>
      <td>3</td>
      <td>74</td>
      <td>0.59</td>
      <td>1</td>
      <td>129</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>48782</th>
      <td>36425863</td>
      <td>83554966</td>
      <td>40.78099</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>1.00</td>
      <td>1</td>
      <td>147</td>
    </tr>
    <tr>
      <th>48790</th>
      <td>36427429</td>
      <td>257683179</td>
      <td>40.75104</td>
      <td>45</td>
      <td>1</td>
      <td>1</td>
      <td>1.00</td>
      <td>6</td>
      <td>339</td>
    </tr>
    <tr>
      <th>48799</th>
      <td>36438336</td>
      <td>211644523</td>
      <td>40.54179</td>
      <td>235</td>
      <td>1</td>
      <td>1</td>
      <td>1.00</td>
      <td>1</td>
      <td>87</td>
    </tr>
    <tr>
      <th>48805</th>
      <td>36442252</td>
      <td>273841667</td>
      <td>40.80787</td>
      <td>100</td>
      <td>1</td>
      <td>2</td>
      <td>2.00</td>
      <td>1</td>
      <td>40</td>
    </tr>
    <tr>
      <th>48852</th>
      <td>36455809</td>
      <td>74162901</td>
      <td>40.69805</td>
      <td>30</td>
      <td>1</td>
      <td>1</td>
      <td>1.00</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>38821 rows × 9 columns</p>
</div>




```python
vif = [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
pd.DataFrame(vif, index=X_num.columns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>8.424770</td>
    </tr>
    <tr>
      <th>host_id</th>
      <td>2.827543</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>7.297302</td>
    </tr>
    <tr>
      <th>price</th>
      <td>1.538975</td>
    </tr>
    <tr>
      <th>minimum_nights</th>
      <td>1.157468</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>3.215893</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>3.858006</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count</th>
      <td>1.106414</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>2.035592</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Engineering

Say we want to predict pricing, using an ML model. How would you build your features?

Based on the number of null values, what would you do with the `last_review` and `reviews_per_month` column?


```python
X = airbnb.copy()
```


```python
X_cat = X.select_dtypes(include='object')
X_cat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>room_type</th>
      <th>last_review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>John</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>Private room</td>
      <td>2018-10-19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Skylit Midtown Castle</td>
      <td>Jennifer</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>Entire home/apt</td>
      <td>2019-05-21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>Elisabeth</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>Private room</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>LisaRoxanne</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>Entire home/apt</td>
      <td>2019-07-05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>Laura</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>Entire home/apt</td>
      <td>2018-11-19</td>
    </tr>
  </tbody>
</table>
</div>



based on the number of unique columns, we may want to remove `name`, `host_name`, and `last_review`


```python
X_cat.nunique()
```




    name                   47905
    host_name              11452
    neighbourhood_group        5
    neighbourhood            221
    room_type                  3
    last_review             1764
    dtype: int64




```python
X_cat = X_cat.drop(['name', 'host_name', 'last_review'], axis=1)
```


```python
enc = OneHotEncoder()
X_enc = enc.fit_transform(X_cat).toarray()
```


```python
X_num = X.select_dtypes(exclude='object')
```

both `id` and `host_id` will be highly cardinal without telling us much about the behavior of unseen data. We should remove them


```python
X_num = X_num.drop(['id', 'host_id'], axis=1)
```


```python
X_enc_df = pd.DataFrame(X_enc, columns=enc.get_feature_names_out())
```


```python
X_feat = pd.concat((X_enc_df, X_num), axis=1)
```


```python
X_feat = X_feat.drop(['reviews_per_month'], axis=1)
```


```python
X_feat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood_group_Bronx</th>
      <th>neighbourhood_group_Brooklyn</th>
      <th>neighbourhood_group_Manhattan</th>
      <th>neighbourhood_group_Queens</th>
      <th>neighbourhood_group_Staten Island</th>
      <th>neighbourhood_Allerton</th>
      <th>neighbourhood_Arden Heights</th>
      <th>neighbourhood_Arrochar</th>
      <th>neighbourhood_Arverne</th>
      <th>neighbourhood_Astoria</th>
      <th>...</th>
      <th>neighbourhood_Woodrow</th>
      <th>neighbourhood_Woodside</th>
      <th>room_type_Entire home/apt</th>
      <th>room_type_Private room</th>
      <th>room_type_Shared room</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>minimum_nights</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>1</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>1</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>3</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>1</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>48890</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>40.67853</td>
      <td>-73.94995</td>
      <td>2</td>
      <td>2</td>
      <td>9</td>
    </tr>
    <tr>
      <th>48891</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>40.70184</td>
      <td>-73.93317</td>
      <td>4</td>
      <td>2</td>
      <td>36</td>
    </tr>
    <tr>
      <th>48892</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.81475</td>
      <td>-73.94867</td>
      <td>10</td>
      <td>1</td>
      <td>27</td>
    </tr>
    <tr>
      <th>48893</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>40.75751</td>
      <td>-73.99112</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>48894</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>40.76404</td>
      <td>-73.98933</td>
      <td>7</td>
      <td>1</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
<p>48895 rows × 234 columns</p>
</div>



## Feature Transformation

What features do you think will cause the most problems if untransformed? 

Scale and Center all but the target variable, price


```python
scaler = StandardScaler()
y = X_feat.pop('price')
X_std = scaler.fit_transform(X_feat)
```


```python
X_std.shape
```




    (48895, 234)




```python
y.shape
```




    (48895,)



## Model Baseline


```python
X_train, X_test, y_train, y_test = train_test_split(X_std, y, train_size=0.8, random_state=42)
```


```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


```python
r2_score(y_train, model.predict(X_train))
```




    0.10989217084945124




```python
r2_score(y_test, y_pred)
```




    -3.0151033955403084e+24




```python
model = RandomForestRegressor()
model.fit(X_train, y_train)
r2_score(y_train, model.predict(X_train))
```




    0.8671952200657349




```python
r2_score(y_test, model.predict(X_test))
```




    0.12233872053971129



both of these results from the `LinearRegression` and `RandomForest` models indicate overfitting

## Back to Feature Engineering


```python
X_to_pca = X_feat.iloc[:,:-5] # grab one hot encoded features
```


```python
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_to_pca)
```


```python
X_std = scaler.fit_transform(X_feat.iloc[:,-5:])
X_std = np.hstack((X_pca, X_std))
```


```python
pca.explained_variance_
```




    array([0.53964396, 0.41513943, 0.16438329, 0.07812296, 0.06035576,
           0.05073931, 0.04038094, 0.04010025, 0.03907097, 0.03722079])




```python
X_std.shape
```




    (48895, 15)



## Model


```python
X_train, X_test, y_train, y_test = train_test_split(X_std, y, train_size=0.8, random_state=42)
```


```python
model = LinearRegression()
model.fit(X_train, y_train)
```




    LinearRegression()




```python
r2_score(y_train, model.predict(X_train))
```




    0.08879713072296502



linear model still is not performing well


```python
model = RandomForestRegressor()
model.fit(X_train, y_train)
r2_score(y_train, model.predict(X_train))
```




    0.8748702090177007




```python
r2_score(y_test, model.predict(X_test))
```




    0.14440978317556497




```python
plt.plot(y_test, model.predict(X_test), ls='', marker='.')
```




    [<matplotlib.lines.Line2D at 0x7f20938e3eb0>]




    
![png](SOLN_X1_Thinking_Data_files/SOLN_X1_Thinking_Data_69_1.png)
    



```python

```
