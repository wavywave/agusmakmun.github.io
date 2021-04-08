---
layout: post
title:  "Kaggle - Steel Plates Faults"
description: "캐글 데이터 분석"
author: SeungRok OH
categories: [Kaggle]
---


# Steel Plates Faults (철판 제조 공정 데이터)

- 데이터 셋 : https://www.kaggle.com/mahsateimourikia/faults-nna

- FastCampus의 강의해설을 듣고 진행하였다. (https://www.fastcampus.co.kr/data_online_dl300)

- 철판의 여러 특성을 통해 불량을 예측하고자 하는 데이터 셋
- 제조 공정 데이터로, 불량품 예측하여 원인을 제거하거나 재고를 예측하여 수요에 맞는 생산을 진행하는 방식의 데이터가 주를 이룬다.
- 데이터 모이는 과정이 자동화되어 결측치가 적거나 퀄리티가 좋은 경향을 보인다.
- 머신러닝 대부분의 모델 적용 연습

## 라이브러리 설정 및 데이터 불러오기


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./Faults.NNA', delimiter='\t', header=None)

pd.set_option('display.max_columns', None)
```


```python
df.head()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42</td>
      <td>50</td>
      <td>270900</td>
      <td>270944</td>
      <td>267</td>
      <td>17</td>
      <td>44</td>
      <td>24220</td>
      <td>76</td>
      <td>108</td>
      <td>1687</td>
      <td>1</td>
      <td>0</td>
      <td>80</td>
      <td>0.0498</td>
      <td>0.2415</td>
      <td>0.1818</td>
      <td>0.0047</td>
      <td>0.4706</td>
      <td>1.0000</td>
      <td>1.0</td>
      <td>2.4265</td>
      <td>0.9031</td>
      <td>1.6435</td>
      <td>0.8182</td>
      <td>-0.2913</td>
      <td>0.5822</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>645</td>
      <td>651</td>
      <td>2538079</td>
      <td>2538108</td>
      <td>108</td>
      <td>10</td>
      <td>30</td>
      <td>11397</td>
      <td>84</td>
      <td>123</td>
      <td>1687</td>
      <td>1</td>
      <td>0</td>
      <td>80</td>
      <td>0.7647</td>
      <td>0.3793</td>
      <td>0.2069</td>
      <td>0.0036</td>
      <td>0.6000</td>
      <td>0.9667</td>
      <td>1.0</td>
      <td>2.0334</td>
      <td>0.7782</td>
      <td>1.4624</td>
      <td>0.7931</td>
      <td>-0.1756</td>
      <td>0.2984</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>829</td>
      <td>835</td>
      <td>1553913</td>
      <td>1553931</td>
      <td>71</td>
      <td>8</td>
      <td>19</td>
      <td>7972</td>
      <td>99</td>
      <td>125</td>
      <td>1623</td>
      <td>1</td>
      <td>0</td>
      <td>100</td>
      <td>0.9710</td>
      <td>0.3426</td>
      <td>0.3333</td>
      <td>0.0037</td>
      <td>0.7500</td>
      <td>0.9474</td>
      <td>1.0</td>
      <td>1.8513</td>
      <td>0.7782</td>
      <td>1.2553</td>
      <td>0.6667</td>
      <td>-0.1228</td>
      <td>0.2150</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853</td>
      <td>860</td>
      <td>369370</td>
      <td>369415</td>
      <td>176</td>
      <td>13</td>
      <td>45</td>
      <td>18996</td>
      <td>99</td>
      <td>126</td>
      <td>1353</td>
      <td>0</td>
      <td>1</td>
      <td>290</td>
      <td>0.7287</td>
      <td>0.4413</td>
      <td>0.1556</td>
      <td>0.0052</td>
      <td>0.5385</td>
      <td>1.0000</td>
      <td>1.0</td>
      <td>2.2455</td>
      <td>0.8451</td>
      <td>1.6532</td>
      <td>0.8444</td>
      <td>-0.1568</td>
      <td>0.5212</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1289</td>
      <td>1306</td>
      <td>498078</td>
      <td>498335</td>
      <td>2409</td>
      <td>60</td>
      <td>260</td>
      <td>246930</td>
      <td>37</td>
      <td>126</td>
      <td>1353</td>
      <td>0</td>
      <td>1</td>
      <td>185</td>
      <td>0.0695</td>
      <td>0.4486</td>
      <td>0.0662</td>
      <td>0.0126</td>
      <td>0.2833</td>
      <td>0.9885</td>
      <td>1.0</td>
      <td>3.3818</td>
      <td>1.2305</td>
      <td>2.4099</td>
      <td>0.9338</td>
      <td>-0.1992</td>
      <td>1.0000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>



- 컬럼이 숫자로 되어 있는데 정해져 있는 이름값으로 바꿔준다.


```python
a_name = pd.read_csv('Faults27x7_var', delimiter=' ', header=None)
df.columns = a_name[0]
```


```python
df.tail()
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
      <th>X_Minimum</th>
      <th>X_Maximum</th>
      <th>Y_Minimum</th>
      <th>Y_Maximum</th>
      <th>Pixels_Areas</th>
      <th>X_Perimeter</th>
      <th>Y_Perimeter</th>
      <th>Sum_of_Luminosity</th>
      <th>Minimum_of_Luminosity</th>
      <th>Maximum_of_Luminosity</th>
      <th>Length_of_Conveyer</th>
      <th>TypeOfSteel_A300</th>
      <th>TypeOfSteel_A400</th>
      <th>Steel_Plate_Thickness</th>
      <th>Edges_Index</th>
      <th>Empty_Index</th>
      <th>Square_Index</th>
      <th>Outside_X_Index</th>
      <th>Edges_X_Index</th>
      <th>Edges_Y_Index</th>
      <th>Outside_Global_Index</th>
      <th>LogOfAreas</th>
      <th>Log_X_Index</th>
      <th>Log_Y_Index</th>
      <th>Orientation_Index</th>
      <th>Luminosity_Index</th>
      <th>SigmoidOfAreas</th>
      <th>Pastry</th>
      <th>Z_Scratch</th>
      <th>K_Scatch</th>
      <th>Stains</th>
      <th>Dirtiness</th>
      <th>Bumps</th>
      <th>Other_Faults</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1936</th>
      <td>249</td>
      <td>277</td>
      <td>325780</td>
      <td>325796</td>
      <td>273</td>
      <td>54</td>
      <td>22</td>
      <td>35033</td>
      <td>119</td>
      <td>141</td>
      <td>1360</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.3662</td>
      <td>0.3906</td>
      <td>0.5714</td>
      <td>0.0206</td>
      <td>0.5185</td>
      <td>0.7273</td>
      <td>0.0</td>
      <td>2.4362</td>
      <td>1.4472</td>
      <td>1.2041</td>
      <td>-0.4286</td>
      <td>0.0026</td>
      <td>0.7254</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>144</td>
      <td>175</td>
      <td>340581</td>
      <td>340598</td>
      <td>287</td>
      <td>44</td>
      <td>24</td>
      <td>34599</td>
      <td>112</td>
      <td>133</td>
      <td>1360</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.2118</td>
      <td>0.4554</td>
      <td>0.5484</td>
      <td>0.0228</td>
      <td>0.7046</td>
      <td>0.7083</td>
      <td>0.0</td>
      <td>2.4579</td>
      <td>1.4914</td>
      <td>1.2305</td>
      <td>-0.4516</td>
      <td>-0.0582</td>
      <td>0.8173</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>145</td>
      <td>174</td>
      <td>386779</td>
      <td>386794</td>
      <td>292</td>
      <td>40</td>
      <td>22</td>
      <td>37572</td>
      <td>120</td>
      <td>140</td>
      <td>1360</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.2132</td>
      <td>0.3287</td>
      <td>0.5172</td>
      <td>0.0213</td>
      <td>0.7250</td>
      <td>0.6818</td>
      <td>0.0</td>
      <td>2.4654</td>
      <td>1.4624</td>
      <td>1.1761</td>
      <td>-0.4828</td>
      <td>0.0052</td>
      <td>0.7079</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>137</td>
      <td>170</td>
      <td>422497</td>
      <td>422528</td>
      <td>419</td>
      <td>97</td>
      <td>47</td>
      <td>52715</td>
      <td>117</td>
      <td>140</td>
      <td>1360</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.2015</td>
      <td>0.5904</td>
      <td>0.9394</td>
      <td>0.0243</td>
      <td>0.3402</td>
      <td>0.6596</td>
      <td>0.0</td>
      <td>2.6222</td>
      <td>1.5185</td>
      <td>1.4914</td>
      <td>-0.0606</td>
      <td>-0.0171</td>
      <td>0.9919</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>1261</td>
      <td>1281</td>
      <td>87951</td>
      <td>87967</td>
      <td>103</td>
      <td>26</td>
      <td>22</td>
      <td>11682</td>
      <td>101</td>
      <td>133</td>
      <td>1360</td>
      <td>1</td>
      <td>0</td>
      <td>80</td>
      <td>0.1162</td>
      <td>0.6781</td>
      <td>0.8000</td>
      <td>0.0147</td>
      <td>0.7692</td>
      <td>0.7273</td>
      <td>0.0</td>
      <td>2.0128</td>
      <td>1.3010</td>
      <td>1.2041</td>
      <td>-0.2000</td>
      <td>-0.1139</td>
      <td>0.5296</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>




```python
print(df.shape)
```

    (1941, 34)



```python
import os
n_cpu = os.cpu_count()
print(n_cpu)
n_thread = n_cpu*2
print(n_thread)
```

    4
    8


- 계산량이 많을 수 있어 cpu를 나누어 계산하는 방법이 있다.
- 전부 숫자형 데이터이다.

종속변수 7개 (철판에 어떠한 불량이 생겼는지)

- Pastry
- Z_Scratch
- K_Scatch
- Stains
- Dirtiness
- Bumps
- Other_Faults

설명변수 27개 (철판의 길이, 반짝이는 정도 - 두께 타입 등 다양한 변수)

- X_Minimum
- X_Maximum
- Y_Minimum
- Y_Maximum
- Pixels_Areas
- X_Perimeter
- Y_Perimeter
- Sum_of_Luminosity
- Minimum_of_Luminosity
- Maximum_of_Luminosity
- Length_of_Conveyer
- TypeOfSteel_A300
- TypeOfSteel_A400
- Steel_Plate_Thickness
- Edges_Index
- Empty_Index
- Square_Index
- Outside_X_Index
- Edges_X_Index
- Edges_Y_Index
- Outside_Global_Index
- LogOfAreas
- Log_X_Index
- Log_Y_Index
- Orientation_Index
- Luminosity_Index
- SigmoidOfAreas

## 데이터 전처리 및 EDA-기초통계분석

- 종속변수(철판이상을 나타내는)가 7가지가 있기에 7개를 이어 붙인 리스트를 만들어준다.


```python
conditions = [df['Pastry'].astype(bool),
             df['Z_Scratch'].astype(bool),
             df['K_Scatch'].astype(bool),
             df['Stains'].astype(bool),
             df['Dirtiness'].astype(bool),
             df['Bumps'].astype(bool),
             df['Other_Faults'].astype(bool)]

# 사실 뭐 astype 안쓰고 conditions = list(map(lambda i : i.astype(bool), 'astype안쓴 데이터프레임')) 이렇게도 가능.
```


```python
print(type(conditions))
print(type(conditions[0]))
print(len(conditions))
print(len(conditions[0]))
```

    <class 'list'>
    <class 'pandas.core.series.Series'>
    7
    1941



```python
choices = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

df['class'] = np.select(conditions, choices)

# np.select 를 이용해 True에 해당하는 값을 출력하는 column을 만들어줌.
```


```python
df.sample(10)
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
      <th>X_Minimum</th>
      <th>X_Maximum</th>
      <th>Y_Minimum</th>
      <th>Y_Maximum</th>
      <th>Pixels_Areas</th>
      <th>X_Perimeter</th>
      <th>Y_Perimeter</th>
      <th>Sum_of_Luminosity</th>
      <th>Minimum_of_Luminosity</th>
      <th>Maximum_of_Luminosity</th>
      <th>Length_of_Conveyer</th>
      <th>TypeOfSteel_A300</th>
      <th>TypeOfSteel_A400</th>
      <th>Steel_Plate_Thickness</th>
      <th>Edges_Index</th>
      <th>Empty_Index</th>
      <th>Square_Index</th>
      <th>Outside_X_Index</th>
      <th>Edges_X_Index</th>
      <th>Edges_Y_Index</th>
      <th>Outside_Global_Index</th>
      <th>LogOfAreas</th>
      <th>Log_X_Index</th>
      <th>Log_Y_Index</th>
      <th>Orientation_Index</th>
      <th>Luminosity_Index</th>
      <th>SigmoidOfAreas</th>
      <th>Pastry</th>
      <th>Z_Scratch</th>
      <th>K_Scatch</th>
      <th>Stains</th>
      <th>Dirtiness</th>
      <th>Bumps</th>
      <th>Other_Faults</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>504</th>
      <td>41</td>
      <td>190</td>
      <td>2254407</td>
      <td>2254463</td>
      <td>4995</td>
      <td>232</td>
      <td>124</td>
      <td>527906</td>
      <td>39</td>
      <td>127</td>
      <td>1402</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.0585</td>
      <td>0.4014</td>
      <td>0.3758</td>
      <td>0.1063</td>
      <td>0.6422</td>
      <td>0.4516</td>
      <td>0.0</td>
      <td>3.6985</td>
      <td>2.1732</td>
      <td>1.7482</td>
      <td>-0.6242</td>
      <td>-0.1743</td>
      <td>1.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>K_Scatch</td>
    </tr>
    <tr>
      <th>1363</th>
      <td>867</td>
      <td>1104</td>
      <td>949655</td>
      <td>949669</td>
      <td>1695</td>
      <td>247</td>
      <td>106</td>
      <td>197365</td>
      <td>103</td>
      <td>132</td>
      <td>1668</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.6763</td>
      <td>0.4891</td>
      <td>0.0591</td>
      <td>0.1421</td>
      <td>0.9595</td>
      <td>0.1321</td>
      <td>0.0</td>
      <td>3.2292</td>
      <td>2.3747</td>
      <td>1.1461</td>
      <td>-0.9409</td>
      <td>-0.0903</td>
      <td>1.0000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Other_Faults</td>
    </tr>
    <tr>
      <th>1604</th>
      <td>1287</td>
      <td>1302</td>
      <td>3204829</td>
      <td>3204845</td>
      <td>191</td>
      <td>17</td>
      <td>16</td>
      <td>20590</td>
      <td>67</td>
      <td>133</td>
      <td>1627</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.3995</td>
      <td>0.2042</td>
      <td>0.9375</td>
      <td>0.0092</td>
      <td>0.8824</td>
      <td>1.0000</td>
      <td>1.0</td>
      <td>2.2810</td>
      <td>1.1761</td>
      <td>1.2041</td>
      <td>0.0625</td>
      <td>-0.1578</td>
      <td>0.3977</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Other_Faults</td>
    </tr>
    <tr>
      <th>1566</th>
      <td>741</td>
      <td>756</td>
      <td>614880</td>
      <td>614919</td>
      <td>392</td>
      <td>19</td>
      <td>39</td>
      <td>40498</td>
      <td>77</td>
      <td>127</td>
      <td>1360</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0.8882</td>
      <td>0.3299</td>
      <td>0.3846</td>
      <td>0.0110</td>
      <td>0.7895</td>
      <td>1.0000</td>
      <td>1.0</td>
      <td>2.5933</td>
      <td>1.1761</td>
      <td>1.5911</td>
      <td>0.6154</td>
      <td>-0.1929</td>
      <td>0.8682</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Other_Faults</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>1058</td>
      <td>1080</td>
      <td>751892</td>
      <td>751911</td>
      <td>133</td>
      <td>35</td>
      <td>20</td>
      <td>17085</td>
      <td>112</td>
      <td>142</td>
      <td>1658</td>
      <td>1</td>
      <td>0</td>
      <td>143</td>
      <td>0.6972</td>
      <td>0.6818</td>
      <td>0.8636</td>
      <td>0.0133</td>
      <td>0.6286</td>
      <td>0.9500</td>
      <td>0.0</td>
      <td>2.1239</td>
      <td>1.3424</td>
      <td>1.2787</td>
      <td>-0.1364</td>
      <td>0.0036</td>
      <td>0.6839</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Other_Faults</td>
    </tr>
    <tr>
      <th>1252</th>
      <td>1094</td>
      <td>1124</td>
      <td>12806495</td>
      <td>12806520</td>
      <td>571</td>
      <td>37</td>
      <td>25</td>
      <td>58587</td>
      <td>63</td>
      <td>127</td>
      <td>1362</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.3495</td>
      <td>0.2387</td>
      <td>0.8333</td>
      <td>0.0220</td>
      <td>0.8108</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>2.7566</td>
      <td>1.4771</td>
      <td>1.3979</td>
      <td>-0.1667</td>
      <td>-0.1984</td>
      <td>0.9519</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Bumps</td>
    </tr>
    <tr>
      <th>273</th>
      <td>322</td>
      <td>357</td>
      <td>1747625</td>
      <td>1747660</td>
      <td>314</td>
      <td>84</td>
      <td>50</td>
      <td>38597</td>
      <td>110</td>
      <td>140</td>
      <td>1356</td>
      <td>1</td>
      <td>0</td>
      <td>70</td>
      <td>0.4749</td>
      <td>0.7437</td>
      <td>1.0000</td>
      <td>0.0258</td>
      <td>0.4167</td>
      <td>0.7000</td>
      <td>0.5</td>
      <td>2.4969</td>
      <td>1.5441</td>
      <td>1.5441</td>
      <td>0.0000</td>
      <td>-0.0397</td>
      <td>0.9979</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Z_Scratch</td>
    </tr>
    <tr>
      <th>373</th>
      <td>919</td>
      <td>930</td>
      <td>561832</td>
      <td>561842</td>
      <td>88</td>
      <td>11</td>
      <td>10</td>
      <td>14538</td>
      <td>134</td>
      <td>183</td>
      <td>1387</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.6590</td>
      <td>0.2000</td>
      <td>0.9091</td>
      <td>0.0079</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0</td>
      <td>1.9445</td>
      <td>1.0414</td>
      <td>1.0000</td>
      <td>-0.0909</td>
      <td>0.2907</td>
      <td>0.2173</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>K_Scatch</td>
    </tr>
    <tr>
      <th>625</th>
      <td>41</td>
      <td>214</td>
      <td>1775604</td>
      <td>1775675</td>
      <td>6686</td>
      <td>273</td>
      <td>132</td>
      <td>687345</td>
      <td>36</td>
      <td>124</td>
      <td>1356</td>
      <td>0</td>
      <td>1</td>
      <td>40</td>
      <td>0.0605</td>
      <td>0.4557</td>
      <td>0.4104</td>
      <td>0.1276</td>
      <td>0.6337</td>
      <td>0.5379</td>
      <td>0.0</td>
      <td>3.8252</td>
      <td>2.2380</td>
      <td>1.8513</td>
      <td>-0.5896</td>
      <td>-0.1969</td>
      <td>1.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>K_Scatch</td>
    </tr>
    <tr>
      <th>1671</th>
      <td>705</td>
      <td>727</td>
      <td>5071808</td>
      <td>5071831</td>
      <td>282</td>
      <td>38</td>
      <td>27</td>
      <td>30746</td>
      <td>77</td>
      <td>126</td>
      <td>1356</td>
      <td>1</td>
      <td>0</td>
      <td>200</td>
      <td>0.9277</td>
      <td>0.4427</td>
      <td>0.9565</td>
      <td>0.0162</td>
      <td>0.5789</td>
      <td>0.8518</td>
      <td>1.0</td>
      <td>2.4502</td>
      <td>1.3424</td>
      <td>1.3617</td>
      <td>0.0435</td>
      <td>-0.1482</td>
      <td>0.7955</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Other_Faults</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.isnull().sum()
```




    0
    X_Minimum                0
    X_Maximum                0
    Y_Minimum                0
    Y_Maximum                0
    Pixels_Areas             0
    X_Perimeter              0
    Y_Perimeter              0
    Sum_of_Luminosity        0
    Minimum_of_Luminosity    0
    Maximum_of_Luminosity    0
    Length_of_Conveyer       0
    TypeOfSteel_A300         0
    TypeOfSteel_A400         0
    Steel_Plate_Thickness    0
    Edges_Index              0
    Empty_Index              0
    Square_Index             0
    Outside_X_Index          0
    Edges_X_Index            0
    Edges_Y_Index            0
    Outside_Global_Index     0
    LogOfAreas               0
    Log_X_Index              0
    Log_Y_Index              0
    Orientation_Index        0
    Luminosity_Index         0
    SigmoidOfAreas           0
    Pastry                   0
    Z_Scratch                0
    K_Scatch                 0
    Stains                   0
    Dirtiness                0
    Bumps                    0
    Other_Faults             0
    class                    0
    dtype: int64




```python
df.describe()
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
      <th>X_Minimum</th>
      <th>X_Maximum</th>
      <th>Y_Minimum</th>
      <th>Y_Maximum</th>
      <th>Pixels_Areas</th>
      <th>X_Perimeter</th>
      <th>Y_Perimeter</th>
      <th>Sum_of_Luminosity</th>
      <th>Minimum_of_Luminosity</th>
      <th>Maximum_of_Luminosity</th>
      <th>Length_of_Conveyer</th>
      <th>TypeOfSteel_A300</th>
      <th>TypeOfSteel_A400</th>
      <th>Steel_Plate_Thickness</th>
      <th>Edges_Index</th>
      <th>Empty_Index</th>
      <th>Square_Index</th>
      <th>Outside_X_Index</th>
      <th>Edges_X_Index</th>
      <th>Edges_Y_Index</th>
      <th>Outside_Global_Index</th>
      <th>LogOfAreas</th>
      <th>Log_X_Index</th>
      <th>Log_Y_Index</th>
      <th>Orientation_Index</th>
      <th>Luminosity_Index</th>
      <th>SigmoidOfAreas</th>
      <th>Pastry</th>
      <th>Z_Scratch</th>
      <th>K_Scatch</th>
      <th>Stains</th>
      <th>Dirtiness</th>
      <th>Bumps</th>
      <th>Other_Faults</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1.941000e+03</td>
      <td>1.941000e+03</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1.941000e+03</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>571.136012</td>
      <td>617.964451</td>
      <td>1.650685e+06</td>
      <td>1.650739e+06</td>
      <td>1893.878413</td>
      <td>111.855229</td>
      <td>82.965997</td>
      <td>2.063121e+05</td>
      <td>84.548686</td>
      <td>130.193715</td>
      <td>1459.160227</td>
      <td>0.400309</td>
      <td>0.599691</td>
      <td>78.737764</td>
      <td>0.331715</td>
      <td>0.414203</td>
      <td>0.570767</td>
      <td>0.033361</td>
      <td>0.610529</td>
      <td>0.813472</td>
      <td>0.575734</td>
      <td>2.492388</td>
      <td>1.335686</td>
      <td>1.403271</td>
      <td>0.083288</td>
      <td>-0.131305</td>
      <td>0.585420</td>
      <td>0.081401</td>
      <td>0.097888</td>
      <td>0.201443</td>
      <td>0.037094</td>
      <td>0.028336</td>
      <td>0.207110</td>
      <td>0.346728</td>
    </tr>
    <tr>
      <th>std</th>
      <td>520.690671</td>
      <td>497.627410</td>
      <td>1.774578e+06</td>
      <td>1.774590e+06</td>
      <td>5168.459560</td>
      <td>301.209187</td>
      <td>426.482879</td>
      <td>5.122936e+05</td>
      <td>32.134276</td>
      <td>18.690992</td>
      <td>144.577823</td>
      <td>0.490087</td>
      <td>0.490087</td>
      <td>55.086032</td>
      <td>0.299712</td>
      <td>0.137261</td>
      <td>0.271058</td>
      <td>0.058961</td>
      <td>0.243277</td>
      <td>0.234274</td>
      <td>0.482352</td>
      <td>0.788930</td>
      <td>0.481612</td>
      <td>0.454345</td>
      <td>0.500868</td>
      <td>0.148767</td>
      <td>0.339452</td>
      <td>0.273521</td>
      <td>0.297239</td>
      <td>0.401181</td>
      <td>0.189042</td>
      <td>0.165973</td>
      <td>0.405339</td>
      <td>0.476051</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>6.712000e+03</td>
      <td>6.724000e+03</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.500000e+02</td>
      <td>0.000000</td>
      <td>37.000000</td>
      <td>1227.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.008300</td>
      <td>0.001500</td>
      <td>0.014400</td>
      <td>0.048400</td>
      <td>0.000000</td>
      <td>0.301000</td>
      <td>0.301000</td>
      <td>0.000000</td>
      <td>-0.991000</td>
      <td>-0.998900</td>
      <td>0.119000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>51.000000</td>
      <td>192.000000</td>
      <td>4.712530e+05</td>
      <td>4.712810e+05</td>
      <td>84.000000</td>
      <td>15.000000</td>
      <td>13.000000</td>
      <td>9.522000e+03</td>
      <td>63.000000</td>
      <td>124.000000</td>
      <td>1358.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>0.060400</td>
      <td>0.315800</td>
      <td>0.361300</td>
      <td>0.006600</td>
      <td>0.411800</td>
      <td>0.596800</td>
      <td>0.000000</td>
      <td>1.924300</td>
      <td>1.000000</td>
      <td>1.079200</td>
      <td>-0.333300</td>
      <td>-0.195000</td>
      <td>0.248200</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>435.000000</td>
      <td>467.000000</td>
      <td>1.204128e+06</td>
      <td>1.204136e+06</td>
      <td>174.000000</td>
      <td>26.000000</td>
      <td>25.000000</td>
      <td>1.920200e+04</td>
      <td>90.000000</td>
      <td>127.000000</td>
      <td>1364.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>70.000000</td>
      <td>0.227300</td>
      <td>0.412100</td>
      <td>0.555600</td>
      <td>0.010100</td>
      <td>0.636400</td>
      <td>0.947400</td>
      <td>1.000000</td>
      <td>2.240600</td>
      <td>1.176100</td>
      <td>1.322200</td>
      <td>0.095200</td>
      <td>-0.133000</td>
      <td>0.506300</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1053.000000</td>
      <td>1072.000000</td>
      <td>2.183073e+06</td>
      <td>2.183084e+06</td>
      <td>822.000000</td>
      <td>84.000000</td>
      <td>83.000000</td>
      <td>8.301100e+04</td>
      <td>106.000000</td>
      <td>140.000000</td>
      <td>1650.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>0.573800</td>
      <td>0.501600</td>
      <td>0.818200</td>
      <td>0.023500</td>
      <td>0.800000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.914900</td>
      <td>1.518500</td>
      <td>1.732400</td>
      <td>0.511600</td>
      <td>-0.066600</td>
      <td>0.999800</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1705.000000</td>
      <td>1713.000000</td>
      <td>1.298766e+07</td>
      <td>1.298769e+07</td>
      <td>152655.000000</td>
      <td>10449.000000</td>
      <td>18152.000000</td>
      <td>1.159141e+07</td>
      <td>203.000000</td>
      <td>253.000000</td>
      <td>1794.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>300.000000</td>
      <td>0.995200</td>
      <td>0.943900</td>
      <td>1.000000</td>
      <td>0.875900</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.183700</td>
      <td>3.074100</td>
      <td>4.258700</td>
      <td>0.991700</td>
      <td>0.642100</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>

</div>




```python
df['class'].value_counts()
```




    Other_Faults    673
    Bumps           402
    K_Scatch        391
    Z_Scratch       190
    Pastry          158
    Stains           72
    Dirtiness        55
    Name: class, dtype: int64



#### 산점도 통해서 각 변수간의 관계 파악


```python
color_code = {'Pastry' : 'Red', 'Z_Scratch' : 'Blue', 'K_Scatch' : 'Green', 'Stains' : 'Black', 'Dirtiness' : 'Pink', 'Bumps' : 'Brown', 'Other_Faults' : 'Gold'}
color_list = [color_code.get(i) for i in df.loc[:,'class']]
```


```python
pd.plotting.scatter_matrix(df.loc[:, df.columns!='class'], c=color_list, figsize=[30,30], alpha=0.3, s= 50, diagonal='hist')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000002C49A32E348>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49AF4E688>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49AF80D08>,
            ...,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49B5D02C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49B608408>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49B63F508>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000002C49B6755C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49B6AE708>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49B6E6808>,
            ...,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49BD48608>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49BD84E88>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49BDB87C8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000002C49BDF08C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49BE279C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49BE60B08>,
            ...,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49C4C1848>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49C4F7948>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C49C530A48>],
           ...,
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000002C4AE5E9588>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4AE6216C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4AE658808>,
            ...,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4AFC88B88>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4AFCC0CC8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4AFCF6E48>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000002C4AFD2CF88>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4AFD6D0C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4AFDA5248>,
            ...,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4B0404748>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4B043D848>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4B04759C8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000002C4B04ACB08>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4B04E3C48>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4B051ADC8>,
            ...,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4B0B81288>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4B0BB73C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002C4B0BF1508>]],
          dtype=object)




![output_24_1](https://user-images.githubusercontent.com/77723966/114031847-bac6e980-98b6-11eb-847f-c3fa3065584b.png)



#### 위의 범주형 변수 시각화


```python
sns.set_style('white')

g = sns.catplot(data=df, x='class', kind='count', palette='YlGnBu', height=6)
g.ax.xaxis.set_label_text('Type of Defect')
g.ax.yaxis.set_label_text('Count')
g.ax.set_title('The number of Defects by Defect type')

for p in g.ax.patches: # ax의 patches(각각의 기둥들= p)
    g.ax.annotate((p.get_height()), (p.get_x()+0.2, p.get_height()+10)) # 높이에 해당하는 값을 annotate할건데 좌표는 x와 y값
```


![output_26_0](https://user-images.githubusercontent.com/77723966/114031830-b6023580-98b6-11eb-9e87-34d052ad5f98.png)



#### 상관계수를 통해 각 변수간의 관계 파악 + Heatmap


```python
df.columns
```




    Index(['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
           'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity',
           'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
           'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness',
           'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
           'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
           'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
           'SigmoidOfAreas', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
           'Dirtiness', 'Bumps', 'Other_Faults', 'class'],
          dtype='object', name=0)




```python
df_corTarget = df[['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
       'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity',
       'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
       'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness',
       'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
       'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
       'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
       'SigmoidOfAreas']]

corr = df_corTarget.corr()
corr
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
      <th>X_Minimum</th>
      <th>X_Maximum</th>
      <th>Y_Minimum</th>
      <th>Y_Maximum</th>
      <th>Pixels_Areas</th>
      <th>X_Perimeter</th>
      <th>Y_Perimeter</th>
      <th>Sum_of_Luminosity</th>
      <th>Minimum_of_Luminosity</th>
      <th>Maximum_of_Luminosity</th>
      <th>Length_of_Conveyer</th>
      <th>TypeOfSteel_A300</th>
      <th>TypeOfSteel_A400</th>
      <th>Steel_Plate_Thickness</th>
      <th>Edges_Index</th>
      <th>Empty_Index</th>
      <th>Square_Index</th>
      <th>Outside_X_Index</th>
      <th>Edges_X_Index</th>
      <th>Edges_Y_Index</th>
      <th>Outside_Global_Index</th>
      <th>LogOfAreas</th>
      <th>Log_X_Index</th>
      <th>Log_Y_Index</th>
      <th>Orientation_Index</th>
      <th>Luminosity_Index</th>
      <th>SigmoidOfAreas</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X_Minimum</th>
      <td>1.000000</td>
      <td>0.988314</td>
      <td>0.041821</td>
      <td>0.041807</td>
      <td>-0.307322</td>
      <td>-0.258937</td>
      <td>-0.118757</td>
      <td>-0.339045</td>
      <td>0.237637</td>
      <td>-0.075554</td>
      <td>0.316662</td>
      <td>0.144319</td>
      <td>-0.144319</td>
      <td>0.136625</td>
      <td>0.278075</td>
      <td>-0.198461</td>
      <td>0.063658</td>
      <td>-0.361160</td>
      <td>0.154778</td>
      <td>0.367907</td>
      <td>0.147282</td>
      <td>-0.428553</td>
      <td>-0.437944</td>
      <td>-0.326851</td>
      <td>0.178585</td>
      <td>-0.031578</td>
      <td>-0.355251</td>
    </tr>
    <tr>
      <th>X_Maximum</th>
      <td>0.988314</td>
      <td>1.000000</td>
      <td>0.052147</td>
      <td>0.052135</td>
      <td>-0.225399</td>
      <td>-0.186326</td>
      <td>-0.090138</td>
      <td>-0.247052</td>
      <td>0.168649</td>
      <td>-0.062392</td>
      <td>0.299390</td>
      <td>0.112009</td>
      <td>-0.112009</td>
      <td>0.106119</td>
      <td>0.242846</td>
      <td>-0.152680</td>
      <td>0.048575</td>
      <td>-0.214930</td>
      <td>0.149259</td>
      <td>0.271915</td>
      <td>0.099253</td>
      <td>-0.332169</td>
      <td>-0.324012</td>
      <td>-0.265990</td>
      <td>0.115019</td>
      <td>-0.038996</td>
      <td>-0.286736</td>
    </tr>
    <tr>
      <th>Y_Minimum</th>
      <td>0.041821</td>
      <td>0.052147</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.017670</td>
      <td>0.023843</td>
      <td>0.024150</td>
      <td>0.007362</td>
      <td>-0.065703</td>
      <td>-0.067785</td>
      <td>-0.049211</td>
      <td>0.075164</td>
      <td>-0.075164</td>
      <td>-0.207640</td>
      <td>0.021314</td>
      <td>-0.043117</td>
      <td>-0.006135</td>
      <td>0.054165</td>
      <td>0.066085</td>
      <td>-0.036543</td>
      <td>-0.062911</td>
      <td>0.044952</td>
      <td>0.070406</td>
      <td>-0.008442</td>
      <td>-0.086497</td>
      <td>-0.090654</td>
      <td>0.025257</td>
    </tr>
    <tr>
      <th>Y_Maximum</th>
      <td>0.041807</td>
      <td>0.052135</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.017840</td>
      <td>0.024038</td>
      <td>0.024380</td>
      <td>0.007499</td>
      <td>-0.065733</td>
      <td>-0.067776</td>
      <td>-0.049219</td>
      <td>0.075151</td>
      <td>-0.075151</td>
      <td>-0.207644</td>
      <td>0.021300</td>
      <td>-0.043085</td>
      <td>-0.006152</td>
      <td>0.054185</td>
      <td>0.066051</td>
      <td>-0.036549</td>
      <td>-0.062901</td>
      <td>0.044994</td>
      <td>0.070432</td>
      <td>-0.008382</td>
      <td>-0.086480</td>
      <td>-0.090666</td>
      <td>0.025284</td>
    </tr>
    <tr>
      <th>Pixels_Areas</th>
      <td>-0.307322</td>
      <td>-0.225399</td>
      <td>0.017670</td>
      <td>0.017840</td>
      <td>1.000000</td>
      <td>0.966644</td>
      <td>0.827199</td>
      <td>0.978952</td>
      <td>-0.497204</td>
      <td>0.110063</td>
      <td>-0.155853</td>
      <td>-0.235591</td>
      <td>0.235591</td>
      <td>-0.183735</td>
      <td>-0.275289</td>
      <td>0.272808</td>
      <td>0.017865</td>
      <td>0.588606</td>
      <td>-0.294673</td>
      <td>-0.463571</td>
      <td>-0.109655</td>
      <td>0.650234</td>
      <td>0.603072</td>
      <td>0.578342</td>
      <td>-0.137604</td>
      <td>-0.043449</td>
      <td>0.422947</td>
    </tr>
    <tr>
      <th>X_Perimeter</th>
      <td>-0.258937</td>
      <td>-0.186326</td>
      <td>0.023843</td>
      <td>0.024038</td>
      <td>0.966644</td>
      <td>1.000000</td>
      <td>0.912436</td>
      <td>0.912956</td>
      <td>-0.400427</td>
      <td>0.111363</td>
      <td>-0.134240</td>
      <td>-0.189250</td>
      <td>0.189250</td>
      <td>-0.147712</td>
      <td>-0.227590</td>
      <td>0.306348</td>
      <td>0.004507</td>
      <td>0.517098</td>
      <td>-0.293039</td>
      <td>-0.412100</td>
      <td>-0.079106</td>
      <td>0.563036</td>
      <td>0.524716</td>
      <td>0.523472</td>
      <td>-0.101731</td>
      <td>-0.032617</td>
      <td>0.380605</td>
    </tr>
    <tr>
      <th>Y_Perimeter</th>
      <td>-0.118757</td>
      <td>-0.090138</td>
      <td>0.024150</td>
      <td>0.024380</td>
      <td>0.827199</td>
      <td>0.912436</td>
      <td>1.000000</td>
      <td>0.704876</td>
      <td>-0.213758</td>
      <td>0.061809</td>
      <td>-0.063825</td>
      <td>-0.095154</td>
      <td>0.095154</td>
      <td>-0.058889</td>
      <td>-0.111240</td>
      <td>0.188825</td>
      <td>-0.047511</td>
      <td>0.209160</td>
      <td>-0.195162</td>
      <td>-0.136723</td>
      <td>0.013438</td>
      <td>0.294040</td>
      <td>0.228485</td>
      <td>0.344378</td>
      <td>0.031381</td>
      <td>-0.047778</td>
      <td>0.191772</td>
    </tr>
    <tr>
      <th>Sum_of_Luminosity</th>
      <td>-0.339045</td>
      <td>-0.247052</td>
      <td>0.007362</td>
      <td>0.007499</td>
      <td>0.978952</td>
      <td>0.912956</td>
      <td>0.704876</td>
      <td>1.000000</td>
      <td>-0.540566</td>
      <td>0.136515</td>
      <td>-0.169331</td>
      <td>-0.263632</td>
      <td>0.263632</td>
      <td>-0.204812</td>
      <td>-0.301452</td>
      <td>0.293691</td>
      <td>0.049607</td>
      <td>0.658339</td>
      <td>-0.327728</td>
      <td>-0.529745</td>
      <td>-0.121090</td>
      <td>0.712128</td>
      <td>0.667736</td>
      <td>0.618795</td>
      <td>-0.158483</td>
      <td>-0.014067</td>
      <td>0.464248</td>
    </tr>
    <tr>
      <th>Minimum_of_Luminosity</th>
      <td>0.237637</td>
      <td>0.168649</td>
      <td>-0.065703</td>
      <td>-0.065733</td>
      <td>-0.497204</td>
      <td>-0.400427</td>
      <td>-0.213758</td>
      <td>-0.540566</td>
      <td>1.000000</td>
      <td>0.429605</td>
      <td>-0.023579</td>
      <td>0.042048</td>
      <td>-0.042048</td>
      <td>0.103393</td>
      <td>0.358915</td>
      <td>-0.044111</td>
      <td>0.066748</td>
      <td>-0.487574</td>
      <td>0.252256</td>
      <td>0.316610</td>
      <td>0.035462</td>
      <td>-0.678762</td>
      <td>-0.567655</td>
      <td>-0.588208</td>
      <td>0.057123</td>
      <td>0.669534</td>
      <td>-0.514797</td>
    </tr>
    <tr>
      <th>Maximum_of_Luminosity</th>
      <td>-0.075554</td>
      <td>-0.062392</td>
      <td>-0.067785</td>
      <td>-0.067776</td>
      <td>0.110063</td>
      <td>0.111363</td>
      <td>0.061809</td>
      <td>0.136515</td>
      <td>0.429605</td>
      <td>1.000000</td>
      <td>-0.098009</td>
      <td>-0.216339</td>
      <td>0.216339</td>
      <td>-0.128397</td>
      <td>0.149675</td>
      <td>0.031425</td>
      <td>0.065517</td>
      <td>0.099300</td>
      <td>0.093522</td>
      <td>-0.167441</td>
      <td>-0.124039</td>
      <td>0.007672</td>
      <td>0.092823</td>
      <td>-0.069522</td>
      <td>-0.169747</td>
      <td>0.870160</td>
      <td>-0.039651</td>
    </tr>
    <tr>
      <th>Length_of_Conveyer</th>
      <td>0.316662</td>
      <td>0.299390</td>
      <td>-0.049211</td>
      <td>-0.049219</td>
      <td>-0.155853</td>
      <td>-0.134240</td>
      <td>-0.063825</td>
      <td>-0.169331</td>
      <td>-0.023579</td>
      <td>-0.098009</td>
      <td>1.000000</td>
      <td>0.378542</td>
      <td>-0.378542</td>
      <td>0.214769</td>
      <td>0.135152</td>
      <td>-0.230601</td>
      <td>0.073694</td>
      <td>-0.217417</td>
      <td>0.123585</td>
      <td>0.235732</td>
      <td>0.128663</td>
      <td>-0.193247</td>
      <td>-0.219973</td>
      <td>-0.157057</td>
      <td>0.120715</td>
      <td>-0.149769</td>
      <td>-0.197543</td>
    </tr>
    <tr>
      <th>TypeOfSteel_A300</th>
      <td>0.144319</td>
      <td>0.112009</td>
      <td>0.075164</td>
      <td>0.075151</td>
      <td>-0.235591</td>
      <td>-0.189250</td>
      <td>-0.095154</td>
      <td>-0.263632</td>
      <td>0.042048</td>
      <td>-0.216339</td>
      <td>0.378542</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.125649</td>
      <td>0.112140</td>
      <td>-0.091954</td>
      <td>0.164156</td>
      <td>-0.244765</td>
      <td>0.173836</td>
      <td>0.240634</td>
      <td>0.022142</td>
      <td>-0.329614</td>
      <td>-0.266955</td>
      <td>-0.311796</td>
      <td>0.010630</td>
      <td>-0.252818</td>
      <td>-0.308910</td>
    </tr>
    <tr>
      <th>TypeOfSteel_A400</th>
      <td>-0.144319</td>
      <td>-0.112009</td>
      <td>-0.075164</td>
      <td>-0.075151</td>
      <td>0.235591</td>
      <td>0.189250</td>
      <td>0.095154</td>
      <td>0.263632</td>
      <td>-0.042048</td>
      <td>0.216339</td>
      <td>-0.378542</td>
      <td>-1.000000</td>
      <td>1.000000</td>
      <td>-0.125649</td>
      <td>-0.112140</td>
      <td>0.091954</td>
      <td>-0.164156</td>
      <td>0.244765</td>
      <td>-0.173836</td>
      <td>-0.240634</td>
      <td>-0.022142</td>
      <td>0.329614</td>
      <td>0.266955</td>
      <td>0.311796</td>
      <td>-0.010630</td>
      <td>0.252818</td>
      <td>0.308910</td>
    </tr>
    <tr>
      <th>Steel_Plate_Thickness</th>
      <td>0.136625</td>
      <td>0.106119</td>
      <td>-0.207640</td>
      <td>-0.207644</td>
      <td>-0.183735</td>
      <td>-0.147712</td>
      <td>-0.058889</td>
      <td>-0.204812</td>
      <td>0.103393</td>
      <td>-0.128397</td>
      <td>0.214769</td>
      <td>0.125649</td>
      <td>-0.125649</td>
      <td>1.000000</td>
      <td>0.063449</td>
      <td>0.012526</td>
      <td>-0.124382</td>
      <td>-0.228352</td>
      <td>-0.077408</td>
      <td>0.251985</td>
      <td>0.221244</td>
      <td>-0.176639</td>
      <td>-0.252822</td>
      <td>-0.037287</td>
      <td>0.274097</td>
      <td>-0.116499</td>
      <td>-0.085159</td>
    </tr>
    <tr>
      <th>Edges_Index</th>
      <td>0.278075</td>
      <td>0.242846</td>
      <td>0.021314</td>
      <td>0.021300</td>
      <td>-0.275289</td>
      <td>-0.227590</td>
      <td>-0.111240</td>
      <td>-0.301452</td>
      <td>0.358915</td>
      <td>0.149675</td>
      <td>0.135152</td>
      <td>0.112140</td>
      <td>-0.112140</td>
      <td>0.063449</td>
      <td>1.000000</td>
      <td>-0.180739</td>
      <td>0.149498</td>
      <td>-0.296510</td>
      <td>0.250178</td>
      <td>0.285302</td>
      <td>0.008282</td>
      <td>-0.408619</td>
      <td>-0.355853</td>
      <td>-0.371989</td>
      <td>0.020548</td>
      <td>0.207516</td>
      <td>-0.330006</td>
    </tr>
    <tr>
      <th>Empty_Index</th>
      <td>-0.198461</td>
      <td>-0.152680</td>
      <td>-0.043117</td>
      <td>-0.043085</td>
      <td>0.272808</td>
      <td>0.306348</td>
      <td>0.188825</td>
      <td>0.293691</td>
      <td>-0.044111</td>
      <td>0.031425</td>
      <td>-0.230601</td>
      <td>-0.091954</td>
      <td>0.091954</td>
      <td>0.012526</td>
      <td>-0.180739</td>
      <td>1.000000</td>
      <td>-0.076439</td>
      <td>0.334996</td>
      <td>-0.389342</td>
      <td>-0.459800</td>
      <td>-0.165293</td>
      <td>0.356685</td>
      <td>0.448864</td>
      <td>0.397289</td>
      <td>-0.139420</td>
      <td>0.061608</td>
      <td>0.481738</td>
    </tr>
    <tr>
      <th>Square_Index</th>
      <td>0.063658</td>
      <td>0.048575</td>
      <td>-0.006135</td>
      <td>-0.006152</td>
      <td>0.017865</td>
      <td>0.004507</td>
      <td>-0.047511</td>
      <td>0.049607</td>
      <td>0.066748</td>
      <td>0.065517</td>
      <td>0.073694</td>
      <td>0.164156</td>
      <td>-0.164156</td>
      <td>-0.124382</td>
      <td>0.149498</td>
      <td>-0.076439</td>
      <td>1.000000</td>
      <td>-0.113627</td>
      <td>0.242779</td>
      <td>0.081488</td>
      <td>-0.069913</td>
      <td>-0.189340</td>
      <td>-0.082846</td>
      <td>-0.257661</td>
      <td>-0.162034</td>
      <td>0.111977</td>
      <td>-0.292251</td>
    </tr>
    <tr>
      <th>Outside_X_Index</th>
      <td>-0.361160</td>
      <td>-0.214930</td>
      <td>0.054165</td>
      <td>0.054185</td>
      <td>0.588606</td>
      <td>0.517098</td>
      <td>0.209160</td>
      <td>0.658339</td>
      <td>-0.487574</td>
      <td>0.099300</td>
      <td>-0.217417</td>
      <td>-0.244765</td>
      <td>0.244765</td>
      <td>-0.228352</td>
      <td>-0.296510</td>
      <td>0.334996</td>
      <td>-0.113627</td>
      <td>1.000000</td>
      <td>-0.076663</td>
      <td>-0.689867</td>
      <td>-0.337173</td>
      <td>0.710837</td>
      <td>0.820223</td>
      <td>0.464860</td>
      <td>-0.440358</td>
      <td>-0.035721</td>
      <td>0.518910</td>
    </tr>
    <tr>
      <th>Edges_X_Index</th>
      <td>0.154778</td>
      <td>0.149259</td>
      <td>0.066085</td>
      <td>0.066051</td>
      <td>-0.294673</td>
      <td>-0.293039</td>
      <td>-0.195162</td>
      <td>-0.327728</td>
      <td>0.252256</td>
      <td>0.093522</td>
      <td>0.123585</td>
      <td>0.173836</td>
      <td>-0.173836</td>
      <td>-0.077408</td>
      <td>0.250178</td>
      <td>-0.389342</td>
      <td>0.242779</td>
      <td>-0.076663</td>
      <td>1.000000</td>
      <td>0.108144</td>
      <td>-0.419383</td>
      <td>-0.496206</td>
      <td>-0.189262</td>
      <td>-0.748892</td>
      <td>-0.550302</td>
      <td>0.126460</td>
      <td>-0.558426</td>
    </tr>
    <tr>
      <th>Edges_Y_Index</th>
      <td>0.367907</td>
      <td>0.271915</td>
      <td>-0.036543</td>
      <td>-0.036549</td>
      <td>-0.463571</td>
      <td>-0.412100</td>
      <td>-0.136723</td>
      <td>-0.529745</td>
      <td>0.316610</td>
      <td>-0.167441</td>
      <td>0.235732</td>
      <td>0.240634</td>
      <td>-0.240634</td>
      <td>0.251985</td>
      <td>0.285302</td>
      <td>-0.459800</td>
      <td>0.081488</td>
      <td>-0.689867</td>
      <td>0.108144</td>
      <td>1.000000</td>
      <td>0.537565</td>
      <td>-0.642991</td>
      <td>-0.855414</td>
      <td>-0.321892</td>
      <td>0.658049</td>
      <td>-0.094368</td>
      <td>-0.545393</td>
    </tr>
    <tr>
      <th>Outside_Global_Index</th>
      <td>0.147282</td>
      <td>0.099253</td>
      <td>-0.062911</td>
      <td>-0.062901</td>
      <td>-0.109655</td>
      <td>-0.079106</td>
      <td>0.013438</td>
      <td>-0.121090</td>
      <td>0.035462</td>
      <td>-0.124039</td>
      <td>0.128663</td>
      <td>0.022142</td>
      <td>-0.022142</td>
      <td>0.221244</td>
      <td>0.008282</td>
      <td>-0.165293</td>
      <td>-0.069913</td>
      <td>-0.337173</td>
      <td>-0.419383</td>
      <td>0.537565</td>
      <td>1.000000</td>
      <td>-0.097762</td>
      <td>-0.428060</td>
      <td>0.241898</td>
      <td>0.862670</td>
      <td>-0.122321</td>
      <td>-0.053770</td>
    </tr>
    <tr>
      <th>LogOfAreas</th>
      <td>-0.428553</td>
      <td>-0.332169</td>
      <td>0.044952</td>
      <td>0.044994</td>
      <td>0.650234</td>
      <td>0.563036</td>
      <td>0.294040</td>
      <td>0.712128</td>
      <td>-0.678762</td>
      <td>0.007672</td>
      <td>-0.193247</td>
      <td>-0.329614</td>
      <td>0.329614</td>
      <td>-0.176639</td>
      <td>-0.408619</td>
      <td>0.356685</td>
      <td>-0.189340</td>
      <td>0.710837</td>
      <td>-0.496206</td>
      <td>-0.642991</td>
      <td>-0.097762</td>
      <td>1.000000</td>
      <td>0.888919</td>
      <td>0.882974</td>
      <td>-0.123898</td>
      <td>-0.175879</td>
      <td>0.877768</td>
    </tr>
    <tr>
      <th>Log_X_Index</th>
      <td>-0.437944</td>
      <td>-0.324012</td>
      <td>0.070406</td>
      <td>0.070432</td>
      <td>0.603072</td>
      <td>0.524716</td>
      <td>0.228485</td>
      <td>0.667736</td>
      <td>-0.567655</td>
      <td>0.092823</td>
      <td>-0.219973</td>
      <td>-0.266955</td>
      <td>0.266955</td>
      <td>-0.252822</td>
      <td>-0.355853</td>
      <td>0.448864</td>
      <td>-0.082846</td>
      <td>0.820223</td>
      <td>-0.189262</td>
      <td>-0.855414</td>
      <td>-0.428060</td>
      <td>0.888919</td>
      <td>1.000000</td>
      <td>0.598652</td>
      <td>-0.536629</td>
      <td>-0.064923</td>
      <td>0.757343</td>
    </tr>
    <tr>
      <th>Log_Y_Index</th>
      <td>-0.326851</td>
      <td>-0.265990</td>
      <td>-0.008442</td>
      <td>-0.008382</td>
      <td>0.578342</td>
      <td>0.523472</td>
      <td>0.344378</td>
      <td>0.618795</td>
      <td>-0.588208</td>
      <td>-0.069522</td>
      <td>-0.157057</td>
      <td>-0.311796</td>
      <td>0.311796</td>
      <td>-0.037287</td>
      <td>-0.371989</td>
      <td>0.397289</td>
      <td>-0.257661</td>
      <td>0.464860</td>
      <td>-0.748892</td>
      <td>-0.321892</td>
      <td>0.241898</td>
      <td>0.882974</td>
      <td>0.598652</td>
      <td>1.000000</td>
      <td>0.316792</td>
      <td>-0.219110</td>
      <td>0.838188</td>
    </tr>
    <tr>
      <th>Orientation_Index</th>
      <td>0.178585</td>
      <td>0.115019</td>
      <td>-0.086497</td>
      <td>-0.086480</td>
      <td>-0.137604</td>
      <td>-0.101731</td>
      <td>0.031381</td>
      <td>-0.158483</td>
      <td>0.057123</td>
      <td>-0.169747</td>
      <td>0.120715</td>
      <td>0.010630</td>
      <td>-0.010630</td>
      <td>0.274097</td>
      <td>0.020548</td>
      <td>-0.139420</td>
      <td>-0.162034</td>
      <td>-0.440358</td>
      <td>-0.550302</td>
      <td>0.658049</td>
      <td>0.862670</td>
      <td>-0.123898</td>
      <td>-0.536629</td>
      <td>0.316792</td>
      <td>1.000000</td>
      <td>-0.153464</td>
      <td>-0.023978</td>
    </tr>
    <tr>
      <th>Luminosity_Index</th>
      <td>-0.031578</td>
      <td>-0.038996</td>
      <td>-0.090654</td>
      <td>-0.090666</td>
      <td>-0.043449</td>
      <td>-0.032617</td>
      <td>-0.047778</td>
      <td>-0.014067</td>
      <td>0.669534</td>
      <td>0.870160</td>
      <td>-0.149769</td>
      <td>-0.252818</td>
      <td>0.252818</td>
      <td>-0.116499</td>
      <td>0.207516</td>
      <td>0.061608</td>
      <td>0.111977</td>
      <td>-0.035721</td>
      <td>0.126460</td>
      <td>-0.094368</td>
      <td>-0.122321</td>
      <td>-0.175879</td>
      <td>-0.064923</td>
      <td>-0.219110</td>
      <td>-0.153464</td>
      <td>1.000000</td>
      <td>-0.184840</td>
    </tr>
    <tr>
      <th>SigmoidOfAreas</th>
      <td>-0.355251</td>
      <td>-0.286736</td>
      <td>0.025257</td>
      <td>0.025284</td>
      <td>0.422947</td>
      <td>0.380605</td>
      <td>0.191772</td>
      <td>0.464248</td>
      <td>-0.514797</td>
      <td>-0.039651</td>
      <td>-0.197543</td>
      <td>-0.308910</td>
      <td>0.308910</td>
      <td>-0.085159</td>
      <td>-0.330006</td>
      <td>0.481738</td>
      <td>-0.292251</td>
      <td>0.518910</td>
      <td>-0.558426</td>
      <td>-0.545393</td>
      <td>-0.053770</td>
      <td>0.877768</td>
      <td>0.757343</td>
      <td>0.838188</td>
      <td>-0.023978</td>
      <td>-0.184840</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>

</div>




```python
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# 대각 기준 위 값 아래 값 똑같으니 하나만 표현


f, ax = plt.subplots(figsize = (11,9))
cmap = sns.diverging_palette(1,200, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, linewidths=2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2c4c246a308>




![output_30_1](https://user-images.githubusercontent.com/77723966/114031773-a71b8300-98b6-11eb-9fc4-98c4fe39e222.png)



### 학습 및 테스트 데이터 분리


```python
x = df[['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
       'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity',
       'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
       'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness',
       'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
       'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
       'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
       'SigmoidOfAreas']]

y = df['K_Scatch']
```


```python
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1, stratify=y)
# stratify : y비율이 train, test에서 비율이 맞게끔
```


```python
# 표준화 작업 sklearn.preprocessing import StandardScaler 와 같은 역할
x_train = x_train.apply(zscore)
x_test = x_test.apply(zscore)
```

## 로지스틱 분류 모형 (Grid Search 구축 , Lidge-Lasso penalty /Threshold)


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics
```


```python
lm = LogisticRegression(solver= 'liblinear')

# liblinear로 지정해야 이후 ridge,lasso 모델에도 알고리즘 적용가능
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
```

- 그리드 서치를 통해 최적의 파라미터 탐색


```python
parameters = {'penalty' : ['l1', 'l2'], 'C':[0.01, 0.1, 0.5, 0.9, 1, 5, 10], 'tol': [1e-4, 1e-2, 1, 1e2]}
```


```python
GSLR = GridSearchCV(lm, parameters, cv=10, n_jobs=n_thread, scoring='accuracy')
```


```python
GSLR.fit(x_train, y_train)
```




    GridSearchCV(cv=10, error_score=nan,
                 estimator=LogisticRegression(C=1.0, class_weight=None, dual=False,
                                              fit_intercept=True,
                                              intercept_scaling=1, l1_ratio=None,
                                              max_iter=100, multi_class='auto',
                                              n_jobs=None, penalty='l2',
                                              random_state=None, solver='liblinear',
                                              tol=0.0001, verbose=0,
                                              warm_start=False),
                 iid='deprecated', n_jobs=8,
                 param_grid={'C': [0.01, 0.1, 0.5, 0.9, 1, 5, 10],
                             'penalty': ['l1', 'l2'],
                             'tol': [0.0001, 0.01, 1, 100.0]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)




```python
print('final params' , GSLR.best_params_)
print('best score', GSLR.best_score_)
```

    final params {'C': 5, 'penalty': 'l1', 'tol': 0.01}
    best score 0.9729321753515301


- 최적 파라미터 경우 범위 극단에 있을 경우 범위 바깥에 있는 것도 시도해 보아야 한다.

### 모형 평가 및 모형 구축


```python
predicted=GSLR.predict(x_test)

cMatrix = confusion_matrix(y_test, predicted)
print(cMatrix)
print('\n Accuracy:', GSLR.score(x_test, y_test))
```

    [[305   6]
     [  5  73]]
    
     Accuracy: 0.9717223650385605



```python
print(metrics.classification_report(y_test, predicted))
```

                  precision    recall  f1-score   support
    
               0       0.98      0.98      0.98       311
               1       0.92      0.94      0.93        78
    
        accuracy                           0.97       389
       macro avg       0.95      0.96      0.96       389
    weighted avg       0.97      0.97      0.97       389


​    

- 파라미터 값 시각화 통해서 파라미터가 바뀔수록 정확도가 어떻게 변화하는가


```python
means = GSLR.cv_results_['mean_test_score']
stds = GSLR.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, GSLR.cv_results_['params']):
    print ('%0.3f (+/-%0.03f) for %r' % (mean, std *2, params))
print()
```

    0.945 (+/-0.037) for {'C': 0.01, 'penalty': 'l1', 'tol': 0.0001}
    0.946 (+/-0.036) for {'C': 0.01, 'penalty': 'l1', 'tol': 0.01}
    0.942 (+/-0.036) for {'C': 0.01, 'penalty': 'l1', 'tol': 1}
    0.798 (+/-0.005) for {'C': 0.01, 'penalty': 'l1', 'tol': 100.0}
    0.950 (+/-0.033) for {'C': 0.01, 'penalty': 'l2', 'tol': 0.0001}
    0.950 (+/-0.033) for {'C': 0.01, 'penalty': 'l2', 'tol': 0.01}
    0.954 (+/-0.035) for {'C': 0.01, 'penalty': 'l2', 'tol': 1}
    0.798 (+/-0.005) for {'C': 0.01, 'penalty': 'l2', 'tol': 100.0}
    0.964 (+/-0.028) for {'C': 0.1, 'penalty': 'l1', 'tol': 0.0001}
    0.963 (+/-0.028) for {'C': 0.1, 'penalty': 'l1', 'tol': 0.01}
    0.954 (+/-0.040) for {'C': 0.1, 'penalty': 'l1', 'tol': 1}
    0.798 (+/-0.005) for {'C': 0.1, 'penalty': 'l1', 'tol': 100.0}
    0.966 (+/-0.021) for {'C': 0.1, 'penalty': 'l2', 'tol': 0.0001}
    0.966 (+/-0.021) for {'C': 0.1, 'penalty': 'l2', 'tol': 0.01}
    0.957 (+/-0.032) for {'C': 0.1, 'penalty': 'l2', 'tol': 1}
    0.798 (+/-0.005) for {'C': 0.1, 'penalty': 'l2', 'tol': 100.0}
    0.969 (+/-0.028) for {'C': 0.5, 'penalty': 'l1', 'tol': 0.0001}
    0.970 (+/-0.025) for {'C': 0.5, 'penalty': 'l1', 'tol': 0.01}
    0.957 (+/-0.031) for {'C': 0.5, 'penalty': 'l1', 'tol': 1}
    0.798 (+/-0.005) for {'C': 0.5, 'penalty': 'l1', 'tol': 100.0}
    0.969 (+/-0.023) for {'C': 0.5, 'penalty': 'l2', 'tol': 0.0001}
    0.968 (+/-0.020) for {'C': 0.5, 'penalty': 'l2', 'tol': 0.01}
    0.958 (+/-0.030) for {'C': 0.5, 'penalty': 'l2', 'tol': 1}
    0.798 (+/-0.005) for {'C': 0.5, 'penalty': 'l2', 'tol': 100.0}
    0.969 (+/-0.030) for {'C': 0.9, 'penalty': 'l1', 'tol': 0.0001}
    0.968 (+/-0.030) for {'C': 0.9, 'penalty': 'l1', 'tol': 0.01}
    0.957 (+/-0.026) for {'C': 0.9, 'penalty': 'l1', 'tol': 1}
    0.798 (+/-0.005) for {'C': 0.9, 'penalty': 'l1', 'tol': 100.0}
    0.971 (+/-0.022) for {'C': 0.9, 'penalty': 'l2', 'tol': 0.0001}
    0.972 (+/-0.025) for {'C': 0.9, 'penalty': 'l2', 'tol': 0.01}
    0.958 (+/-0.030) for {'C': 0.9, 'penalty': 'l2', 'tol': 1}
    0.798 (+/-0.005) for {'C': 0.9, 'penalty': 'l2', 'tol': 100.0}
    0.970 (+/-0.030) for {'C': 1, 'penalty': 'l1', 'tol': 0.0001}
    0.970 (+/-0.026) for {'C': 1, 'penalty': 'l1', 'tol': 0.01}
    0.955 (+/-0.035) for {'C': 1, 'penalty': 'l1', 'tol': 1}
    0.798 (+/-0.005) for {'C': 1, 'penalty': 'l1', 'tol': 100.0}
    0.972 (+/-0.023) for {'C': 1, 'penalty': 'l2', 'tol': 0.0001}
    0.972 (+/-0.025) for {'C': 1, 'penalty': 'l2', 'tol': 0.01}
    0.958 (+/-0.030) for {'C': 1, 'penalty': 'l2', 'tol': 1}
    0.798 (+/-0.005) for {'C': 1, 'penalty': 'l2', 'tol': 100.0}
    0.970 (+/-0.027) for {'C': 5, 'penalty': 'l1', 'tol': 0.0001}
    0.973 (+/-0.025) for {'C': 5, 'penalty': 'l1', 'tol': 0.01}
    0.951 (+/-0.047) for {'C': 5, 'penalty': 'l1', 'tol': 1}
    0.798 (+/-0.005) for {'C': 5, 'penalty': 'l1', 'tol': 100.0}
    0.972 (+/-0.028) for {'C': 5, 'penalty': 'l2', 'tol': 0.0001}
    0.972 (+/-0.028) for {'C': 5, 'penalty': 'l2', 'tol': 0.01}
    0.959 (+/-0.030) for {'C': 5, 'penalty': 'l2', 'tol': 1}
    0.798 (+/-0.005) for {'C': 5, 'penalty': 'l2', 'tol': 100.0}
    0.972 (+/-0.025) for {'C': 10, 'penalty': 'l1', 'tol': 0.0001}
    0.971 (+/-0.028) for {'C': 10, 'penalty': 'l1', 'tol': 0.01}
    0.955 (+/-0.032) for {'C': 10, 'penalty': 'l1', 'tol': 1}
    0.798 (+/-0.005) for {'C': 10, 'penalty': 'l1', 'tol': 100.0}
    0.972 (+/-0.024) for {'C': 10, 'penalty': 'l2', 'tol': 0.0001}
    0.971 (+/-0.024) for {'C': 10, 'penalty': 'l2', 'tol': 0.01}
    0.959 (+/-0.030) for {'C': 10, 'penalty': 'l2', 'tol': 1}
    0.798 (+/-0.005) for {'C': 10, 'penalty': 'l2', 'tol': 100.0}


​    

## 의사결정나무


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
dt = DecisionTreeClassifier()

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
```


```python
parameters = {'criterion' : ['gini', 'entropy'],'min_samples_split' : [2, 5, 10, 15], 'max_depth' : [None, 2], 'min_samples_leaf':[1,3,10,15], 'max_features':[None, 'sqrt', 'log2']}
```


```python
GSDT = GridSearchCV(dt, parameters, cv=10, n_jobs=n_thread, scoring='accuracy')
GSDT.fit(x_train, y_train)
```




    GridSearchCV(cv=10, error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=None,
                                                  splitter='best'),
                 iid='deprecated', n_jobs=8,
                 param_grid={'criterion': ['gini', 'entropy'],
                             'max_depth': [None, 2],
                             'max_features': [None, 'sqrt', 'log2'],
                             'min_samples_leaf': [1, 3, 10, 15],
                             'min_samples_split': [2, 5, 10, 15]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)




```python
print('final params', GSDT.best_params_)
print('ACC', GSDT.best_score_)
```

    final params {'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'min_samples_leaf': 3, 'min_samples_split': 2}
    ACC 0.9780976013234077



```python
predicted = GSDT.predict(x_test)
cMatrix = confusion_matrix(y_test, predicted)
print(cMatrix)
print(round(GSDT.score(x_test, y_test), 3))
print(metrics.classification_report(y_test, predicted))
```

    [[308   3]
     [  8  70]]
    0.972
                  precision    recall  f1-score   support
    
               0       0.97      0.99      0.98       311
               1       0.96      0.90      0.93        78
    
        accuracy                           0.97       389
       macro avg       0.97      0.94      0.95       389
    weighted avg       0.97      0.97      0.97       389


​    

## 랜덤포레스트


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rf = RandomForestClassifier()

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
```


```python
parameters = {'n_estimators':[20,50,100], 'criterion':['entropy'], 'min_samples_split':[2,5], 'max_depth':[None, 2], 'min_samples_leaf':[1, 3, 10], 'max_features':['sqrt']}
GSRF = GridSearchCV(rf, parameters, cv=10, n_jobs=n_thread, scoring='accuracy')
GSRF.fit(x_train, y_train)
```




    GridSearchCV(cv=10, error_score=nan,
                 estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                                  class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features='auto',
                                                  max_leaf_nodes=None,
                                                  max_samples=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  n_estimators=100, n_jobs=None,
                                                  oob_score=False,
                                                  random_state=None, verbose=0,
                                                  warm_start=False),
                 iid='deprecated', n_jobs=8,
                 param_grid={'criterion': ['entropy'], 'max_depth': [None, 2],
                             'max_features': ['sqrt'],
                             'min_samples_leaf': [1, 3, 10],
                             'min_samples_split': [2, 5],
                             'n_estimators': [20, 50, 100]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)




```python
print('final params', GSRF.best_params_)
print('best score', GSRF.best_score_)
```

    final params {'criterion': 'entropy', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
    best score 0.9851736972704715



```python
predicted = GSRF.predict(x_test)
cMatrix = confusion_matrix(y_test, predicted)
print(cMatrix)
print(metrics.classification_report(y_test, predicted))
```

    [[311   0]
     [  4  74]]
                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99       311
               1       1.00      0.95      0.97        78
    
        accuracy                           0.99       389
       macro avg       0.99      0.97      0.98       389
    weighted avg       0.99      0.99      0.99       389


​    

- 랜덤포레스트가 가장 많은 정확도를 가지고 있지만 랜덤포레스트의 경우 모델에 대한 해석이 쉽지 않다. 즉 어떤 요인으로 인해 'K_Scatch'가 가장 많이 발생했는지 그 요인을 명쾌하기 설명해내기 어렵다.

## SVM (서포트 벡터 머신)


```python
from sklearn import svm

svc = svm.SVC()

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
```


```python
parameters = {'C':[0.01, 0.1, 0.5, 0.9, 1.5, 10], 'kernel':['linear','rbf','poly'], 'gamma':[0.1, 1, 10]}
GS_SVM = GridSearchCV(svc, parameters, cv=10, n_jobs=n_thread, scoring='accuracy')
GS_SVM.fit(x_train, y_train)
```




    GridSearchCV(cv=10, error_score=nan,
                 estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                               class_weight=None, coef0=0.0,
                               decision_function_shape='ovr', degree=3,
                               gamma='scale', kernel='rbf', max_iter=-1,
                               probability=False, random_state=None, shrinking=True,
                               tol=0.001, verbose=False),
                 iid='deprecated', n_jobs=8,
                 param_grid={'C': [0.01, 0.1, 0.5, 0.9, 1.5, 10],
                             'gamma': [0.1, 1, 10],
                             'kernel': ['linear', 'rbf', 'poly']},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)




```python
print('final params', GS_SVM.best_params_)
print('final score', GS_SVM.best_score_)
```

    final params {'C': 1.5, 'gamma': 0.1, 'kernel': 'rbf'}
    final score 0.9819602977667493



```python
predicted = GS_SVM.predict(x_test)
cMatrix = confusion_matrix(y_test, predicted)
print(cMatrix)
print(metrics.classification_report(y_test, predicted))
```

    [[311   0]
     [  8  70]]
                  precision    recall  f1-score   support
    
               0       0.97      1.00      0.99       311
               1       1.00      0.90      0.95        78
    
        accuracy                           0.98       389
       macro avg       0.99      0.95      0.97       389
    weighted avg       0.98      0.98      0.98       389


​    

## 인공 신경망 모형

- 참고 : playground.tensorflow.org/


```python
from sklearn.neural_network import MLPClassifier
```


```python
nn_model = MLPClassifier(random_state=1)

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
```

- 인공 신경망의 경우 Hidden Layer 설정하는것이 관건. 
- 방법은 없지만 보편적인 경향성을 말할 수 있음. 보통 1개로도 충분함- 1개에서 시작
- 히든레이어의 노드의 수는 보통 GridSerach를 통해 파악.
- 히든레이어 노드 가이드 라인. Number Of Neurons = Trading Data Samples / Factor * (Input Neurons + Output Neurons) Factor 수를 크게 하면 노드의 수가 줄어들며 적게하면 반대의 경우


```python
x_train.shape
```




    (1552, 27)




```python
a = 1552/(10 * (27+1))  # 27 = Input 변수개수 1 = output y변수
b = 1552 /(1 * (27+1)) # Factor가 1인경우
print(a, b)
```

    5.542857142857143 55.42857142857143


- Hidden Layer 가 하나라고 한다면 Factor 1-10이므로 노드의 수를 5~55개로 설정하며 그리드 서치 실행


```python
parameters = {'alpha':[1e-3, 1e-1, 1e1], 'hidden_layer_sizes':[(5),(30),(56)], 'activation':['tanh', 'relu'], 'solver':['adam', 'lbfgs']}
GS_NN = GridSearchCV(nn_model, parameters, cv=10, n_jobs=n_thread, scoring='accuracy')

# alpha, Searching Space를 찾아야 할 필요가 있다. 범위가 넓으므로
```


```python
GS_NN.fit(x_train, y_train)
```

    C:\Users\dissi\anaconda31\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:571: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)





    GridSearchCV(cv=10, error_score=nan,
                 estimator=MLPClassifier(activation='relu', alpha=0.0001,
                                         batch_size='auto', beta_1=0.9,
                                         beta_2=0.999, early_stopping=False,
                                         epsilon=1e-08, hidden_layer_sizes=(100,),
                                         learning_rate='constant',
                                         learning_rate_init=0.001, max_fun=15000,
                                         max_iter=200, momentum=0.9,
                                         n_iter_no_change=10,
                                         nesterovs_momentum=True, power_t=0.5,
                                         random_state=1, shuffle=True,
                                         solver='adam', tol=0.0001,
                                         validation_fraction=0.1, verbose=False,
                                         warm_start=False),
                 iid='deprecated', n_jobs=8,
                 param_grid={'activation': ['tanh', 'relu'],
                             'alpha': [0.001, 0.1, 10.0],
                             'hidden_layer_sizes': [5, 30, 56],
                             'solver': ['adam', 'lbfgs']},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)




```python
print('final params', GS_NN.best_params_)
print('best score', GS_NN.best_score_)
```

    final params {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 56, 'solver': 'adam'}
    best score 0.9781058726220018



```python
means = GS_NN.cv_results_['mean_test_score']
stds = GS_NN.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, GS_NN.cv_results_['params']):
    print ('%0.3f (+/-%0.03f) for %r' % (mean, std *2, params))
print()
```

    0.971 (+/-0.023) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 5, 'solver': 'adam'}
    0.971 (+/-0.029) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 5, 'solver': 'lbfgs'}
    0.976 (+/-0.025) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 30, 'solver': 'adam'}
    0.978 (+/-0.018) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 30, 'solver': 'lbfgs'}
    0.976 (+/-0.024) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 56, 'solver': 'adam'}
    0.975 (+/-0.029) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 56, 'solver': 'lbfgs'}
    0.971 (+/-0.023) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 5, 'solver': 'adam'}
    0.972 (+/-0.024) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 5, 'solver': 'lbfgs'}
    0.975 (+/-0.025) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 30, 'solver': 'adam'}
    0.976 (+/-0.024) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 30, 'solver': 'lbfgs'}
    0.976 (+/-0.024) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 56, 'solver': 'adam'}
    0.977 (+/-0.031) for {'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': 56, 'solver': 'lbfgs'}
    0.957 (+/-0.028) for {'activation': 'tanh', 'alpha': 10.0, 'hidden_layer_sizes': 5, 'solver': 'adam'}
    0.972 (+/-0.026) for {'activation': 'tanh', 'alpha': 10.0, 'hidden_layer_sizes': 5, 'solver': 'lbfgs'}
    0.956 (+/-0.028) for {'activation': 'tanh', 'alpha': 10.0, 'hidden_layer_sizes': 30, 'solver': 'adam'}
    0.972 (+/-0.027) for {'activation': 'tanh', 'alpha': 10.0, 'hidden_layer_sizes': 30, 'solver': 'lbfgs'}
    0.958 (+/-0.032) for {'activation': 'tanh', 'alpha': 10.0, 'hidden_layer_sizes': 56, 'solver': 'adam'}
    0.972 (+/-0.027) for {'activation': 'tanh', 'alpha': 10.0, 'hidden_layer_sizes': 56, 'solver': 'lbfgs'}
    0.962 (+/-0.036) for {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 5, 'solver': 'adam'}
    0.966 (+/-0.022) for {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 5, 'solver': 'lbfgs'}
    0.976 (+/-0.022) for {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 30, 'solver': 'adam'}
    0.972 (+/-0.018) for {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 30, 'solver': 'lbfgs'}
    0.978 (+/-0.022) for {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 56, 'solver': 'adam'}
    0.975 (+/-0.018) for {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 56, 'solver': 'lbfgs'}
    0.961 (+/-0.036) for {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': 5, 'solver': 'adam'}
    0.970 (+/-0.028) for {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': 5, 'solver': 'lbfgs'}
    0.976 (+/-0.022) for {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': 30, 'solver': 'adam'}
    0.978 (+/-0.027) for {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': 30, 'solver': 'lbfgs'}
    0.977 (+/-0.022) for {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': 56, 'solver': 'adam'}
    0.978 (+/-0.024) for {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': 56, 'solver': 'lbfgs'}
    0.960 (+/-0.033) for {'activation': 'relu', 'alpha': 10.0, 'hidden_layer_sizes': 5, 'solver': 'adam'}
    0.973 (+/-0.026) for {'activation': 'relu', 'alpha': 10.0, 'hidden_layer_sizes': 5, 'solver': 'lbfgs'}
    0.959 (+/-0.028) for {'activation': 'relu', 'alpha': 10.0, 'hidden_layer_sizes': 30, 'solver': 'adam'}
    0.974 (+/-0.024) for {'activation': 'relu', 'alpha': 10.0, 'hidden_layer_sizes': 30, 'solver': 'lbfgs'}
    0.957 (+/-0.030) for {'activation': 'relu', 'alpha': 10.0, 'hidden_layer_sizes': 56, 'solver': 'adam'}
    0.975 (+/-0.023) for {'activation': 'relu', 'alpha': 10.0, 'hidden_layer_sizes': 56, 'solver': 'lbfgs'}


​    

- solver의 경우 adam 보다는 lbfgs일 경우 정확도가 높다. 
- Hidden Layer를 늘려서 진행할 수도 있다.

- Hidden Layer 구성 어려움- 노드 설정에 있어 가이드 라인 존재. Hidden Layer 일단 하나부터 시작.
- 노드 범위를 옮겨가며 최적의 조건을 찾아가기


```python
predicted = GS_NN.predict(x_test)
cMatrix = confusion_matrix(y_test, predicted)
print(cMatrix)
print(metrics.classification_report(y_test, predicted))
```

    [[307   4]
     [  4  74]]
                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99       311
               1       0.95      0.95      0.95        78
    
        accuracy                           0.98       389
       macro avg       0.97      0.97      0.97       389
    weighted avg       0.98      0.98      0.98       389


​    

## 부스트 (XGBoost, LightGBM)

#### xgboost


```python
import xgboost as xgb
from sklearn.metrics import accuracy_score
```


```python
xgb_model = xgb.XGBClassifier(objective='binary:logistic')

# https://xgboost.readthedocs.io/en/latest/parameter.html
# https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
```


```python
parameters = {
    'max_depth' : [5,8],
    'min_child_weight' :[1,5],
    'gamma':[0,1],
    'colsample_bytree': [0.8, 1],
    'colsample_bylevel': [0.9, 1],
    'n_estimators': [50, 100]
}

GS_xgb = GridSearchCV(xgb_model, param_grid = parameters, cv=10, n_jobs=n_thread, scoring='accuracy')
GS_xgb.fit(x_train, y_train)
```

    C:\Users\dissi\anaconda31\lib\site-packages\xgboost\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)


    [21:34:51] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.





    GridSearchCV(cv=10, error_score=nan,
                 estimator=XGBClassifier(base_score=None, booster=None,
                                         colsample_bylevel=None,
                                         colsample_bynode=None,
                                         colsample_bytree=None, gamma=None,
                                         gpu_id=None, importance_type='gain',
                                         interaction_constraints=None,
                                         learning_rate=None, max_delta_step=None,
                                         max_depth=None, min_child_weight=None,
                                         missing=nan, monotone_constraints=None,
                                         n_esti...
                                         subsample=None, tree_method=None,
                                         use_label_encoder=True,
                                         validate_parameters=None, verbosity=None),
                 iid='deprecated', n_jobs=8,
                 param_grid={'colsample_bylevel': [0.9, 1],
                             'colsample_bytree': [0.8, 1], 'gamma': [0, 1],
                             'max_depth': [5, 8], 'min_child_weight': [1, 5],
                             'n_estimators': [50, 100]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)




```python
print('final params', GS_xgb.best_params_)
print('best score', GS_xgb.best_score_)
```

    final params {'colsample_bylevel': 0.9, 'colsample_bytree': 0.8, 'gamma': 0, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 50}
    best score 0.9864681555004138



```python
predicted = GS_xgb.predict(x_test)
cMatrix = confusion_matrix(y_test, predicted)
print(cMatrix)
print(metrics.classification_report(y_test, predicted))
```

    [[310   1]
     [  4  74]]
                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99       311
               1       0.99      0.95      0.97        78
    
        accuracy                           0.99       389
       macro avg       0.99      0.97      0.98       389
    weighted avg       0.99      0.99      0.99       389


​    

#### lightgbm


```python
import lightgbm as lgb

lgbm_model = lgb.LGBMClassifier(objective='binary')

# https://lightgbm.readthedocs.io/en/latest/Parameters.html
# https://lightgbm.readthedocs.io/en/latest/Parameters_Tuning.html
```


```python
parameters ={
    'num_leaves' : [32, 64, 128],
    'min_data_in_leaf' : [1, 5, 10],
    'colsample_byree' : [0.8, 1],
    'n_estimators' : [100, 150]
    }

GS_lgbm = GridSearchCV(lgbm_model, parameters, cv=10, n_jobs = n_thread, scoring='accuracy')
GS_lgbm.fit(x_train, y_train)
```

    [LightGBM] [Warning] Unknown parameter: colsample_byree
    [LightGBM] [Warning] min_data_in_leaf is set=10, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=10





    GridSearchCV(cv=10, error_score=nan,
                 estimator=LGBMClassifier(boosting_type='gbdt', class_weight=None,
                                          colsample_bytree=1.0,
                                          importance_type='split',
                                          learning_rate=0.1, max_depth=-1,
                                          min_child_samples=20,
                                          min_child_weight=0.001,
                                          min_split_gain=0.0, n_estimators=100,
                                          n_jobs=-1, num_leaves=31,
                                          objective='binary', random_state=None,
                                          reg_alpha=0.0, reg_lambda=0.0,
                                          silent=True, subsample=1.0,
                                          subsample_for_bin=200000,
                                          subsample_freq=0),
                 iid='deprecated', n_jobs=8,
                 param_grid={'colsample_byree': [0.8, 1],
                             'min_data_in_leaf': [1, 5, 10],
                             'n_estimators': [100, 150],
                             'num_leaves': [32, 64, 128]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=0)




```python
print('final params', GS_lgbm.best_params_)
print('best score', GS_lgbm.best_score_)
```

    final params {'colsample_byree': 0.8, 'min_data_in_leaf': 10, 'n_estimators': 100, 'num_leaves': 64}
    best score 0.985827129859388



```python
predicted = GS_lgbm.predict(x_test)
cMatrix = confusion_matrix(y_test, predicted)
print(cMatrix)
print(metrics.classification_report(y_test, predicted))
```

    [[310   1]
     [  5  73]]
                  precision    recall  f1-score   support
    
               0       0.98      1.00      0.99       311
               1       0.99      0.94      0.96        78
    
        accuracy                           0.98       389
       macro avg       0.99      0.97      0.98       389
    weighted avg       0.98      0.98      0.98       389


​    
