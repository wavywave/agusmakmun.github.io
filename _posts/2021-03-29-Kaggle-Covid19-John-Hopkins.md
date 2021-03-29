---
layout: post
title:  "Kaggle - Covid19 John Hopkins"
description: "캐글 데이터 분석"
author: SeungRok OH
categories: [Kaggle]
---

# Covid 19 from John Hopkins Dataset

- 데이터 셋 : https://www.kaggle.com/antgoldbloom/covid19-data-from-john-hopkins-university
- 실시간으로 업데이트 되는 라이브 데이터
- 시계열 데이터가 있음. (시간별 변화 추이가 중요)

## 라이브러리 및 데이터 불러오기


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_case = pd.read_csv('RAW_global_confirmed_cases.csv')
df_death = pd.read_csv('RAW_global_deaths.csv')

pd.set_option('display.max_columns', None)
```


```python
df_case.head()
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
      <th>Country/Region</th>
      <th>Province/State</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>2/1/20</th>
      <th>2/2/20</th>
      <th>2/3/20</th>
      <th>2/4/20</th>
      <th>2/5/20</th>
      <th>2/6/20</th>
      <th>2/7/20</th>
      <th>2/8/20</th>
      <th>2/9/20</th>
      <th>2/10/20</th>
      <th>2/11/20</th>
      <th>2/12/20</th>
      <th>2/13/20</th>
      <th>2/14/20</th>
      <th>2/15/20</th>
      <th>2/16/20</th>
      <th>2/17/20</th>
      <th>2/18/20</th>
      <th>2/19/20</th>
      <th>2/20/20</th>
      <th>2/21/20</th>
      <th>2/22/20</th>
      <th>2/23/20</th>
      <th>2/24/20</th>
      <th>2/25/20</th>
      <th>2/26/20</th>
      <th>2/27/20</th>
      <th>2/28/20</th>
      <th>2/29/20</th>
      <th>3/1/20</th>
      <th>3/2/20</th>
      <th>3/3/20</th>
      <th>3/4/20</th>
      <th>3/5/20</th>
      <th>3/6/20</th>
      <th>3/7/20</th>
      <th>3/8/20</th>
      <th>3/9/20</th>
      <th>3/10/20</th>
      <th>3/11/20</th>
      <th>3/12/20</th>
      <th>3/13/20</th>
      <th>3/14/20</th>
      <th>3/15/20</th>
      <th>3/16/20</th>
      <th>3/17/20</th>
      <th>3/18/20</th>
      <th>3/19/20</th>
      <th>3/20/20</th>
      <th>3/21/20</th>
      <th>3/22/20</th>
      <th>3/23/20</th>
      <th>3/24/20</th>
      <th>3/25/20</th>
      <th>3/26/20</th>
      <th>3/27/20</th>
      <th>3/28/20</th>
      <th>3/29/20</th>
      <th>3/30/20</th>
      <th>3/31/20</th>
      <th>4/1/20</th>
      <th>4/2/20</th>
      <th>4/3/20</th>
      <th>4/4/20</th>
      <th>4/5/20</th>
      <th>4/6/20</th>
      <th>4/7/20</th>
      <th>4/8/20</th>
      <th>4/9/20</th>
      <th>4/10/20</th>
      <th>4/11/20</th>
      <th>4/12/20</th>
      <th>4/13/20</th>
      <th>4/14/20</th>
      <th>4/15/20</th>
      <th>4/16/20</th>
      <th>4/17/20</th>
      <th>4/18/20</th>
      <th>4/19/20</th>
      <th>4/20/20</th>
      <th>4/21/20</th>
      <th>4/22/20</th>
      <th>4/23/20</th>
      <th>4/24/20</th>
      <th>4/25/20</th>
      <th>4/26/20</th>
      <th>4/27/20</th>
      <th>4/28/20</th>
      <th>4/29/20</th>
      <th>4/30/20</th>
      <th>5/1/20</th>
      <th>5/2/20</th>
      <th>5/3/20</th>
      <th>5/4/20</th>
      <th>5/5/20</th>
      <th>5/6/20</th>
      <th>5/7/20</th>
      <th>5/8/20</th>
      <th>5/9/20</th>
      <th>5/10/20</th>
      <th>5/11/20</th>
      <th>5/12/20</th>
      <th>5/13/20</th>
      <th>5/14/20</th>
      <th>5/15/20</th>
      <th>5/16/20</th>
      <th>5/17/20</th>
      <th>5/18/20</th>
      <th>5/19/20</th>
      <th>5/20/20</th>
      <th>5/21/20</th>
      <th>5/22/20</th>
      <th>5/23/20</th>
      <th>5/24/20</th>
      <th>5/25/20</th>
      <th>5/26/20</th>
      <th>5/27/20</th>
      <th>5/28/20</th>
      <th>5/29/20</th>
      <th>5/30/20</th>
      <th>5/31/20</th>
      <th>6/1/20</th>
      <th>6/2/20</th>
      <th>6/3/20</th>
      <th>6/4/20</th>
      <th>6/5/20</th>
      <th>6/6/20</th>
      <th>6/7/20</th>
      <th>6/8/20</th>
      <th>6/9/20</th>
      <th>6/10/20</th>
      <th>6/11/20</th>
      <th>6/12/20</th>
      <th>6/13/20</th>
      <th>6/14/20</th>
      <th>6/15/20</th>
      <th>6/16/20</th>
      <th>6/17/20</th>
      <th>6/18/20</th>
      <th>6/19/20</th>
      <th>6/20/20</th>
      <th>6/21/20</th>
      <th>6/22/20</th>
      <th>6/23/20</th>
      <th>6/24/20</th>
      <th>6/25/20</th>
      <th>6/26/20</th>
      <th>6/27/20</th>
      <th>6/28/20</th>
      <th>6/29/20</th>
      <th>6/30/20</th>
      <th>7/1/20</th>
      <th>7/2/20</th>
      <th>7/3/20</th>
      <th>7/4/20</th>
      <th>7/5/20</th>
      <th>7/6/20</th>
      <th>7/7/20</th>
      <th>7/8/20</th>
      <th>7/9/20</th>
      <th>7/10/20</th>
      <th>7/11/20</th>
      <th>7/12/20</th>
      <th>7/13/20</th>
      <th>7/14/20</th>
      <th>7/15/20</th>
      <th>7/16/20</th>
      <th>7/17/20</th>
      <th>7/18/20</th>
      <th>7/19/20</th>
      <th>7/20/20</th>
      <th>7/21/20</th>
      <th>7/22/20</th>
      <th>7/23/20</th>
      <th>7/24/20</th>
      <th>7/25/20</th>
      <th>7/26/20</th>
      <th>7/27/20</th>
      <th>7/28/20</th>
      <th>7/29/20</th>
      <th>7/30/20</th>
      <th>7/31/20</th>
      <th>8/1/20</th>
      <th>8/2/20</th>
      <th>8/3/20</th>
      <th>8/4/20</th>
      <th>8/5/20</th>
      <th>8/6/20</th>
      <th>8/7/20</th>
      <th>8/8/20</th>
      <th>8/9/20</th>
      <th>8/10/20</th>
      <th>8/11/20</th>
      <th>8/12/20</th>
      <th>8/13/20</th>
      <th>8/14/20</th>
      <th>8/15/20</th>
      <th>8/16/20</th>
      <th>8/17/20</th>
      <th>8/18/20</th>
      <th>8/19/20</th>
      <th>8/20/20</th>
      <th>8/21/20</th>
      <th>8/22/20</th>
      <th>8/23/20</th>
      <th>8/24/20</th>
      <th>8/25/20</th>
      <th>8/26/20</th>
      <th>8/27/20</th>
      <th>8/28/20</th>
      <th>8/29/20</th>
      <th>8/30/20</th>
      <th>8/31/20</th>
      <th>9/1/20</th>
      <th>9/2/20</th>
      <th>9/3/20</th>
      <th>9/4/20</th>
      <th>9/5/20</th>
      <th>9/6/20</th>
      <th>9/7/20</th>
      <th>9/8/20</th>
      <th>9/9/20</th>
      <th>9/10/20</th>
      <th>9/11/20</th>
      <th>9/12/20</th>
      <th>9/13/20</th>
      <th>9/14/20</th>
      <th>9/15/20</th>
      <th>9/16/20</th>
      <th>9/17/20</th>
      <th>9/18/20</th>
      <th>9/19/20</th>
      <th>9/20/20</th>
      <th>9/21/20</th>
      <th>9/22/20</th>
      <th>9/23/20</th>
      <th>9/24/20</th>
      <th>9/25/20</th>
      <th>9/26/20</th>
      <th>9/27/20</th>
      <th>9/28/20</th>
      <th>9/29/20</th>
      <th>9/30/20</th>
      <th>10/1/20</th>
      <th>10/2/20</th>
      <th>10/3/20</th>
      <th>10/4/20</th>
      <th>10/5/20</th>
      <th>10/6/20</th>
      <th>10/7/20</th>
      <th>10/8/20</th>
      <th>10/9/20</th>
      <th>10/10/20</th>
      <th>10/11/20</th>
      <th>10/12/20</th>
      <th>10/13/20</th>
      <th>10/14/20</th>
      <th>10/15/20</th>
      <th>10/16/20</th>
      <th>10/17/20</th>
      <th>10/18/20</th>
      <th>10/19/20</th>
      <th>10/20/20</th>
      <th>10/21/20</th>
      <th>10/22/20</th>
      <th>10/23/20</th>
      <th>10/24/20</th>
      <th>10/25/20</th>
      <th>10/26/20</th>
      <th>10/27/20</th>
      <th>10/28/20</th>
      <th>10/29/20</th>
      <th>10/30/20</th>
      <th>10/31/20</th>
      <th>11/1/20</th>
      <th>11/2/20</th>
      <th>11/3/20</th>
      <th>11/4/20</th>
      <th>11/5/20</th>
      <th>11/6/20</th>
      <th>11/7/20</th>
      <th>11/8/20</th>
      <th>11/9/20</th>
      <th>11/10/20</th>
      <th>11/11/20</th>
      <th>11/12/20</th>
      <th>11/13/20</th>
      <th>11/14/20</th>
      <th>11/15/20</th>
      <th>11/16/20</th>
      <th>11/17/20</th>
      <th>11/18/20</th>
      <th>11/19/20</th>
      <th>11/20/20</th>
      <th>11/21/20</th>
      <th>11/22/20</th>
      <th>11/23/20</th>
      <th>11/24/20</th>
      <th>11/25/20</th>
      <th>11/26/20</th>
      <th>11/27/20</th>
      <th>11/28/20</th>
      <th>11/29/20</th>
      <th>11/30/20</th>
      <th>12/1/20</th>
      <th>12/2/20</th>
      <th>12/3/20</th>
      <th>12/4/20</th>
      <th>12/5/20</th>
      <th>12/6/20</th>
      <th>12/7/20</th>
      <th>12/8/20</th>
      <th>12/9/20</th>
      <th>12/10/20</th>
      <th>12/11/20</th>
      <th>12/12/20</th>
      <th>12/13/20</th>
      <th>12/14/20</th>
      <th>12/15/20</th>
      <th>12/16/20</th>
      <th>12/17/20</th>
      <th>12/18/20</th>
      <th>12/19/20</th>
      <th>12/20/20</th>
      <th>12/21/20</th>
      <th>12/22/20</th>
      <th>12/23/20</th>
      <th>12/24/20</th>
      <th>12/25/20</th>
      <th>12/26/20</th>
      <th>12/27/20</th>
      <th>12/28/20</th>
      <th>12/29/20</th>
      <th>12/30/20</th>
      <th>12/31/20</th>
      <th>1/1/21</th>
      <th>1/2/21</th>
      <th>1/3/21</th>
      <th>1/4/21</th>
      <th>1/5/21</th>
      <th>1/6/21</th>
      <th>1/7/21</th>
      <th>1/8/21</th>
      <th>1/9/21</th>
      <th>1/10/21</th>
      <th>1/11/21</th>
      <th>1/12/21</th>
      <th>1/13/21</th>
      <th>1/14/21</th>
      <th>1/15/21</th>
      <th>1/16/21</th>
      <th>1/17/21</th>
      <th>1/18/21</th>
      <th>1/19/21</th>
      <th>1/20/21</th>
      <th>1/21/21</th>
      <th>1/22/21</th>
      <th>1/23/21</th>
      <th>1/24/21</th>
      <th>1/25/21</th>
      <th>1/26/21</th>
      <th>1/27/21</th>
      <th>1/28/21</th>
      <th>1/29/21</th>
      <th>1/30/21</th>
      <th>1/31/21</th>
      <th>2/1/21</th>
      <th>2/2/21</th>
      <th>2/3/21</th>
      <th>2/4/21</th>
      <th>2/5/21</th>
      <th>2/6/21</th>
      <th>2/7/21</th>
      <th>2/8/21</th>
      <th>2/9/21</th>
      <th>2/10/21</th>
      <th>2/11/21</th>
      <th>2/12/21</th>
      <th>2/13/21</th>
      <th>2/14/21</th>
      <th>2/15/21</th>
      <th>2/16/21</th>
      <th>2/17/21</th>
      <th>2/18/21</th>
      <th>2/19/21</th>
      <th>2/20/21</th>
      <th>2/21/21</th>
      <th>2/22/21</th>
      <th>2/23/21</th>
      <th>2/24/21</th>
      <th>2/25/21</th>
      <th>2/26/21</th>
      <th>2/27/21</th>
      <th>2/28/21</th>
      <th>3/1/21</th>
      <th>3/2/21</th>
      <th>3/3/21</th>
      <th>3/4/21</th>
      <th>3/5/21</th>
      <th>3/6/21</th>
      <th>3/7/21</th>
      <th>3/8/21</th>
      <th>3/9/21</th>
      <th>3/10/21</th>
      <th>3/11/21</th>
      <th>3/12/21</th>
      <th>3/13/21</th>
      <th>3/14/21</th>
      <th>3/15/21</th>
      <th>3/16/21</th>
      <th>3/17/21</th>
      <th>3/18/21</th>
      <th>3/19/21</th>
      <th>3/20/21</th>
      <th>3/21/21</th>
      <th>3/22/21</th>
      <th>3/23/21</th>
      <th>3/24/21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>33.93911</td>
      <td>67.709953</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>8</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
      <td>15</td>
      <td>16</td>
      <td>18</td>
      <td>20</td>
      <td>24</td>
      <td>25</td>
      <td>29</td>
      <td>30</td>
      <td>34</td>
      <td>41</td>
      <td>43</td>
      <td>76</td>
      <td>80</td>
      <td>91</td>
      <td>107</td>
      <td>118</td>
      <td>146</td>
      <td>175</td>
      <td>197</td>
      <td>240</td>
      <td>275</td>
      <td>300</td>
      <td>338</td>
      <td>368</td>
      <td>424</td>
      <td>445</td>
      <td>485</td>
      <td>532</td>
      <td>556</td>
      <td>608</td>
      <td>666</td>
      <td>715</td>
      <td>785</td>
      <td>841</td>
      <td>907</td>
      <td>934</td>
      <td>997</td>
      <td>1027</td>
      <td>1093</td>
      <td>1177</td>
      <td>1236</td>
      <td>1331</td>
      <td>1464</td>
      <td>1532</td>
      <td>1704</td>
      <td>1830</td>
      <td>1940</td>
      <td>2127</td>
      <td>2291</td>
      <td>2470</td>
      <td>2705</td>
      <td>2895</td>
      <td>3225</td>
      <td>3393</td>
      <td>3564</td>
      <td>3781</td>
      <td>4042</td>
      <td>4403</td>
      <td>4687</td>
      <td>4968</td>
      <td>5227</td>
      <td>5640</td>
      <td>6054</td>
      <td>6403</td>
      <td>6665</td>
      <td>7073</td>
      <td>7654</td>
      <td>8146</td>
      <td>8677</td>
      <td>9219</td>
      <td>10001</td>
      <td>10585</td>
      <td>11176</td>
      <td>11834</td>
      <td>12459</td>
      <td>13039</td>
      <td>13662</td>
      <td>14528</td>
      <td>15208</td>
      <td>15753</td>
      <td>16512</td>
      <td>17270</td>
      <td>18057</td>
      <td>18972</td>
      <td>19554</td>
      <td>20345</td>
      <td>20920</td>
      <td>21462</td>
      <td>22146</td>
      <td>22894</td>
      <td>23550</td>
      <td>24106</td>
      <td>24770</td>
      <td>25531</td>
      <td>26314</td>
      <td>26878</td>
      <td>27536</td>
      <td>27882</td>
      <td>28428</td>
      <td>28837</td>
      <td>29147</td>
      <td>29471</td>
      <td>29705</td>
      <td>30165</td>
      <td>30441</td>
      <td>30606</td>
      <td>30957</td>
      <td>31228</td>
      <td>31507</td>
      <td>31826</td>
      <td>32012</td>
      <td>32314</td>
      <td>32662</td>
      <td>32941</td>
      <td>33180</td>
      <td>33374</td>
      <td>33584</td>
      <td>33898</td>
      <td>34184</td>
      <td>34356</td>
      <td>34441</td>
      <td>34595</td>
      <td>34730</td>
      <td>34984</td>
      <td>35060</td>
      <td>35219</td>
      <td>35279</td>
      <td>35453</td>
      <td>35493</td>
      <td>35605</td>
      <td>35717</td>
      <td>35918</td>
      <td>35978</td>
      <td>36026</td>
      <td>36147</td>
      <td>36253</td>
      <td>36358</td>
      <td>36463</td>
      <td>36532</td>
      <td>36665</td>
      <td>36700</td>
      <td>36701</td>
      <td>36737</td>
      <td>36773</td>
      <td>36820</td>
      <td>36928</td>
      <td>37006</td>
      <td>37046</td>
      <td>37083</td>
      <td>37153</td>
      <td>37260</td>
      <td>37336</td>
      <td>37422</td>
      <td>37497</td>
      <td>37542</td>
      <td>37590</td>
      <td>37667</td>
      <td>37710</td>
      <td>37750</td>
      <td>37852</td>
      <td>37885</td>
      <td>37944</td>
      <td>37990</td>
      <td>38045</td>
      <td>38061</td>
      <td>38103</td>
      <td>38119</td>
      <td>38130</td>
      <td>38133</td>
      <td>38155</td>
      <td>38159</td>
      <td>38193</td>
      <td>38243</td>
      <td>38288</td>
      <td>38304</td>
      <td>38324</td>
      <td>38398</td>
      <td>38494</td>
      <td>38520</td>
      <td>38544</td>
      <td>38572</td>
      <td>38606</td>
      <td>38641</td>
      <td>38716</td>
      <td>38772</td>
      <td>38815</td>
      <td>38855</td>
      <td>38872</td>
      <td>38897</td>
      <td>38919</td>
      <td>39044</td>
      <td>39074</td>
      <td>39096</td>
      <td>39145</td>
      <td>39170</td>
      <td>39186</td>
      <td>39192</td>
      <td>39227</td>
      <td>39239</td>
      <td>39254</td>
      <td>39268</td>
      <td>39285</td>
      <td>39290</td>
      <td>39297</td>
      <td>39341</td>
      <td>39422</td>
      <td>39486</td>
      <td>39548</td>
      <td>39616</td>
      <td>39693</td>
      <td>39703</td>
      <td>39799</td>
      <td>39870</td>
      <td>39928</td>
      <td>39994</td>
      <td>40026</td>
      <td>40088</td>
      <td>40141</td>
      <td>40200</td>
      <td>40287</td>
      <td>40369</td>
      <td>40510</td>
      <td>40626</td>
      <td>40687</td>
      <td>40768</td>
      <td>40833</td>
      <td>40937</td>
      <td>41032</td>
      <td>41145</td>
      <td>41268</td>
      <td>41334</td>
      <td>41425</td>
      <td>41501</td>
      <td>41633</td>
      <td>41728</td>
      <td>41814</td>
      <td>41935</td>
      <td>41975</td>
      <td>42033</td>
      <td>42159</td>
      <td>42297</td>
      <td>42463</td>
      <td>42609</td>
      <td>42795</td>
      <td>42969</td>
      <td>43035</td>
      <td>43240</td>
      <td>43468</td>
      <td>43681</td>
      <td>43924</td>
      <td>44177</td>
      <td>44363</td>
      <td>44503</td>
      <td>44706</td>
      <td>44988</td>
      <td>45174</td>
      <td>45384</td>
      <td>45600</td>
      <td>45723</td>
      <td>45844</td>
      <td>46116</td>
      <td>46274</td>
      <td>46516</td>
      <td>46718</td>
      <td>46837</td>
      <td>46837</td>
      <td>47072</td>
      <td>47306</td>
      <td>47516</td>
      <td>47716</td>
      <td>47851</td>
      <td>48053</td>
      <td>48116</td>
      <td>48229</td>
      <td>48527</td>
      <td>48718</td>
      <td>48952</td>
      <td>49161</td>
      <td>49378</td>
      <td>49621</td>
      <td>49681</td>
      <td>49817</td>
      <td>50013</td>
      <td>50190</td>
      <td>50433</td>
      <td>50655</td>
      <td>50810</td>
      <td>50886</td>
      <td>51039</td>
      <td>51280</td>
      <td>51350</td>
      <td>51405</td>
      <td>51526</td>
      <td>51526</td>
      <td>51526</td>
      <td>51526</td>
      <td>53011</td>
      <td>53105</td>
      <td>53105</td>
      <td>53207</td>
      <td>53332</td>
      <td>53400</td>
      <td>53489</td>
      <td>53538</td>
      <td>53584</td>
      <td>53584</td>
      <td>53775</td>
      <td>53831</td>
      <td>53938</td>
      <td>53984</td>
      <td>54062</td>
      <td>54141</td>
      <td>54278</td>
      <td>54403</td>
      <td>54483</td>
      <td>54559</td>
      <td>54595</td>
      <td>54672</td>
      <td>54750</td>
      <td>54854</td>
      <td>54891</td>
      <td>54939</td>
      <td>55008</td>
      <td>55023</td>
      <td>55059</td>
      <td>55121</td>
      <td>55174</td>
      <td>55231</td>
      <td>55265</td>
      <td>55330</td>
      <td>55335</td>
      <td>55359</td>
      <td>55384</td>
      <td>55402</td>
      <td>55420</td>
      <td>55445</td>
      <td>55473</td>
      <td>55492</td>
      <td>55514</td>
      <td>55518</td>
      <td>55540</td>
      <td>55557</td>
      <td>55575</td>
      <td>55580</td>
      <td>55604</td>
      <td>55617</td>
      <td>55646</td>
      <td>55664</td>
      <td>55680</td>
      <td>55696</td>
      <td>55707</td>
      <td>55714</td>
      <td>55733</td>
      <td>55759</td>
      <td>55770</td>
      <td>55775</td>
      <td>55827</td>
      <td>55840</td>
      <td>55847</td>
      <td>55876</td>
      <td>55876</td>
      <td>55894</td>
      <td>55917</td>
      <td>55959</td>
      <td>55959</td>
      <td>55985</td>
      <td>55985</td>
      <td>55995</td>
      <td>56016</td>
      <td>56044</td>
      <td>56069</td>
      <td>56093</td>
      <td>56103</td>
      <td>56153</td>
      <td>56177</td>
      <td>56192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>NaN</td>
      <td>41.15330</td>
      <td>20.168300</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>10</td>
      <td>12</td>
      <td>23</td>
      <td>33</td>
      <td>38</td>
      <td>42</td>
      <td>51</td>
      <td>55</td>
      <td>59</td>
      <td>64</td>
      <td>70</td>
      <td>76</td>
      <td>89</td>
      <td>104</td>
      <td>123</td>
      <td>146</td>
      <td>174</td>
      <td>186</td>
      <td>197</td>
      <td>212</td>
      <td>223</td>
      <td>243</td>
      <td>259</td>
      <td>277</td>
      <td>304</td>
      <td>333</td>
      <td>361</td>
      <td>377</td>
      <td>383</td>
      <td>400</td>
      <td>409</td>
      <td>416</td>
      <td>433</td>
      <td>446</td>
      <td>467</td>
      <td>475</td>
      <td>494</td>
      <td>518</td>
      <td>539</td>
      <td>548</td>
      <td>562</td>
      <td>584</td>
      <td>609</td>
      <td>634</td>
      <td>663</td>
      <td>678</td>
      <td>712</td>
      <td>726</td>
      <td>736</td>
      <td>750</td>
      <td>766</td>
      <td>773</td>
      <td>782</td>
      <td>789</td>
      <td>795</td>
      <td>803</td>
      <td>820</td>
      <td>832</td>
      <td>842</td>
      <td>850</td>
      <td>856</td>
      <td>868</td>
      <td>872</td>
      <td>876</td>
      <td>880</td>
      <td>898</td>
      <td>916</td>
      <td>933</td>
      <td>946</td>
      <td>948</td>
      <td>949</td>
      <td>964</td>
      <td>969</td>
      <td>981</td>
      <td>989</td>
      <td>998</td>
      <td>1004</td>
      <td>1029</td>
      <td>1050</td>
      <td>1076</td>
      <td>1099</td>
      <td>1122</td>
      <td>1137</td>
      <td>1143</td>
      <td>1164</td>
      <td>1184</td>
      <td>1197</td>
      <td>1212</td>
      <td>1232</td>
      <td>1246</td>
      <td>1263</td>
      <td>1299</td>
      <td>1341</td>
      <td>1385</td>
      <td>1416</td>
      <td>1464</td>
      <td>1521</td>
      <td>1590</td>
      <td>1672</td>
      <td>1722</td>
      <td>1788</td>
      <td>1838</td>
      <td>1891</td>
      <td>1962</td>
      <td>1995</td>
      <td>2047</td>
      <td>2114</td>
      <td>2192</td>
      <td>2269</td>
      <td>2330</td>
      <td>2402</td>
      <td>2466</td>
      <td>2535</td>
      <td>2580</td>
      <td>2662</td>
      <td>2752</td>
      <td>2819</td>
      <td>2893</td>
      <td>2964</td>
      <td>3038</td>
      <td>3106</td>
      <td>3188</td>
      <td>3278</td>
      <td>3371</td>
      <td>3454</td>
      <td>3571</td>
      <td>3667</td>
      <td>3752</td>
      <td>3851</td>
      <td>3906</td>
      <td>4008</td>
      <td>4090</td>
      <td>4171</td>
      <td>4290</td>
      <td>4358</td>
      <td>4466</td>
      <td>4570</td>
      <td>4637</td>
      <td>4763</td>
      <td>4880</td>
      <td>4997</td>
      <td>5105</td>
      <td>5197</td>
      <td>5276</td>
      <td>5396</td>
      <td>5519</td>
      <td>5620</td>
      <td>5750</td>
      <td>5889</td>
      <td>6016</td>
      <td>6151</td>
      <td>6275</td>
      <td>6411</td>
      <td>6536</td>
      <td>6676</td>
      <td>6817</td>
      <td>6971</td>
      <td>7117</td>
      <td>7260</td>
      <td>7380</td>
      <td>7499</td>
      <td>7654</td>
      <td>7812</td>
      <td>7967</td>
      <td>8119</td>
      <td>8275</td>
      <td>8427</td>
      <td>8605</td>
      <td>8759</td>
      <td>8927</td>
      <td>9083</td>
      <td>9195</td>
      <td>9279</td>
      <td>9380</td>
      <td>9513</td>
      <td>9606</td>
      <td>9728</td>
      <td>9844</td>
      <td>9967</td>
      <td>10102</td>
      <td>10255</td>
      <td>10406</td>
      <td>10553</td>
      <td>10704</td>
      <td>10860</td>
      <td>11021</td>
      <td>11185</td>
      <td>11353</td>
      <td>11520</td>
      <td>11672</td>
      <td>11816</td>
      <td>11948</td>
      <td>12073</td>
      <td>12226</td>
      <td>12385</td>
      <td>12535</td>
      <td>12666</td>
      <td>12787</td>
      <td>12921</td>
      <td>13045</td>
      <td>13153</td>
      <td>13259</td>
      <td>13391</td>
      <td>13518</td>
      <td>13649</td>
      <td>13806</td>
      <td>13965</td>
      <td>14117</td>
      <td>14266</td>
      <td>14410</td>
      <td>14568</td>
      <td>14730</td>
      <td>14899</td>
      <td>15066</td>
      <td>15231</td>
      <td>15399</td>
      <td>15570</td>
      <td>15752</td>
      <td>15955</td>
      <td>16212</td>
      <td>16501</td>
      <td>16774</td>
      <td>17055</td>
      <td>17350</td>
      <td>17651</td>
      <td>17948</td>
      <td>18250</td>
      <td>18556</td>
      <td>18858</td>
      <td>19157</td>
      <td>19445</td>
      <td>19729</td>
      <td>20040</td>
      <td>20315</td>
      <td>20634</td>
      <td>20875</td>
      <td>21202</td>
      <td>21523</td>
      <td>21904</td>
      <td>22300</td>
      <td>22721</td>
      <td>23210</td>
      <td>23705</td>
      <td>24206</td>
      <td>24731</td>
      <td>25294</td>
      <td>25801</td>
      <td>26211</td>
      <td>26701</td>
      <td>27233</td>
      <td>27830</td>
      <td>28432</td>
      <td>29126</td>
      <td>29837</td>
      <td>30623</td>
      <td>31459</td>
      <td>32196</td>
      <td>32761</td>
      <td>33556</td>
      <td>34300</td>
      <td>34944</td>
      <td>35600</td>
      <td>36245</td>
      <td>36790</td>
      <td>37625</td>
      <td>38182</td>
      <td>39014</td>
      <td>39719</td>
      <td>40501</td>
      <td>41302</td>
      <td>42148</td>
      <td>42988</td>
      <td>43683</td>
      <td>44436</td>
      <td>45188</td>
      <td>46061</td>
      <td>46863</td>
      <td>47742</td>
      <td>48530</td>
      <td>49191</td>
      <td>50000</td>
      <td>50637</td>
      <td>51424</td>
      <td>52004</td>
      <td>52542</td>
      <td>53003</td>
      <td>53425</td>
      <td>53814</td>
      <td>54317</td>
      <td>54827</td>
      <td>55380</td>
      <td>55755</td>
      <td>56254</td>
      <td>56572</td>
      <td>57146</td>
      <td>57727</td>
      <td>58316</td>
      <td>58316</td>
      <td>58991</td>
      <td>59438</td>
      <td>59623</td>
      <td>60283</td>
      <td>61008</td>
      <td>61705</td>
      <td>62378</td>
      <td>63033</td>
      <td>63595</td>
      <td>63971</td>
      <td>64627</td>
      <td>65334</td>
      <td>65994</td>
      <td>66635</td>
      <td>67216</td>
      <td>67690</td>
      <td>67982</td>
      <td>68568</td>
      <td>69238</td>
      <td>69916</td>
      <td>70655</td>
      <td>71441</td>
      <td>72274</td>
      <td>72812</td>
      <td>73691</td>
      <td>74567</td>
      <td>75454</td>
      <td>76350</td>
      <td>77251</td>
      <td>78127</td>
      <td>78992</td>
      <td>79934</td>
      <td>80941</td>
      <td>81993</td>
      <td>83082</td>
      <td>84212</td>
      <td>85336</td>
      <td>86289</td>
      <td>87528</td>
      <td>88671</td>
      <td>89776</td>
      <td>90835</td>
      <td>91987</td>
      <td>93075</td>
      <td>93850</td>
      <td>94651</td>
      <td>95726</td>
      <td>96838</td>
      <td>97909</td>
      <td>99062</td>
      <td>100246</td>
      <td>101285</td>
      <td>102306</td>
      <td>103327</td>
      <td>104313</td>
      <td>105229</td>
      <td>106215</td>
      <td>107167</td>
      <td>107931</td>
      <td>108823</td>
      <td>109674</td>
      <td>110521</td>
      <td>111301</td>
      <td>112078</td>
      <td>112897</td>
      <td>113580</td>
      <td>114209</td>
      <td>114840</td>
      <td>115442</td>
      <td>116123</td>
      <td>116821</td>
      <td>117474</td>
      <td>118017</td>
      <td>118492</td>
      <td>118938</td>
      <td>119528</td>
      <td>120022</td>
      <td>120541</td>
      <td>121200</td>
      <td>121544</td>
      <td>121847</td>
      <td>122295</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>NaN</td>
      <td>28.03390</td>
      <td>1.659600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>12</td>
      <td>12</td>
      <td>17</td>
      <td>17</td>
      <td>19</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>24</td>
      <td>26</td>
      <td>37</td>
      <td>48</td>
      <td>54</td>
      <td>60</td>
      <td>74</td>
      <td>87</td>
      <td>90</td>
      <td>139</td>
      <td>201</td>
      <td>230</td>
      <td>264</td>
      <td>302</td>
      <td>367</td>
      <td>409</td>
      <td>454</td>
      <td>511</td>
      <td>584</td>
      <td>716</td>
      <td>847</td>
      <td>986</td>
      <td>1171</td>
      <td>1251</td>
      <td>1320</td>
      <td>1423</td>
      <td>1468</td>
      <td>1572</td>
      <td>1666</td>
      <td>1761</td>
      <td>1825</td>
      <td>1914</td>
      <td>1983</td>
      <td>2070</td>
      <td>2160</td>
      <td>2268</td>
      <td>2418</td>
      <td>2534</td>
      <td>2629</td>
      <td>2718</td>
      <td>2811</td>
      <td>2910</td>
      <td>3007</td>
      <td>3127</td>
      <td>3256</td>
      <td>3382</td>
      <td>3517</td>
      <td>3649</td>
      <td>3848</td>
      <td>4006</td>
      <td>4154</td>
      <td>4295</td>
      <td>4474</td>
      <td>4648</td>
      <td>4838</td>
      <td>4997</td>
      <td>5182</td>
      <td>5369</td>
      <td>5558</td>
      <td>5723</td>
      <td>5891</td>
      <td>6067</td>
      <td>6253</td>
      <td>6442</td>
      <td>6629</td>
      <td>6821</td>
      <td>7019</td>
      <td>7201</td>
      <td>7377</td>
      <td>7542</td>
      <td>7728</td>
      <td>7918</td>
      <td>8113</td>
      <td>8306</td>
      <td>8503</td>
      <td>8697</td>
      <td>8857</td>
      <td>8997</td>
      <td>9134</td>
      <td>9267</td>
      <td>9394</td>
      <td>9513</td>
      <td>9626</td>
      <td>9733</td>
      <td>9831</td>
      <td>9935</td>
      <td>10050</td>
      <td>10154</td>
      <td>10265</td>
      <td>10382</td>
      <td>10484</td>
      <td>10589</td>
      <td>10698</td>
      <td>10810</td>
      <td>10919</td>
      <td>11031</td>
      <td>11147</td>
      <td>11268</td>
      <td>11385</td>
      <td>11504</td>
      <td>11631</td>
      <td>11771</td>
      <td>11920</td>
      <td>12076</td>
      <td>12248</td>
      <td>12445</td>
      <td>12685</td>
      <td>12968</td>
      <td>13273</td>
      <td>13571</td>
      <td>13907</td>
      <td>14272</td>
      <td>14657</td>
      <td>15070</td>
      <td>15500</td>
      <td>15941</td>
      <td>16404</td>
      <td>16879</td>
      <td>17348</td>
      <td>17808</td>
      <td>18242</td>
      <td>18712</td>
      <td>19195</td>
      <td>19689</td>
      <td>20216</td>
      <td>20770</td>
      <td>21355</td>
      <td>21948</td>
      <td>22549</td>
      <td>23084</td>
      <td>23691</td>
      <td>24278</td>
      <td>24872</td>
      <td>25484</td>
      <td>26159</td>
      <td>26764</td>
      <td>27357</td>
      <td>27973</td>
      <td>28615</td>
      <td>29229</td>
      <td>29831</td>
      <td>30394</td>
      <td>30950</td>
      <td>31465</td>
      <td>31972</td>
      <td>32504</td>
      <td>33055</td>
      <td>33626</td>
      <td>34155</td>
      <td>34693</td>
      <td>35160</td>
      <td>35712</td>
      <td>36204</td>
      <td>36699</td>
      <td>37187</td>
      <td>37664</td>
      <td>38133</td>
      <td>38583</td>
      <td>39025</td>
      <td>39444</td>
      <td>39847</td>
      <td>40258</td>
      <td>40667</td>
      <td>41068</td>
      <td>41460</td>
      <td>41858</td>
      <td>42228</td>
      <td>42619</td>
      <td>43016</td>
      <td>43403</td>
      <td>43781</td>
      <td>44146</td>
      <td>44494</td>
      <td>44833</td>
      <td>45158</td>
      <td>45469</td>
      <td>45773</td>
      <td>46071</td>
      <td>46364</td>
      <td>46653</td>
      <td>46938</td>
      <td>47216</td>
      <td>47488</td>
      <td>47752</td>
      <td>48007</td>
      <td>48254</td>
      <td>48496</td>
      <td>48734</td>
      <td>48966</td>
      <td>49194</td>
      <td>49413</td>
      <td>49623</td>
      <td>49826</td>
      <td>50023</td>
      <td>50214</td>
      <td>50400</td>
      <td>50579</td>
      <td>50754</td>
      <td>50914</td>
      <td>51067</td>
      <td>51213</td>
      <td>51368</td>
      <td>51530</td>
      <td>51690</td>
      <td>51847</td>
      <td>51995</td>
      <td>52136</td>
      <td>52270</td>
      <td>52399</td>
      <td>52520</td>
      <td>52658</td>
      <td>52804</td>
      <td>52940</td>
      <td>53072</td>
      <td>53325</td>
      <td>53399</td>
      <td>53584</td>
      <td>53777</td>
      <td>53998</td>
      <td>54203</td>
      <td>54402</td>
      <td>54616</td>
      <td>54829</td>
      <td>55081</td>
      <td>55357</td>
      <td>55630</td>
      <td>55880</td>
      <td>56143</td>
      <td>56419</td>
      <td>56706</td>
      <td>57026</td>
      <td>57332</td>
      <td>57651</td>
      <td>57942</td>
      <td>58272</td>
      <td>58574</td>
      <td>58979</td>
      <td>59527</td>
      <td>60169</td>
      <td>60800</td>
      <td>61381</td>
      <td>62051</td>
      <td>62693</td>
      <td>63446</td>
      <td>64257</td>
      <td>65108</td>
      <td>65975</td>
      <td>66819</td>
      <td>67679</td>
      <td>68589</td>
      <td>69591</td>
      <td>70629</td>
      <td>71652</td>
      <td>72755</td>
      <td>73774</td>
      <td>74862</td>
      <td>75867</td>
      <td>77000</td>
      <td>78025</td>
      <td>79110</td>
      <td>80168</td>
      <td>81212</td>
      <td>82221</td>
      <td>83199</td>
      <td>84152</td>
      <td>85084</td>
      <td>85927</td>
      <td>86730</td>
      <td>87502</td>
      <td>88252</td>
      <td>88825</td>
      <td>89416</td>
      <td>90014</td>
      <td>90579</td>
      <td>91121</td>
      <td>91638</td>
      <td>92102</td>
      <td>92597</td>
      <td>93065</td>
      <td>93507</td>
      <td>93933</td>
      <td>94371</td>
      <td>94781</td>
      <td>95203</td>
      <td>95659</td>
      <td>96069</td>
      <td>96549</td>
      <td>97007</td>
      <td>97441</td>
      <td>97857</td>
      <td>98249</td>
      <td>98631</td>
      <td>98988</td>
      <td>99311</td>
      <td>99610</td>
      <td>99897</td>
      <td>100159</td>
      <td>100408</td>
      <td>100645</td>
      <td>100873</td>
      <td>101120</td>
      <td>101382</td>
      <td>101657</td>
      <td>101913</td>
      <td>102144</td>
      <td>102369</td>
      <td>102641</td>
      <td>102860</td>
      <td>103127</td>
      <td>103381</td>
      <td>103611</td>
      <td>103833</td>
      <td>104092</td>
      <td>104341</td>
      <td>104606</td>
      <td>104852</td>
      <td>105124</td>
      <td>105369</td>
      <td>105596</td>
      <td>105854</td>
      <td>106097</td>
      <td>106359</td>
      <td>106610</td>
      <td>106887</td>
      <td>107122</td>
      <td>107339</td>
      <td>107578</td>
      <td>107841</td>
      <td>108116</td>
      <td>108381</td>
      <td>108629</td>
      <td>108629</td>
      <td>109088</td>
      <td>109313</td>
      <td>109559</td>
      <td>109782</td>
      <td>110049</td>
      <td>110303</td>
      <td>110513</td>
      <td>110711</td>
      <td>110894</td>
      <td>111069</td>
      <td>111247</td>
      <td>111418</td>
      <td>111600</td>
      <td>111764</td>
      <td>111917</td>
      <td>112094</td>
      <td>112279</td>
      <td>112461</td>
      <td>112622</td>
      <td>112805</td>
      <td>112960</td>
      <td>113092</td>
      <td>113255</td>
      <td>113430</td>
      <td>113593</td>
      <td>113761</td>
      <td>113948</td>
      <td>114104</td>
      <td>114234</td>
      <td>114382</td>
      <td>114543</td>
      <td>114681</td>
      <td>114851</td>
      <td>115008</td>
      <td>115143</td>
      <td>115265</td>
      <td>115410</td>
      <td>115540</td>
      <td>115688</td>
      <td>115842</td>
      <td>115970</td>
      <td>116066</td>
      <td>116157</td>
      <td>116255</td>
      <td>116349</td>
      <td>116438</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>NaN</td>
      <td>42.50630</td>
      <td>1.521800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>39</td>
      <td>39</td>
      <td>53</td>
      <td>75</td>
      <td>88</td>
      <td>113</td>
      <td>133</td>
      <td>164</td>
      <td>188</td>
      <td>224</td>
      <td>267</td>
      <td>308</td>
      <td>334</td>
      <td>370</td>
      <td>376</td>
      <td>390</td>
      <td>428</td>
      <td>439</td>
      <td>466</td>
      <td>501</td>
      <td>525</td>
      <td>545</td>
      <td>564</td>
      <td>583</td>
      <td>601</td>
      <td>601</td>
      <td>638</td>
      <td>646</td>
      <td>659</td>
      <td>673</td>
      <td>673</td>
      <td>696</td>
      <td>704</td>
      <td>713</td>
      <td>717</td>
      <td>717</td>
      <td>723</td>
      <td>723</td>
      <td>731</td>
      <td>738</td>
      <td>738</td>
      <td>743</td>
      <td>743</td>
      <td>743</td>
      <td>745</td>
      <td>745</td>
      <td>747</td>
      <td>748</td>
      <td>750</td>
      <td>751</td>
      <td>751</td>
      <td>752</td>
      <td>752</td>
      <td>754</td>
      <td>755</td>
      <td>755</td>
      <td>758</td>
      <td>760</td>
      <td>761</td>
      <td>761</td>
      <td>761</td>
      <td>761</td>
      <td>761</td>
      <td>761</td>
      <td>762</td>
      <td>762</td>
      <td>762</td>
      <td>762</td>
      <td>762</td>
      <td>763</td>
      <td>763</td>
      <td>763</td>
      <td>763</td>
      <td>764</td>
      <td>764</td>
      <td>764</td>
      <td>765</td>
      <td>844</td>
      <td>851</td>
      <td>852</td>
      <td>852</td>
      <td>852</td>
      <td>852</td>
      <td>852</td>
      <td>852</td>
      <td>852</td>
      <td>852</td>
      <td>853</td>
      <td>853</td>
      <td>853</td>
      <td>853</td>
      <td>854</td>
      <td>854</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>855</td>
      <td>858</td>
      <td>861</td>
      <td>862</td>
      <td>877</td>
      <td>880</td>
      <td>880</td>
      <td>880</td>
      <td>884</td>
      <td>884</td>
      <td>889</td>
      <td>889</td>
      <td>897</td>
      <td>897</td>
      <td>897</td>
      <td>907</td>
      <td>907</td>
      <td>918</td>
      <td>922</td>
      <td>925</td>
      <td>925</td>
      <td>925</td>
      <td>937</td>
      <td>939</td>
      <td>939</td>
      <td>944</td>
      <td>955</td>
      <td>955</td>
      <td>955</td>
      <td>963</td>
      <td>963</td>
      <td>977</td>
      <td>981</td>
      <td>989</td>
      <td>989</td>
      <td>989</td>
      <td>1005</td>
      <td>1005</td>
      <td>1024</td>
      <td>1024</td>
      <td>1045</td>
      <td>1045</td>
      <td>1045</td>
      <td>1060</td>
      <td>1060</td>
      <td>1098</td>
      <td>1098</td>
      <td>1124</td>
      <td>1124</td>
      <td>1124</td>
      <td>1176</td>
      <td>1184</td>
      <td>1199</td>
      <td>1199</td>
      <td>1215</td>
      <td>1215</td>
      <td>1215</td>
      <td>1261</td>
      <td>1261</td>
      <td>1301</td>
      <td>1301</td>
      <td>1344</td>
      <td>1344</td>
      <td>1344</td>
      <td>1438</td>
      <td>1438</td>
      <td>1483</td>
      <td>1483</td>
      <td>1564</td>
      <td>1564</td>
      <td>1564</td>
      <td>1681</td>
      <td>1681</td>
      <td>1753</td>
      <td>1753</td>
      <td>1836</td>
      <td>1836</td>
      <td>1836</td>
      <td>1966</td>
      <td>1966</td>
      <td>2050</td>
      <td>2050</td>
      <td>2110</td>
      <td>2110</td>
      <td>2110</td>
      <td>2370</td>
      <td>2370</td>
      <td>2568</td>
      <td>2568</td>
      <td>2696</td>
      <td>2696</td>
      <td>2696</td>
      <td>2995</td>
      <td>2995</td>
      <td>3190</td>
      <td>3190</td>
      <td>3377</td>
      <td>3377</td>
      <td>3377</td>
      <td>3623</td>
      <td>3623</td>
      <td>3811</td>
      <td>3811</td>
      <td>4038</td>
      <td>4038</td>
      <td>4038</td>
      <td>4325</td>
      <td>4410</td>
      <td>4517</td>
      <td>4567</td>
      <td>4665</td>
      <td>4756</td>
      <td>4825</td>
      <td>4888</td>
      <td>4910</td>
      <td>5045</td>
      <td>5135</td>
      <td>5135</td>
      <td>5319</td>
      <td>5383</td>
      <td>5437</td>
      <td>5477</td>
      <td>5567</td>
      <td>5616</td>
      <td>5725</td>
      <td>5725</td>
      <td>5872</td>
      <td>5914</td>
      <td>5951</td>
      <td>6018</td>
      <td>6066</td>
      <td>6142</td>
      <td>6207</td>
      <td>6256</td>
      <td>6304</td>
      <td>6351</td>
      <td>6428</td>
      <td>6534</td>
      <td>6610</td>
      <td>6610</td>
      <td>6712</td>
      <td>6745</td>
      <td>6790</td>
      <td>6842</td>
      <td>6904</td>
      <td>6955</td>
      <td>7005</td>
      <td>7050</td>
      <td>7084</td>
      <td>7127</td>
      <td>7162</td>
      <td>7190</td>
      <td>7236</td>
      <td>7288</td>
      <td>7338</td>
      <td>7382</td>
      <td>7382</td>
      <td>7446</td>
      <td>7466</td>
      <td>7519</td>
      <td>7560</td>
      <td>7577</td>
      <td>7602</td>
      <td>7633</td>
      <td>7669</td>
      <td>7699</td>
      <td>7756</td>
      <td>7806</td>
      <td>7821</td>
      <td>7875</td>
      <td>7919</td>
      <td>7983</td>
      <td>8049</td>
      <td>8117</td>
      <td>8166</td>
      <td>8192</td>
      <td>8249</td>
      <td>8308</td>
      <td>8348</td>
      <td>8348</td>
      <td>8489</td>
      <td>8586</td>
      <td>8586</td>
      <td>8586</td>
      <td>8682</td>
      <td>8818</td>
      <td>8868</td>
      <td>8946</td>
      <td>9038</td>
      <td>9083</td>
      <td>9083</td>
      <td>9194</td>
      <td>9308</td>
      <td>9379</td>
      <td>9416</td>
      <td>9499</td>
      <td>9549</td>
      <td>9596</td>
      <td>9638</td>
      <td>9716</td>
      <td>9779</td>
      <td>9837</td>
      <td>9885</td>
      <td>9937</td>
      <td>9972</td>
      <td>10017</td>
      <td>10070</td>
      <td>10137</td>
      <td>10172</td>
      <td>10206</td>
      <td>10251</td>
      <td>10275</td>
      <td>10312</td>
      <td>10352</td>
      <td>10391</td>
      <td>10427</td>
      <td>10463</td>
      <td>10503</td>
      <td>10538</td>
      <td>10555</td>
      <td>10583</td>
      <td>10610</td>
      <td>10645</td>
      <td>10672</td>
      <td>10699</td>
      <td>10712</td>
      <td>10739</td>
      <td>10775</td>
      <td>10799</td>
      <td>10822</td>
      <td>10849</td>
      <td>10866</td>
      <td>10889</td>
      <td>10908</td>
      <td>10948</td>
      <td>10976</td>
      <td>10998</td>
      <td>11019</td>
      <td>11042</td>
      <td>11069</td>
      <td>11089</td>
      <td>11130</td>
      <td>11130</td>
      <td>11199</td>
      <td>11228</td>
      <td>11266</td>
      <td>11289</td>
      <td>11319</td>
      <td>11360</td>
      <td>11393</td>
      <td>11431</td>
      <td>11481</td>
      <td>11517</td>
      <td>11545</td>
      <td>11591</td>
      <td>11638</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>NaN</td>
      <td>-11.20270</td>
      <td>17.873900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>10</td>
      <td>14</td>
      <td>16</td>
      <td>17</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>26</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>30</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>43</td>
      <td>43</td>
      <td>45</td>
      <td>45</td>
      <td>45</td>
      <td>45</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>50</td>
      <td>52</td>
      <td>52</td>
      <td>58</td>
      <td>60</td>
      <td>61</td>
      <td>69</td>
      <td>70</td>
      <td>70</td>
      <td>71</td>
      <td>74</td>
      <td>81</td>
      <td>84</td>
      <td>86</td>
      <td>86</td>
      <td>86</td>
      <td>86</td>
      <td>86</td>
      <td>86</td>
      <td>88</td>
      <td>91</td>
      <td>92</td>
      <td>96</td>
      <td>113</td>
      <td>118</td>
      <td>130</td>
      <td>138</td>
      <td>140</td>
      <td>142</td>
      <td>148</td>
      <td>155</td>
      <td>166</td>
      <td>172</td>
      <td>176</td>
      <td>183</td>
      <td>186</td>
      <td>189</td>
      <td>197</td>
      <td>212</td>
      <td>212</td>
      <td>259</td>
      <td>267</td>
      <td>276</td>
      <td>284</td>
      <td>291</td>
      <td>315</td>
      <td>328</td>
      <td>346</td>
      <td>346</td>
      <td>346</td>
      <td>386</td>
      <td>386</td>
      <td>396</td>
      <td>458</td>
      <td>462</td>
      <td>506</td>
      <td>525</td>
      <td>541</td>
      <td>576</td>
      <td>607</td>
      <td>638</td>
      <td>687</td>
      <td>705</td>
      <td>749</td>
      <td>779</td>
      <td>812</td>
      <td>851</td>
      <td>880</td>
      <td>916</td>
      <td>932</td>
      <td>950</td>
      <td>1000</td>
      <td>1078</td>
      <td>1109</td>
      <td>1148</td>
      <td>1164</td>
      <td>1199</td>
      <td>1280</td>
      <td>1344</td>
      <td>1395</td>
      <td>1483</td>
      <td>1538</td>
      <td>1572</td>
      <td>1672</td>
      <td>1679</td>
      <td>1735</td>
      <td>1762</td>
      <td>1815</td>
      <td>1852</td>
      <td>1879</td>
      <td>1906</td>
      <td>1935</td>
      <td>1966</td>
      <td>2015</td>
      <td>2044</td>
      <td>2068</td>
      <td>2134</td>
      <td>2171</td>
      <td>2222</td>
      <td>2283</td>
      <td>2332</td>
      <td>2415</td>
      <td>2471</td>
      <td>2551</td>
      <td>2624</td>
      <td>2654</td>
      <td>2729</td>
      <td>2777</td>
      <td>2805</td>
      <td>2876</td>
      <td>2935</td>
      <td>2965</td>
      <td>2981</td>
      <td>3033</td>
      <td>3092</td>
      <td>3217</td>
      <td>3279</td>
      <td>3335</td>
      <td>3388</td>
      <td>3439</td>
      <td>3569</td>
      <td>3675</td>
      <td>3789</td>
      <td>3848</td>
      <td>3901</td>
      <td>3991</td>
      <td>4117</td>
      <td>4236</td>
      <td>4363</td>
      <td>4475</td>
      <td>4590</td>
      <td>4672</td>
      <td>4718</td>
      <td>4797</td>
      <td>4905</td>
      <td>4972</td>
      <td>5114</td>
      <td>5211</td>
      <td>5370</td>
      <td>5402</td>
      <td>5530</td>
      <td>5725</td>
      <td>5725</td>
      <td>5958</td>
      <td>6031</td>
      <td>6246</td>
      <td>6366</td>
      <td>6488</td>
      <td>6680</td>
      <td>6846</td>
      <td>7096</td>
      <td>7222</td>
      <td>7462</td>
      <td>7622</td>
      <td>7829</td>
      <td>8049</td>
      <td>8338</td>
      <td>8582</td>
      <td>8829</td>
      <td>9026</td>
      <td>9381</td>
      <td>9644</td>
      <td>9871</td>
      <td>10074</td>
      <td>10269</td>
      <td>10558</td>
      <td>10805</td>
      <td>11035</td>
      <td>11228</td>
      <td>11577</td>
      <td>11813</td>
      <td>12102</td>
      <td>12223</td>
      <td>12335</td>
      <td>12433</td>
      <td>12680</td>
      <td>12816</td>
      <td>12953</td>
      <td>13053</td>
      <td>13228</td>
      <td>13374</td>
      <td>13451</td>
      <td>13615</td>
      <td>13818</td>
      <td>13922</td>
      <td>14134</td>
      <td>14267</td>
      <td>14413</td>
      <td>14493</td>
      <td>14634</td>
      <td>14742</td>
      <td>14821</td>
      <td>14920</td>
      <td>15008</td>
      <td>15087</td>
      <td>15103</td>
      <td>15139</td>
      <td>15251</td>
      <td>15319</td>
      <td>15361</td>
      <td>15493</td>
      <td>15536</td>
      <td>15591</td>
      <td>15648</td>
      <td>15729</td>
      <td>15804</td>
      <td>15925</td>
      <td>16061</td>
      <td>16161</td>
      <td>16188</td>
      <td>16277</td>
      <td>16362</td>
      <td>16407</td>
      <td>16484</td>
      <td>16562</td>
      <td>16626</td>
      <td>16644</td>
      <td>16686</td>
      <td>16802</td>
      <td>16931</td>
      <td>17029</td>
      <td>17099</td>
      <td>17149</td>
      <td>17240</td>
      <td>17296</td>
      <td>17371</td>
      <td>17433</td>
      <td>17553</td>
      <td>17568</td>
      <td>17608</td>
      <td>17642</td>
      <td>17684</td>
      <td>17756</td>
      <td>17864</td>
      <td>17974</td>
      <td>18066</td>
      <td>18156</td>
      <td>18193</td>
      <td>18254</td>
      <td>18343</td>
      <td>18425</td>
      <td>18613</td>
      <td>18679</td>
      <td>18765</td>
      <td>18875</td>
      <td>18926</td>
      <td>19011</td>
      <td>19093</td>
      <td>19177</td>
      <td>19269</td>
      <td>19367</td>
      <td>19399</td>
      <td>19476</td>
      <td>19553</td>
      <td>19580</td>
      <td>19672</td>
      <td>19723</td>
      <td>19782</td>
      <td>19796</td>
      <td>19829</td>
      <td>19900</td>
      <td>19937</td>
      <td>19996</td>
      <td>20030</td>
      <td>20062</td>
      <td>20086</td>
      <td>20112</td>
      <td>20163</td>
      <td>20210</td>
      <td>20261</td>
      <td>20294</td>
      <td>20329</td>
      <td>20366</td>
      <td>20381</td>
      <td>20389</td>
      <td>20400</td>
      <td>20452</td>
      <td>20478</td>
      <td>20499</td>
      <td>20519</td>
      <td>20548</td>
      <td>20584</td>
      <td>20640</td>
      <td>20695</td>
      <td>20759</td>
      <td>20782</td>
      <td>20807</td>
      <td>20854</td>
      <td>20882</td>
      <td>20923</td>
      <td>20981</td>
      <td>21026</td>
      <td>21055</td>
      <td>21086</td>
      <td>21108</td>
      <td>21114</td>
      <td>21161</td>
      <td>21205</td>
      <td>21265</td>
      <td>21323</td>
      <td>21380</td>
      <td>21407</td>
      <td>21446</td>
      <td>21489</td>
      <td>21558</td>
      <td>21642</td>
      <td>21696</td>
      <td>21733</td>
      <td>21757</td>
      <td>21774</td>
      <td>21836</td>
    </tr>
  </tbody>
</table>

</div>



## 데이터 구조 변경, 시각화

- 위-경도 사용하지 않고 큰 나라의 경우 지방이 많아 나뉘어져 있는데 묶어서 분석하는것이 용이할 듯.
- date columns이 단순 문자열로 입력되어 있는데 이를 datetime 객체로 변환하여 date가 index가 되도록 변경.

### 구조변경


```python
df_case['Country/Region'].value_counts()
```




    China             33
    Canada            16
    United Kingdom    12
    France            12
    Australia          8
                      ..
    Ethiopia           1
    Montenegro         1
    Uzbekistan         1
    Nicaragua          1
    Chad               1
    Name: Country/Region, Length: 192, dtype: int64




```python
def fix_dataframe(df):
    df = df.drop(['Lat', 'Long'], axis=1).groupby('Country/Region').sum()
# 나라 기준으로 묶고 Province당 경우를 합하여 표현한다.

    df = df.transpose()
# 날짜가 인덱스가 되게끔 바꾸어준다.

    df.index.name = 'Date'
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].apply(lambda s : pd.to_datetime(str(s)))

    df.set_index('Date', inplace=True)
    return df
```

- 데이터 프레임 구조를 바꾸고 date를 datetime 타입으로 바꾸어 인덱스로 만드는 함수를 만들어준다.


```python
df_case = fix_dataframe(df_case)
df_death = fix_dataframe(df_death)
```

### 시각화


```python
# 확진자 상위 10개국

ten_cases = df_case.loc[df_case.index[-1]].sort_values(ascending=False)[:10]
ten_cases
```




    Country/Region
    US                30011839
    Brazil            12220011
    India             11787534
    Russia             4433364
    France             4374774
    United Kingdom     4326645
    Italy              3440862
    Spain              3234319
    Turkey             3091282
    Germany            2722988
    Name: 2021-03-24 00:00:00, dtype: int64




```python
sns.barplot(x=ten_cases.index, y=ten_cases)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x211e06a8288>




![output_14_1](https://user-images.githubusercontent.com/77723966/112796291-bb84a200-90a4-11eb-9d7c-1f1da2b84645.png)




```python
ten_death = df_death.loc[df_death.index[-1]][ten_cases.index] 
#sort_values 말고 확진자 10개국의 사망자를 확인해보자.
ten_death
```




    Country/Region
    US                545264
    Brazil            300685
    India             160692
    Russia             94624
    France             93083
    United Kingdom    126621
    Italy             106339
    Spain              73744
    Turkey             30462
    Germany            75484
    Name: 2021-03-24 00:00:00, dtype: int64




```python
# 확진자와 사망자를 동시에 보여주기.
plt.figure(figsize=(7,5))

sns.barplot(x=ten_cases.index, y=ten_cases, color='black')
plt.xticks(rotation=90, size=15)
plt.ylabel('Total Confirmed Cases', size=15)
plt.xlabel('')
plt.title('Total Confirmed Cases (%s)' %ten_cases.name.strftime('%Y-%m-%d'), size=15)

ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(ten_death.index, ten_death, 'r--')
ax2.set_ylabel('Total Deaths', color='red', size=15)
plt.show()
```


![output_16_0](https://user-images.githubusercontent.com/77723966/112796303-c0495600-90a4-11eb-8806-abdcf42a447e.png)




```python
# 미국만 알아보자.
plt.figure(figsize=(10,7))

plt.plot(df_case.index, df_case['US'], 'b-')
plt.ylabel('Confirmed Cases', color='blue')
plt.xlabel('Date')
plt.xlim(right=df_case.index[-1])
plt.ylim(0, df_case['US'].max()*1.1)
plt.title('US' + 'Cases & Deaths')

ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(df_death.index, df_death['US'], 'r--')
ax2.set_ylabel('Deaths', color='red')
ax2.set_ylim(0, df_death['US'].max()*1.3)

plt.show()
```


![output_17_0](https://user-images.githubusercontent.com/77723966/112796314-c3444680-90a4-11eb-9fd3-2e3d9e720386.png)




```python
# 재미로 일본도

plt.figure(figsize=(6,4))

plt.plot(df_case.index, df_case['Japan'], 'b-')
plt.ylabel('Confirmed Cases', color='blue')
plt.xlabel('Date')
plt.xlim(right=df_case.index[-1])
plt.ylim(0, df_case['Japan'].max()*1.1)
plt.title('US' + 'Cases & Deaths')

ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(df_death.index, df_death['Japan'], 'r--')
ax2.set_ylabel('Deaths', color='red')
ax2.set_ylim(0, df_death['Japan'].max()*1.3)

plt.show()
```


![output_18_0](https://user-images.githubusercontent.com/77723966/112796363-d6efad00-90a4-11eb-9862-cf0bcd25d670.png)




```python
# 한국
plt.figure(figsize=(6,4))

plt.plot(df_case.index, df_case['Korea, South'], 'b-')
plt.ylabel('Confirmed Cases', color='blue')
plt.xlabel('Date')
plt.xlim(right=df_case.index[-1])
plt.ylim(0, df_case['Korea, South'].max()*1.1)
plt.title('US' + 'Cases & Deaths')

ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(df_death.index, df_death['Korea, South'], 'r--')
ax2.set_ylabel('Deaths', color='red')
ax2.set_ylim(0, df_death['Korea, South'].max()*1.3)

plt.show()
```


![output_19_0](https://user-images.githubusercontent.com/77723966/112796369-d9520700-90a4-11eb-87d8-d0d91f878ba1.png)




```python
# 누적 말고 일일데이터는?
plt.plot(df_case.index, df_case['Korea, South'].diff(), 'b-')
plt.ylabel('Confirmed Cases', color='blue')
plt.xlabel('Date')
plt.xlim(right=df_case.index[-1])
plt.ylim(bottom = 0)
plt.title('US' + 'Cases & Deaths')

ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(df_death.index, df_death['Korea, South'].diff(), 'r--')
ax2.set_ylabel('Deaths', color='red')
ax2.set_ylim(bottom=0)

plt.show()
```


![output_20_0](https://user-images.githubusercontent.com/77723966/112796377-dce58e00-90a4-11eb-9337-c6c81a7e52d4.png)



- diff()만 붙히면 해결되는데, diff는 기준과 그 전의 차이를 나타내고, df_case[나라]가 누적확진자이므로 일일확진자로 표현할 수 있다.
- 작년 3월 대구 사건이 2번째로 영향이 크다. 사망자는 당연하게도 확진자가 급증한 이후에 간격을두고 역시 늘어났다.

## 전처리

- FBProhphet 사용하기 위해 전처리
- FBProhphet 사용은 공식 레퍼런스를 참조 함 : https://facebook.github.io/prophet/docs/quick_start.html#python-api


```python
df_case.reset_index()[['Date', 'Korea, South']]
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
      <th>Country/Region</th>
      <th>Date</th>
      <th>Korea, South</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-25</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-26</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>423</th>
      <td>2021-03-20</td>
      <td>98665</td>
    </tr>
    <tr>
      <th>424</th>
      <td>2021-03-21</td>
      <td>99075</td>
    </tr>
    <tr>
      <th>425</th>
      <td>2021-03-22</td>
      <td>99421</td>
    </tr>
    <tr>
      <th>426</th>
      <td>2021-03-23</td>
      <td>99846</td>
    </tr>
    <tr>
      <th>427</th>
      <td>2021-03-24</td>
      <td>100276</td>
    </tr>
  </tbody>
</table>
<p>428 rows × 2 columns</p>

</div>




```python
df = pd.DataFrame(df_case.reset_index()[['Date', 'Korea, South']].to_numpy(), columns=['ds', 'y'])
df
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-25</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-26</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>423</th>
      <td>2021-03-20</td>
      <td>98665</td>
    </tr>
    <tr>
      <th>424</th>
      <td>2021-03-21</td>
      <td>99075</td>
    </tr>
    <tr>
      <th>425</th>
      <td>2021-03-22</td>
      <td>99421</td>
    </tr>
    <tr>
      <th>426</th>
      <td>2021-03-23</td>
      <td>99846</td>
    </tr>
    <tr>
      <th>427</th>
      <td>2021-03-24</td>
      <td>100276</td>
    </tr>
  </tbody>
</table>
<p>428 rows × 2 columns</p>

</div>




```python
from math import floor

def train_test_split_df(df, test_size):
    div = floor(df.shape[0] * (1 - test_size))
    return df.loc[:div], df.loc[div + 1:]

train_df, test_df = train_test_split_df(df, 0.1)
```


```python
train_df.shape
```




    (386, 2)




```python
train_df.tail()
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>381</th>
      <td>2021-02-06</td>
      <td>80896</td>
    </tr>
    <tr>
      <th>382</th>
      <td>2021-02-07</td>
      <td>81185</td>
    </tr>
    <tr>
      <th>383</th>
      <td>2021-02-08</td>
      <td>81487</td>
    </tr>
    <tr>
      <th>384</th>
      <td>2021-02-09</td>
      <td>81930</td>
    </tr>
    <tr>
      <th>385</th>
      <td>2021-02-10</td>
      <td>82434</td>
    </tr>
  </tbody>
</table>

</div>




```python
test_df.head()
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>386</th>
      <td>2021-02-11</td>
      <td>82837</td>
    </tr>
    <tr>
      <th>387</th>
      <td>2021-02-12</td>
      <td>83199</td>
    </tr>
    <tr>
      <th>388</th>
      <td>2021-02-13</td>
      <td>83525</td>
    </tr>
    <tr>
      <th>389</th>
      <td>2021-02-14</td>
      <td>83869</td>
    </tr>
    <tr>
      <th>390</th>
      <td>2021-02-15</td>
      <td>84325</td>
    </tr>
  </tbody>
</table>

</div>



## 모델 학습(Prophet)


```python
from fbprophet import Prophet

model = Prophet(changepoint_range = 1.0)
model.fit(train_df)
```

    INFO:fbprophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    C:\Users\dissi\anaconda31\lib\site-packages\pystan\misc.py:399: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      elif np.issubdtype(np.asarray(v).dtype, float):





    <fbprophet.forecaster.Prophet at 0x211e324c848>




```python
from fbprophet.plot import add_changepoints_to_plot
```


```python
pred = model.predict(test_df)
model.plot(pred);
```


![output_33_0](https://user-images.githubusercontent.com/77723966/112796426-ebcc4080-90a4-11eb-8120-a8780e457c80.png)



```python
model.plot_components(pred);
# 주마다 경향
```


![output_34_0](https://user-images.githubusercontent.com/77723966/112796433-ee2e9a80-90a4-11eb-89ee-9003f5a01034.png)




```python
fig = model.plot(pred)
plt.plot(test_df['ds'], test_df['y'], 'g-', label='actual')
add_changepoints_to_plot(fig.gca(), model, pred)
plt.legend()
```




    <matplotlib.legend.Legend at 0x211e346c048>




![output_35_1](https://user-images.githubusercontent.com/77723966/112796446-f1c22180-90a4-11eb-871d-fae6120813d3.png)



- 실제랑 거의 동일
- 실제 양상이 변칙적인 상황없이 예상대로 흘러감을 뜻함.

## 모델평가


```python
from sklearn.metrics import r2_score
print('R2 Score : ', r2_score(test_df['y'], pred['yhat']))
```

    R2 Score :  0.9928405923770204


## 앞으로 30일 예측


```python
model = Prophet(changepoint_range=1.0)
model.fit(df)
future = model.make_future_dataframe(30)
pred = model.predict(future)
model.plot(pred);
```

    INFO:fbprophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    C:\Users\dissi\anaconda31\lib\site-packages\pystan\misc.py:399: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      elif np.issubdtype(np.asarray(v).dtype, float):



![output_40_1](https://user-images.githubusercontent.com/77723966/112796477-fbe42000-90a4-11eb-967e-5ac6d5c23256.png)



### 과거 데이터

- 몇 가지 특수한 사건을 통해 코로나가 급증하게 되었는데 만약 이 사건이 벌어지지 않았다면 지금쯤 어떤 양상을 보일지 분석해보고자 한다.


```python
df.loc[26:32]

# 2월 20일 기준으로 급격히 늘어남.
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>2020-02-17</td>
      <td>30</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2020-02-18</td>
      <td>31</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2020-02-19</td>
      <td>31</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2020-02-20</td>
      <td>104</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2020-02-21</td>
      <td>204</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2020-02-22</td>
      <td>433</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020-02-23</td>
      <td>602</td>
    </tr>
  </tbody>
</table>

</div>




```python
model = Prophet(changepoint_range=1.0)
model.fit(df.loc[:28])
future2 = model.make_future_dataframe(30)
pred = model.predict(future2)
model.plot(pred);

plt.plot(df.loc[:58]['ds'], df.loc[:58]['y'], 'g-')
plt.show()
```

    INFO:fbprophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    C:\Users\dissi\anaconda31\lib\site-packages\pystan\misc.py:399: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      elif np.issubdtype(np.asarray(v).dtype, float):



![output_44_1](https://user-images.githubusercontent.com/77723966/112796487-01416a80-90a5-11eb-9e60-bab7b60c2f9f.png)



- 극단적인 차이가 남.
- 작은사건 하나가 미치는 큰 영향을 확인할 수 있으며, 모든걸 통제하고 예측해야하는 방역 체계가 얼마나 힘든지 알 수 있다.
- 모두 조심한다고 해도 조그만 사건을 통해 이런 무서운 결과가 발생할 수 있음은 방역에 대한 매너리즘에서 빠져나올 수 있게 해준다.
