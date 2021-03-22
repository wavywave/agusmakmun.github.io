---
layout: post
title:  "Kaggle - League of Legends"
description: "캐글 데이터 분석"
author: SeungRok OH
categories: [Kaggle]
---


# League of Legends

## 라이브러리 설정, 데이터 읽어들이기

- 데이터 셋 : https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min

- League of Legends 게임 시작 10분이내 일어나는 여러 상황을 통해 승패를 예측하는 데이터 셋이다.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('high_diamond_ranked_10min.csv')
```


```python
pd.set_option('display.max_columns', None)
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
      <th>gameId</th>
      <th>blueWins</th>
      <th>blueWardsPlaced</th>
      <th>blueWardsDestroyed</th>
      <th>blueFirstBlood</th>
      <th>blueKills</th>
      <th>blueDeaths</th>
      <th>blueAssists</th>
      <th>blueEliteMonsters</th>
      <th>blueDragons</th>
      <th>blueHeralds</th>
      <th>blueTowersDestroyed</th>
      <th>blueTotalGold</th>
      <th>blueAvgLevel</th>
      <th>blueTotalExperience</th>
      <th>blueTotalMinionsKilled</th>
      <th>blueTotalJungleMinionsKilled</th>
      <th>blueGoldDiff</th>
      <th>blueExperienceDiff</th>
      <th>blueCSPerMin</th>
      <th>blueGoldPerMin</th>
      <th>redWardsPlaced</th>
      <th>redWardsDestroyed</th>
      <th>redFirstBlood</th>
      <th>redKills</th>
      <th>redDeaths</th>
      <th>redAssists</th>
      <th>redEliteMonsters</th>
      <th>redDragons</th>
      <th>redHeralds</th>
      <th>redTowersDestroyed</th>
      <th>redTotalGold</th>
      <th>redAvgLevel</th>
      <th>redTotalExperience</th>
      <th>redTotalMinionsKilled</th>
      <th>redTotalJungleMinionsKilled</th>
      <th>redGoldDiff</th>
      <th>redExperienceDiff</th>
      <th>redCSPerMin</th>
      <th>redGoldPerMin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4555</th>
      <td>4445985804</td>
      <td>1</td>
      <td>19</td>
      <td>4</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>17459</td>
      <td>7.2</td>
      <td>19021</td>
      <td>231</td>
      <td>64</td>
      <td>2929</td>
      <td>1296</td>
      <td>23.1</td>
      <td>1745.9</td>
      <td>15</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14530</td>
      <td>7.0</td>
      <td>17725</td>
      <td>191</td>
      <td>64</td>
      <td>-2929</td>
      <td>-1296</td>
      <td>19.1</td>
      <td>1453.0</td>
    </tr>
    <tr>
      <th>3756</th>
      <td>4505375338</td>
      <td>1</td>
      <td>21</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>16709</td>
      <td>7.0</td>
      <td>18855</td>
      <td>222</td>
      <td>68</td>
      <td>728</td>
      <td>2095</td>
      <td>22.2</td>
      <td>1670.9</td>
      <td>15</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15981</td>
      <td>6.8</td>
      <td>16760</td>
      <td>224</td>
      <td>44</td>
      <td>-728</td>
      <td>-2095</td>
      <td>22.4</td>
      <td>1598.1</td>
    </tr>
    <tr>
      <th>5222</th>
      <td>4495636315</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17262</td>
      <td>7.0</td>
      <td>18608</td>
      <td>248</td>
      <td>52</td>
      <td>1673</td>
      <td>1200</td>
      <td>24.8</td>
      <td>1726.2</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>15589</td>
      <td>6.8</td>
      <td>17408</td>
      <td>197</td>
      <td>69</td>
      <td>-1673</td>
      <td>-1200</td>
      <td>19.7</td>
      <td>1558.9</td>
    </tr>
    <tr>
      <th>2104</th>
      <td>4490390200</td>
      <td>1</td>
      <td>13</td>
      <td>3</td>
      <td>0</td>
      <td>7</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>17247</td>
      <td>7.2</td>
      <td>19189</td>
      <td>249</td>
      <td>63</td>
      <td>2152</td>
      <td>2575</td>
      <td>24.9</td>
      <td>1724.7</td>
      <td>17</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15095</td>
      <td>6.8</td>
      <td>16614</td>
      <td>178</td>
      <td>40</td>
      <td>-2152</td>
      <td>-2575</td>
      <td>17.8</td>
      <td>1509.5</td>
    </tr>
    <tr>
      <th>2400</th>
      <td>4381068813</td>
      <td>0</td>
      <td>42</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>16146</td>
      <td>6.6</td>
      <td>16204</td>
      <td>196</td>
      <td>43</td>
      <td>2037</td>
      <td>-281</td>
      <td>19.6</td>
      <td>1614.6</td>
      <td>19</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14109</td>
      <td>6.4</td>
      <td>16485</td>
      <td>172</td>
      <td>57</td>
      <td>-2037</td>
      <td>281</td>
      <td>17.2</td>
      <td>1410.9</td>
    </tr>
    <tr>
      <th>3512</th>
      <td>4419275391</td>
      <td>0</td>
      <td>15</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15856</td>
      <td>7.0</td>
      <td>17723</td>
      <td>224</td>
      <td>47</td>
      <td>-460</td>
      <td>60</td>
      <td>22.4</td>
      <td>1585.6</td>
      <td>16</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>16316</td>
      <td>6.8</td>
      <td>17663</td>
      <td>238</td>
      <td>36</td>
      <td>460</td>
      <td>-60</td>
      <td>23.8</td>
      <td>1631.6</td>
    </tr>
    <tr>
      <th>5749</th>
      <td>4464945993</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>7</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>17898</td>
      <td>7.4</td>
      <td>19411</td>
      <td>232</td>
      <td>58</td>
      <td>942</td>
      <td>1300</td>
      <td>23.2</td>
      <td>1789.8</td>
      <td>16</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16956</td>
      <td>7.0</td>
      <td>18111</td>
      <td>245</td>
      <td>43</td>
      <td>-942</td>
      <td>-1300</td>
      <td>24.5</td>
      <td>1695.6</td>
    </tr>
    <tr>
      <th>9082</th>
      <td>4518444031</td>
      <td>0</td>
      <td>16</td>
      <td>5</td>
      <td>1</td>
      <td>10</td>
      <td>8</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>17357</td>
      <td>7.0</td>
      <td>18388</td>
      <td>191</td>
      <td>59</td>
      <td>-310</td>
      <td>570</td>
      <td>19.1</td>
      <td>1735.7</td>
      <td>27</td>
      <td>2</td>
      <td>0</td>
      <td>8</td>
      <td>10</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>17667</td>
      <td>7.2</td>
      <td>17818</td>
      <td>194</td>
      <td>56</td>
      <td>310</td>
      <td>-570</td>
      <td>19.4</td>
      <td>1766.7</td>
    </tr>
    <tr>
      <th>7260</th>
      <td>4479460532</td>
      <td>1</td>
      <td>21</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>14850</td>
      <td>6.8</td>
      <td>17967</td>
      <td>226</td>
      <td>57</td>
      <td>-858</td>
      <td>847</td>
      <td>22.6</td>
      <td>1485.0</td>
      <td>21</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15708</td>
      <td>6.8</td>
      <td>17120</td>
      <td>239</td>
      <td>44</td>
      <td>858</td>
      <td>-847</td>
      <td>23.9</td>
      <td>1570.8</td>
    </tr>
    <tr>
      <th>4272</th>
      <td>4504894173</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16763</td>
      <td>6.6</td>
      <td>17181</td>
      <td>209</td>
      <td>36</td>
      <td>3371</td>
      <td>1173</td>
      <td>20.9</td>
      <td>1676.3</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13392</td>
      <td>6.4</td>
      <td>16008</td>
      <td>195</td>
      <td>50</td>
      <td>-3371</td>
      <td>-1173</td>
      <td>19.5</td>
      <td>1339.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9879 entries, 0 to 9878
    Data columns (total 40 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   gameId                        9879 non-null   int64  
     1   blueWins                      9879 non-null   int64  
     2   blueWardsPlaced               9879 non-null   int64  
     3   blueWardsDestroyed            9879 non-null   int64  
     4   blueFirstBlood                9879 non-null   int64  
     5   blueKills                     9879 non-null   int64  
     6   blueDeaths                    9879 non-null   int64  
     7   blueAssists                   9879 non-null   int64  
     8   blueEliteMonsters             9879 non-null   int64  
     9   blueDragons                   9879 non-null   int64  
     10  blueHeralds                   9879 non-null   int64  
     11  blueTowersDestroyed           9879 non-null   int64  
     12  blueTotalGold                 9879 non-null   int64  
     13  blueAvgLevel                  9879 non-null   float64
     14  blueTotalExperience           9879 non-null   int64  
     15  blueTotalMinionsKilled        9879 non-null   int64  
     16  blueTotalJungleMinionsKilled  9879 non-null   int64  
     17  blueGoldDiff                  9879 non-null   int64  
     18  blueExperienceDiff            9879 non-null   int64  
     19  blueCSPerMin                  9879 non-null   float64
     20  blueGoldPerMin                9879 non-null   float64
     21  redWardsPlaced                9879 non-null   int64  
     22  redWardsDestroyed             9879 non-null   int64  
     23  redFirstBlood                 9879 non-null   int64  
     24  redKills                      9879 non-null   int64  
     25  redDeaths                     9879 non-null   int64  
     26  redAssists                    9879 non-null   int64  
     27  redEliteMonsters              9879 non-null   int64  
     28  redDragons                    9879 non-null   int64  
     29  redHeralds                    9879 non-null   int64  
     30  redTowersDestroyed            9879 non-null   int64  
     31  redTotalGold                  9879 non-null   int64  
     32  redAvgLevel                   9879 non-null   float64
     33  redTotalExperience            9879 non-null   int64  
     34  redTotalMinionsKilled         9879 non-null   int64  
     35  redTotalJungleMinionsKilled   9879 non-null   int64  
     36  redGoldDiff                   9879 non-null   int64  
     37  redExperienceDiff             9879 non-null   int64  
     38  redCSPerMin                   9879 non-null   float64
     39  redGoldPerMin                 9879 non-null   float64
    dtypes: float64(6), int64(34)
    memory usage: 3.0 MB
    

## EDA 및 기초 통계 분석


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
      <th>gameId</th>
      <th>blueWins</th>
      <th>blueWardsPlaced</th>
      <th>blueWardsDestroyed</th>
      <th>blueFirstBlood</th>
      <th>blueKills</th>
      <th>blueDeaths</th>
      <th>blueAssists</th>
      <th>blueEliteMonsters</th>
      <th>blueDragons</th>
      <th>blueHeralds</th>
      <th>blueTowersDestroyed</th>
      <th>blueTotalGold</th>
      <th>blueAvgLevel</th>
      <th>blueTotalExperience</th>
      <th>blueTotalMinionsKilled</th>
      <th>blueTotalJungleMinionsKilled</th>
      <th>blueGoldDiff</th>
      <th>blueExperienceDiff</th>
      <th>blueCSPerMin</th>
      <th>blueGoldPerMin</th>
      <th>redWardsPlaced</th>
      <th>redWardsDestroyed</th>
      <th>redFirstBlood</th>
      <th>redKills</th>
      <th>redDeaths</th>
      <th>redAssists</th>
      <th>redEliteMonsters</th>
      <th>redDragons</th>
      <th>redHeralds</th>
      <th>redTowersDestroyed</th>
      <th>redTotalGold</th>
      <th>redAvgLevel</th>
      <th>redTotalExperience</th>
      <th>redTotalMinionsKilled</th>
      <th>redTotalJungleMinionsKilled</th>
      <th>redGoldDiff</th>
      <th>redExperienceDiff</th>
      <th>redCSPerMin</th>
      <th>redGoldPerMin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.879000e+03</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
      <td>9879.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.500084e+09</td>
      <td>0.499038</td>
      <td>22.288288</td>
      <td>2.824881</td>
      <td>0.504808</td>
      <td>6.183925</td>
      <td>6.137666</td>
      <td>6.645106</td>
      <td>0.549954</td>
      <td>0.361980</td>
      <td>0.187974</td>
      <td>0.051422</td>
      <td>16503.455512</td>
      <td>6.916004</td>
      <td>17928.110133</td>
      <td>216.699565</td>
      <td>50.509667</td>
      <td>14.414111</td>
      <td>-33.620306</td>
      <td>21.669956</td>
      <td>1650.345551</td>
      <td>22.367952</td>
      <td>2.723150</td>
      <td>0.495192</td>
      <td>6.137666</td>
      <td>6.183925</td>
      <td>6.662112</td>
      <td>0.573135</td>
      <td>0.413098</td>
      <td>0.160036</td>
      <td>0.043021</td>
      <td>16489.041401</td>
      <td>6.925316</td>
      <td>17961.730438</td>
      <td>217.349226</td>
      <td>51.313088</td>
      <td>-14.414111</td>
      <td>33.620306</td>
      <td>21.734923</td>
      <td>1648.904140</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.757328e+07</td>
      <td>0.500024</td>
      <td>18.019177</td>
      <td>2.174998</td>
      <td>0.500002</td>
      <td>3.011028</td>
      <td>2.933818</td>
      <td>4.064520</td>
      <td>0.625527</td>
      <td>0.480597</td>
      <td>0.390712</td>
      <td>0.244369</td>
      <td>1535.446636</td>
      <td>0.305146</td>
      <td>1200.523764</td>
      <td>21.858437</td>
      <td>9.898282</td>
      <td>2453.349179</td>
      <td>1920.370438</td>
      <td>2.185844</td>
      <td>153.544664</td>
      <td>18.457427</td>
      <td>2.138356</td>
      <td>0.500002</td>
      <td>2.933818</td>
      <td>3.011028</td>
      <td>4.060612</td>
      <td>0.626482</td>
      <td>0.492415</td>
      <td>0.366658</td>
      <td>0.216900</td>
      <td>1490.888406</td>
      <td>0.305311</td>
      <td>1198.583912</td>
      <td>21.911668</td>
      <td>10.027885</td>
      <td>2453.349179</td>
      <td>1920.370438</td>
      <td>2.191167</td>
      <td>149.088841</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.295358e+09</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10730.000000</td>
      <td>4.600000</td>
      <td>10098.000000</td>
      <td>90.000000</td>
      <td>0.000000</td>
      <td>-10830.000000</td>
      <td>-9333.000000</td>
      <td>9.000000</td>
      <td>1073.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11212.000000</td>
      <td>4.800000</td>
      <td>10465.000000</td>
      <td>107.000000</td>
      <td>4.000000</td>
      <td>-11467.000000</td>
      <td>-8348.000000</td>
      <td>10.700000</td>
      <td>1121.200000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.483301e+09</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15415.500000</td>
      <td>6.800000</td>
      <td>17168.000000</td>
      <td>202.000000</td>
      <td>44.000000</td>
      <td>-1585.500000</td>
      <td>-1290.500000</td>
      <td>20.200000</td>
      <td>1541.550000</td>
      <td>14.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15427.500000</td>
      <td>6.800000</td>
      <td>17209.500000</td>
      <td>203.000000</td>
      <td>44.000000</td>
      <td>-1596.000000</td>
      <td>-1212.000000</td>
      <td>20.300000</td>
      <td>1542.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.510920e+09</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16398.000000</td>
      <td>7.000000</td>
      <td>17951.000000</td>
      <td>218.000000</td>
      <td>50.000000</td>
      <td>14.000000</td>
      <td>-28.000000</td>
      <td>21.800000</td>
      <td>1639.800000</td>
      <td>16.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16378.000000</td>
      <td>7.000000</td>
      <td>17974.000000</td>
      <td>218.000000</td>
      <td>51.000000</td>
      <td>-14.000000</td>
      <td>28.000000</td>
      <td>21.800000</td>
      <td>1637.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.521733e+09</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17459.000000</td>
      <td>7.200000</td>
      <td>18724.000000</td>
      <td>232.000000</td>
      <td>56.000000</td>
      <td>1596.000000</td>
      <td>1212.000000</td>
      <td>23.200000</td>
      <td>1745.900000</td>
      <td>20.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17418.500000</td>
      <td>7.200000</td>
      <td>18764.500000</td>
      <td>233.000000</td>
      <td>57.000000</td>
      <td>1585.500000</td>
      <td>1290.500000</td>
      <td>23.300000</td>
      <td>1741.850000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.527991e+09</td>
      <td>1.000000</td>
      <td>250.000000</td>
      <td>27.000000</td>
      <td>1.000000</td>
      <td>22.000000</td>
      <td>22.000000</td>
      <td>29.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>23701.000000</td>
      <td>8.000000</td>
      <td>22224.000000</td>
      <td>283.000000</td>
      <td>92.000000</td>
      <td>11467.000000</td>
      <td>8348.000000</td>
      <td>28.300000</td>
      <td>2370.100000</td>
      <td>276.000000</td>
      <td>24.000000</td>
      <td>1.000000</td>
      <td>22.000000</td>
      <td>22.000000</td>
      <td>28.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>22732.000000</td>
      <td>8.200000</td>
      <td>22269.000000</td>
      <td>289.000000</td>
      <td>92.000000</td>
      <td>10830.000000</td>
      <td>9333.000000</td>
      <td>28.900000</td>
      <td>2273.200000</td>
    </tr>
  </tbody>
</table>
</div>



- 당장 보이는 것은 blue 경우 정글에서 red보다 herald를 죽일 확률이 높고 red는 그 반대.


```python
# correlation 히트맵 시각화

fig = plt.figure(figsize=(4,10))
sns.heatmap(df.corr()[['blueWins']], annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d802aed648>




![output_11_1](https://user-images.githubusercontent.com/77723966/112033656-ffe1e080-8b80-11eb-9c3c-ffac75f35ffc.png)


- 전체 골드 수급량, 골드-경험치 수급이 승리로 이어질 가능성이 높다.


```python
df.corr()
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
      <th>gameId</th>
      <th>blueWins</th>
      <th>blueWardsPlaced</th>
      <th>blueWardsDestroyed</th>
      <th>blueFirstBlood</th>
      <th>blueKills</th>
      <th>blueDeaths</th>
      <th>blueAssists</th>
      <th>blueEliteMonsters</th>
      <th>blueDragons</th>
      <th>blueHeralds</th>
      <th>blueTowersDestroyed</th>
      <th>blueTotalGold</th>
      <th>blueAvgLevel</th>
      <th>blueTotalExperience</th>
      <th>blueTotalMinionsKilled</th>
      <th>blueTotalJungleMinionsKilled</th>
      <th>blueGoldDiff</th>
      <th>blueExperienceDiff</th>
      <th>blueCSPerMin</th>
      <th>blueGoldPerMin</th>
      <th>redWardsPlaced</th>
      <th>redWardsDestroyed</th>
      <th>redFirstBlood</th>
      <th>redKills</th>
      <th>redDeaths</th>
      <th>redAssists</th>
      <th>redEliteMonsters</th>
      <th>redDragons</th>
      <th>redHeralds</th>
      <th>redTowersDestroyed</th>
      <th>redTotalGold</th>
      <th>redAvgLevel</th>
      <th>redTotalExperience</th>
      <th>redTotalMinionsKilled</th>
      <th>redTotalJungleMinionsKilled</th>
      <th>redGoldDiff</th>
      <th>redExperienceDiff</th>
      <th>redCSPerMin</th>
      <th>redGoldPerMin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gameId</th>
      <td>1.000000</td>
      <td>0.000985</td>
      <td>0.005361</td>
      <td>-0.012057</td>
      <td>-0.011577</td>
      <td>-0.038993</td>
      <td>-0.013160</td>
      <td>-0.023329</td>
      <td>0.016599</td>
      <td>0.008962</td>
      <td>0.015551</td>
      <td>-0.007467</td>
      <td>-0.033754</td>
      <td>-0.040956</td>
      <td>-0.040852</td>
      <td>-0.002917</td>
      <td>-0.004193</td>
      <td>-0.014670</td>
      <td>-0.012315</td>
      <td>-0.002917</td>
      <td>-0.033754</td>
      <td>0.007405</td>
      <td>-0.001197</td>
      <td>0.011577</td>
      <td>-0.013160</td>
      <td>-0.038993</td>
      <td>-0.008664</td>
      <td>0.017296</td>
      <td>0.017416</td>
      <td>0.006163</td>
      <td>0.003557</td>
      <td>-0.010622</td>
      <td>-0.012419</td>
      <td>-0.021187</td>
      <td>-0.005118</td>
      <td>0.006040</td>
      <td>0.014670</td>
      <td>0.012315</td>
      <td>-0.005118</td>
      <td>-0.010622</td>
    </tr>
    <tr>
      <th>blueWins</th>
      <td>0.000985</td>
      <td>1.000000</td>
      <td>0.000087</td>
      <td>0.044247</td>
      <td>0.201769</td>
      <td>0.337358</td>
      <td>-0.339297</td>
      <td>0.276685</td>
      <td>0.221944</td>
      <td>0.213768</td>
      <td>0.092385</td>
      <td>0.115566</td>
      <td>0.417213</td>
      <td>0.357820</td>
      <td>0.396141</td>
      <td>0.224909</td>
      <td>0.131445</td>
      <td>0.511119</td>
      <td>0.489558</td>
      <td>0.224909</td>
      <td>0.417213</td>
      <td>-0.023671</td>
      <td>-0.055400</td>
      <td>-0.201769</td>
      <td>-0.339297</td>
      <td>0.337358</td>
      <td>-0.271047</td>
      <td>-0.221551</td>
      <td>-0.209516</td>
      <td>-0.097172</td>
      <td>-0.103696</td>
      <td>-0.411396</td>
      <td>-0.352127</td>
      <td>-0.387588</td>
      <td>-0.212171</td>
      <td>-0.110994</td>
      <td>-0.511119</td>
      <td>-0.489558</td>
      <td>-0.212171</td>
      <td>-0.411396</td>
    </tr>
    <tr>
      <th>blueWardsPlaced</th>
      <td>0.005361</td>
      <td>0.000087</td>
      <td>1.000000</td>
      <td>0.034447</td>
      <td>0.003228</td>
      <td>0.018138</td>
      <td>-0.002612</td>
      <td>0.033217</td>
      <td>0.019892</td>
      <td>0.017676</td>
      <td>0.010104</td>
      <td>0.009462</td>
      <td>0.019725</td>
      <td>0.034349</td>
      <td>0.031719</td>
      <td>-0.033925</td>
      <td>0.010501</td>
      <td>0.015800</td>
      <td>0.027943</td>
      <td>-0.033925</td>
      <td>0.019725</td>
      <td>-0.012906</td>
      <td>0.115549</td>
      <td>-0.003228</td>
      <td>-0.002612</td>
      <td>0.018138</td>
      <td>-0.009009</td>
      <td>-0.022817</td>
      <td>-0.020121</td>
      <td>-0.011964</td>
      <td>-0.008225</td>
      <td>-0.005685</td>
      <td>-0.008882</td>
      <td>-0.013000</td>
      <td>-0.012395</td>
      <td>0.001224</td>
      <td>-0.015800</td>
      <td>-0.027943</td>
      <td>-0.012395</td>
      <td>-0.005685</td>
    </tr>
    <tr>
      <th>blueWardsDestroyed</th>
      <td>-0.012057</td>
      <td>0.044247</td>
      <td>0.034447</td>
      <td>1.000000</td>
      <td>0.017717</td>
      <td>0.033748</td>
      <td>-0.073182</td>
      <td>0.067793</td>
      <td>0.041700</td>
      <td>0.040504</td>
      <td>0.016940</td>
      <td>-0.009150</td>
      <td>0.060054</td>
      <td>0.060294</td>
      <td>0.067462</td>
      <td>0.111028</td>
      <td>-0.023452</td>
      <td>0.078585</td>
      <td>0.077946</td>
      <td>0.111028</td>
      <td>0.060054</td>
      <td>0.135966</td>
      <td>0.123919</td>
      <td>-0.017717</td>
      <td>-0.073182</td>
      <td>0.033748</td>
      <td>-0.046212</td>
      <td>-0.034509</td>
      <td>-0.034439</td>
      <td>-0.012712</td>
      <td>-0.023943</td>
      <td>-0.067467</td>
      <td>-0.059090</td>
      <td>-0.057314</td>
      <td>0.040023</td>
      <td>-0.035732</td>
      <td>-0.078585</td>
      <td>-0.077946</td>
      <td>0.040023</td>
      <td>-0.067467</td>
    </tr>
    <tr>
      <th>blueFirstBlood</th>
      <td>-0.011577</td>
      <td>0.201769</td>
      <td>0.003228</td>
      <td>0.017717</td>
      <td>1.000000</td>
      <td>0.269425</td>
      <td>-0.247929</td>
      <td>0.229485</td>
      <td>0.151603</td>
      <td>0.134309</td>
      <td>0.077509</td>
      <td>0.083316</td>
      <td>0.312058</td>
      <td>0.177617</td>
      <td>0.190365</td>
      <td>0.125642</td>
      <td>0.018190</td>
      <td>0.378511</td>
      <td>0.240665</td>
      <td>0.125642</td>
      <td>0.312058</td>
      <td>-0.019142</td>
      <td>-0.043304</td>
      <td>-1.000000</td>
      <td>-0.247929</td>
      <td>0.269425</td>
      <td>-0.201140</td>
      <td>-0.141627</td>
      <td>-0.135327</td>
      <td>-0.060246</td>
      <td>-0.069584</td>
      <td>-0.301479</td>
      <td>-0.182602</td>
      <td>-0.194920</td>
      <td>-0.156711</td>
      <td>-0.024559</td>
      <td>-0.378511</td>
      <td>-0.240665</td>
      <td>-0.156711</td>
      <td>-0.301479</td>
    </tr>
    <tr>
      <th>blueKills</th>
      <td>-0.038993</td>
      <td>0.337358</td>
      <td>0.018138</td>
      <td>0.033748</td>
      <td>0.269425</td>
      <td>1.000000</td>
      <td>0.004044</td>
      <td>0.813667</td>
      <td>0.178540</td>
      <td>0.170436</td>
      <td>0.076195</td>
      <td>0.180314</td>
      <td>0.888751</td>
      <td>0.434867</td>
      <td>0.472155</td>
      <td>-0.030880</td>
      <td>-0.112506</td>
      <td>0.654148</td>
      <td>0.583730</td>
      <td>-0.030880</td>
      <td>0.888751</td>
      <td>-0.034239</td>
      <td>-0.092278</td>
      <td>-0.269425</td>
      <td>0.004044</td>
      <td>1.000000</td>
      <td>-0.020344</td>
      <td>-0.224564</td>
      <td>-0.207949</td>
      <td>-0.104423</td>
      <td>-0.082491</td>
      <td>-0.161127</td>
      <td>-0.412219</td>
      <td>-0.462333</td>
      <td>-0.472203</td>
      <td>-0.214454</td>
      <td>-0.654148</td>
      <td>-0.583730</td>
      <td>-0.472203</td>
      <td>-0.161127</td>
    </tr>
    <tr>
      <th>blueDeaths</th>
      <td>-0.013160</td>
      <td>-0.339297</td>
      <td>-0.002612</td>
      <td>-0.073182</td>
      <td>-0.247929</td>
      <td>0.004044</td>
      <td>1.000000</td>
      <td>-0.026372</td>
      <td>-0.204764</td>
      <td>-0.188852</td>
      <td>-0.095527</td>
      <td>-0.071441</td>
      <td>-0.162572</td>
      <td>-0.414755</td>
      <td>-0.460122</td>
      <td>-0.468560</td>
      <td>-0.228102</td>
      <td>-0.640000</td>
      <td>-0.577613</td>
      <td>-0.468560</td>
      <td>-0.162572</td>
      <td>0.008102</td>
      <td>0.038672</td>
      <td>0.247929</td>
      <td>1.000000</td>
      <td>0.004044</td>
      <td>0.804023</td>
      <td>0.163340</td>
      <td>0.150746</td>
      <td>0.076639</td>
      <td>0.156780</td>
      <td>0.885728</td>
      <td>0.433383</td>
      <td>0.464584</td>
      <td>-0.040521</td>
      <td>-0.100271</td>
      <td>0.640000</td>
      <td>0.577613</td>
      <td>-0.040521</td>
      <td>0.885728</td>
    </tr>
    <tr>
      <th>blueAssists</th>
      <td>-0.023329</td>
      <td>0.276685</td>
      <td>0.033217</td>
      <td>0.067793</td>
      <td>0.229485</td>
      <td>0.813667</td>
      <td>-0.026372</td>
      <td>1.000000</td>
      <td>0.149043</td>
      <td>0.170873</td>
      <td>0.028434</td>
      <td>0.123663</td>
      <td>0.748352</td>
      <td>0.292661</td>
      <td>0.303022</td>
      <td>-0.062035</td>
      <td>-0.134023</td>
      <td>0.549761</td>
      <td>0.437002</td>
      <td>-0.062035</td>
      <td>0.748352</td>
      <td>-0.032474</td>
      <td>-0.064501</td>
      <td>-0.229485</td>
      <td>-0.026372</td>
      <td>0.813667</td>
      <td>-0.007481</td>
      <td>-0.182985</td>
      <td>-0.189563</td>
      <td>-0.058074</td>
      <td>-0.060880</td>
      <td>-0.133948</td>
      <td>-0.356928</td>
      <td>-0.396652</td>
      <td>-0.337515</td>
      <td>-0.160915</td>
      <td>-0.549761</td>
      <td>-0.437002</td>
      <td>-0.337515</td>
      <td>-0.133948</td>
    </tr>
    <tr>
      <th>blueEliteMonsters</th>
      <td>0.016599</td>
      <td>0.221944</td>
      <td>0.019892</td>
      <td>0.041700</td>
      <td>0.151603</td>
      <td>0.178540</td>
      <td>-0.204764</td>
      <td>0.149043</td>
      <td>1.000000</td>
      <td>0.781039</td>
      <td>0.640271</td>
      <td>0.166644</td>
      <td>0.239396</td>
      <td>0.203530</td>
      <td>0.232774</td>
      <td>0.118762</td>
      <td>0.198378</td>
      <td>0.281464</td>
      <td>0.263991</td>
      <td>0.118762</td>
      <td>0.239396</td>
      <td>-0.017292</td>
      <td>-0.005288</td>
      <td>-0.151603</td>
      <td>-0.204764</td>
      <td>0.178540</td>
      <td>-0.156764</td>
      <td>-0.455139</td>
      <td>-0.471754</td>
      <td>-0.144104</td>
      <td>-0.052029</td>
      <td>-0.216616</td>
      <td>-0.169649</td>
      <td>-0.189816</td>
      <td>-0.074838</td>
      <td>-0.087893</td>
      <td>-0.281464</td>
      <td>-0.263991</td>
      <td>-0.074838</td>
      <td>-0.216616</td>
    </tr>
    <tr>
      <th>blueDragons</th>
      <td>0.008962</td>
      <td>0.213768</td>
      <td>0.017676</td>
      <td>0.040504</td>
      <td>0.134309</td>
      <td>0.170436</td>
      <td>-0.188852</td>
      <td>0.170873</td>
      <td>0.781039</td>
      <td>1.000000</td>
      <td>0.020381</td>
      <td>0.039750</td>
      <td>0.186413</td>
      <td>0.160683</td>
      <td>0.179083</td>
      <td>0.086686</td>
      <td>0.159595</td>
      <td>0.233875</td>
      <td>0.211496</td>
      <td>0.086686</td>
      <td>0.186413</td>
      <td>-0.027102</td>
      <td>-0.023049</td>
      <td>-0.134309</td>
      <td>-0.188852</td>
      <td>0.170436</td>
      <td>-0.162406</td>
      <td>-0.506546</td>
      <td>-0.631930</td>
      <td>-0.016827</td>
      <td>-0.032865</td>
      <td>-0.192871</td>
      <td>-0.149806</td>
      <td>-0.159485</td>
      <td>-0.059803</td>
      <td>-0.098446</td>
      <td>-0.233875</td>
      <td>-0.211496</td>
      <td>-0.059803</td>
      <td>-0.192871</td>
    </tr>
    <tr>
      <th>blueHeralds</th>
      <td>0.015551</td>
      <td>0.092385</td>
      <td>0.010104</td>
      <td>0.016940</td>
      <td>0.077509</td>
      <td>0.076195</td>
      <td>-0.095527</td>
      <td>0.028434</td>
      <td>0.640271</td>
      <td>0.020381</td>
      <td>1.000000</td>
      <td>0.217901</td>
      <td>0.153974</td>
      <td>0.128201</td>
      <td>0.152386</td>
      <td>0.083509</td>
      <td>0.121291</td>
      <td>0.162943</td>
      <td>0.162496</td>
      <td>0.083509</td>
      <td>0.153974</td>
      <td>0.005653</td>
      <td>0.019885</td>
      <td>-0.077509</td>
      <td>-0.095527</td>
      <td>0.076195</td>
      <td>-0.051209</td>
      <td>-0.105593</td>
      <td>0.022035</td>
      <td>-0.210012</td>
      <td>-0.042872</td>
      <td>-0.109557</td>
      <td>-0.087337</td>
      <td>-0.107718</td>
      <td>-0.046253</td>
      <td>-0.019622</td>
      <td>-0.162943</td>
      <td>-0.162496</td>
      <td>-0.046253</td>
      <td>-0.109557</td>
    </tr>
    <tr>
      <th>blueTowersDestroyed</th>
      <td>-0.007467</td>
      <td>0.115566</td>
      <td>0.009462</td>
      <td>-0.009150</td>
      <td>0.083316</td>
      <td>0.180314</td>
      <td>-0.071441</td>
      <td>0.123663</td>
      <td>0.166644</td>
      <td>0.039750</td>
      <td>0.217901</td>
      <td>1.000000</td>
      <td>0.350941</td>
      <td>0.124453</td>
      <td>0.139398</td>
      <td>0.092291</td>
      <td>0.008165</td>
      <td>0.294060</td>
      <td>0.218320</td>
      <td>0.092291</td>
      <td>0.350941</td>
      <td>0.003660</td>
      <td>-0.038623</td>
      <td>-0.083316</td>
      <td>-0.071441</td>
      <td>0.180314</td>
      <td>-0.036254</td>
      <td>-0.041099</td>
      <td>-0.028482</td>
      <td>-0.031973</td>
      <td>0.011738</td>
      <td>-0.122465</td>
      <td>-0.204429</td>
      <td>-0.210167</td>
      <td>-0.186879</td>
      <td>-0.038505</td>
      <td>-0.294060</td>
      <td>-0.218320</td>
      <td>-0.186879</td>
      <td>-0.122465</td>
    </tr>
    <tr>
      <th>blueTotalGold</th>
      <td>-0.033754</td>
      <td>0.417213</td>
      <td>0.019725</td>
      <td>0.060054</td>
      <td>0.312058</td>
      <td>0.888751</td>
      <td>-0.162572</td>
      <td>0.748352</td>
      <td>0.239396</td>
      <td>0.186413</td>
      <td>0.153974</td>
      <td>0.350941</td>
      <td>1.000000</td>
      <td>0.616968</td>
      <td>0.676193</td>
      <td>0.284902</td>
      <td>0.090769</td>
      <td>0.816803</td>
      <td>0.729345</td>
      <td>0.284902</td>
      <td>1.000000</td>
      <td>-0.020069</td>
      <td>-0.090611</td>
      <td>-0.312058</td>
      <td>-0.162572</td>
      <td>0.888751</td>
      <td>-0.128921</td>
      <td>-0.227236</td>
      <td>-0.201794</td>
      <td>-0.117257</td>
      <td>-0.119579</td>
      <td>-0.314212</td>
      <td>-0.447049</td>
      <td>-0.491268</td>
      <td>-0.452593</td>
      <td>-0.175275</td>
      <td>-0.816803</td>
      <td>-0.729345</td>
      <td>-0.452593</td>
      <td>-0.314212</td>
    </tr>
    <tr>
      <th>blueAvgLevel</th>
      <td>-0.040956</td>
      <td>0.357820</td>
      <td>0.034349</td>
      <td>0.060294</td>
      <td>0.177617</td>
      <td>0.434867</td>
      <td>-0.414755</td>
      <td>0.292661</td>
      <td>0.203530</td>
      <td>0.160683</td>
      <td>0.128201</td>
      <td>0.124453</td>
      <td>0.616968</td>
      <td>1.000000</td>
      <td>0.901297</td>
      <td>0.506279</td>
      <td>0.371371</td>
      <td>0.653538</td>
      <td>0.718822</td>
      <td>0.506279</td>
      <td>0.616968</td>
      <td>0.001020</td>
      <td>-0.052770</td>
      <td>-0.177617</td>
      <td>-0.414755</td>
      <td>0.434867</td>
      <td>-0.366039</td>
      <td>-0.167348</td>
      <td>-0.137854</td>
      <td>-0.100798</td>
      <td>-0.183090</td>
      <td>-0.440031</td>
      <td>-0.228466</td>
      <td>-0.248941</td>
      <td>-0.123316</td>
      <td>-0.013128</td>
      <td>-0.653538</td>
      <td>-0.718822</td>
      <td>-0.123316</td>
      <td>-0.440031</td>
    </tr>
    <tr>
      <th>blueTotalExperience</th>
      <td>-0.040852</td>
      <td>0.396141</td>
      <td>0.031719</td>
      <td>0.067462</td>
      <td>0.190365</td>
      <td>0.472155</td>
      <td>-0.460122</td>
      <td>0.303022</td>
      <td>0.232774</td>
      <td>0.179083</td>
      <td>0.152386</td>
      <td>0.139398</td>
      <td>0.676193</td>
      <td>0.901297</td>
      <td>1.000000</td>
      <td>0.570850</td>
      <td>0.412967</td>
      <td>0.717968</td>
      <td>0.800815</td>
      <td>0.570850</td>
      <td>0.676193</td>
      <td>-0.006032</td>
      <td>-0.057446</td>
      <td>-0.190365</td>
      <td>-0.460122</td>
      <td>0.472155</td>
      <td>-0.397254</td>
      <td>-0.186937</td>
      <td>-0.145501</td>
      <td>-0.124000</td>
      <td>-0.187414</td>
      <td>-0.485059</td>
      <td>-0.254508</td>
      <td>-0.281446</td>
      <td>-0.141276</td>
      <td>-0.010528</td>
      <td>-0.717968</td>
      <td>-0.800815</td>
      <td>-0.141276</td>
      <td>-0.485059</td>
    </tr>
    <tr>
      <th>blueTotalMinionsKilled</th>
      <td>-0.002917</td>
      <td>0.224909</td>
      <td>-0.033925</td>
      <td>0.111028</td>
      <td>0.125642</td>
      <td>-0.030880</td>
      <td>-0.468560</td>
      <td>-0.062035</td>
      <td>0.118762</td>
      <td>0.086686</td>
      <td>0.083509</td>
      <td>0.092291</td>
      <td>0.284902</td>
      <td>0.506279</td>
      <td>0.570850</td>
      <td>1.000000</td>
      <td>0.172282</td>
      <td>0.450497</td>
      <td>0.447264</td>
      <td>1.000000</td>
      <td>0.284902</td>
      <td>0.014210</td>
      <td>0.030234</td>
      <td>-0.125642</td>
      <td>-0.468560</td>
      <td>-0.030880</td>
      <td>-0.337314</td>
      <td>-0.069986</td>
      <td>-0.053958</td>
      <td>-0.047115</td>
      <td>-0.145974</td>
      <td>-0.447904</td>
      <td>-0.142399</td>
      <td>-0.144832</td>
      <td>0.000484</td>
      <td>0.092225</td>
      <td>-0.450497</td>
      <td>-0.447264</td>
      <td>0.000484</td>
      <td>-0.447904</td>
    </tr>
    <tr>
      <th>blueTotalJungleMinionsKilled</th>
      <td>-0.004193</td>
      <td>0.131445</td>
      <td>0.010501</td>
      <td>-0.023452</td>
      <td>0.018190</td>
      <td>-0.112506</td>
      <td>-0.228102</td>
      <td>-0.134023</td>
      <td>0.198378</td>
      <td>0.159595</td>
      <td>0.121291</td>
      <td>0.008165</td>
      <td>0.090769</td>
      <td>0.371371</td>
      <td>0.412967</td>
      <td>0.172282</td>
      <td>1.000000</td>
      <td>0.167510</td>
      <td>0.265443</td>
      <td>0.172282</td>
      <td>0.090769</td>
      <td>0.004671</td>
      <td>-0.018008</td>
      <td>-0.018190</td>
      <td>-0.228102</td>
      <td>-0.112506</td>
      <td>-0.169318</td>
      <td>-0.075076</td>
      <td>-0.053295</td>
      <td>-0.056702</td>
      <td>-0.048078</td>
      <td>-0.182167</td>
      <td>-0.013881</td>
      <td>-0.011657</td>
      <td>0.109806</td>
      <td>-0.026363</td>
      <td>-0.167510</td>
      <td>-0.265443</td>
      <td>0.109806</td>
      <td>-0.182167</td>
    </tr>
    <tr>
      <th>blueGoldDiff</th>
      <td>-0.014670</td>
      <td>0.511119</td>
      <td>0.015800</td>
      <td>0.078585</td>
      <td>0.378511</td>
      <td>0.654148</td>
      <td>-0.640000</td>
      <td>0.549761</td>
      <td>0.281464</td>
      <td>0.233875</td>
      <td>0.162943</td>
      <td>0.294060</td>
      <td>0.816803</td>
      <td>0.653538</td>
      <td>0.717968</td>
      <td>0.450497</td>
      <td>0.167510</td>
      <td>1.000000</td>
      <td>0.894729</td>
      <td>0.450497</td>
      <td>0.816803</td>
      <td>-0.019042</td>
      <td>-0.099725</td>
      <td>-0.378511</td>
      <td>-0.640000</td>
      <td>0.654148</td>
      <td>-0.528081</td>
      <td>-0.281296</td>
      <td>-0.234566</td>
      <td>-0.165611</td>
      <td>-0.273861</td>
      <td>-0.804347</td>
      <td>-0.652929</td>
      <td>-0.714405</td>
      <td>-0.452633</td>
      <td>-0.172066</td>
      <td>-1.000000</td>
      <td>-0.894729</td>
      <td>-0.452633</td>
      <td>-0.804347</td>
    </tr>
    <tr>
      <th>blueExperienceDiff</th>
      <td>-0.012315</td>
      <td>0.489558</td>
      <td>0.027943</td>
      <td>0.077946</td>
      <td>0.240665</td>
      <td>0.583730</td>
      <td>-0.577613</td>
      <td>0.437002</td>
      <td>0.263991</td>
      <td>0.211496</td>
      <td>0.162496</td>
      <td>0.218320</td>
      <td>0.729345</td>
      <td>0.718822</td>
      <td>0.800815</td>
      <td>0.447264</td>
      <td>0.265443</td>
      <td>0.894729</td>
      <td>1.000000</td>
      <td>0.447264</td>
      <td>0.729345</td>
      <td>-0.026556</td>
      <td>-0.085829</td>
      <td>-0.240665</td>
      <td>-0.577613</td>
      <td>0.583730</td>
      <td>-0.422972</td>
      <td>-0.269283</td>
      <td>-0.218872</td>
      <td>-0.166162</td>
      <td>-0.197678</td>
      <td>-0.721190</td>
      <td>-0.721925</td>
      <td>-0.800089</td>
      <td>-0.437205</td>
      <td>-0.273224</td>
      <td>-0.894729</td>
      <td>-1.000000</td>
      <td>-0.437205</td>
      <td>-0.721190</td>
    </tr>
    <tr>
      <th>blueCSPerMin</th>
      <td>-0.002917</td>
      <td>0.224909</td>
      <td>-0.033925</td>
      <td>0.111028</td>
      <td>0.125642</td>
      <td>-0.030880</td>
      <td>-0.468560</td>
      <td>-0.062035</td>
      <td>0.118762</td>
      <td>0.086686</td>
      <td>0.083509</td>
      <td>0.092291</td>
      <td>0.284902</td>
      <td>0.506279</td>
      <td>0.570850</td>
      <td>1.000000</td>
      <td>0.172282</td>
      <td>0.450497</td>
      <td>0.447264</td>
      <td>1.000000</td>
      <td>0.284902</td>
      <td>0.014210</td>
      <td>0.030234</td>
      <td>-0.125642</td>
      <td>-0.468560</td>
      <td>-0.030880</td>
      <td>-0.337314</td>
      <td>-0.069986</td>
      <td>-0.053958</td>
      <td>-0.047115</td>
      <td>-0.145974</td>
      <td>-0.447904</td>
      <td>-0.142399</td>
      <td>-0.144832</td>
      <td>0.000484</td>
      <td>0.092225</td>
      <td>-0.450497</td>
      <td>-0.447264</td>
      <td>0.000484</td>
      <td>-0.447904</td>
    </tr>
    <tr>
      <th>blueGoldPerMin</th>
      <td>-0.033754</td>
      <td>0.417213</td>
      <td>0.019725</td>
      <td>0.060054</td>
      <td>0.312058</td>
      <td>0.888751</td>
      <td>-0.162572</td>
      <td>0.748352</td>
      <td>0.239396</td>
      <td>0.186413</td>
      <td>0.153974</td>
      <td>0.350941</td>
      <td>1.000000</td>
      <td>0.616968</td>
      <td>0.676193</td>
      <td>0.284902</td>
      <td>0.090769</td>
      <td>0.816803</td>
      <td>0.729345</td>
      <td>0.284902</td>
      <td>1.000000</td>
      <td>-0.020069</td>
      <td>-0.090611</td>
      <td>-0.312058</td>
      <td>-0.162572</td>
      <td>0.888751</td>
      <td>-0.128921</td>
      <td>-0.227236</td>
      <td>-0.201794</td>
      <td>-0.117257</td>
      <td>-0.119579</td>
      <td>-0.314212</td>
      <td>-0.447049</td>
      <td>-0.491268</td>
      <td>-0.452593</td>
      <td>-0.175275</td>
      <td>-0.816803</td>
      <td>-0.729345</td>
      <td>-0.452593</td>
      <td>-0.314212</td>
    </tr>
    <tr>
      <th>redWardsPlaced</th>
      <td>0.007405</td>
      <td>-0.023671</td>
      <td>-0.012906</td>
      <td>0.135966</td>
      <td>-0.019142</td>
      <td>-0.034239</td>
      <td>0.008102</td>
      <td>-0.032474</td>
      <td>-0.017292</td>
      <td>-0.027102</td>
      <td>0.005653</td>
      <td>0.003660</td>
      <td>-0.020069</td>
      <td>0.001020</td>
      <td>-0.006032</td>
      <td>0.014210</td>
      <td>0.004671</td>
      <td>-0.019042</td>
      <td>-0.026556</td>
      <td>0.014210</td>
      <td>-0.020069</td>
      <td>1.000000</td>
      <td>0.019784</td>
      <td>0.019142</td>
      <td>0.008102</td>
      <td>-0.034239</td>
      <td>0.023791</td>
      <td>0.027452</td>
      <td>0.018717</td>
      <td>0.021769</td>
      <td>-0.006230</td>
      <td>0.010666</td>
      <td>0.041737</td>
      <td>0.036506</td>
      <td>-0.021842</td>
      <td>0.004666</td>
      <td>0.019042</td>
      <td>0.026556</td>
      <td>-0.021842</td>
      <td>0.010666</td>
    </tr>
    <tr>
      <th>redWardsDestroyed</th>
      <td>-0.001197</td>
      <td>-0.055400</td>
      <td>0.115549</td>
      <td>0.123919</td>
      <td>-0.043304</td>
      <td>-0.092278</td>
      <td>0.038672</td>
      <td>-0.064501</td>
      <td>-0.005288</td>
      <td>-0.023049</td>
      <td>0.019885</td>
      <td>-0.038623</td>
      <td>-0.090611</td>
      <td>-0.052770</td>
      <td>-0.057446</td>
      <td>0.030234</td>
      <td>-0.018008</td>
      <td>-0.099725</td>
      <td>-0.085829</td>
      <td>0.030234</td>
      <td>-0.090611</td>
      <td>0.019784</td>
      <td>1.000000</td>
      <td>0.043304</td>
      <td>0.038672</td>
      <td>-0.092278</td>
      <td>0.055798</td>
      <td>0.039335</td>
      <td>0.046132</td>
      <td>0.005255</td>
      <td>0.003855</td>
      <td>0.070784</td>
      <td>0.075537</td>
      <td>0.079975</td>
      <td>0.128062</td>
      <td>-0.009313</td>
      <td>0.099725</td>
      <td>0.085829</td>
      <td>0.128062</td>
      <td>0.070784</td>
    </tr>
    <tr>
      <th>redFirstBlood</th>
      <td>0.011577</td>
      <td>-0.201769</td>
      <td>-0.003228</td>
      <td>-0.017717</td>
      <td>-1.000000</td>
      <td>-0.269425</td>
      <td>0.247929</td>
      <td>-0.229485</td>
      <td>-0.151603</td>
      <td>-0.134309</td>
      <td>-0.077509</td>
      <td>-0.083316</td>
      <td>-0.312058</td>
      <td>-0.177617</td>
      <td>-0.190365</td>
      <td>-0.125642</td>
      <td>-0.018190</td>
      <td>-0.378511</td>
      <td>-0.240665</td>
      <td>-0.125642</td>
      <td>-0.312058</td>
      <td>0.019142</td>
      <td>0.043304</td>
      <td>1.000000</td>
      <td>0.247929</td>
      <td>-0.269425</td>
      <td>0.201140</td>
      <td>0.141627</td>
      <td>0.135327</td>
      <td>0.060246</td>
      <td>0.069584</td>
      <td>0.301479</td>
      <td>0.182602</td>
      <td>0.194920</td>
      <td>0.156711</td>
      <td>0.024559</td>
      <td>0.378511</td>
      <td>0.240665</td>
      <td>0.156711</td>
      <td>0.301479</td>
    </tr>
    <tr>
      <th>redKills</th>
      <td>-0.013160</td>
      <td>-0.339297</td>
      <td>-0.002612</td>
      <td>-0.073182</td>
      <td>-0.247929</td>
      <td>0.004044</td>
      <td>1.000000</td>
      <td>-0.026372</td>
      <td>-0.204764</td>
      <td>-0.188852</td>
      <td>-0.095527</td>
      <td>-0.071441</td>
      <td>-0.162572</td>
      <td>-0.414755</td>
      <td>-0.460122</td>
      <td>-0.468560</td>
      <td>-0.228102</td>
      <td>-0.640000</td>
      <td>-0.577613</td>
      <td>-0.468560</td>
      <td>-0.162572</td>
      <td>0.008102</td>
      <td>0.038672</td>
      <td>0.247929</td>
      <td>1.000000</td>
      <td>0.004044</td>
      <td>0.804023</td>
      <td>0.163340</td>
      <td>0.150746</td>
      <td>0.076639</td>
      <td>0.156780</td>
      <td>0.885728</td>
      <td>0.433383</td>
      <td>0.464584</td>
      <td>-0.040521</td>
      <td>-0.100271</td>
      <td>0.640000</td>
      <td>0.577613</td>
      <td>-0.040521</td>
      <td>0.885728</td>
    </tr>
    <tr>
      <th>redDeaths</th>
      <td>-0.038993</td>
      <td>0.337358</td>
      <td>0.018138</td>
      <td>0.033748</td>
      <td>0.269425</td>
      <td>1.000000</td>
      <td>0.004044</td>
      <td>0.813667</td>
      <td>0.178540</td>
      <td>0.170436</td>
      <td>0.076195</td>
      <td>0.180314</td>
      <td>0.888751</td>
      <td>0.434867</td>
      <td>0.472155</td>
      <td>-0.030880</td>
      <td>-0.112506</td>
      <td>0.654148</td>
      <td>0.583730</td>
      <td>-0.030880</td>
      <td>0.888751</td>
      <td>-0.034239</td>
      <td>-0.092278</td>
      <td>-0.269425</td>
      <td>0.004044</td>
      <td>1.000000</td>
      <td>-0.020344</td>
      <td>-0.224564</td>
      <td>-0.207949</td>
      <td>-0.104423</td>
      <td>-0.082491</td>
      <td>-0.161127</td>
      <td>-0.412219</td>
      <td>-0.462333</td>
      <td>-0.472203</td>
      <td>-0.214454</td>
      <td>-0.654148</td>
      <td>-0.583730</td>
      <td>-0.472203</td>
      <td>-0.161127</td>
    </tr>
    <tr>
      <th>redAssists</th>
      <td>-0.008664</td>
      <td>-0.271047</td>
      <td>-0.009009</td>
      <td>-0.046212</td>
      <td>-0.201140</td>
      <td>-0.020344</td>
      <td>0.804023</td>
      <td>-0.007481</td>
      <td>-0.156764</td>
      <td>-0.162406</td>
      <td>-0.051209</td>
      <td>-0.036254</td>
      <td>-0.128921</td>
      <td>-0.366039</td>
      <td>-0.397254</td>
      <td>-0.337314</td>
      <td>-0.169318</td>
      <td>-0.528081</td>
      <td>-0.422972</td>
      <td>-0.337314</td>
      <td>-0.128921</td>
      <td>0.023791</td>
      <td>0.055798</td>
      <td>0.201140</td>
      <td>0.804023</td>
      <td>-0.020344</td>
      <td>1.000000</td>
      <td>0.129698</td>
      <td>0.142671</td>
      <td>0.030000</td>
      <td>0.107425</td>
      <td>0.736215</td>
      <td>0.277040</td>
      <td>0.279788</td>
      <td>-0.078234</td>
      <td>-0.130417</td>
      <td>0.528081</td>
      <td>0.422972</td>
      <td>-0.078234</td>
      <td>0.736215</td>
    </tr>
    <tr>
      <th>redEliteMonsters</th>
      <td>0.017296</td>
      <td>-0.221551</td>
      <td>-0.022817</td>
      <td>-0.034509</td>
      <td>-0.141627</td>
      <td>-0.224564</td>
      <td>0.163340</td>
      <td>-0.182985</td>
      <td>-0.455139</td>
      <td>-0.506546</td>
      <td>-0.105593</td>
      <td>-0.041099</td>
      <td>-0.227236</td>
      <td>-0.167348</td>
      <td>-0.186937</td>
      <td>-0.069986</td>
      <td>-0.075076</td>
      <td>-0.281296</td>
      <td>-0.269283</td>
      <td>-0.069986</td>
      <td>-0.227236</td>
      <td>0.027452</td>
      <td>0.039335</td>
      <td>0.141627</td>
      <td>0.163340</td>
      <td>-0.224564</td>
      <td>0.129698</td>
      <td>1.000000</td>
      <td>0.811234</td>
      <td>0.619153</td>
      <td>0.158999</td>
      <td>0.228861</td>
      <td>0.222537</td>
      <td>0.244205</td>
      <td>0.129705</td>
      <td>0.216969</td>
      <td>0.281296</td>
      <td>0.269283</td>
      <td>0.129705</td>
      <td>0.228861</td>
    </tr>
    <tr>
      <th>redDragons</th>
      <td>0.017416</td>
      <td>-0.209516</td>
      <td>-0.020121</td>
      <td>-0.034439</td>
      <td>-0.135327</td>
      <td>-0.207949</td>
      <td>0.150746</td>
      <td>-0.189563</td>
      <td>-0.471754</td>
      <td>-0.631930</td>
      <td>0.022035</td>
      <td>-0.028482</td>
      <td>-0.201794</td>
      <td>-0.137854</td>
      <td>-0.145501</td>
      <td>-0.053958</td>
      <td>-0.053295</td>
      <td>-0.234566</td>
      <td>-0.218872</td>
      <td>-0.053958</td>
      <td>-0.201794</td>
      <td>0.018717</td>
      <td>0.046132</td>
      <td>0.135327</td>
      <td>0.150746</td>
      <td>-0.207949</td>
      <td>0.142671</td>
      <td>0.811234</td>
      <td>1.000000</td>
      <td>0.043114</td>
      <td>0.026950</td>
      <td>0.178168</td>
      <td>0.191497</td>
      <td>0.204941</td>
      <td>0.103151</td>
      <td>0.214187</td>
      <td>0.234566</td>
      <td>0.218872</td>
      <td>0.103151</td>
      <td>0.178168</td>
    </tr>
    <tr>
      <th>redHeralds</th>
      <td>0.006163</td>
      <td>-0.097172</td>
      <td>-0.011964</td>
      <td>-0.012712</td>
      <td>-0.060246</td>
      <td>-0.104423</td>
      <td>0.076639</td>
      <td>-0.058074</td>
      <td>-0.144104</td>
      <td>-0.016827</td>
      <td>-0.210012</td>
      <td>-0.031973</td>
      <td>-0.117257</td>
      <td>-0.100798</td>
      <td>-0.124000</td>
      <td>-0.047115</td>
      <td>-0.056702</td>
      <td>-0.165611</td>
      <td>-0.166162</td>
      <td>-0.047115</td>
      <td>-0.117257</td>
      <td>0.021769</td>
      <td>0.005255</td>
      <td>0.060246</td>
      <td>0.076639</td>
      <td>-0.104423</td>
      <td>0.030000</td>
      <td>0.619153</td>
      <td>0.043114</td>
      <td>1.000000</td>
      <td>0.235475</td>
      <td>0.151762</td>
      <td>0.123056</td>
      <td>0.142024</td>
      <td>0.083087</td>
      <td>0.083068</td>
      <td>0.165611</td>
      <td>0.166162</td>
      <td>0.083087</td>
      <td>0.151762</td>
    </tr>
    <tr>
      <th>redTowersDestroyed</th>
      <td>0.003557</td>
      <td>-0.103696</td>
      <td>-0.008225</td>
      <td>-0.023943</td>
      <td>-0.069584</td>
      <td>-0.082491</td>
      <td>0.156780</td>
      <td>-0.060880</td>
      <td>-0.052029</td>
      <td>-0.032865</td>
      <td>-0.042872</td>
      <td>0.011738</td>
      <td>-0.119579</td>
      <td>-0.183090</td>
      <td>-0.187414</td>
      <td>-0.145974</td>
      <td>-0.048078</td>
      <td>-0.273861</td>
      <td>-0.197678</td>
      <td>-0.145974</td>
      <td>-0.119579</td>
      <td>-0.006230</td>
      <td>0.003855</td>
      <td>0.069584</td>
      <td>0.156780</td>
      <td>-0.082491</td>
      <td>0.107425</td>
      <td>0.158999</td>
      <td>0.026950</td>
      <td>0.235475</td>
      <td>1.000000</td>
      <td>0.327503</td>
      <td>0.113035</td>
      <td>0.129002</td>
      <td>0.092564</td>
      <td>0.006374</td>
      <td>0.273861</td>
      <td>0.197678</td>
      <td>0.092564</td>
      <td>0.327503</td>
    </tr>
    <tr>
      <th>redTotalGold</th>
      <td>-0.010622</td>
      <td>-0.411396</td>
      <td>-0.005685</td>
      <td>-0.067467</td>
      <td>-0.301479</td>
      <td>-0.161127</td>
      <td>0.885728</td>
      <td>-0.133948</td>
      <td>-0.216616</td>
      <td>-0.192871</td>
      <td>-0.109557</td>
      <td>-0.122465</td>
      <td>-0.314212</td>
      <td>-0.440031</td>
      <td>-0.485059</td>
      <td>-0.447904</td>
      <td>-0.182167</td>
      <td>-0.804347</td>
      <td>-0.721190</td>
      <td>-0.447904</td>
      <td>-0.314212</td>
      <td>0.010666</td>
      <td>0.070784</td>
      <td>0.301479</td>
      <td>0.885728</td>
      <td>-0.161127</td>
      <td>0.736215</td>
      <td>0.228861</td>
      <td>0.178168</td>
      <td>0.151762</td>
      <td>0.327503</td>
      <td>1.000000</td>
      <td>0.614025</td>
      <td>0.669646</td>
      <td>0.278715</td>
      <td>0.102632</td>
      <td>0.804347</td>
      <td>0.721190</td>
      <td>0.278715</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>redAvgLevel</th>
      <td>-0.012419</td>
      <td>-0.352127</td>
      <td>-0.008882</td>
      <td>-0.059090</td>
      <td>-0.182602</td>
      <td>-0.412219</td>
      <td>0.433383</td>
      <td>-0.356928</td>
      <td>-0.169649</td>
      <td>-0.149806</td>
      <td>-0.087337</td>
      <td>-0.204429</td>
      <td>-0.447049</td>
      <td>-0.228466</td>
      <td>-0.254508</td>
      <td>-0.142399</td>
      <td>-0.013881</td>
      <td>-0.652929</td>
      <td>-0.721925</td>
      <td>-0.142399</td>
      <td>-0.447049</td>
      <td>0.041737</td>
      <td>0.075537</td>
      <td>0.182602</td>
      <td>0.433383</td>
      <td>-0.412219</td>
      <td>0.277040</td>
      <td>0.222537</td>
      <td>0.191497</td>
      <td>0.123056</td>
      <td>0.113035</td>
      <td>0.614025</td>
      <td>1.000000</td>
      <td>0.901748</td>
      <td>0.489672</td>
      <td>0.382009</td>
      <td>0.652929</td>
      <td>0.721925</td>
      <td>0.489672</td>
      <td>0.614025</td>
    </tr>
    <tr>
      <th>redTotalExperience</th>
      <td>-0.021187</td>
      <td>-0.387588</td>
      <td>-0.013000</td>
      <td>-0.057314</td>
      <td>-0.194920</td>
      <td>-0.462333</td>
      <td>0.464584</td>
      <td>-0.396652</td>
      <td>-0.189816</td>
      <td>-0.159485</td>
      <td>-0.107718</td>
      <td>-0.210167</td>
      <td>-0.491268</td>
      <td>-0.248941</td>
      <td>-0.281446</td>
      <td>-0.144832</td>
      <td>-0.011657</td>
      <td>-0.714405</td>
      <td>-0.800089</td>
      <td>-0.144832</td>
      <td>-0.491268</td>
      <td>0.036506</td>
      <td>0.079975</td>
      <td>0.194920</td>
      <td>0.464584</td>
      <td>-0.462333</td>
      <td>0.279788</td>
      <td>0.244205</td>
      <td>0.204941</td>
      <td>0.142024</td>
      <td>0.129002</td>
      <td>0.669646</td>
      <td>0.901748</td>
      <td>1.000000</td>
      <td>0.558985</td>
      <td>0.427214</td>
      <td>0.714405</td>
      <td>0.800089</td>
      <td>0.558985</td>
      <td>0.669646</td>
    </tr>
    <tr>
      <th>redTotalMinionsKilled</th>
      <td>-0.005118</td>
      <td>-0.212171</td>
      <td>-0.012395</td>
      <td>0.040023</td>
      <td>-0.156711</td>
      <td>-0.472203</td>
      <td>-0.040521</td>
      <td>-0.337515</td>
      <td>-0.074838</td>
      <td>-0.059803</td>
      <td>-0.046253</td>
      <td>-0.186879</td>
      <td>-0.452593</td>
      <td>-0.123316</td>
      <td>-0.141276</td>
      <td>0.000484</td>
      <td>0.109806</td>
      <td>-0.452633</td>
      <td>-0.437205</td>
      <td>0.000484</td>
      <td>-0.452593</td>
      <td>-0.021842</td>
      <td>0.128062</td>
      <td>0.156711</td>
      <td>-0.040521</td>
      <td>-0.472203</td>
      <td>-0.078234</td>
      <td>0.129705</td>
      <td>0.103151</td>
      <td>0.083087</td>
      <td>0.092564</td>
      <td>0.278715</td>
      <td>0.489672</td>
      <td>0.558985</td>
      <td>1.000000</td>
      <td>0.165652</td>
      <td>0.452633</td>
      <td>0.437205</td>
      <td>1.000000</td>
      <td>0.278715</td>
    </tr>
    <tr>
      <th>redTotalJungleMinionsKilled</th>
      <td>0.006040</td>
      <td>-0.110994</td>
      <td>0.001224</td>
      <td>-0.035732</td>
      <td>-0.024559</td>
      <td>-0.214454</td>
      <td>-0.100271</td>
      <td>-0.160915</td>
      <td>-0.087893</td>
      <td>-0.098446</td>
      <td>-0.019622</td>
      <td>-0.038505</td>
      <td>-0.175275</td>
      <td>-0.013128</td>
      <td>-0.010528</td>
      <td>0.092225</td>
      <td>-0.026363</td>
      <td>-0.172066</td>
      <td>-0.273224</td>
      <td>0.092225</td>
      <td>-0.175275</td>
      <td>0.004666</td>
      <td>-0.009313</td>
      <td>0.024559</td>
      <td>-0.100271</td>
      <td>-0.214454</td>
      <td>-0.130417</td>
      <td>0.216969</td>
      <td>0.214187</td>
      <td>0.083068</td>
      <td>0.006374</td>
      <td>0.102632</td>
      <td>0.382009</td>
      <td>0.427214</td>
      <td>0.165652</td>
      <td>1.000000</td>
      <td>0.172066</td>
      <td>0.273224</td>
      <td>0.165652</td>
      <td>0.102632</td>
    </tr>
    <tr>
      <th>redGoldDiff</th>
      <td>0.014670</td>
      <td>-0.511119</td>
      <td>-0.015800</td>
      <td>-0.078585</td>
      <td>-0.378511</td>
      <td>-0.654148</td>
      <td>0.640000</td>
      <td>-0.549761</td>
      <td>-0.281464</td>
      <td>-0.233875</td>
      <td>-0.162943</td>
      <td>-0.294060</td>
      <td>-0.816803</td>
      <td>-0.653538</td>
      <td>-0.717968</td>
      <td>-0.450497</td>
      <td>-0.167510</td>
      <td>-1.000000</td>
      <td>-0.894729</td>
      <td>-0.450497</td>
      <td>-0.816803</td>
      <td>0.019042</td>
      <td>0.099725</td>
      <td>0.378511</td>
      <td>0.640000</td>
      <td>-0.654148</td>
      <td>0.528081</td>
      <td>0.281296</td>
      <td>0.234566</td>
      <td>0.165611</td>
      <td>0.273861</td>
      <td>0.804347</td>
      <td>0.652929</td>
      <td>0.714405</td>
      <td>0.452633</td>
      <td>0.172066</td>
      <td>1.000000</td>
      <td>0.894729</td>
      <td>0.452633</td>
      <td>0.804347</td>
    </tr>
    <tr>
      <th>redExperienceDiff</th>
      <td>0.012315</td>
      <td>-0.489558</td>
      <td>-0.027943</td>
      <td>-0.077946</td>
      <td>-0.240665</td>
      <td>-0.583730</td>
      <td>0.577613</td>
      <td>-0.437002</td>
      <td>-0.263991</td>
      <td>-0.211496</td>
      <td>-0.162496</td>
      <td>-0.218320</td>
      <td>-0.729345</td>
      <td>-0.718822</td>
      <td>-0.800815</td>
      <td>-0.447264</td>
      <td>-0.265443</td>
      <td>-0.894729</td>
      <td>-1.000000</td>
      <td>-0.447264</td>
      <td>-0.729345</td>
      <td>0.026556</td>
      <td>0.085829</td>
      <td>0.240665</td>
      <td>0.577613</td>
      <td>-0.583730</td>
      <td>0.422972</td>
      <td>0.269283</td>
      <td>0.218872</td>
      <td>0.166162</td>
      <td>0.197678</td>
      <td>0.721190</td>
      <td>0.721925</td>
      <td>0.800089</td>
      <td>0.437205</td>
      <td>0.273224</td>
      <td>0.894729</td>
      <td>1.000000</td>
      <td>0.437205</td>
      <td>0.721190</td>
    </tr>
    <tr>
      <th>redCSPerMin</th>
      <td>-0.005118</td>
      <td>-0.212171</td>
      <td>-0.012395</td>
      <td>0.040023</td>
      <td>-0.156711</td>
      <td>-0.472203</td>
      <td>-0.040521</td>
      <td>-0.337515</td>
      <td>-0.074838</td>
      <td>-0.059803</td>
      <td>-0.046253</td>
      <td>-0.186879</td>
      <td>-0.452593</td>
      <td>-0.123316</td>
      <td>-0.141276</td>
      <td>0.000484</td>
      <td>0.109806</td>
      <td>-0.452633</td>
      <td>-0.437205</td>
      <td>0.000484</td>
      <td>-0.452593</td>
      <td>-0.021842</td>
      <td>0.128062</td>
      <td>0.156711</td>
      <td>-0.040521</td>
      <td>-0.472203</td>
      <td>-0.078234</td>
      <td>0.129705</td>
      <td>0.103151</td>
      <td>0.083087</td>
      <td>0.092564</td>
      <td>0.278715</td>
      <td>0.489672</td>
      <td>0.558985</td>
      <td>1.000000</td>
      <td>0.165652</td>
      <td>0.452633</td>
      <td>0.437205</td>
      <td>1.000000</td>
      <td>0.278715</td>
    </tr>
    <tr>
      <th>redGoldPerMin</th>
      <td>-0.010622</td>
      <td>-0.411396</td>
      <td>-0.005685</td>
      <td>-0.067467</td>
      <td>-0.301479</td>
      <td>-0.161127</td>
      <td>0.885728</td>
      <td>-0.133948</td>
      <td>-0.216616</td>
      <td>-0.192871</td>
      <td>-0.109557</td>
      <td>-0.122465</td>
      <td>-0.314212</td>
      <td>-0.440031</td>
      <td>-0.485059</td>
      <td>-0.447904</td>
      <td>-0.182167</td>
      <td>-0.804347</td>
      <td>-0.721190</td>
      <td>-0.447904</td>
      <td>-0.314212</td>
      <td>0.010666</td>
      <td>0.070784</td>
      <td>0.301479</td>
      <td>0.885728</td>
      <td>-0.161127</td>
      <td>0.736215</td>
      <td>0.228861</td>
      <td>0.178168</td>
      <td>0.151762</td>
      <td>0.327503</td>
      <td>1.000000</td>
      <td>0.614025</td>
      <td>0.669646</td>
      <td>0.278715</td>
      <td>0.102632</td>
      <td>0.804347</td>
      <td>0.721190</td>
      <td>0.278715</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



-blue kill 과 red death (반대도 마찬가지), firstblood 간의 관계는 대척 관계이므로 모두 변수에 넣는다면 다중공선성 발생


```python
sns.histplot(data=df, x='blueGoldDiff', hue='blueWins', palette='RdBu', kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d802543a08>




![output_15_1](https://user-images.githubusercontent.com/77723966/112033686-0a9c7580-8b81-11eb-85ac-ff8e3580c213.png)



- blueGoldDiff 변수에 따라 승패가 명확하게 갈린다.


```python
sns.histplot(data=df, x='blueKills', hue='blueWins', palette='RdBu', kde=True, bins=8)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d803583308>




![output_17_1](https://user-images.githubusercontent.com/77723966/112033696-0e2ffc80-8b81-11eb-9094-e43c1290bb32.png)



- blue kills 어느정도 상관성이 있음


```python
sns.jointplot(data=df, x='blueKills', y='blueGoldDiff', hue='blueWins')
```




    <seaborn.axisgrid.JointGrid at 0x1d8036438c8>




![output_19_1](https://user-images.githubusercontent.com/77723966/112033710-11c38380-8b81-11eb-8538-cd078538ec2f.png)



- 두 가지 feature가 상보적이다. 상관성이 높다.


```python
sns.jointplot(data=df, x='blueExperienceDiff', y='blueGoldDiff', hue='blueWins')
```




    <seaborn.axisgrid.JointGrid at 0x1d80376dbc8>




![output_21_1](https://user-images.githubusercontent.com/77723966/112033727-15570a80-8b81-11eb-9b7b-cccbc8a2f483.png)



- 골드 수급과 경험치 수급은 상관관계가 아주 높다. (당연하게도)


```python
sns.countplot(data=df, x='blueDragons', hue='blueWins', palette='RdBu')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d804886508>




![output_23_1](https://user-images.githubusercontent.com/77723966/112033754-19832800-8b81-11eb-9d62-6091dcad865b.png)




```python
sns.countplot(data=df, x='redDragons', hue='blueWins', palette='RdBu')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d8048ca0c8>




![output_24_1](https://user-images.githubusercontent.com/77723966/112033765-1d16af00-8b81-11eb-9fd6-abf642485f02.png)



- blue가 Dragons을 죽였을때 승리확률이 높음
- red는 Dragons 죽이든 죽이지 않든 승패에 크게 영향을 미치지 않음. Dragons의 경우 red보다 Blue에 더 영향을 미침.


```python
sns.countplot(data=df, x='blueFirstBlood', hue='blueWins',palette='RdBu')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d80493b848>




![output_26_1](https://user-images.githubusercontent.com/77723966/112033779-2142cc80-8b81-11eb-848e-70bce4c89341.png)



- blue 첫 킬을 달성한다면 blue가 이기는 사건이 많음. 반대도 마찬가지, 균형이 맞는 변수.

## 데이터 전처리


```python
# 수치형 데이터 표준화
from sklearn.preprocessing import StandardScaler

df.columns
```




    Index(['gameId', 'blueWins', 'blueWardsPlaced', 'blueWardsDestroyed',
           'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
           'blueEliteMonsters', 'blueDragons', 'blueHeralds',
           'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
           'blueTotalExperience', 'blueTotalMinionsKilled',
           'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
           'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
           'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
           'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
           'redTotalGold', 'redAvgLevel', 'redTotalExperience',
           'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff',
           'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin'],
          dtype='object')




```python
# 다중공선성을 불러일으킬만한 변수 제거
df.drop(['gameId','redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
       'redTotalGold', 'redTotalExperience', 'redGoldDiff','redExperienceDiff'], axis=1, inplace=True)

df.columns
```




    Index(['blueWins', 'blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood',
           'blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters',
           'blueDragons', 'blueHeralds', 'blueTowersDestroyed', 'blueTotalGold',
           'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled',
           'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
           'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
           'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
           'redAvgLevel', 'redTotalMinionsKilled', 'redTotalJungleMinionsKilled',
           'redCSPerMin', 'redGoldPerMin'],
          dtype='object')




```python
X_num = df[['blueWardsPlaced', 'blueWardsDestroyed', 
       'blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters',
      'blueTowersDestroyed', 'blueTotalGold',
       'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
       'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
       'redEliteMonsters', 'redTowersDestroyed',
       'redAvgLevel', 'redTotalMinionsKilled', 'redTotalJungleMinionsKilled',
       'redCSPerMin', 'redGoldPerMin']]
X_cat = df[['blueFirstBlood', 'blueDragons', 'blueHeralds', 'redDragons', 'redHeralds']]

scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns=X_num.columns)

X = pd.concat([X_scaled, X_cat], axis=1) 
y = df['blueWins']

X
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
      <th>blueWardsPlaced</th>
      <th>blueWardsDestroyed</th>
      <th>blueKills</th>
      <th>blueDeaths</th>
      <th>blueAssists</th>
      <th>blueEliteMonsters</th>
      <th>blueTowersDestroyed</th>
      <th>blueTotalGold</th>
      <th>blueAvgLevel</th>
      <th>blueTotalExperience</th>
      <th>blueTotalMinionsKilled</th>
      <th>blueTotalJungleMinionsKilled</th>
      <th>blueGoldDiff</th>
      <th>blueExperienceDiff</th>
      <th>blueCSPerMin</th>
      <th>blueGoldPerMin</th>
      <th>redWardsPlaced</th>
      <th>redWardsDestroyed</th>
      <th>redEliteMonsters</th>
      <th>redTowersDestroyed</th>
      <th>redAvgLevel</th>
      <th>redTotalMinionsKilled</th>
      <th>redTotalJungleMinionsKilled</th>
      <th>redCSPerMin</th>
      <th>redGoldPerMin</th>
      <th>blueFirstBlood</th>
      <th>blueDragons</th>
      <th>blueHeralds</th>
      <th>redDragons</th>
      <th>redHeralds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.316996</td>
      <td>-0.379275</td>
      <td>0.935301</td>
      <td>-0.046926</td>
      <td>1.071495</td>
      <td>-0.879231</td>
      <td>-0.210439</td>
      <td>0.460179</td>
      <td>-1.035635</td>
      <td>-0.740639</td>
      <td>-0.992782</td>
      <td>-1.465951</td>
      <td>0.256228</td>
      <td>0.013342</td>
      <td>-0.992782</td>
      <td>0.460179</td>
      <td>-0.399207</td>
      <td>1.532493</td>
      <td>-0.914893</td>
      <td>-0.198353</td>
      <td>-0.410475</td>
      <td>-0.928741</td>
      <td>0.367685</td>
      <td>-0.928741</td>
      <td>0.052293</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.570992</td>
      <td>-0.839069</td>
      <td>-0.393216</td>
      <td>-0.387796</td>
      <td>-0.404768</td>
      <td>-0.879231</td>
      <td>-0.210439</td>
      <td>-1.166792</td>
      <td>-1.035635</td>
      <td>-1.385391</td>
      <td>-1.953558</td>
      <td>-0.758722</td>
      <td>-1.191254</td>
      <td>-0.593342</td>
      <td>-1.953558</td>
      <td>-1.166792</td>
      <td>-0.561751</td>
      <td>-0.805870</td>
      <td>2.277700</td>
      <td>4.412301</td>
      <td>-0.410475</td>
      <td>1.033784</td>
      <td>0.068504</td>
      <td>1.033784</td>
      <td>0.758619</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.404494</td>
      <td>-1.298863</td>
      <td>0.271042</td>
      <td>1.657424</td>
      <td>-0.650812</td>
      <td>0.719503</td>
      <td>-0.210439</td>
      <td>-0.254307</td>
      <td>-1.691092</td>
      <td>-1.422043</td>
      <td>-1.404543</td>
      <td>-0.455624</td>
      <td>-0.483614</td>
      <td>-0.520436</td>
      <td>-1.404543</td>
      <td>-0.254307</td>
      <td>-0.399207</td>
      <td>0.129475</td>
      <td>-0.914893</td>
      <td>-0.198353</td>
      <td>-0.410475</td>
      <td>-0.654900</td>
      <td>-2.324944</td>
      <td>-0.654900</td>
      <td>0.533909</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.149484</td>
      <td>-0.839069</td>
      <td>-0.725346</td>
      <td>-0.387796</td>
      <td>-0.404768</td>
      <td>0.719503</td>
      <td>-0.210439</td>
      <td>-0.876959</td>
      <td>0.275280</td>
      <td>0.021567</td>
      <td>-0.718275</td>
      <td>0.453671</td>
      <td>-0.544350</td>
      <td>0.013863</td>
      <td>-0.718275</td>
      <td>-0.876959</td>
      <td>-0.399207</td>
      <td>-0.338198</td>
      <td>-0.914893</td>
      <td>-0.198353</td>
      <td>0.244627</td>
      <td>0.805583</td>
      <td>-0.430131</td>
      <td>0.805583</td>
      <td>-0.007406</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.925460</td>
      <td>0.540312</td>
      <td>-0.061087</td>
      <td>-0.046926</td>
      <td>-0.158724</td>
      <td>-0.879231</td>
      <td>-0.210439</td>
      <td>-0.067382</td>
      <td>0.275280</td>
      <td>0.512211</td>
      <td>-0.306513</td>
      <td>0.655736</td>
      <td>-0.415133</td>
      <td>0.137283</td>
      <td>-0.306513</td>
      <td>-0.067382</td>
      <td>-0.290844</td>
      <td>-0.338198</td>
      <td>0.681403</td>
      <td>-0.198353</td>
      <td>0.244627</td>
      <td>0.349182</td>
      <td>1.564408</td>
      <td>0.349182</td>
      <td>0.613731</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <th>9874</th>
      <td>-0.293496</td>
      <td>-0.379275</td>
      <td>0.271042</td>
      <td>-0.728666</td>
      <td>-0.404768</td>
      <td>0.719503</td>
      <td>-0.210439</td>
      <td>0.821656</td>
      <td>0.930738</td>
      <td>0.865408</td>
      <td>-0.260762</td>
      <td>1.868129</td>
      <td>1.020936</td>
      <td>1.303263</td>
      <td>-0.260762</td>
      <td>0.821656</td>
      <td>1.280419</td>
      <td>0.129475</td>
      <td>-0.914893</td>
      <td>-0.198353</td>
      <td>-0.410475</td>
      <td>0.531742</td>
      <td>-1.726582</td>
      <td>0.531742</td>
      <td>-0.833801</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9875</th>
      <td>1.759976</td>
      <td>-1.298863</td>
      <td>-0.061087</td>
      <td>-0.728666</td>
      <td>0.333364</td>
      <td>0.719503</td>
      <td>-0.210439</td>
      <td>-0.172894</td>
      <td>0.930738</td>
      <td>1.105315</td>
      <td>0.745765</td>
      <td>-0.253559</td>
      <td>0.312888</td>
      <td>0.479942</td>
      <td>0.745765</td>
      <td>-0.172894</td>
      <td>-0.561751</td>
      <td>8.547582</td>
      <td>-0.914893</td>
      <td>-0.198353</td>
      <td>0.244627</td>
      <td>-0.517980</td>
      <td>0.467412</td>
      <td>-0.517980</td>
      <td>-0.692938</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9876</th>
      <td>0.039499</td>
      <td>-0.839069</td>
      <td>-0.061087</td>
      <td>0.293944</td>
      <td>-0.404768</td>
      <td>-0.879231</td>
      <td>-0.210439</td>
      <td>-0.391082</td>
      <td>0.275280</td>
      <td>0.086541</td>
      <td>-0.306513</td>
      <td>-0.556657</td>
      <td>-0.990702</td>
      <td>-0.959957</td>
      <td>-0.306513</td>
      <td>-0.391082</td>
      <td>-0.453388</td>
      <td>-1.273543</td>
      <td>0.681403</td>
      <td>-0.198353</td>
      <td>1.554831</td>
      <td>1.992226</td>
      <td>0.866319</td>
      <td>1.992226</td>
      <td>1.227490</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9877</th>
      <td>-0.459994</td>
      <td>0.540312</td>
      <td>-1.389604</td>
      <td>-1.069536</td>
      <td>-0.896856</td>
      <td>0.719503</td>
      <td>-0.210439</td>
      <td>-1.331573</td>
      <td>-1.035635</td>
      <td>-0.582367</td>
      <td>0.334004</td>
      <td>-0.253559</td>
      <td>-0.347874</td>
      <td>-0.547516</td>
      <td>0.334004</td>
      <td>-1.331573</td>
      <td>2.364049</td>
      <td>0.597148</td>
      <td>-0.914893</td>
      <td>-0.198353</td>
      <td>0.899729</td>
      <td>1.353264</td>
      <td>-1.128220</td>
      <td>1.353264</td>
      <td>-0.798921</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9878</th>
      <td>-0.237997</td>
      <td>-1.298863</td>
      <td>-0.061087</td>
      <td>-0.046926</td>
      <td>-0.404768</td>
      <td>-0.879231</td>
      <td>-0.210439</td>
      <td>-0.154657</td>
      <td>0.275280</td>
      <td>-0.505730</td>
      <td>-0.443767</td>
      <td>-0.657690</td>
      <td>0.371994</td>
      <td>-0.012696</td>
      <td>-0.443767</td>
      <td>-0.154657</td>
      <td>-0.724295</td>
      <td>-0.338198</td>
      <td>0.681403</td>
      <td>-0.198353</td>
      <td>-0.410475</td>
      <td>-0.746180</td>
      <td>-0.529858</td>
      <td>-0.746180</td>
      <td>-0.771419</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>9879 rows × 30 columns</p>
</div>



### 학습 테스트 데이터 분리


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## 분류하기

### 회귀분석 모델


```python
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
from sklearn.metrics import classification_report

pred = model_lr.predict(X_test)
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
               0       0.73      0.75      0.74      1469
               1       0.75      0.73      0.74      1495
    
        accuracy                           0.74      2964
       macro avg       0.74      0.74      0.74      2964
    weighted avg       0.74      0.74      0.74      2964
    
    

### 앙상블 모델 적용(XGBoost)


```python
from xgboost import XGBClassifier

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
```

    C:\Users\dissi\anaconda31\lib\site-packages\xgboost\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [15:25:49] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=4, num_parallel_tree=1,
                  objective='binary:logistic', random_state=0, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', use_label_encoder=True,
                  validate_parameters=1, verbosity=None)




```python
pred = model_xgb.predict(X_test)
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
               0       0.71      0.71      0.71      1469
               1       0.72      0.72      0.72      1495
    
        accuracy                           0.72      2964
       macro avg       0.72      0.72      0.72      2964
    weighted avg       0.72      0.72      0.72      2964
    
    

## 모델평가

### 회귀분석 모델


```python
model_coef = pd.DataFrame(data=model_lr.coef_[0], index=X.columns, columns=['model coefficient'])
model_coef.sort_values(by='model coefficient', inplace=True)
plt.figure(figsize=(12, 10))
plt.barh(model_coef.index, model_coef['model coefficient'])
plt.show()
```


![output_43_0](https://user-images.githubusercontent.com/77723966/112033826-2d2e8e80-8b81-11eb-9f91-4d874e33cde5.png)



### xgboost 모델


```python
plt.figure(figsize=(12,10))
plt.barh(X.columns, model_xgb.feature_importances_)
plt.show()
```


![output_45_0](https://user-images.githubusercontent.com/77723966/112033843-30c21580-8b81-11eb-9996-c85fbdeadb8c.png)




```python

```


```python

```


```python

```
