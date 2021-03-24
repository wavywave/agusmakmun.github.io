---
layout: post
title:  "Kaggle - Happiness Report"
description: "캐글 데이터 분석"
author: SeungRok OH
categories: [Kaggle]
---

# Happiness Report Dataset

- 데이터 셋 : https://www.kaggle.com/mathurinache/world-happiness-report
- GDP, 기대수명과 같은 평가기준을 통해 각 나라의 행복도를 정리한 데이터. 전문기관에서 조사한 데이터이기에 잘 정리 되어 있고 점수로 표시 되어 있다는 특징이 있다. 변수간의 상관관계와 행복에 미치는 영향을 알아보고자 한다.
- 또한 변수들을 정리하여 표로 시각화하는 연습을 진행하고자 한다.

## 라이브러리 설정 및 데이터 불러오기


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = dict()
df['2015'] = pd.read_csv('./happiness/2015.csv')
df['2016'] = pd.read_csv('./happiness/2016.csv')
df['2017'] = pd.read_csv('./happiness/2017.csv')
df['2018'] = pd.read_csv('./happiness/2018.csv')
df['2019'] = pd.read_csv('./happiness/2019.csv')
df['2020'] = pd.read_csv('./happiness/2020.csv')

pd.set_option('display.max_columns', None)
```


```python
df['2020'].head()
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
      <th>Country name</th>
      <th>Regional indicator</th>
      <th>Ladder score</th>
      <th>Standard error of ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>7.8087</td>
      <td>0.031156</td>
      <td>7.869766</td>
      <td>7.747634</td>
      <td>10.639267</td>
      <td>0.954330</td>
      <td>71.900825</td>
      <td>0.949172</td>
      <td>-0.059482</td>
      <td>0.195445</td>
      <td>1.972317</td>
      <td>1.285190</td>
      <td>1.499526</td>
      <td>0.961271</td>
      <td>0.662317</td>
      <td>0.159670</td>
      <td>0.477857</td>
      <td>2.762835</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>7.6456</td>
      <td>0.033492</td>
      <td>7.711245</td>
      <td>7.579955</td>
      <td>10.774001</td>
      <td>0.955991</td>
      <td>72.402504</td>
      <td>0.951444</td>
      <td>0.066202</td>
      <td>0.168489</td>
      <td>1.972317</td>
      <td>1.326949</td>
      <td>1.503449</td>
      <td>0.979333</td>
      <td>0.665040</td>
      <td>0.242793</td>
      <td>0.495260</td>
      <td>2.432741</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>7.5599</td>
      <td>0.035014</td>
      <td>7.628528</td>
      <td>7.491272</td>
      <td>10.979933</td>
      <td>0.942847</td>
      <td>74.102448</td>
      <td>0.921337</td>
      <td>0.105911</td>
      <td>0.303728</td>
      <td>1.972317</td>
      <td>1.390774</td>
      <td>1.472403</td>
      <td>1.040533</td>
      <td>0.628954</td>
      <td>0.269056</td>
      <td>0.407946</td>
      <td>2.350267</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>7.5045</td>
      <td>0.059616</td>
      <td>7.621347</td>
      <td>7.387653</td>
      <td>10.772559</td>
      <td>0.974670</td>
      <td>73.000000</td>
      <td>0.948892</td>
      <td>0.246944</td>
      <td>0.711710</td>
      <td>1.972317</td>
      <td>1.326502</td>
      <td>1.547567</td>
      <td>1.000843</td>
      <td>0.661981</td>
      <td>0.362330</td>
      <td>0.144541</td>
      <td>2.460688</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>7.4880</td>
      <td>0.034837</td>
      <td>7.556281</td>
      <td>7.419719</td>
      <td>11.087804</td>
      <td>0.952487</td>
      <td>73.200783</td>
      <td>0.955750</td>
      <td>0.134533</td>
      <td>0.263218</td>
      <td>1.972317</td>
      <td>1.424207</td>
      <td>1.495173</td>
      <td>1.008072</td>
      <td>0.670201</td>
      <td>0.287985</td>
      <td>0.434101</td>
      <td>2.168266</td>
    </tr>
  </tbody>
</table>

</div>




```python
for key in df:
    print(key, df[key].columns)
```

    2015 Index(['Country', 'Region', 'Happiness Rank', 'Happiness Score',
           'Standard Error', 'Economy (GDP per Capita)', 'Family',
           'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
           'Generosity', 'Dystopia Residual'],
          dtype='object')
    2016 Index(['Country', 'Region', 'Happiness Rank', 'Happiness Score',
           'Lower Confidence Interval', 'Upper Confidence Interval',
           'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
           'Freedom', 'Trust (Government Corruption)', 'Generosity',
           'Dystopia Residual'],
          dtype='object')
    2017 Index(['Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.high',
           'Whisker.low', 'Economy..GDP.per.Capita.', 'Family',
           'Health..Life.Expectancy.', 'Freedom', 'Generosity',
           'Trust..Government.Corruption.', 'Dystopia.Residual'],
          dtype='object')
    2018 Index(['Overall rank', 'Country or region', 'Score', 'GDP per capita',
           'Social support', 'Healthy life expectancy',
           'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption'],
          dtype='object')
    2019 Index(['Overall rank', 'Country or region', 'Score', 'GDP per capita',
           'Social support', 'Healthy life expectancy',
           'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption'],
          dtype='object')
    2020 Index(['Country name', 'Regional indicator', 'Ladder score',
           'Standard error of ladder score', 'upperwhisker', 'lowerwhisker',
           'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
           'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption', 'Ladder score in Dystopia',
           'Explained by: Log GDP per capita', 'Explained by: Social support',
           'Explained by: Healthy life expectancy',
           'Explained by: Freedom to make life choices',
           'Explained by: Generosity', 'Explained by: Perceptions of corruption',
           'Dystopia + residual'],
          dtype='object')


- 각 데이터프레임 컬럼 확인 (연도마다 컬럼이 각각 다름.)
- 연도마다 컬럼이 다르므로 통합이 필요하다.

- Country : 국가
- Region : 국가의 지역
- Happiness Rank : 행복지수 순위
- Happiness Score : 행복지수점수
- GDP per capita : 1인당 GDP
- Healthy Life Expectancy : 건강 기대수명
- Social Support : 사회적 지원
- Freedom to make life choices : 삶에 대한 선택의 자유
- Generosity : 관용
- Corruption Perception : 부정부패
- Dystopia + Residual : 그 외 / Score 점수에서 나머지를 빼면 됨.


```python
# Columns 표준화
cols = ['country', 'score', 'economy', 'family', 'health', 'freedom', 'generosity', 'trust', 'residual']
```


```python
# Happiness Rank 경우 이 과정을 통해 측정.

df['2015'].drop(['Region', 'Happiness Rank', 'Standard Error'], axis=1, inplace=True) #Trust와 Generosity 순서반대

df['2016'].drop(['Region', 'Happiness Rank', 'Lower Confidence Interval',
                 'Upper Confidence Interval'], axis=1, inplace=True) #Trust와 Generosity 순서반대

df['2017'].drop(['Happiness.Rank', 'Whisker.high', 'Whisker.low'], axis=1, inplace=True) 

df['2018'].drop(['Overall rank'], axis=1, inplace=True) # residual 없음.

df['2019'].drop(['Overall rank'], axis=1, inplace=True) # residual 없음.

df['2020'].drop(['Regional indicator', 'Standard error of ladder score', 
                 'upperwhisker', 'lowerwhisker', 'Logged GDP per capita', 
                 'Social support', 'Healthy life expectancy', 
                 'Freedom to make life choices', 'Generosity',
                 'Perceptions of corruption', 'Ladder score in Dystopia'], axis=1, inplace=True) 

```

- 먼저 2018년도 2019년도 'residual' column을 만들어준다.


```python
df['2018'].columns
```




    Index(['Country or region', 'Score', 'GDP per capita', 'Social support',
           'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption'],
          dtype='object')




```python
df['2018'][['GDP per capita', 'Social support',
       'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']].sum(axis=1)
```




    0      5.047
    1      5.211
    2      5.184
    3      5.069
    4      5.169
           ...  
    151    2.249
    152    2.675
    153    1.564
    154    0.595
    155    1.153
    Length: 156, dtype: float64




```python
df['2019'].columns
```




    Index(['Country or region', 'Score', 'GDP per capita', 'Social support',
           'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption'],
          dtype='object')




```python
df['2018']['residual'] = df['2018']['Score'] - df['2018'][['GDP per capita', 'Social support',
       'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']].sum(axis=1)

df['2019']['residual'] = df['2019']['Score'] - df['2019'][['GDP per capita', 'Social support',
       'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']].sum(axis=1)
```

- 다음으로 2015년도, 2016년도 Trust와 Generosity 순서를 바꾸어야 한다.


```python
df['2015'].columns
```




    Index(['Country', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
           'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
           'Generosity', 'Dystopia Residual'],
          dtype='object')




```python
df['2016'].columns
```




    Index(['Country', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
           'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
           'Generosity', 'Dystopia Residual'],
          dtype='object')




```python
df['2015'] = df['2015'][['Country', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Freedom','Generosity', 'Trust (Government Corruption)',
       'Dystopia Residual']]

df['2016'] = df['2016'][['Country', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)',
        'Dystopia Residual']]
```

- 모든 연도의 컬럼명을 통일해준다.


```python
for year in df:
    df[year].columns = cols
```


```python
df['2015'].head()
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
      <th>country</th>
      <th>score</th>
      <th>economy</th>
      <th>family</th>
      <th>health</th>
      <th>freedom</th>
      <th>generosity</th>
      <th>trust</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Switzerland</td>
      <td>7.587</td>
      <td>1.39651</td>
      <td>1.34951</td>
      <td>0.94143</td>
      <td>0.66557</td>
      <td>0.29678</td>
      <td>0.41978</td>
      <td>2.51738</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iceland</td>
      <td>7.561</td>
      <td>1.30232</td>
      <td>1.40223</td>
      <td>0.94784</td>
      <td>0.62877</td>
      <td>0.43630</td>
      <td>0.14145</td>
      <td>2.70201</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>7.527</td>
      <td>1.32548</td>
      <td>1.36058</td>
      <td>0.87464</td>
      <td>0.64938</td>
      <td>0.34139</td>
      <td>0.48357</td>
      <td>2.49204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Norway</td>
      <td>7.522</td>
      <td>1.45900</td>
      <td>1.33095</td>
      <td>0.88521</td>
      <td>0.66973</td>
      <td>0.34699</td>
      <td>0.36503</td>
      <td>2.46531</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>7.427</td>
      <td>1.32629</td>
      <td>1.32261</td>
      <td>0.90563</td>
      <td>0.63297</td>
      <td>0.45811</td>
      <td>0.32957</td>
      <td>2.45176</td>
    </tr>
  </tbody>
</table>

</div>



### 데이터 통합


```python
df_all = pd.concat(df, axis=0)
# df를 dict타입으로 만듦. dict의 key(년도)가 index로 사용.

df_all.index.names = ['year', 'rank']
# 년도를 year로, 각 나라를 rank로 명명화

df_all.reset_index(inplace=True)
df_all['rank'] += 1
df_all
# 다시 year와 rank를 일반 column으로 변경. rank를 1부터 설정
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
      <th>year</th>
      <th>rank</th>
      <th>country</th>
      <th>score</th>
      <th>economy</th>
      <th>family</th>
      <th>health</th>
      <th>freedom</th>
      <th>generosity</th>
      <th>trust</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>1</td>
      <td>Switzerland</td>
      <td>7.5870</td>
      <td>1.396510</td>
      <td>1.349510</td>
      <td>0.941430</td>
      <td>0.665570</td>
      <td>0.296780</td>
      <td>0.419780</td>
      <td>2.517380</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>2</td>
      <td>Iceland</td>
      <td>7.5610</td>
      <td>1.302320</td>
      <td>1.402230</td>
      <td>0.947840</td>
      <td>0.628770</td>
      <td>0.436300</td>
      <td>0.141450</td>
      <td>2.702010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>3</td>
      <td>Denmark</td>
      <td>7.5270</td>
      <td>1.325480</td>
      <td>1.360580</td>
      <td>0.874640</td>
      <td>0.649380</td>
      <td>0.341390</td>
      <td>0.483570</td>
      <td>2.492040</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>4</td>
      <td>Norway</td>
      <td>7.5220</td>
      <td>1.459000</td>
      <td>1.330950</td>
      <td>0.885210</td>
      <td>0.669730</td>
      <td>0.346990</td>
      <td>0.365030</td>
      <td>2.465310</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>5</td>
      <td>Canada</td>
      <td>7.4270</td>
      <td>1.326290</td>
      <td>1.322610</td>
      <td>0.905630</td>
      <td>0.632970</td>
      <td>0.458110</td>
      <td>0.329570</td>
      <td>2.451760</td>
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
    </tr>
    <tr>
      <th>930</th>
      <td>2020</td>
      <td>149</td>
      <td>Central African Republic</td>
      <td>3.4759</td>
      <td>0.041072</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.292814</td>
      <td>0.253513</td>
      <td>0.028265</td>
      <td>2.860198</td>
    </tr>
    <tr>
      <th>931</th>
      <td>2020</td>
      <td>150</td>
      <td>Rwanda</td>
      <td>3.3123</td>
      <td>0.343243</td>
      <td>0.522876</td>
      <td>0.572383</td>
      <td>0.604088</td>
      <td>0.235705</td>
      <td>0.485542</td>
      <td>0.548445</td>
    </tr>
    <tr>
      <th>932</th>
      <td>2020</td>
      <td>151</td>
      <td>Zimbabwe</td>
      <td>3.2992</td>
      <td>0.425564</td>
      <td>1.047835</td>
      <td>0.375038</td>
      <td>0.377405</td>
      <td>0.151349</td>
      <td>0.080929</td>
      <td>0.841031</td>
    </tr>
    <tr>
      <th>933</th>
      <td>2020</td>
      <td>152</td>
      <td>South Sudan</td>
      <td>2.8166</td>
      <td>0.289083</td>
      <td>0.553279</td>
      <td>0.208809</td>
      <td>0.065609</td>
      <td>0.209935</td>
      <td>0.111157</td>
      <td>1.378751</td>
    </tr>
    <tr>
      <th>934</th>
      <td>2020</td>
      <td>153</td>
      <td>Afghanistan</td>
      <td>2.5669</td>
      <td>0.300706</td>
      <td>0.356434</td>
      <td>0.266052</td>
      <td>0.000000</td>
      <td>0.135235</td>
      <td>0.001226</td>
      <td>1.507236</td>
    </tr>
  </tbody>
</table>
<p>935 rows × 11 columns</p>

</div>




```python
rank_table = df_all.pivot(index='country', columns='year', values='rank')
rank_table
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
      <th>year</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
    </tr>
    <tr>
      <th>country</th>
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
      <th>Afghanistan</th>
      <td>153.0</td>
      <td>154.0</td>
      <td>141.0</td>
      <td>145.0</td>
      <td>154.0</td>
      <td>153.0</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>95.0</td>
      <td>109.0</td>
      <td>109.0</td>
      <td>112.0</td>
      <td>107.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>68.0</td>
      <td>38.0</td>
      <td>53.0</td>
      <td>84.0</td>
      <td>88.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>137.0</td>
      <td>141.0</td>
      <td>140.0</td>
      <td>142.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>30.0</td>
      <td>26.0</td>
      <td>24.0</td>
      <td>29.0</td>
      <td>47.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Venezuela</th>
      <td>23.0</td>
      <td>44.0</td>
      <td>82.0</td>
      <td>102.0</td>
      <td>108.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>Vietnam</th>
      <td>75.0</td>
      <td>96.0</td>
      <td>94.0</td>
      <td>95.0</td>
      <td>94.0</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>136.0</td>
      <td>147.0</td>
      <td>146.0</td>
      <td>152.0</td>
      <td>151.0</td>
      <td>146.0</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>85.0</td>
      <td>106.0</td>
      <td>116.0</td>
      <td>125.0</td>
      <td>138.0</td>
      <td>141.0</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>115.0</td>
      <td>131.0</td>
      <td>138.0</td>
      <td>144.0</td>
      <td>146.0</td>
      <td>151.0</td>
    </tr>
  </tbody>
</table>
<p>172 rows × 6 columns</p>

</div>



- 각 나라의 연도별로 행복 순위를 데이터프레임화 해봄.
- 매번 모든 나라가 순위에 드는 것이 아니기에 Nan값이 존재.
- index 기준으로 정리되어 있어 가시화가 미흡.


```python
rank_table.sort_values('2020', inplace=True) # sort_values / values = '2020' 기준으로 정리 
rank_table.head(10)
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
      <th>year</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
    </tr>
    <tr>
      <th>country</th>
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
      <th>Finland</th>
      <td>6.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Denmark</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Iceland</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Netherlands</th>
      <td>7.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>8.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>New Zealand</th>
      <td>9.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>13.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Luxembourg</th>
      <td>17.0</td>
      <td>20.0</td>
      <td>18.0</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>

</div>



## 시각화


```python
rank_table.max() # 기본이 axis=0 
```




    year
    2015    158.0
    2016    157.0
    2017    155.0
    2018    156.0
    2019    156.0
    2020    153.0
    dtype: float64



#### 각 나라별 기준 연도별 순위


```python
plt.figure(figsize=(10,50))
rank2020 = rank_table['2020'].dropna()

for c in rank2020.index: # c = 2020년 순위별 나라이름
    t = rank_table.loc[c].dropna() # 2020년 순위별기준 연도별 순위를 보여준다. dropna통해서 빈 데이터 그냥 드랍.
    plt.plot(t.index, t, '.-')
    
plt.xlim(['2015', '2020'])
plt.ylim([0, rank_table.max().max() +1 ]) #연도별 최고점 그 중에서도 최고점을 끝으로 정한다. +1은 타이트함 방지
plt.yticks(rank2020, rank2020.index)
ax = plt.gca() # 시각화자료를 받아오는 것.
ax.invert_yaxis() # 순위 정반대로 (1순위부터)

ax.yaxis.set_label_position('right')
ax.yaxis.tick_right() # y축 라벨과 점수를 같이 이동시켜줘야 보기에 용이

plt.tight_layout()
plt.show()
```


![output_30_0](https://user-images.githubusercontent.com/77723966/112319578-8f0e0600-8cf1-11eb-825a-b5a6df795c1c.png)



#### 분야별 상위 20개국 점수 현황


```python
data = df_all[df_all['year'] == '2020']
data = data.loc[data.index[:20]]
d = data[data.columns[4:]].cumsum(axis=1) 
#  누적합 pd.cumsum()

d
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
      <th>economy</th>
      <th>family</th>
      <th>health</th>
      <th>freedom</th>
      <th>generosity</th>
      <th>trust</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>782</th>
      <td>1.285190</td>
      <td>2.784715</td>
      <td>3.745987</td>
      <td>4.408304</td>
      <td>4.567974</td>
      <td>5.045831</td>
      <td>7.808666</td>
    </tr>
    <tr>
      <th>783</th>
      <td>1.326949</td>
      <td>2.830398</td>
      <td>3.809730</td>
      <td>4.474770</td>
      <td>4.717564</td>
      <td>5.212824</td>
      <td>7.645565</td>
    </tr>
    <tr>
      <th>784</th>
      <td>1.390774</td>
      <td>2.863178</td>
      <td>3.903711</td>
      <td>4.532665</td>
      <td>4.801721</td>
      <td>5.209667</td>
      <td>7.559934</td>
    </tr>
    <tr>
      <th>785</th>
      <td>1.326502</td>
      <td>2.874069</td>
      <td>3.874913</td>
      <td>4.536893</td>
      <td>4.899223</td>
      <td>5.043764</td>
      <td>7.504452</td>
    </tr>
    <tr>
      <th>786</th>
      <td>1.424207</td>
      <td>2.919380</td>
      <td>3.927452</td>
      <td>4.597653</td>
      <td>4.885638</td>
      <td>5.319738</td>
      <td>7.488005</td>
    </tr>
    <tr>
      <th>787</th>
      <td>1.338946</td>
      <td>2.802592</td>
      <td>3.778268</td>
      <td>4.391894</td>
      <td>4.728212</td>
      <td>5.096781</td>
      <td>7.448898</td>
    </tr>
    <tr>
      <th>788</th>
      <td>1.322235</td>
      <td>2.755583</td>
      <td>3.742053</td>
      <td>4.392351</td>
      <td>4.665179</td>
      <td>5.107245</td>
      <td>7.353545</td>
    </tr>
    <tr>
      <th>789</th>
      <td>1.242318</td>
      <td>2.729536</td>
      <td>3.737675</td>
      <td>4.384465</td>
      <td>4.710191</td>
      <td>5.171459</td>
      <td>7.299567</td>
    </tr>
    <tr>
      <th>790</th>
      <td>1.317286</td>
      <td>2.754730</td>
      <td>3.755664</td>
      <td>4.359033</td>
      <td>4.614543</td>
      <td>4.895799</td>
      <td>7.294245</td>
    </tr>
    <tr>
      <th>791</th>
      <td>1.536676</td>
      <td>2.924204</td>
      <td>3.910647</td>
      <td>4.520784</td>
      <td>4.716738</td>
      <td>5.083780</td>
      <td>7.237480</td>
    </tr>
    <tr>
      <th>792</th>
      <td>1.301648</td>
      <td>2.737040</td>
      <td>3.759542</td>
      <td>4.403570</td>
      <td>4.685099</td>
      <td>5.036800</td>
      <td>7.232070</td>
    </tr>
    <tr>
      <th>793</th>
      <td>1.310396</td>
      <td>2.787543</td>
      <td>3.810150</td>
      <td>4.432028</td>
      <td>4.757001</td>
      <td>5.092998</td>
      <td>7.222802</td>
    </tr>
    <tr>
      <th>794</th>
      <td>1.273061</td>
      <td>2.730906</td>
      <td>3.706606</td>
      <td>4.231775</td>
      <td>4.605208</td>
      <td>4.927810</td>
      <td>7.164532</td>
    </tr>
    <tr>
      <th>795</th>
      <td>1.216464</td>
      <td>2.619720</td>
      <td>3.627773</td>
      <td>4.048473</td>
      <td>4.315335</td>
      <td>4.415233</td>
      <td>7.128592</td>
    </tr>
    <tr>
      <th>796</th>
      <td>0.981108</td>
      <td>2.355961</td>
      <td>3.295597</td>
      <td>3.940614</td>
      <td>4.071881</td>
      <td>4.168243</td>
      <td>7.121378</td>
    </tr>
    <tr>
      <th>797</th>
      <td>1.446887</td>
      <td>2.917483</td>
      <td>3.893154</td>
      <td>4.480934</td>
      <td>4.776361</td>
      <td>5.149794</td>
      <td>7.093672</td>
    </tr>
    <tr>
      <th>798</th>
      <td>1.314185</td>
      <td>2.682728</td>
      <td>3.654843</td>
      <td>4.219117</td>
      <td>4.471155</td>
      <td>4.780517</td>
      <td>7.075767</td>
    </tr>
    <tr>
      <th>799</th>
      <td>1.373987</td>
      <td>2.778774</td>
      <td>3.610392</td>
      <td>4.145000</td>
      <td>4.443143</td>
      <td>4.595428</td>
      <td>6.939552</td>
    </tr>
    <tr>
      <th>800</th>
      <td>1.212322</td>
      <td>2.617609</td>
      <td>3.512173</td>
      <td>4.017918</td>
      <td>4.064244</td>
      <td>4.114047</td>
      <td>6.910855</td>
    </tr>
    <tr>
      <th>801</th>
      <td>1.295843</td>
      <td>2.694520</td>
      <td>3.659422</td>
      <td>4.159227</td>
      <td>4.306193</td>
      <td>4.514917</td>
      <td>6.863544</td>
    </tr>
  </tbody>
</table>

</div>




```python
# 누적합 가시화를 위해 d 뒤집어줌. residual 변수가 가장 많은 파이를 차지하기 때문.

data = df_all[df_all['year'] == '2020']
data = data.loc[data.index[:20]]
d = data[data.columns[4:]].cumsum(axis=1) 
#  누적합 pd.cumsum()

plt.figure(figsize=(6,8))

d = d[d.columns[::-1]]
d['country'] = data['country']

sns.set_color_codes('muted')
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'purple'][::-1]
for idx, c in enumerate(d.columns[:-1]):
    sns.barplot(data=d, x=c, y='country', label=c, color=colors[idx])
    
plt.legend(loc= 'lower right')
plt.title('Top 20 Happiness Scores in Details')
plt.xlabel('Happiness Score')
sns.despine(left=True, bottom=True)
```


![output_33_0](https://user-images.githubusercontent.com/77723966/112319603-96351400-8cf1-11eb-889a-1402c3f5dc2b.png)



#### 컬럼 간 상관성 시각화


```python
sns.heatmap(df_all.drop('rank', axis=1).corr(), annot=True, cmap='YlOrRd')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f0d101d308>




![output_35_1](https://user-images.githubusercontent.com/77723966/112319618-99c89b00-8cf1-11eb-84a4-72814777593e.png)


- 점수가 경제와 건강과 상관성이 높다. 그 다음으로는 가족과 자유정도 나머지 요소들 순이다.
- Residual 변수와 다른 변수간의 상관관계가 낮게 표현되는데 이는 나머지 변수에서 포착하지 못했던 것들을 residual이 가지고 있기 때문이다. 
- 경제와 건강간의 높은 상관성이 주목할만하다.

## 데이터 전처리

- 애초에 score는 다 표현되었기에 큰 의미가 있진 않음. 나머지 변수들로 residual을 예측해보는것이 의미가 있을 수 있음.


```python
col_in= ['economy', 'family', 'health', 'freedom', 'generosity', 'trust']
col_out = 'residual'
```

### 학습데이터 테스트 데이터 분리

- 2015~2019년도를 학습데이터 2020년도를 테스트데이터로 분리한다.


```python
df_train = df_all[df_all['year'] != '2020']
df_test = df_all[df_all['year'] == '2020']

X_train = df_train[col_in]
y_train = df_train[col_out]
X_test = df_test[col_in]
y_test = df_test[col_out]
```

### 수치형 데이터 표준화


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_norm = scaler.transform(X_train)
X_train = pd.DataFrame(X_norm, index=X_train.index, columns=X_train.columns)

X_norm = scaler.transform(X_test)
X_test = pd.DataFrame(X_norm, index=X_test.index, columns=X_test.columns)
```

- 지금까지 진행한 표준화 과정이랑 다른데 기존에 학습-테스트 데이터 분리가 자동적으로 이루어졌다면 지금은 2020년도를 테스트 데이터로 수동적으로 설정. 이에 표준화 과정 역시 X를 한꺼번에 하는것이 아닌 나누어서 진행하였다.
- scaler.fit 역시 학습데이터로만 진행하였는데 이 의미는 테스트 데이터를 더욱 철저하게 하겠다는 의미와 동일하고 모델을 통해 결과값을 예측한다기보다 검증적인 측면이 크기 때문에 이 방법 역시 유효하다 할 수 있다.

## 모델 학습 및 평가

### Linear 모델


```python
from sklearn.linear_model import LinearRegression
```


```python
df_all[df_all['trust'].isna()]
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
      <th>year</th>
      <th>rank</th>
      <th>country</th>
      <th>score</th>
      <th>economy</th>
      <th>family</th>
      <th>health</th>
      <th>freedom</th>
      <th>generosity</th>
      <th>trust</th>
      <th>residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>489</th>
      <td>2018</td>
      <td>20</td>
      <td>United Arab Emirates</td>
      <td>6.774</td>
      <td>2.096</td>
      <td>0.776</td>
      <td>0.67</td>
      <td>0.284</td>
      <td>0.186</td>
      <td>NaN</td>
      <td>2.762</td>
    </tr>
  </tbody>
</table>

</div>



- 해당 데이터 때문에 모델 적용이 불가능했는데 연산하여 보면 score 점수가 맞음.
- 즉 trust 점수가 0점. 0점으로 바꾸어준다.


```python
X_train.fillna(0, inplace=True)
```


```python
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

pred = model_lr.predict(X_test)
print(mean_absolute_error(y_test, pred))
print(sqrt(mean_squared_error(y_test, pred)))
```

    0.44680181487187914
    0.5629470546494035


### XGBoost 모델


```python
from xgboost import XGBRegressor

model_xgb = XGBRegressor()
model_xgb.fit(X_train, y_train)
```




    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                 min_child_weight=1, missing=nan, monotone_constraints='()',
                 n_estimators=100, n_jobs=4, num_parallel_tree=1,
                 objective='reg:squarederror', random_state=0, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
                 validate_parameters=1, verbosity=None)




```python
pred = model_xgb.predict(X_test)
print(mean_absolute_error(y_test, pred))
print(sqrt(mean_squared_error(y_test, pred)))
```

    0.4846164752194342
    0.6176769268710504

