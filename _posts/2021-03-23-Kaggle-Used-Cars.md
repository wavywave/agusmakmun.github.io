---
layout: post
title:  "Kaggle - Used Cars"
description: "캐글 데이터 분석"
author: SeungRok OH
categories: [Kaggle]
---

# Used Cars Dataset

- 데이터 셋 : https://www.kaggle.com/austinreese/craigslist-carstrucks-data

- 중고차가 가진 여러가지 변수를 통해 중고차의 가격을 예측하고자 하는 데이터 셋이다.

## 라이브러리 설정 및 데이터 읽어들이기


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('vehicles.csv')

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
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>url</th>
      <th>region</th>
      <th>region_url</th>
      <th>price</th>
      <th>year</th>
      <th>manufacturer</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>title_status</th>
      <th>transmission</th>
      <th>VIN</th>
      <th>drive</th>
      <th>size</th>
      <th>type</th>
      <th>paint_color</th>
      <th>image_url</th>
      <th>description</th>
      <th>state</th>
      <th>lat</th>
      <th>long</th>
      <th>posting_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7240372487</td>
      <td>https://auburn.craigslist.org/ctd/d/auburn-uni...</td>
      <td>auburn</td>
      <td>https://auburn.craigslist.org</td>
      <td>35990</td>
      <td>2010.0</td>
      <td>chevrolet</td>
      <td>corvette grand sport</td>
      <td>good</td>
      <td>8 cylinders</td>
      <td>gas</td>
      <td>32742.0</td>
      <td>clean</td>
      <td>other</td>
      <td>1G1YU3DW1A5106980</td>
      <td>rwd</td>
      <td>NaN</td>
      <td>other</td>
      <td>NaN</td>
      <td>https://images.craigslist.org/00N0N_ipkbHVZYf4...</td>
      <td>Carvana is the safer way to buy a car During t...</td>
      <td>al</td>
      <td>32.590000</td>
      <td>-85.480000</td>
      <td>2020-12-02T08:11:30-0600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7240309422</td>
      <td>https://auburn.craigslist.org/cto/d/auburn-201...</td>
      <td>auburn</td>
      <td>https://auburn.craigslist.org</td>
      <td>7500</td>
      <td>2014.0</td>
      <td>hyundai</td>
      <td>sonata</td>
      <td>excellent</td>
      <td>4 cylinders</td>
      <td>gas</td>
      <td>93600.0</td>
      <td>clean</td>
      <td>automatic</td>
      <td>5NPEC4AB0EH813529</td>
      <td>fwd</td>
      <td>NaN</td>
      <td>sedan</td>
      <td>NaN</td>
      <td>https://images.craigslist.org/00s0s_gBHYmJ5o7y...</td>
      <td>I'll move to another city and try to sell my c...</td>
      <td>al</td>
      <td>32.547500</td>
      <td>-85.468200</td>
      <td>2020-12-02T02:11:50-0600</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>7240224296</td>
      <td>https://auburn.craigslist.org/cto/d/auburn-200...</td>
      <td>auburn</td>
      <td>https://auburn.craigslist.org</td>
      <td>4900</td>
      <td>2006.0</td>
      <td>bmw</td>
      <td>x3 3.0i</td>
      <td>good</td>
      <td>6 cylinders</td>
      <td>gas</td>
      <td>87046.0</td>
      <td>clean</td>
      <td>automatic</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>SUV</td>
      <td>blue</td>
      <td>https://images.craigslist.org/00B0B_5zgEGWPOrt...</td>
      <td>Clean 2006 BMW X3 3.0I.  Beautiful and rare Bl...</td>
      <td>al</td>
      <td>32.616807</td>
      <td>-85.464149</td>
      <td>2020-12-01T19:50:41-0600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>7240103965</td>
      <td>https://auburn.craigslist.org/cto/d/lanett-tru...</td>
      <td>auburn</td>
      <td>https://auburn.craigslist.org</td>
      <td>2000</td>
      <td>1974.0</td>
      <td>chevrolet</td>
      <td>c-10</td>
      <td>good</td>
      <td>4 cylinders</td>
      <td>gas</td>
      <td>190000.0</td>
      <td>clean</td>
      <td>automatic</td>
      <td>NaN</td>
      <td>rwd</td>
      <td>full-size</td>
      <td>pickup</td>
      <td>blue</td>
      <td>https://images.craigslist.org/00M0M_6o7KcDpArw...</td>
      <td>1974 chev. truck (LONG BED) NEW starter front ...</td>
      <td>al</td>
      <td>32.861600</td>
      <td>-85.216100</td>
      <td>2020-12-01T15:54:45-0600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>7239983776</td>
      <td>https://auburn.craigslist.org/cto/d/auburn-200...</td>
      <td>auburn</td>
      <td>https://auburn.craigslist.org</td>
      <td>19500</td>
      <td>2005.0</td>
      <td>ford</td>
      <td>f350 lariat</td>
      <td>excellent</td>
      <td>8 cylinders</td>
      <td>diesel</td>
      <td>116000.0</td>
      <td>lien</td>
      <td>automatic</td>
      <td>NaN</td>
      <td>4wd</td>
      <td>full-size</td>
      <td>pickup</td>
      <td>blue</td>
      <td>https://images.craigslist.org/00p0p_b95l1EgUfl...</td>
      <td>2005 Ford F350 Lariat (Bullet Proofed). This t...</td>
      <td>al</td>
      <td>32.547500</td>
      <td>-85.468200</td>
      <td>2020-12-01T12:53:56-0600</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 458213 entries, 0 to 458212
    Data columns (total 26 columns):
     #   Column        Non-Null Count   Dtype  
    ---  ------        --------------   -----  
     0   Unnamed: 0    458213 non-null  int64  
     1   id            458213 non-null  int64  
     2   url           458213 non-null  object 
     3   region        458213 non-null  object 
     4   region_url    458213 non-null  object 
     5   price         458213 non-null  int64  
     6   year          457163 non-null  float64
     7   manufacturer  439993 non-null  object 
     8   model         453367 non-null  object 
     9   condition     265273 non-null  object 
     10  cylinders     287073 non-null  object 
     11  fuel          454976 non-null  object 
     12  odometer      402910 non-null  float64
     13  title_status  455636 non-null  object 
     14  transmission  455771 non-null  object 
     15  VIN           270664 non-null  object 
     16  drive         324025 non-null  object 
     17  size          136865 non-null  object 
     18  type          345475 non-null  object 
     19  paint_color   317370 non-null  object 
     20  image_url     458185 non-null  object 
     21  description   458143 non-null  object 
     22  state         458213 non-null  object 
     23  lat           450765 non-null  float64
     24  long          450765 non-null  float64
     25  posting_date  458185 non-null  object 
    dtypes: float64(4), int64(3), object(19)
    memory usage: 90.9+ MB


## EDA 및 기초 통계 분석


```python
df.isna().sum()
```




    Unnamed: 0           0
    id                   0
    url                  0
    region               0
    region_url           0
    price                0
    year              1050
    manufacturer     18220
    model             4846
    condition       192940
    cylinders       171140
    fuel              3237
    odometer         55303
    title_status      2577
    transmission      2442
    VIN             187549
    drive           134188
    size            321348
    type            112738
    paint_color     140843
    image_url           28
    description         70
    state                0
    lat               7448
    long              7448
    posting_date        28
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
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>price</th>
      <th>year</th>
      <th>odometer</th>
      <th>lat</th>
      <th>long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>458213.000000</td>
      <td>4.582130e+05</td>
      <td>4.582130e+05</td>
      <td>457163.000000</td>
      <td>4.029100e+05</td>
      <td>450765.000000</td>
      <td>450765.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>229106.000000</td>
      <td>7.235233e+09</td>
      <td>4.042093e+04</td>
      <td>2010.746067</td>
      <td>1.016698e+05</td>
      <td>38.531925</td>
      <td>-94.375824</td>
    </tr>
    <tr>
      <th>std</th>
      <td>132274.843786</td>
      <td>4.594362e+06</td>
      <td>8.194599e+06</td>
      <td>8.868136</td>
      <td>3.228623e+06</td>
      <td>5.857378</td>
      <td>18.076225</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>7.208550e+09</td>
      <td>0.000000e+00</td>
      <td>1900.000000</td>
      <td>0.000000e+00</td>
      <td>-82.607549</td>
      <td>-164.091797</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>114553.000000</td>
      <td>7.231953e+09</td>
      <td>4.900000e+03</td>
      <td>2008.000000</td>
      <td>4.087700e+04</td>
      <td>34.600000</td>
      <td>-110.890427</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>229106.000000</td>
      <td>7.236409e+09</td>
      <td>1.099500e+04</td>
      <td>2013.000000</td>
      <td>8.764100e+04</td>
      <td>39.244500</td>
      <td>-88.314889</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>343659.000000</td>
      <td>7.239321e+09</td>
      <td>2.149500e+04</td>
      <td>2016.000000</td>
      <td>1.340000e+05</td>
      <td>42.484503</td>
      <td>-81.015022</td>
    </tr>
    <tr>
      <th>max</th>
      <td>458212.000000</td>
      <td>7.241019e+09</td>
      <td>3.615215e+09</td>
      <td>2021.000000</td>
      <td>2.043756e+09</td>
      <td>82.049255</td>
      <td>150.898969</td>
    </tr>
  </tbody>
</table>

</div>



- 가격/ 평균값이 4만달러, 중위값이 천만달러 
- 가격 /최소값이 0, 최대값이 터무니없이 높음(잘못설정된것(아웃라이어))
- 주행거리 역시 잘못된 데이터가 있음.

### 불필요한 column 제거

- year를 연식(age)값으로 바꿈.


```python
df.columns
```




    Index(['Unnamed: 0', 'id', 'url', 'region', 'region_url', 'price', 'year',
           'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'odometer',
           'title_status', 'transmission', 'VIN', 'drive', 'size', 'type',
           'paint_color', 'image_url', 'description', 'state', 'lat', 'long',
           'posting_date'],
          dtype='object')




```python
df.drop(['Unnamed: 0', 'id', 'url', 'region_url', 'VIN',
         'image_url', 'description', 'state', 'lat', 'long',
         'posting_date'], axis=1, inplace=True)
```


```python
df['age'] = 2021 - df['year']
df.drop('year', axis=1, inplace=True)
```

### 범주형 데이터 분석


```python
len(df['manufacturer'].value_counts())
```




    43




```python
plt.figure(figsize=(8,10))
sns.countplot(data=df.fillna('n/a'), y='manufacturer', order=df.fillna('n/a')['manufacturer'].value_counts().index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c5b58bfb88>




![output_17_1](https://user-images.githubusercontent.com/77723966/112030636-cf4c7780-8b7d-11eb-9d2b-2ac117cbd00f.png)




```python
len(df['model'].value_counts())

# for model, num in zip(df['model'].value_counts().index df['model'].value_counts()):
# print(model, num)
```




    31520




```python
len(df['condition'].value_counts())
```




    6




```python
len(df['cylinders'].value_counts())
```




    8




```python
sns.countplot(data=df.fillna('n/a'), y='transmission', order=df.fillna('n/a')['transmission'].value_counts().index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c5b5961c08>




![output_21_1](https://user-images.githubusercontent.com/77723966/112030681-da9fa300-8b7d-11eb-91aa-8d28e957bde8.png)




```python
sns.countplot(data=df.fillna('n/a'), y='drive', order=df.fillna('n/a')['drive'].value_counts().index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c5b594f788>




![output_22_1](https://user-images.githubusercontent.com/77723966/112030696-decbc080-8b7d-11eb-8224-c39132607ca2.png)




```python
sns.countplot(data=df.fillna('n/a'), y='size', order=df.fillna('n/a')['size'].value_counts().index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c5ba1cfc88>




![output_23_1](https://user-images.githubusercontent.com/77723966/112030720-e3907480-8b7d-11eb-944a-2e0cf2cec69c.png)




```python
sns.countplot(data=df.fillna('n/a'), y='type', order=df.fillna('n/a')['type'].value_counts().index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c5b9daf8c8>




![output_24_1](https://user-images.githubusercontent.com/77723966/112030735-e723fb80-8b7d-11eb-9cbb-e660ffa19902.png)




```python
sns.countplot(data=df.fillna('n/a'), y='paint_color', order=df.fillna('n/a')['paint_color'].value_counts().index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c5a9223d88>




![output_25_1](https://user-images.githubusercontent.com/77723966/112030747-eb501900-8b7d-11eb-824c-749833cba526.png)



### 수치형 데이터 분석


```python
df.columns
```




    Index(['region', 'price', 'manufacturer', 'model', 'condition', 'cylinders',
           'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size',
           'type', 'paint_color', 'age'],
          dtype='object')




```python
plt.figure(figsize=(8,2))
sns.rugplot(data=df, x='odometer', height=1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c5b4522d48>




![output_28_1](https://user-images.githubusercontent.com/77723966/112030783-f3a85400-8b7d-11eb-8f1c-894815864c2a.png)




```python
sns.histplot(data=df, x='age', bins=20, kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c59ed64a48>




![output_29_1](https://user-images.githubusercontent.com/77723966/112030799-f86d0800-8b7d-11eb-8590-38ed07f3a82d.png)



## 데이터 전처리

#### 범주형 데이터 전처리


```python
sns.boxplot(data=df.fillna('n/a'), x='manufacturer', y='price')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c5b5162c48>



![output_32_1](https://user-images.githubusercontent.com/77723966/112030825-fe62e900-8b7d-11eb-851c-9d417a03516b.png)


- 아웃라이어가 심각


```python
df.columns
```




    Index(['region', 'price', 'manufacturer', 'model', 'condition', 'cylinders',
           'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size',
           'type', 'paint_color', 'age'],
          dtype='object')



- 해당 Columns 들이 어느정도까지가 유의미한 데이터인지 확인하기 위해 plot를 그림.


```python
col= 'manufacturer'
counts = df[col].fillna('others').value_counts()
plt.grid()
plt.plot(range(len(counts)), counts)

# 10 정도가 적당해보임.
```




    [<matplotlib.lines.Line2D at 0x1c6857feec8>]




![output_36_1](https://user-images.githubusercontent.com/77723966/112030842-03c03380-8b7e-11eb-90d5-a661505fa958.png)




```python
# 상위 10개에 포함되지 않는 분류의 경우 others로 취급
n_categorical = 10 
counts.index[n_categorical:]
df[col] = df[col].apply(lambda s : s if str(s) not in counts.index[n_categorical:] else 'others')
```


```python
df['manufacturer'].value_counts()
```




    others       134392
    ford          79666
    chevrolet     64977
    toyota        38577
    honda         25868
    nissan        23654
    jeep          21165
    ram           17697
    gmc           17267
    dodge         16730
    Name: manufacturer, dtype: int64




```python
col= 'model'
counts = df[col].fillna('others').value_counts()
plt.grid()
plt.plot(range(len(counts)), counts)
```




    [<matplotlib.lines.Line2D at 0x1c684f1e808>]



![output_39_1](https://user-images.githubusercontent.com/77723966/112030884-0f135f00-8b7e-11eb-807c-1ab2e5c1963e.png)




```python
col= 'model'
counts = df[col].fillna('others').value_counts()
plt.grid()
plt.plot(range(len(counts[:20])), counts[:20])
```




    [<matplotlib.lines.Line2D at 0x1c684fad2c8>]




![output_40_1](https://user-images.githubusercontent.com/77723966/112030898-13d81300-8b7e-11eb-930c-d1c65241f36e.png)




```python
# manufacturer 와 달리 분류가 엄청 많아서 lambda 실행으로 느리게 작동할 수 있음.
n_categorical = 10 
#counts.index[n_categorical:]
# df[col] = df[col].apply(lambda s : s if str(s) not in counts.index[n_categorical:] else 'others')
others = counts.index[n_categorical:]
df[col] = df[col].apply(lambda s : s if str(s) not in others else 'others')

df['model'].value_counts()
```




    others            413556
    f-150               8370
    silverado 1500      5964
    1500                4211
    camry               4033
    accord              3730
    altima              3490
    civic               3479
    escape              3444
    silverado           3090
    Name: model, dtype: int64




```python
col = 'condition'
counts = df[col].fillna('others').value_counts()
n_categorical = 3 
others = counts.index[n_categorical:]
df[col] = df[col].apply(lambda s : s if str(s) not in others else 'others')
```


```python
col = 'cylinders'
counts = df[col].fillna('others').value_counts()
n_categorical = 4 
others = counts.index[n_categorical:]
df[col] = df[col].apply(lambda s : s if str(s) not in others else 'others')
```


```python
col = 'fuel'
# counts = df[col].fillna('others').value_counts()
# counts.index

n_categorical = 2
others = counts.index[n_categorical:]
df[col] = df[col].apply(lambda s : s if str(s) not in others else 'others')
```


```python
df.drop('title_status', axis=1, inplace=True)
```


```python
col = 'transmission'
counts = df[col].fillna('others').value_counts()
n_categorical = 3
others = counts.index[n_categorical:]
df[col] = df[col].apply(lambda s : s if str(s) not in others else 'others')
```


```python
col = 'drive'
df[col].fillna('others', inplace=True)
```


```python
col = 'size'
counts = df[col].fillna('others').value_counts()
n_categorical = 2
others = counts.index[n_categorical:]
df[col] = df[col].apply(lambda s : s if str(s) not in others else 'others')
```


```python
col = 'type'
counts = df[col].fillna('others').value_counts()
n_categorical = 8
others = counts.index[n_categorical:]
df[col] = df[col].apply(lambda s : s if str(s) not in others else 'others')

df.loc[df[col] == 'other', col] = 'others'

# other이란 분류도 있어서 others에 편입.
```


```python
col = 'paint_color'
counts = df[col].fillna('others').value_counts()
n_categorical = 7
others = counts.index[n_categorical:]
df[col] = df[col].apply(lambda s : s if str(s) not in others else 'others')
```

#### 수치형 데이터 전처리


```python
# age는 양호, odometer와 price 조정 필요.
p1 = df['price'].quantile(0.99)
p2 = df['price'].quantile(0.1)
print(p1, p2)

# 가격 = 상위 1% , 하위 10%를 제거하여 아웃라이어 제거.
```

    59900.0 651.0



```python
df = df[(p1 > df['price']) & (df['price'] > p2)]
```


```python
o1 = df['odometer'].quantile(0.99)
o2 = df['odometer'].quantile(0.1)
print(o1, o2)

# 주행거리 = 상위, 하위 1% 제거하여 아웃라이어 제거.
```

    270000.0 17553.0



```python
df = df[(o1 > df['odometer']) & (df['odometer'] > o2)]
```


```python
df.describe()

# 아웃라이어 제거 하여 수치형 데이터 개괄
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
      <th>price</th>
      <th>odometer</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>324382.000000</td>
      <td>324382.000000</td>
      <td>323860.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>15314.530106</td>
      <td>102569.319602</td>
      <td>10.174001</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11298.917484</td>
      <td>55165.135400</td>
      <td>7.076283</td>
    </tr>
    <tr>
      <th>min</th>
      <td>652.000000</td>
      <td>17555.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6500.000000</td>
      <td>56199.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12388.000000</td>
      <td>98146.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>21000.000000</td>
      <td>140482.750000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>59895.000000</td>
      <td>269930.000000</td>
      <td>121.000000</td>
    </tr>
  </tbody>
</table>

</div>




```python
plt.figure(figsize=(10,8))
sns.boxplot(data=df, x='manufacturer', y='price')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c684c59cc8>




![output_57_1](https://user-images.githubusercontent.com/77723966/112030935-1e92a800-8b7e-11eb-9fec-7e273416271c.png)



- 전체적인 범위가 다르지 않지만 평균 값들을 비교할 만하다.


```python
plt.figure(figsize=(10,8))
sns.boxplot(data=df, x='model', y='price')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c684c6db08>




![output_59_1](https://user-images.githubusercontent.com/77723966/112030949-23575c00-8b7e-11eb-92ba-6d42e05d4794.png)



- 같은 모델이라도 상태, 주행거리에 따라 가격이 크게 달라짐. 물론 모델마다 다르기도함.


```python
sns.heatmap(df.corr(), annot=True, cmap='YlOrRd')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c684ef1408>




![output_61_1](https://user-images.githubusercontent.com/77723966/112030962-281c1000-8b7e-11eb-9df7-ca614a4ab5e3.png)



- 상관성은 높으나 주행거리, 연식 둘다 가격에 역방향으로 영향을 줌.
- 주행거리와 연식도 당연하게도 영향이 있음.
- 두개 모두 사용하면 비효율적인 모델이 될 수 있음.


```python
from sklearn.preprocessing import StandardScaler

X_num = df[['odometer', 'age']]

scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns=X_num.columns)

# 범주형 데이터 one-hot 벡터로
X_cat = df.drop(['price', 'odometer', 'age'], axis=1)
X_cat = pd.get_dummies(X_cat)

X = pd.concat([X_scaled, X_cat], axis=1)
y = df['price']
```


```python
X.head()
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
      <th>odometer</th>
      <th>age</th>
      <th>region_SF bay area</th>
      <th>region_abilene</th>
      <th>region_akron / canton</th>
      <th>region_albany</th>
      <th>region_albuquerque</th>
      <th>region_altoona-johnstown</th>
      <th>region_amarillo</th>
      <th>region_ames</th>
      <th>region_anchorage / mat-su</th>
      <th>region_ann arbor</th>
      <th>region_annapolis</th>
      <th>region_appleton-oshkosh-FDL</th>
      <th>region_asheville</th>
      <th>region_ashtabula</th>
      <th>region_athens</th>
      <th>region_atlanta</th>
      <th>region_auburn</th>
      <th>region_augusta</th>
      <th>region_austin</th>
      <th>region_bakersfield</th>
      <th>region_baltimore</th>
      <th>region_baton rouge</th>
      <th>region_battle creek</th>
      <th>region_beaumont / port arthur</th>
      <th>region_bellingham</th>
      <th>region_bemidji</th>
      <th>region_bend</th>
      <th>region_billings</th>
      <th>region_binghamton</th>
      <th>region_birmingham</th>
      <th>region_bismarck</th>
      <th>region_bloomington</th>
      <th>region_bloomington-normal</th>
      <th>region_boise</th>
      <th>region_boone</th>
      <th>region_boston</th>
      <th>region_boulder</th>
      <th>region_bowling green</th>
      <th>region_bozeman</th>
      <th>region_brainerd</th>
      <th>region_brownsville</th>
      <th>region_brunswick</th>
      <th>region_buffalo</th>
      <th>region_butte</th>
      <th>region_cape cod / islands</th>
      <th>region_catskills</th>
      <th>region_cedar rapids</th>
      <th>region_central NJ</th>
      <th>region_central louisiana</th>
      <th>region_central michigan</th>
      <th>region_champaign urbana</th>
      <th>region_charleston</th>
      <th>region_charlotte</th>
      <th>region_charlottesville</th>
      <th>region_chattanooga</th>
      <th>region_chautauqua</th>
      <th>region_chicago</th>
      <th>region_chico</th>
      <th>region_chillicothe</th>
      <th>region_cincinnati</th>
      <th>region_clarksville</th>
      <th>region_cleveland</th>
      <th>region_clovis / portales</th>
      <th>region_college station</th>
      <th>region_colorado springs</th>
      <th>region_columbia</th>
      <th>region_columbia / jeff city</th>
      <th>region_columbus</th>
      <th>region_cookeville</th>
      <th>region_corpus christi</th>
      <th>region_corvallis/albany</th>
      <th>region_cumberland valley</th>
      <th>region_dallas / fort worth</th>
      <th>region_danville</th>
      <th>region_dayton / springfield</th>
      <th>region_daytona beach</th>
      <th>region_decatur</th>
      <th>region_deep east texas</th>
      <th>region_del rio / eagle pass</th>
      <th>region_delaware</th>
      <th>region_denver</th>
      <th>region_des moines</th>
      <th>region_detroit metro</th>
      <th>region_dothan</th>
      <th>region_dubuque</th>
      <th>region_duluth / superior</th>
      <th>region_east idaho</th>
      <th>region_east oregon</th>
      <th>region_eastern CO</th>
      <th>region_eastern CT</th>
      <th>region_eastern NC</th>
      <th>region_eastern kentucky</th>
      <th>region_eastern montana</th>
      <th>region_eastern panhandle</th>
      <th>region_eastern shore</th>
      <th>region_eau claire</th>
      <th>region_el paso</th>
      <th>region_elko</th>
      <th>region_elmira-corning</th>
      <th>region_erie</th>
      <th>region_eugene</th>
      <th>region_evansville</th>
      <th>region_fairbanks</th>
      <th>region_fargo / moorhead</th>
      <th>region_farmington</th>
      <th>region_fayetteville</th>
      <th>region_finger lakes</th>
      <th>region_flagstaff / sedona</th>
      <th>region_flint</th>
      <th>region_florence</th>
      <th>region_florence / muscle shoals</th>
      <th>region_florida keys</th>
      <th>region_fort collins / north CO</th>
      <th>region_fort dodge</th>
      <th>region_fort smith</th>
      <th>region_fort smith, AR</th>
      <th>region_fort wayne</th>
      <th>region_frederick</th>
      <th>region_fredericksburg</th>
      <th>region_fresno / madera</th>
      <th>region_ft myers / SW florida</th>
      <th>region_gadsden-anniston</th>
      <th>region_gainesville</th>
      <th>region_galveston</th>
      <th>region_glens falls</th>
      <th>region_gold country</th>
      <th>region_grand forks</th>
      <th>region_grand island</th>
      <th>region_grand rapids</th>
      <th>region_great falls</th>
      <th>region_green bay</th>
      <th>region_greensboro</th>
      <th>region_greenville / upstate</th>
      <th>region_gulfport / biloxi</th>
      <th>region_hanford-corcoran</th>
      <th>region_harrisburg</th>
      <th>region_harrisonburg</th>
      <th>region_hartford</th>
      <th>region_hattiesburg</th>
      <th>region_hawaii</th>
      <th>region_heartland florida</th>
      <th>region_helena</th>
      <th>region_hickory / lenoir</th>
      <th>region_high rockies</th>
      <th>region_hilton head</th>
      <th>region_holland</th>
      <th>region_houma</th>
      <th>region_houston</th>
      <th>region_hudson valley</th>
      <th>region_humboldt county</th>
      <th>region_huntington-ashland</th>
      <th>region_huntsville / decatur</th>
      <th>region_imperial county</th>
      <th>region_indianapolis</th>
      <th>region_inland empire</th>
      <th>region_iowa city</th>
      <th>region_ithaca</th>
      <th>region_jackson</th>
      <th>region_jacksonville</th>
      <th>region_janesville</th>
      <th>region_jersey shore</th>
      <th>region_jonesboro</th>
      <th>region_joplin</th>
      <th>region_kalamazoo</th>
      <th>region_kalispell</th>
      <th>region_kansas city</th>
      <th>region_kansas city, MO</th>
      <th>region_kenai peninsula</th>
      <th>region_kennewick-pasco-richland</th>
      <th>region_kenosha-racine</th>
      <th>region_killeen / temple / ft hood</th>
      <th>region_kirksville</th>
      <th>region_klamath falls</th>
      <th>region_knoxville</th>
      <th>region_kokomo</th>
      <th>region_la crosse</th>
      <th>region_la salle co</th>
      <th>region_lafayette</th>
      <th>region_lafayette / west lafayette</th>
      <th>region_lake charles</th>
      <th>region_lake of the ozarks</th>
      <th>region_lakeland</th>
      <th>region_lancaster</th>
      <th>region_lansing</th>
      <th>region_laredo</th>
      <th>region_las cruces</th>
      <th>region_las vegas</th>
      <th>region_lawrence</th>
      <th>region_lawton</th>
      <th>region_lehigh valley</th>
      <th>region_lewiston / clarkston</th>
      <th>region_lexington</th>
      <th>region_lima / findlay</th>
      <th>region_lincoln</th>
      <th>region_little rock</th>
      <th>region_logan</th>
      <th>region_long island</th>
      <th>region_los angeles</th>
      <th>region_louisville</th>
      <th>region_lubbock</th>
      <th>region_lynchburg</th>
      <th>region_macon / warner robins</th>
      <th>region_madison</th>
      <th>region_maine</th>
      <th>region_manhattan</th>
      <th>region_mankato</th>
      <th>region_mansfield</th>
      <th>region_mason city</th>
      <th>region_mattoon-charleston</th>
      <th>region_mcallen / edinburg</th>
      <th>region_meadville</th>
      <th>region_medford-ashland</th>
      <th>region_memphis</th>
      <th>region_mendocino county</th>
      <th>region_merced</th>
      <th>region_meridian</th>
      <th>region_milwaukee</th>
      <th>region_minneapolis / st paul</th>
      <th>region_missoula</th>
      <th>region_mobile</th>
      <th>region_modesto</th>
      <th>region_mohave county</th>
      <th>region_monroe</th>
      <th>region_monterey bay</th>
      <th>region_montgomery</th>
      <th>region_morgantown</th>
      <th>region_moses lake</th>
      <th>region_muncie / anderson</th>
      <th>region_muskegon</th>
      <th>region_myrtle beach</th>
      <th>region_nashville</th>
      <th>region_new hampshire</th>
      <th>region_new haven</th>
      <th>region_new orleans</th>
      <th>region_new river valley</th>
      <th>region_new york city</th>
      <th>region_norfolk / hampton roads</th>
      <th>region_north central FL</th>
      <th>region_north dakota</th>
      <th>region_north jersey</th>
      <th>region_north mississippi</th>
      <th>region_north platte</th>
      <th>region_northeast SD</th>
      <th>region_northern WI</th>
      <th>region_northern michigan</th>
      <th>region_northern panhandle</th>
      <th>region_northwest CT</th>
      <th>region_northwest GA</th>
      <th>region_northwest KS</th>
      <th>region_northwest OK</th>
      <th>region_ocala</th>
      <th>region_odessa / midland</th>
      <th>region_ogden-clearfield</th>
      <th>region_okaloosa / walton</th>
      <th>region_oklahoma city</th>
      <th>region_olympic peninsula</th>
      <th>region_omaha / council bluffs</th>
      <th>region_oneonta</th>
      <th>region_orange county</th>
      <th>region_oregon coast</th>
      <th>region_orlando</th>
      <th>region_outer banks</th>
      <th>region_owensboro</th>
      <th>region_palm springs</th>
      <th>region_panama city</th>
      <th>region_parkersburg-marietta</th>
      <th>region_pensacola</th>
      <th>region_peoria</th>
      <th>region_philadelphia</th>
      <th>region_phoenix</th>
      <th>region_pierre / central SD</th>
      <th>region_pittsburgh</th>
      <th>region_plattsburgh-adirondacks</th>
      <th>region_poconos</th>
      <th>region_port huron</th>
      <th>region_portland</th>
      <th>region_potsdam-canton-massena</th>
      <th>region_prescott</th>
      <th>region_provo / orem</th>
      <th>region_pueblo</th>
      <th>region_pullman / moscow</th>
      <th>region_quad cities, IA/IL</th>
      <th>region_raleigh / durham / CH</th>
      <th>region_rapid city / west SD</th>
      <th>region_reading</th>
      <th>region_redding</th>
      <th>region_reno / tahoe</th>
      <th>region_rhode island</th>
      <th>region_richmond</th>
      <th>region_roanoke</th>
      <th>region_rochester</th>
      <th>region_rockford</th>
      <th>region_roseburg</th>
      <th>region_roswell / carlsbad</th>
      <th>region_sacramento</th>
      <th>region_saginaw-midland-baycity</th>
      <th>region_salem</th>
      <th>region_salina</th>
      <th>region_salt lake city</th>
      <th>region_san angelo</th>
      <th>region_san antonio</th>
      <th>region_san diego</th>
      <th>region_san luis obispo</th>
      <th>region_san marcos</th>
      <th>region_sandusky</th>
      <th>region_santa barbara</th>
      <th>region_santa fe / taos</th>
      <th>region_santa maria</th>
      <th>region_sarasota-bradenton</th>
      <th>region_savannah / hinesville</th>
      <th>region_scottsbluff / panhandle</th>
      <th>region_scranton / wilkes-barre</th>
      <th>region_seattle-tacoma</th>
      <th>region_sheboygan</th>
      <th>region_show low</th>
      <th>region_shreveport</th>
      <th>region_sierra vista</th>
      <th>region_sioux city</th>
      <th>region_sioux city, IA</th>
      <th>region_sioux falls / SE SD</th>
      <th>region_siskiyou county</th>
      <th>region_skagit / island / SJI</th>
      <th>region_south bend / michiana</th>
      <th>region_south coast</th>
      <th>region_south dakota</th>
      <th>region_south florida</th>
      <th>region_south jersey</th>
      <th>region_southeast IA</th>
      <th>region_southeast KS</th>
      <th>region_southeast alaska</th>
      <th>region_southeast missouri</th>
      <th>region_southern WV</th>
      <th>region_southern illinois</th>
      <th>region_southern maryland</th>
      <th>region_southwest KS</th>
      <th>region_southwest MN</th>
      <th>region_southwest MS</th>
      <th>region_southwest TX</th>
      <th>region_southwest VA</th>
      <th>region_southwest michigan</th>
      <th>region_space coast</th>
      <th>region_spokane / coeur d'alene</th>
      <th>region_springfield</th>
      <th>region_st augustine</th>
      <th>region_st cloud</th>
      <th>region_st george</th>
      <th>region_st joseph</th>
      <th>region_st louis</th>
      <th>region_st louis, MO</th>
      <th>region_state college</th>
      <th>region_statesboro</th>
      <th>region_stillwater</th>
      <th>region_stockton</th>
      <th>region_susanville</th>
      <th>region_syracuse</th>
      <th>region_tallahassee</th>
      <th>region_tampa bay area</th>
      <th>region_terre haute</th>
      <th>region_texarkana</th>
      <th>region_texoma</th>
      <th>region_the thumb</th>
      <th>region_toledo</th>
      <th>region_topeka</th>
      <th>region_treasure coast</th>
      <th>region_tri-cities</th>
      <th>region_tucson</th>
      <th>region_tulsa</th>
      <th>region_tuscaloosa</th>
      <th>region_tuscarawas co</th>
      <th>region_twin falls</th>
      <th>region_twin tiers NY/PA</th>
      <th>region_tyler / east TX</th>
      <th>region_upper peninsula</th>
      <th>region_utica-rome-oneida</th>
      <th>region_valdosta</th>
      <th>region_ventura county</th>
      <th>region_vermont</th>
      <th>region_victoria</th>
      <th>region_visalia-tulare</th>
      <th>region_waco</th>
      <th>region_washington, DC</th>
      <th>region_waterloo / cedar falls</th>
      <th>region_watertown</th>
      <th>region_wausau</th>
      <th>region_wenatchee</th>
      <th>region_west virginia (old)</th>
      <th>region_western IL</th>
      <th>region_western KY</th>
      <th>region_western maryland</th>
      <th>region_western massachusetts</th>
      <th>region_western slope</th>
      <th>region_wichita</th>
      <th>region_wichita falls</th>
      <th>region_williamsport</th>
      <th>region_wilmington</th>
      <th>region_winchester</th>
      <th>region_winston-salem</th>
      <th>region_worcester / central MA</th>
      <th>region_wyoming</th>
      <th>region_yakima</th>
      <th>region_york</th>
      <th>region_youngstown</th>
      <th>region_yuba-sutter</th>
      <th>region_yuma</th>
      <th>region_zanesville / cambridge</th>
      <th>manufacturer_chevrolet</th>
      <th>manufacturer_dodge</th>
      <th>manufacturer_ford</th>
      <th>manufacturer_gmc</th>
      <th>manufacturer_honda</th>
      <th>manufacturer_jeep</th>
      <th>manufacturer_nissan</th>
      <th>manufacturer_others</th>
      <th>manufacturer_ram</th>
      <th>manufacturer_toyota</th>
      <th>model_1500</th>
      <th>model_accord</th>
      <th>model_altima</th>
      <th>model_camry</th>
      <th>model_civic</th>
      <th>model_escape</th>
      <th>model_f-150</th>
      <th>model_others</th>
      <th>model_silverado</th>
      <th>model_silverado 1500</th>
      <th>condition_excellent</th>
      <th>condition_good</th>
      <th>condition_others</th>
      <th>cylinders_4 cylinders</th>
      <th>cylinders_6 cylinders</th>
      <th>cylinders_8 cylinders</th>
      <th>cylinders_others</th>
      <th>fuel_diesel</th>
      <th>fuel_electric</th>
      <th>fuel_gas</th>
      <th>fuel_hybrid</th>
      <th>fuel_others</th>
      <th>transmission_automatic</th>
      <th>transmission_manual</th>
      <th>transmission_other</th>
      <th>drive_4wd</th>
      <th>drive_fwd</th>
      <th>drive_others</th>
      <th>drive_rwd</th>
      <th>size_full-size</th>
      <th>size_others</th>
      <th>type_SUV</th>
      <th>type_coupe</th>
      <th>type_hatchback</th>
      <th>type_others</th>
      <th>type_pickup</th>
      <th>type_sedan</th>
      <th>type_truck</th>
      <th>paint_color_black</th>
      <th>paint_color_blue</th>
      <th>paint_color_grey</th>
      <th>paint_color_others</th>
      <th>paint_color_red</th>
      <th>paint_color_silver</th>
      <th>paint_color_white</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.265789</td>
      <td>0.116728</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>-0.162591</td>
      <td>-0.448541</td>
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
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.281398</td>
      <td>0.681997</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.584893</td>
      <td>5.204152</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.243464</td>
      <td>0.823315</td>
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
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
X.isna().sum()
```




    odometer                   0
    age                      522
    region_SF bay area         0
    region_abilene             0
    region_akron / canton      0
                            ... 
    paint_color_grey           0
    paint_color_others         0
    paint_color_red            0
    paint_color_silver         0
    paint_color_white          0
    Length: 462, dtype: int64




```python
X['age'].mean()
```




    2.4065666124846312e-15




```python
X['age'].fillna(0.0, inplace=True)

# 평균값이 0에 가깝고 따라서 nan값을 0으로 채워준다.
```

#### 학습데이터 테스트데이터 분리


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## 분류하기 (Regression 모델)

#### XGBoost Regression


```python
from xgboost import XGBRegressor

model_reg = XGBRegressor()
model_reg.fit(X_train, y_train)
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
```


```python
pred = model_reg.predict(X_test)
print(mean_absolute_error(y_test, pred))
print(sqrt(mean_squared_error(y_test, pred)))
```

    3208.492644616665
    4863.626919465305


## 결과 분석


```python
plt.scatter(x=y_test, y=pred, alpha=0.005)
plt.plot([0,60000], [0,60000], 'r-')
```




    [<matplotlib.lines.Line2D at 0x1c5b73ec108>]




![output_76_1](https://user-images.githubusercontent.com/77723966/112031057-4124c100-8b7e-11eb-97e8-3daea7c437b7.png)


- 실제로 값이 엄청 저렴한데 높게 책정하는 경우가 있고 전체적으로 underestimate 하는 경향이 있다.


```python
err = (pred - y_test) / y_test * 100
sns.histplot(err)
plt.xlabel('error(%)')
plt.xlim(-100, 100)

# 오차율에 관한 히스토그램
```




    (-100, 100)




![output_78_1](https://user-images.githubusercontent.com/77723966/112031079-46820b80-8b7e-11eb-83a0-37b2f9fd8cb6.png)


- 위 결론과 마찬가지로 underestimate 되는 경향이 주로 보이고, overestimate되는 값은 오차율이 크다.


```python
sns.histplot(x=y_test, y=pred)
plt.plot([0,60000], [0,60000], 'r-')
```




    [<matplotlib.lines.Line2D at 0x1c681b0a148>]




![output_80_1](https://user-images.githubusercontent.com/77723966/112031099-4b46bf80-8b7e-11eb-9bee-8299d34d3f60.png)

