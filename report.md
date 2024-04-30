---
layout: wide_default
---  
```python
import fnmatch
import glob
import os
import re
from time import sleep
from zipfile import ZipFile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from near_regex import NEAR_regex  # copy this file into the folder this script is in
from tqdm import tqdm  # progress bar on loops

# if you have tqdm issues, run this in terminal or with ! trick
# jupyter nbextension enable --py widgetsnbextension
# jupyter labextension install @jupyter-widgets/jupyterlab-manager
#
# if that fails, you can disable it

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import seaborn as sns
import matplotlib.pyplot as plt
import pandas_datareader as pdr

os.makedirs("output", exist_ok=True)
```

## Summary

The goal of this project is to answer the question of whether 10-K filings contain value-relevant information in the sentiment of the text and if the sentiment impacts stock returns in the days following the report. My hypothesis is that the sentiment score is correlated with the stock returns and that a higher sentiment score will result in a higher stock return around the 10-K's release. 

## Data 

The data for this project consists of the 503 firms that were in the S&P 500 index in the 2022 fiscal year. The list of firms were found on this webpage: [S&P500](https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1130173030). The list of firms was used to download a 10-K file for each individual firm from the SEC EDGAR database. Initially, the plan was to download the 10-Ks using the tickers for each firm, but this became an issue since tickers for companies such as Activision became invalid. This is possible due to mergers and acquisitions and bankruptcy. Instead of using tickers, CIK was used as permanent firm ID variables are more reliable. After the 10-Ks were downloaded, I added 10 sentiment variables to the data. The first four variables were "LM Positive", "LM Negative, "ML Positive" and "ML Negative" from the sentiment dictionary in the inputs folder. The next six variables were created for 3 different topics focused on the sentiment of the text in a 10-K. The topics are employee, litigation, and economy. The data also includes firm buy and hold return. One of them was from the day of the filing to two days later and the other was from 3 days after the filing to 10 days later. 


```python
sp500 = pd.read_csv('output/analysis_sample.csv')
sp500

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
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>filing_date</th>
      <th>...</th>
      <th>mb</th>
      <th>prof_a</th>
      <th>ppe_a</th>
      <th>cash_a</th>
      <th>xrd_a</th>
      <th>dltt_a</th>
      <th>invopps_FG09</th>
      <th>sales_g</th>
      <th>dv_a</th>
      <th>short_debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>2022-02-09</td>
      <td>...</td>
      <td>2.838265</td>
      <td>0.197931</td>
      <td>0.218538</td>
      <td>0.101228</td>
      <td>0.042361</td>
      <td>0.355625</td>
      <td>2.564301</td>
      <td>0.098527</td>
      <td>0.072655</td>
      <td>0.086095</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>2022-02-11</td>
      <td>...</td>
      <td>4.368153</td>
      <td>0.197847</td>
      <td>0.183974</td>
      <td>0.181729</td>
      <td>0.027113</td>
      <td>0.061075</td>
      <td>NaN</td>
      <td>0.222291</td>
      <td>0.048958</td>
      <td>0.080191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>2022-02-18</td>
      <td>...</td>
      <td>3.825614</td>
      <td>0.166285</td>
      <td>0.134475</td>
      <td>0.136297</td>
      <td>0.036465</td>
      <td>0.242726</td>
      <td>3.559664</td>
      <td>0.244654</td>
      <td>0.042582</td>
      <td>0.051893</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>2022-02-18</td>
      <td>...</td>
      <td>2.528878</td>
      <td>0.194433</td>
      <td>0.040074</td>
      <td>0.067086</td>
      <td>0.054911</td>
      <td>0.442929</td>
      <td>2.144449</td>
      <td>0.227438</td>
      <td>0.063203</td>
      <td>0.163364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>2022-10-12</td>
      <td>...</td>
      <td>5.474851</td>
      <td>0.195625</td>
      <td>0.111674</td>
      <td>0.189283</td>
      <td>0.025902</td>
      <td>0.063702</td>
      <td>5.023477</td>
      <td>0.140013</td>
      <td>0.051790</td>
      <td>0.215661</td>
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
      <th>498</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
      <td>2022-02-23</td>
      <td>...</td>
      <td>9.129993</td>
      <td>0.395240</td>
      <td>0.337915</td>
      <td>0.123366</td>
      <td>0.000000</td>
      <td>1.019505</td>
      <td>8.944086</td>
      <td>0.164897</td>
      <td>0.099229</td>
      <td>0.012864</td>
    </tr>
    <tr>
      <th>499</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
      <td>2022-02-10</td>
      <td>...</td>
      <td>5.635335</td>
      <td>0.192759</td>
      <td>0.064843</td>
      <td>0.055350</td>
      <td>0.091231</td>
      <td>0.167820</td>
      <td>5.301699</td>
      <td>0.265063</td>
      <td>0.000000</td>
      <td>0.089083</td>
    </tr>
    <tr>
      <th>500</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
      <td>2022-02-25</td>
      <td>...</td>
      <td>1.592191</td>
      <td>0.092759</td>
      <td>0.097530</td>
      <td>0.020400</td>
      <td>0.021892</td>
      <td>0.242318</td>
      <td>1.415104</td>
      <td>0.115553</td>
      <td>0.008531</td>
      <td>0.227553</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
      <td>2022-02-25</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>2022-02-15</td>
      <td>...</td>
      <td>8.969729</td>
      <td>0.236475</td>
      <td>0.187266</td>
      <td>0.250719</td>
      <td>0.036547</td>
      <td>0.485108</td>
      <td>8.792744</td>
      <td>0.164349</td>
      <td>0.034101</td>
      <td>0.006044</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 89 columns</p>
</div>



### Creating return variables

The first step for creating the return variables was finding the filing date of each 10-K file from SEC EDGAR. Each firm's page could only be accessed by using the accession number and CIK. I already had the CIK from the dataset but I did not have the accession number. I needed to scrape the accession number from the file path of the 10-Ks downloaded. This was completed using this code: ```accession_number = fpath.split('/')[-2] #splits the path by forward slash and select second to last element```

Once I got the filing date, I would need to scan through the file to find the closes matching date: 

```filing_date = soup.find('div', text = 'Filing Date').find_next_sibling('div').text.strip() #looks for the filing date```

Once I was able to scrape the filing dates, I saved it in the sp500 dataframe. Furthermore, I created another for loop to iterrate through each row of the SP500 dataframe to get the correct returns. The returns were located in a different dataframe called sp_ret which consisted of the symbol, date and return. The below are the steps I took to calculate the firm's buy and hold return over each time span: 

1) Within the for loop, I had to locate the parts of the sp500 dataframe and the sp_ret dataframe where they had the same symbol and date. This led to the index of the date.
2) For the for loop, I also ignored weekends and holidays
3) Using the index, I was able to pull the returns from the day of the filing to two indexes after which is day t+2.
4) Then I added 1 and cumulated the returns and put it into a list. The last row of the list was the cumulative returns for that time span.
5) Finally, I created a new column to store the returns from dayt to day t+2. 
6) For version 3 of the buy and hold return, I repeated the same process but I started on the index + 3 (day t+3) and ended on index + 10 (day t+10)

### Sentiment Variables

#### First Four Sentiment Variables

The sentiment variables consists of a positive sentiment and negative sentiment which are words in a 10-K that are either "positive" or "negative". The goal is to measure the percentage of words in the 10-K files that were either positive or negative. For these words, we were given multiple files consisting of different words. These files included the LM sentiment dictionary and the ML sentiment list which comes from a machine learning approach. Here is a count of words in each dictionary:

* LM positive dictionary: 347
* LM negative dictionary: 2345
* ML positive dictionary: 75
* ML negative dictionary: 94


In order to use these files, I opened the text files using ```with open('inputs\ML_negative_unigram.txt', 'r') as file:```. I had to split the positive words into one list and the negative words into a separate list. Once I got each word from the text file, I had to convert the list into the form "(Detroit Lions|Red Wings|Detroit Tigers|Detroit Pistons)" with each word separated by pipes. This was done using the following code: 

```BHR_negative = ['('+'|'.join(BHR_negative)+')']```

I repeated the process for both the ML and LM lists. I also had to make sure that each word was in lowercase. This is to ensure that the format for the list is suitable for the NEAR_regex() function.

The next step was to go through the zip file for each 10-K and convert the html into a form that was readable. 

```     with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
            soup = BeautifulSoup(html,features='lxml-xml') 

        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.text.lower()
        remove_punc = re.sub(r'\W',' ', lower)   # no punctuation
        cleaned = re.sub(r'\s+',' ',remove_punc)  # excess whtiespace
```



This was done to remove punctuation and whitespace and to only include the words in the file. This process is important for the use of the NEAR_regex() function. With the NEAR_regex() function I was able to find positive or negative sentiment words. I counted the number of positive or negative sentiment words in a file and divided it by the total count of words in the file to determine the percentage of positive or negative sentiment words. 



#### Last 6 of the 10 variables

For the last 6 variables, I picked the topics employee, litigation and economy. These topics were picked based on their risk factor and it's ability to represent how well the company is performing. The list of words for each topic needed to encompass the majority of words in the 10-K files. I picked words based on prior knowledge in the topic, additional research and the use of Chat-GPT. Once the words were compiled, I needed to clean the list to fit the parameters for NEAR_regex() similar to the cleaning process for the first four variables. After cleaning the lists, I used NEAR_regex() function to determine how many times a word of that topic appears near a positive or negative sentiment word. 

#### NEAR_regex()

When I used the NEAR_regex() function for the ten sentiment variables, there were different parameters that can be modified to fit the list of words: 

* I set partial = False since I did not want to include words that were not part of the original list of words for the topics
* For max_words_between which specifies how many words between the word list and the target word, I decided to use 6 for this project. This means that it looks for a total of 12 words near the target word. 
    * I believe that the target word and sentiment word would not to be related if the range was greater than 12 words. 

### Summary of Final Analysis

First, I organized and stored the columns for the new variables in a variable called target_var. The final sample size was 503 observations. However, there were a couple of rows that had missing data which could stem from the fact that a 10-K was not filed that year. This would make it difficult to perform an analysis on that dataset including analysis on the returns of the firms and sentiment in the 10-K. As a result, I removed the rows that had missing information. 


```python
new = sp500[['T_to_T2', 'T3_to_T10']]
new.isnull().sum()
```




    T_to_T2      11
    T3_to_T10    11
    dtype: int64




```python
target_var = sp500[['Symbol', 'filing_date', 'BHR_positive', 'BHR_negative', 'LM_positive', 'LM_negative', 'Employee_positive', 'Employee_negative', 'Litigation_negative', 'Litigation_positive', 'Economy_positive', 'Economy_negative']]
target_var.isnull().sum()
```




    Symbol                 0
    filing_date            2
    BHR_positive           2
    BHR_negative           2
    LM_positive            2
    LM_negative            2
    Employee_positive      2
    Employee_negative      2
    Litigation_negative    2
    Litigation_positive    2
    Economy_positive       2
    Economy_negative       2
    dtype: int64




```python
target_var = sp500[['Symbol', 'filing_date', 'BHR_positive', 'BHR_negative', 'LM_positive', 'LM_negative', 'Employee_positive', 'Employee_negative', 'Litigation_negative', 'Litigation_positive', 'Economy_positive', 'Economy_negative', 'T_to_T2', 'T3_to_T10']]
target_var = target_var.dropna()
target_var

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
      <th>Symbol</th>
      <th>filing_date</th>
      <th>BHR_positive</th>
      <th>BHR_negative</th>
      <th>LM_positive</th>
      <th>LM_negative</th>
      <th>Employee_positive</th>
      <th>Employee_negative</th>
      <th>Litigation_negative</th>
      <th>Litigation_positive</th>
      <th>Economy_positive</th>
      <th>Economy_negative</th>
      <th>T_to_T2</th>
      <th>T3_to_T10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>2022-02-09</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>0.003977</td>
      <td>0.023249</td>
      <td>0.001047</td>
      <td>0.001321</td>
      <td>0.002826</td>
      <td>0.001073</td>
      <td>0.002996</td>
      <td>0.003899</td>
      <td>-0.017671</td>
      <td>-0.090256</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>2022-02-11</td>
      <td>0.024460</td>
      <td>0.023602</td>
      <td>0.003756</td>
      <td>0.012984</td>
      <td>0.001124</td>
      <td>0.000769</td>
      <td>0.001686</td>
      <td>0.000769</td>
      <td>0.002662</td>
      <td>0.003401</td>
      <td>0.003184</td>
      <td>-0.053547</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>2022-02-18</td>
      <td>0.021590</td>
      <td>0.024394</td>
      <td>0.003726</td>
      <td>0.012793</td>
      <td>0.000922</td>
      <td>0.001287</td>
      <td>0.001479</td>
      <td>0.000730</td>
      <td>0.002977</td>
      <td>0.002862</td>
      <td>-0.027617</td>
      <td>0.013731</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>2022-02-18</td>
      <td>0.019753</td>
      <td>0.022645</td>
      <td>0.006481</td>
      <td>0.015448</td>
      <td>0.000747</td>
      <td>0.001300</td>
      <td>0.001949</td>
      <td>0.000958</td>
      <td>0.002599</td>
      <td>0.002827</td>
      <td>0.012347</td>
      <td>0.018329</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>2022-10-12</td>
      <td>0.027968</td>
      <td>0.023964</td>
      <td>0.008642</td>
      <td>0.016861</td>
      <td>0.001482</td>
      <td>0.001444</td>
      <td>0.001155</td>
      <td>0.000462</td>
      <td>0.004889</td>
      <td>0.003638</td>
      <td>0.003446</td>
      <td>0.107431</td>
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
    </tr>
    <tr>
      <th>498</th>
      <td>YUM</td>
      <td>2022-02-23</td>
      <td>0.026854</td>
      <td>0.027307</td>
      <td>0.004718</td>
      <td>0.015997</td>
      <td>0.001390</td>
      <td>0.001390</td>
      <td>0.001531</td>
      <td>0.000656</td>
      <td>0.003390</td>
      <td>0.004218</td>
      <td>-0.002821</td>
      <td>-0.062237</td>
    </tr>
    <tr>
      <th>499</th>
      <td>ZBRA</td>
      <td>2022-02-10</td>
      <td>0.028396</td>
      <td>0.026842</td>
      <td>0.006258</td>
      <td>0.014964</td>
      <td>0.002022</td>
      <td>0.001447</td>
      <td>0.001362</td>
      <td>0.001149</td>
      <td>0.003470</td>
      <td>0.004193</td>
      <td>-0.157306</td>
      <td>-0.019582</td>
    </tr>
    <tr>
      <th>500</th>
      <td>ZBH</td>
      <td>2022-02-25</td>
      <td>0.021506</td>
      <td>0.026759</td>
      <td>0.004591</td>
      <td>0.021783</td>
      <td>0.000739</td>
      <td>0.001309</td>
      <td>0.002372</td>
      <td>0.001001</td>
      <td>0.002650</td>
      <td>0.002788</td>
      <td>0.026020</td>
      <td>-0.042855</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZION</td>
      <td>2022-02-25</td>
      <td>0.019965</td>
      <td>0.023898</td>
      <td>0.003962</td>
      <td>0.014992</td>
      <td>0.000812</td>
      <td>0.000841</td>
      <td>0.001083</td>
      <td>0.000570</td>
      <td>0.002736</td>
      <td>0.003534</td>
      <td>-0.030312</td>
      <td>-0.011241</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZTS</td>
      <td>2022-02-15</td>
      <td>0.021790</td>
      <td>0.033508</td>
      <td>0.005036</td>
      <td>0.019980</td>
      <td>0.001237</td>
      <td>0.001836</td>
      <td>0.002435</td>
      <td>0.000854</td>
      <td>0.002563</td>
      <td>0.003723</td>
      <td>-0.017105</td>
      <td>0.015538</td>
    </tr>
  </tbody>
</table>
<p>492 rows × 14 columns</p>
</div>



.describe() is used to visualize the summary statistics for the final analysis sample
* It is important to note that after dropping the rows with missing data, there are 492 observations. 
* There is information for both the return variables and sentiment variables which indicates that there is a mean and variance in the data
* The result of the positive and negative sentiment variables appear to be really small


```python
target_var.describe()
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
      <th>BHR_positive</th>
      <th>BHR_negative</th>
      <th>LM_positive</th>
      <th>LM_negative</th>
      <th>Employee_positive</th>
      <th>Employee_negative</th>
      <th>Litigation_negative</th>
      <th>Litigation_positive</th>
      <th>Economy_positive</th>
      <th>Economy_negative</th>
      <th>T_to_T2</th>
      <th>T3_to_T10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.023956</td>
      <td>0.025908</td>
      <td>0.004985</td>
      <td>0.015875</td>
      <td>0.001097</td>
      <td>0.001270</td>
      <td>0.001505</td>
      <td>0.000712</td>
      <td>0.003305</td>
      <td>0.003632</td>
      <td>0.003420</td>
      <td>-0.009128</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.003505</td>
      <td>0.003395</td>
      <td>0.001321</td>
      <td>0.003659</td>
      <td>0.000388</td>
      <td>0.000420</td>
      <td>0.000589</td>
      <td>0.000271</td>
      <td>0.001030</td>
      <td>0.000992</td>
      <td>0.052119</td>
      <td>0.064819</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.007966</td>
      <td>0.008953</td>
      <td>0.001226</td>
      <td>0.006609</td>
      <td>0.000000</td>
      <td>0.000102</td>
      <td>0.000307</td>
      <td>0.000188</td>
      <td>0.000613</td>
      <td>0.000679</td>
      <td>-0.447499</td>
      <td>-0.288483</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.021896</td>
      <td>0.023977</td>
      <td>0.004092</td>
      <td>0.013297</td>
      <td>0.000833</td>
      <td>0.000977</td>
      <td>0.001100</td>
      <td>0.000507</td>
      <td>0.002628</td>
      <td>0.002937</td>
      <td>-0.025323</td>
      <td>-0.048425</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.024122</td>
      <td>0.025897</td>
      <td>0.004904</td>
      <td>0.015633</td>
      <td>0.001061</td>
      <td>0.001246</td>
      <td>0.001412</td>
      <td>0.000688</td>
      <td>0.003200</td>
      <td>0.003503</td>
      <td>0.000627</td>
      <td>-0.010736</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.026142</td>
      <td>0.027813</td>
      <td>0.005651</td>
      <td>0.017815</td>
      <td>0.001318</td>
      <td>0.001527</td>
      <td>0.001754</td>
      <td>0.000871</td>
      <td>0.003879</td>
      <td>0.004194</td>
      <td>0.028560</td>
      <td>0.028827</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.037982</td>
      <td>0.038030</td>
      <td>0.010899</td>
      <td>0.030185</td>
      <td>0.002608</td>
      <td>0.002859</td>
      <td>0.004612</td>
      <td>0.002170</td>
      <td>0.007401</td>
      <td>0.007128</td>
      <td>0.229167</td>
      <td>0.332299</td>
    </tr>
  </tbody>
</table>
</div>




```python
industry_check = sp500[['Employee_positive', 'Employee_negative', 'Litigation_negative', 'Litigation_positive', 'Economy_positive', 'Economy_negative', 'GICS Sector']]
```


```python
group_sector1 = industry_check.groupby('GICS Sector')[['Employee_positive', 'Employee_negative', 'Litigation_positive', 'Litigation_negative', 'Economy_positive', 'Economy_negative']].mean()
group_sector1 = group_sector1.reset_index().melt(id_vars= ['GICS Sector'], var_name='Sentiment', value_name='Mean Sentiment')
plt.figure(figsize=(12, 6))
sns.barplot(data = group_sector1, x = 'Sentiment', y = 'Mean Sentiment', hue = 'GICS Sector')
```




    <Axes: xlabel='Sentiment', ylabel='Mean Sentiment'>




    
![png](output_23_1.png)
    


Other notes: 
* For the topic of economy, I would expect the consumer industry to talk about the economy the most. 
    * From the bar graph, it appears that this is true as consumer staples has one of the highest mean for both positive and negative sentiments
* I also expected litigation negative to be more focused on the financial industry but it does not appear that way

## Results

### The heatmap below shows the correlation between the sentiment variables the two return variables


```python
corrl_matrix = target_var.corr()
sns.heatmap(corrl_matrix.loc[['Employee_positive', 'Employee_negative', 'Litigation_positive', 'Litigation_negative', 'Economy_positive', 'Economy_negative', 'BHR_positive', 'BHR_negative', 'LM_positive', 'LM_negative'], 
                             ['T_to_T2', 'T3_to_T10']], annot=True, cmap='RdYlGn')
```




    <Axes: >




    
![png](output_27_1.png)
    


From the heatmap, I could not find conclusive evidence that a higher positive sentiment would lead to a higher return. There are several positive sentiment variables that have a negative return. The result in the heatmap could also be an issue with the topics I chose and the list of words for each topic. However, I also noticed an issue with the sentiment variable dictionary such as the ML list and LM list. For BHR_negative, I would expect the return for both time frames to be negative. There is not enough data to make a conclusion about the relationship between the sentiment variables and the two return variables. 

### Scatterplot


```python
sentiment = target_var[['Employee_positive', 'Employee_negative', 'Litigation_negative', 'Litigation_positive', 'Economy_positive', 'Economy_negative', 'BHR_positive', 'BHR_negative', 'LM_positive', 'LM_negative']]
return_var = target_var[['T_to_T2', 'T3_to_T10']]
sns.pairplot(data = target_var, x_vars = sentiment, y_vars = return_var)
```




    <seaborn.axisgrid.PairGrid at 0x139d609fb20>




    
![png](output_30_1.png)
    



```python
corrl_matrix
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
      <th>BHR_positive</th>
      <th>BHR_negative</th>
      <th>LM_positive</th>
      <th>LM_negative</th>
      <th>Employee_positive</th>
      <th>Employee_negative</th>
      <th>Litigation_negative</th>
      <th>Litigation_positive</th>
      <th>Economy_positive</th>
      <th>Economy_negative</th>
      <th>T_to_T2</th>
      <th>T3_to_T10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BHR_positive</th>
      <td>1.000000</td>
      <td>0.370839</td>
      <td>0.277210</td>
      <td>-0.085388</td>
      <td>0.388248</td>
      <td>0.267082</td>
      <td>-0.254454</td>
      <td>-0.140295</td>
      <td>0.644255</td>
      <td>0.450157</td>
      <td>0.043834</td>
      <td>-0.045126</td>
    </tr>
    <tr>
      <th>BHR_negative</th>
      <td>0.370839</td>
      <td>1.000000</td>
      <td>0.082306</td>
      <td>0.431131</td>
      <td>0.109548</td>
      <td>0.308212</td>
      <td>0.351862</td>
      <td>0.284178</td>
      <td>0.409398</td>
      <td>0.596120</td>
      <td>0.083837</td>
      <td>0.051295</td>
    </tr>
    <tr>
      <th>LM_positive</th>
      <td>0.277210</td>
      <td>0.082306</td>
      <td>1.000000</td>
      <td>0.241424</td>
      <td>0.422862</td>
      <td>0.418162</td>
      <td>0.062552</td>
      <td>0.009720</td>
      <td>0.278919</td>
      <td>0.165930</td>
      <td>-0.093768</td>
      <td>-0.042054</td>
    </tr>
    <tr>
      <th>LM_negative</th>
      <td>-0.085388</td>
      <td>0.431131</td>
      <td>0.241424</td>
      <td>1.000000</td>
      <td>0.121071</td>
      <td>0.294292</td>
      <td>0.613984</td>
      <td>0.413640</td>
      <td>0.125797</td>
      <td>0.306711</td>
      <td>-0.014040</td>
      <td>-0.123211</td>
    </tr>
    <tr>
      <th>Employee_positive</th>
      <td>0.388248</td>
      <td>0.109548</td>
      <td>0.422862</td>
      <td>0.121071</td>
      <td>1.000000</td>
      <td>0.736012</td>
      <td>0.065340</td>
      <td>0.065383</td>
      <td>0.196918</td>
      <td>0.167068</td>
      <td>-0.022144</td>
      <td>0.051800</td>
    </tr>
    <tr>
      <th>Employee_negative</th>
      <td>0.267082</td>
      <td>0.308212</td>
      <td>0.418162</td>
      <td>0.294292</td>
      <td>0.736012</td>
      <td>1.000000</td>
      <td>0.205882</td>
      <td>0.110189</td>
      <td>0.190662</td>
      <td>0.198795</td>
      <td>0.033049</td>
      <td>0.042164</td>
    </tr>
    <tr>
      <th>Litigation_negative</th>
      <td>-0.254454</td>
      <td>0.351862</td>
      <td>0.062552</td>
      <td>0.613984</td>
      <td>0.065340</td>
      <td>0.205882</td>
      <td>1.000000</td>
      <td>0.711395</td>
      <td>-0.123946</td>
      <td>0.110639</td>
      <td>0.009125</td>
      <td>0.045859</td>
    </tr>
    <tr>
      <th>Litigation_positive</th>
      <td>-0.140295</td>
      <td>0.284178</td>
      <td>0.009720</td>
      <td>0.413640</td>
      <td>0.065383</td>
      <td>0.110189</td>
      <td>0.711395</td>
      <td>1.000000</td>
      <td>-0.041993</td>
      <td>0.156508</td>
      <td>-0.006482</td>
      <td>0.047871</td>
    </tr>
    <tr>
      <th>Economy_positive</th>
      <td>0.644255</td>
      <td>0.409398</td>
      <td>0.278919</td>
      <td>0.125797</td>
      <td>0.196918</td>
      <td>0.190662</td>
      <td>-0.123946</td>
      <td>-0.041993</td>
      <td>1.000000</td>
      <td>0.790573</td>
      <td>0.028657</td>
      <td>-0.012278</td>
    </tr>
    <tr>
      <th>Economy_negative</th>
      <td>0.450157</td>
      <td>0.596120</td>
      <td>0.165930</td>
      <td>0.306711</td>
      <td>0.167068</td>
      <td>0.198795</td>
      <td>0.110639</td>
      <td>0.156508</td>
      <td>0.790573</td>
      <td>1.000000</td>
      <td>-0.009993</td>
      <td>0.016640</td>
    </tr>
    <tr>
      <th>T_to_T2</th>
      <td>0.043834</td>
      <td>0.083837</td>
      <td>-0.093768</td>
      <td>-0.014040</td>
      <td>-0.022144</td>
      <td>0.033049</td>
      <td>0.009125</td>
      <td>-0.006482</td>
      <td>0.028657</td>
      <td>-0.009993</td>
      <td>1.000000</td>
      <td>0.049714</td>
    </tr>
    <tr>
      <th>T3_to_T10</th>
      <td>-0.045126</td>
      <td>0.051295</td>
      <td>-0.042054</td>
      <td>-0.123211</td>
      <td>0.051800</td>
      <td>0.042164</td>
      <td>0.045859</td>
      <td>0.047871</td>
      <td>-0.012278</td>
      <td>0.016640</td>
      <td>0.049714</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Discussion Topics

#### Compare and Contrast the relationship between return variables and the two "LM Sentiment" variables

From the correlation matrix and the heatmap, it appears that there is no correlation between the return variables and the "LM Sentiment" variables. The correlation matrix shows that the correlation between the return variables and the "LM Sentiment" variables is close to zero. For the positive variable, it appears that it has a negative correlation with the return variables. For the negative variable, there is also a negative correlation with the return variables. Neither the positive variable or the negative variable had a strong correlation with the return. 

#### Conflicts with Table 3 of ML_JFE

My findings of the data was similar to the findings in table 3 of ML_JFE. In table 3, the results for LM positive is similar to my findings as the LM positive variable had a negative return and the LM negative variable also had a negative return. All of the correlations I found were extremely small and close to zero which is also similar to the results in ML_JFE. The reason why the ML_JFE paper included more firms and years and additional controls in their study is because their study was done during the COVID-19 pandemic. Because the pandemic year resulted in several outliers where the data was extremely volatile and hard to predict, there are irrational decisions made by investors causing the data to be different than other years. 

With a larger sample size, the authors of ML_JFE are able to validate their data and the results are more representative of the entire population. It is difficult to make conclusions based on our small sample size of 503 firms. A larger sample size can also provide more insight on the types of correlations between the positive and negative sentiment variables. Using more controls, they are able to determine the factors that may affect the results

#### My sentiment variables

Although the correlations from the sentiment variables that I chose are not zero, there is not enough information to show that there is a relationship. There are too many contradictions in the correlation between sentiment variables and the return variables for the results to be useful. However, I do believe that there is merit to how different sentiment impacts the returns of a firm. If a 10-K is very positive in terms of the context and words they use, then investors will be interested in the firm and becomes a good indicator for buying a stock.
