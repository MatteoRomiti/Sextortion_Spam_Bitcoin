# Spams meet Cryptocurrencies: Sextortion in the Bitcoin Ecosystem

This repository hosts the relevant code and the Bitcoin addresses extracted from the emails at our disposal to reproduce the results of the research paper accepted at [AFT 2019](https://aft.acm.org).

The paper is available [here](https://arxiv.org/abs/1908.01051).
The slides are available [here](https://www.slideshare.net/MatteoRomiti1/spams-meet-cryptocurrencies-184791429).
The Bitcoin addresses are also available [here](https://zenodo.org/record/3515199#.Xa603XUzY3E).
------------

## Abstract

In the past year, a new spamming scheme has emerged: sexual extortion messages requiring payments in the cryptocurrency Bitcoin, also known as sextortion. This scheme represents a first integration of the use of cryptocurrencies by members of the spamming industry. Using a dataset of 4,340,736 sextortion spams, this research aims at understanding such new amalgamation by uncovering spammers’ operations. To do so, a simple, yet effective method for projecting Bitcoin addresses mentioned in sextortion spams onto transaction graph abstractions is computed over the entire Bitcoin blockchain. This allows us to track and investigate monetary flows between involved actors and gain insights into the financial structure of sextortion campaigns. We find that sextortion spammers are somewhat sophisticated, following pricing strategies and benefiting from costs reduction as their operations cut the upper-tail of the spamming supply chain. We discover that one single entity is likely controlling the financial backbone of the majority of the sextortion campaigns and that the 11-month operation studied yielded a lower-bound revenue between \$1,300,620 and \$1,352,266. We conclude that sextortion spamming is a lucrative business and spammers will likely continue to send bulk emails that try to extort money through cryptocurrencies. 

## How to

This work relies on python 3, jupyter notebooks and [GraphSense REST API](https://github.com/graphsense/graphsense-REST).

To install the required libraries:
	
	pip install -r requirements.txt

The raw dataset of emails cannot be published, but in `data` you can find the addresses found in the emails related to the sextortion used in our paper. You can run the cells in the notebook and reproduce/improve the results.

---
**NOTE**

An API token is needed to query the GraphSense API.
