#!/bin/bash
curl -L -o data.zip\
  https://www.kaggle.com/api/v1/datasets/download/carrie1/ecommerce-data

mkdir -p cache
unzip data.zip
rm data.zip
mv data.csv cache/data.csv
