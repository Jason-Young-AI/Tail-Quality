#!/bin/bash

source ./vars.sh

echo "Getting Datasets"
wget -P ${THIS_ASSETS_DIR} -c ${THIS_DATASET_URL}
echo "Unzip Datasets"
unzip -n -q ${THIS_ASSETS_DIR}/datasets.zip -d ${THIS_ASSETS_DIR}
echo "Done"

echo ""

echo "Getting Model Weight"
wget -P ${THIS_ASSETS_DIR}/${THIS_FRAMEWORK} -c ${THIS_MODEL_URL}
echo "Done"