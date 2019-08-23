#!/usr/bin/env bash
if test -z "$1"
then
      echo "Usage ./build-task-images.sh VERSION"
      echo "Version wasn't passed!"
      exit 1
fi

VERSION=$1
docker build -t code-challenge/download-data:$VERSION download_data
docker build -t code-challenge/make-dataset:$VERSION make_dataset
docker build -t code-challenge/model:$VERSION model
