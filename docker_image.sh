#!/usr/bin/env bash

VERSION="v1.0"

cd /home/batu/Master_Thesis/chemtensor_mojo/chemtensor_mojo
docker build -t batuhangunduz/chemtensor-mojo:latest .
docker tag batuhangunduz/chemtensor-mojo:latest batuhangunduz/chemtensor-mojo:$VERSION
docker push batuhangunduz/chemtensor-mojo:$VERSION