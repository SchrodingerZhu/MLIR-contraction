#!/usr/bin/env bash

set -e

GREEN="\033[32m"
RED="\033[31m"
CLEAR="\033[0m"
for file in $(find . -name "*.mlir"); do
    echo -n "Testing $file: "
    mlir-opt $file > /dev/null && echo -e "${GREEN}[OK]${CLEAR}" || echo -e "${RED}[FAIL]${CLEAR}"
done