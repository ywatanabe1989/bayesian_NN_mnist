#!/bin/bash
# Time-stamp: "2021-11-08 05:52:43 (ywatanabe)"

for nt in 100 1000 10000 20000 40000; do
    python classify_mnist.py -nt $nt
done

## EOF
