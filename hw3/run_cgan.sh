#!/bin/bash
python3 main.py --test --prefix cgan --test_tag $1 --noise_file ./noise_file.txt --result_file ./samples/cgan.png
