#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chr\
    ome/74.0.3729.108 Safari/537.36"
}

while True:
    resp = requests.get("http://47.93.5.166:5000", headers=headers)
    print(resp.status_code)



