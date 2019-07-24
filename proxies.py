#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests

proxies = {"http": "http://116.55.116.136:50269"}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chr\
    ome/74.0.3729.108 Safari/537.36"
}

resp = requests.get("http://www.baidu.com", headers=headers, proxies=proxies)

print(resp.status_code)
