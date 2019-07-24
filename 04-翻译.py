#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests


headers = {
    "referer": "https://fanyi.baidu.com/?aldtype=16047",
    "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like G\
    ecko) Version/10.0 Mobile/14E304 Safari/602.1"
}

post_data = {
    "query": "where",
    "from": "en",
    "to": "zh",
    "token": "5ff4a46ddd56957755dac90cf94df3d1",
    "sign": "951029.680388"
}

post_url = "https://fanyi.baidu.com/basetrans"
resp = requests.get(post_url, headers=headers, data=post_data)

print(resp.content.decode())
