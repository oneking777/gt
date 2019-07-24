#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
from retrying import retry

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chr\
    ome/74.0.3729.108 Safari/537.36"
}


@retry(stop_max_attemp_number=3)
def _parse_url(url, method, data):
    if method == "POST":
        resp = requests.post(url, headers=headers, timeout=3, data=data)
    else:
        resp = requests.get(url, headers=headers, timeout=3)
    assert resp.status_code == 200
    return resp.content.decode()


def parse_url(url, method="GET", data=None):
    try:
        html_str = _parse_url(url, method, data)
    except Exception as e:
        html_str = None

    return html_str


if __name__ == "__main__":
    url = "http://www.baidu.com"
    print(parse_url(url))