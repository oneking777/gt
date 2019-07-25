#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
from lxml import etree
import json


class QiuBaiSpider(object):
    """糗事百科推荐段子爬取"""
    def __init__(self):
        self.start_url = "https://www.qiushibaike.com/article/119541531"
        self.basic_url = "https://www.qiushibaike.com"
        self.headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 \
        (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1"}

    def send_start_url(self):
        resp = requests.get(url=self.start_url, headers=self.headers)
        return resp.content.decode()

    def parse_resp_content(self, html_str):
        html = etree.HTML(html_str)
        item = dict()
        item["author_image_url"] = html.xpath('//section[@id="userInfoMain"]//a[1]/img/@src')[0]
        item["author_name"] = html.xpath('//section[@id="userInfoMain"]//span/text()')[0]
        content_list = html.xpath('//div[@class="content-text image"]//text()')
        item["content"] = ""
        for content in content_list:
            item["content"] += content
        try:
            item["content_image_url"] = html.xpath('//div[@class="content-img"]/img/@src')[0]
        except Exception as e:
            print(e)
            item["content_image_url"] = None
        item["comment_name_list"] = html.xpath('//div[@class="comments-list"]//a/text()')
        item["comment_info_list"] = html.xpath('//div[@class="comments-list"]//span/text()')
        try:
            next_url = self.basic_url + html.xpath('//span[@class="btn next"]/@href')[0]
        except Exception as e:
            print(e)
            next_url = None
        return item, next_url

    def save_content(self, item):
        with open("qiubaiduanzi.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False, indent=2))
            f.write("\n")
        print("保存成功")

    def send_next_url(self, next_url):
        resp = requests.get(url=next_url, headers=self.headers)
        print(next_url)
        return resp.content.decode()

    def run(self):
        # 1,start_url
        html_str = self.send_start_url()
        next_url = True
        while next_url:
            # 2,发送请求获取响应
            # 3,接受响应，提取作者头像，名字,正文，正文图片（如果有的话），评论用户名字，评论信息，下一页url
            item, next_url = self.parse_resp_content(html_str)
            # 4，保存数据
            self.save_content(item)
            # 5，发送next_url
            html_str = self.send_next_url(next_url)


if __name__ == "__main__":
    qiubai = QiuBaiSpider()
    qiubai.run()



