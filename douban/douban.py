#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
from lxml import etree


class Kr36Spider(object):
    """爬取36kr网站的新闻信息"""
    def __init__(self, url):
        self.headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 \
        (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1"}
        self.start_url = url

    def parse_url(self):  # 发送请求，获取响应
        resp = requests.get(self.start_url, headers=self.headers)
        return resp.content.decode()

    def get_content_url(self, html_str):  # 提取数据，提取下一条url
        html = etree.HTML(html_str)
        content_list = html.xpath('//div[@id="body-content"]/p/text()')  # 是一个列表，需要遍历
        next_url_list = html.xpath('//div[@class="kr-card-content"]/a[1]/@href')  # p/5228866
        title_list = html.xpath('//div[@class="body-title weight-bold"]/text()')  # title
        return content_list, next_url_list, title_list

    def parse_next_url(self, next_url_list):  # 发送下一条请求
        next_url = "https://36kr.com" + next_url_list[0]
        print(next_url)
        resp = requests.get(next_url, headers=self.headers)
        return resp.content.decode()

    def save_content_list(self, content_list, title_list):
        title = title_list[0]
        try:
            with open('./kr36_news/{}.txt'.format(title), "w", encoding="utf-8") as f:
                for content in content_list:
                    f.write(content)
            print("{}.txt保存成功".format(title))
        except Exception as e:
            print("{}.txt保存失败".format(title))
            print(e)

    def run(self):
        # 1,start_url
        # 2,发送请求，获取响应
        # 3，提取数据，提取下一条url
        #   3.1 数据保存到文件中
        #   3.2 获得下一条url，继续发送请求， 多进程
        num = 0
        html = self.parse_url()
        while True:
            try:
                content_list, next_url_list, title_list = self.get_content_url(html)
                self.save_content_list(content_list, title_list)
                html = self.parse_next_url(next_url_list)
            except Exception as e:
                print(e)
                break
            finally:
                num += 1
                print(num)


if __name__ == "__main__":
    kr36 = Kr36Spider("https://36kr.com/p/5228926")
    kr36.run()


