import requests


class TiebaSpider():
    """实现贴吧爬虫"""
    def __init__(self, tieba_name):
        self.tieba_name = tieba_name
        self.url_temp = "https://tieba.baidu.com/f?kw=" + tieba_name +"&ie=utf-8&pn={}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36"}

    def get_url_list(self):
        # # 构造url_list
        # url_list = list()
        # for i in range(1000):
        #     url_list.append(self.url_temp.format(i*50))
        # return url_list
        return [self.url_temp.format(i*50) for i in range(1000)]

    def parse_url(self, url):
        # 发送请求获取响应
        print(url)
        resp = requests.get(url=url, headers=self.headers)
        return resp.content.decode()

    def save_html(self, html_str, page_num):
        # 保存html_str
        file_path = "{}-第{}页.html".format(self.tieba_name, page_num)
        with open(file_path, "w", encoding="utf-8") as f:  # "lol-4.html"
            f.write(html_str)

    def run(self):
        # 实现主要逻辑
        # 1,构造url_list
        url_list = self.get_url_list()
        # 2.遍历发送请求获取响应
        for url in url_list:
            html_str = self.parse_url(url)
            # 3.保存数据到本地
            page_num = url_list.index(url) + 1  # 页码数
            self.save_html(html_str, page_num)


if __name__ == "__main__":
    tieba_spider = TiebaSpider("lol")
    tieba_spider.run()