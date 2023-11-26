import json
from bs4 import BeautifulSoup
import requests
import re
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from flask import Flask


# 페이지 url 형식에 맞게 바꾸어 주는 함수 만들기
# 입력된 수를 1, 11, 21, 31 ...만들어 주는 함수
def makePgNum(num):
    if num == 1:
        return num
    elif num == 0:
        return num + 1
    else:
        return num + 9 * (num - 1)


# 크롤링할 url 생성하는 함수 만들기(검색어, 크롤링 시작 페이지, 크롤링 종료 페이지)
def makeUrl(search, start_pg, end_pg):
    if start_pg == end_pg:
        start_page = makePgNum(start_pg)
        url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(
            start_page)
        # print("생성url: ", url)
        return url
    else:
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = makePgNum(i)
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(page)
            urls.append(url)
        # print("생성url: ", urls)
        return urls


# html에서 원하는 속성 추출하는 함수 만들기 (기사, 추출하려는 속성값)
def news_attrs_crawler(articles, attrs):
    attrs_content = []
    for i in articles:
        attrs_content.append(i.attrs[attrs])
    return attrs_content


# html생성해서 기사크롤링하는 함수 만들기(url): 링크를 반환
def articles_crawler(i, url):
    # ConnectionError방지
    global headers
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
    # html 불러오기
    original_html = requests.get(i, headers=headers)
    html = BeautifulSoup(original_html.text, "html.parser")

    url_naver = html.select(
        "div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
    url = news_attrs_crawler(url_naver, 'href')
    return url


# 제목, 링크, 내용 1차원 리스트로 꺼내는 함수 생성
def makeList(newlist, content):
    for i in content:
        for j in i:
            newlist.append(j)
    return newlist


def section_news(section, news_list):  # 지역별 뉴스 고르기
    temp = []
    if len(news_list[0]):
        for i in range(len(news_list[0])):
            if (section in np.array(news_list).T[i][1]) or ('도심' in np.array(news_list).T[i][1]):
                temp.append(np.array(news_list).T[i])
    return temp  # 각 행: 날짜-제목


def today_news(news, news_titles):
    news_day = []
    for i in range(len(news)):
        string = news[i]
        news_day.append(string[0:10])
    news_day_title = np.vstack((np.array(news_day), np.array(news_titles)))  # 1행: 날짜/ 2행: 제목

    today_date = datetime.datetime.today().date()
    news_print = []
    for i in range(len(news_day_title[0])):
        if news_day_title[0][i] == str(today_date):
            news_print.append(news_day_title.T[i])  # 1행: 날짜/ 2행: 제목
    return news_print


#####뉴스크롤링 시작#####

def crowling_news(section):
    page = 1
    page2 = 2
    # naver url 생성 makeUrl(검색어, 시작페이지, 종료페이지)
    url_crowded = makeUrl('서울 집회 교통', page, page2)
    url_festival = makeUrl('서울 교통 통제', page, page2)
    url_traffic1 = makeUrl('서울 교통 중상', page, page2)

    urls = []
    urls.append(url_crowded)
    urls.append(url_festival)
    urls.append(url_traffic1)

    # 뉴스 크롤러 실행
    news_titles = []
    news_url = []
    news_contents = []
    news_dates = []
    for url in urls:
        for i in url:
            url = articles_crawler(i, url)
            news_url.append(url)

    # 제목, 링크, 내용 담을 리스트 생성
    news_url_1 = []

    # 1차원 리스트로 만들기(내용 제외)
    makeList(news_url_1, news_url)

    # NAVER 뉴스만 남기기
    final_urls = []
    for i in tqdm(range(len(news_url_1))):
        if "news.naver.com" in news_url_1[i]:
            final_urls.append(news_url_1[i])
        else:
            pass

    # 뉴스 내용 크롤링
    for i in tqdm(final_urls):
        # 각 기사 html get하기
        news = requests.get(i, headers=headers)
        news_html = BeautifulSoup(news.text, "html.parser")

        # 뉴스 제목 가져오기
        title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        if title == None:
            title = news_html.select_one("#content > div.end_ct > div > h2")

        # # 뉴스 본문 가져오기
        # content = news_html.select("article#dic_area")
        # if content == []:
        #     content = news_html.select("#articeBody")

        # # 기사 텍스트만 가져오기
        # # list합치기
        # content = ''.join(str(content))

        # html태그제거 및 텍스트 다듬기
        pattern1 = '<[^>]*>'
        title = re.sub(pattern=pattern1, repl='', string=str(title))
        # content = re.sub(pattern=pattern1, repl='', string=content)
        # pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
        # content = content.replace(pattern2, '')
        news_titles.append(title)
        # news_contents.append(content)

        try:
            html_date = news_html.select_one(
                "div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
            news_date = html_date.attrs['data-date-time']
        except AttributeError:
            news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")
            news_date = re.sub(pattern=pattern1, repl='', string=str(news_date))
        # 날짜 가져오기
        news_dates.append(news_titles)

    # 오늘 날짜 뉴스만 골라오기
    news_day = today_news(news_dates, news_titles)  # 1행: 날짜/ 2행: 제목
    # 지역별 뉴스 고르기
    news_print = section_news(section, news_day)  # 각 행: 날짜-제목

    # 추려진 뉴스 출력
    print("\n[뉴스 제목]")
    for i in range(len(news_print)):
        print(news_print[i][1])

    return np.array(news_print).T[1] # 1행: 날짜/ 2행: 제목


app = Flask(__name__)


@app.route('/<user_section>')
def run(user_section):
    data = crowling_news(user_section)
    data_df = pd.DataFrame(data)
    data_js = data_df.values.tolist()
    data = json.dumps(data, ensure_ascii=False).encode('utf8')
    print(data)
    return data


if __name__ == '__main__':
    app.run('0.0.0.0', port=5020, debug=True)
