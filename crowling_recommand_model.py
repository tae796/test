import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import boto3
import io

import json
from bs4 import BeautifulSoup
import requests
import re
import datetime
from tqdm import tqdm

from flask import Flask

app = Flask(__name__)


class news_crowling():
    # 페이지 url 형식에 맞게 바꾸어 주는 함수 만들기
    # 입력된 수를 1, 11, 21, 31 ...만들어 주는 함수
    def makePgNum(self, num):
        if num == 1:
            return num
        elif num == 0:
            return num + 1
        else:
            return num + 9 * (num - 1)

    # 크롤링할 url 생성하는 함수 만들기(검색어, 크롤링 시작 페이지, 크롤링 종료 페이지)
    def makeUrl(self, search, start_pg, end_pg):
        if start_pg == end_pg:
            start_page = self.makePgNum(start_pg)
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(
                start_page)
            # print("생성url: ", url)
            return url
        else:
            urls = []
            for i in range(start_pg, end_pg + 1):
                page = self.makePgNum(i)
                url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(
                    page)
                urls.append(url)
            # print("생성url: ", urls)
            return urls

    # html에서 원하는 속성 추출하는 함수 만들기 (기사, 추출하려는 속성값)
    def news_attrs_crawler(self, articles, attrs):
        attrs_content = []
        for i in articles:
            attrs_content.append(i.attrs[attrs])
        return attrs_content

    # html생성해서 기사크롤링하는 함수 만들기(url): 링크를 반환
    def articles_crawler(self, i, url):
        # ConnectionError방지
        global headers
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
        # html 불러오기
        original_html = requests.get(i, headers=headers)
        html = BeautifulSoup(original_html.text, "html.parser")

        url_naver = html.select(
            "div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
        url = self.news_attrs_crawler(url_naver, 'href')
        return url

    # 제목, 링크, 내용 1차원 리스트로 꺼내는 함수 생성
    def makeList(self, newlist, content):
        for i in content:
            for j in i:
                newlist.append(j)
        return newlist

    def section_news(self, section, news_list):  # 지역별 뉴스 고르기
        temp = []
        if len(news_list[0]):
            for i in range(len(news_list[0])):
                if (section in np.array(news_list).T[i][1]) or ('도심' in np.array(news_list).T[i][1]):
                    temp.append(np.array(news_list).T[i])
        return temp  # 각 행: 날짜-제목

    def today_news(self, news, news_titles):
        news_day = []
        news_day_title = []
        for i in range(len(news)):
            string = news[i]
            news_day.append(string[0:10])
            news_day_title.append([news_day, news_titles])

        today_date = datetime.datetime.today().date()
        news_print = []
        for i in range(len(news_day_title[0])):
            if news_day_title[0][i] == str(today_date):
                news_print.append(news_day_title.T[i])  # 1행: 날짜/ 2행: 제목
        return news_print

    #####뉴스크롤링 시작###
    def crowling_news(self, section):
        page = 1
        page2 = 2
        # naver url 생성 makeUrl(검색어, 시작페이지, 종료페이지)
        url_crowded = self.makeUrl('서울 집회 교통', page, page2)
        url_festival = self.makeUrl('서울 교통 통제', page, page2)
        url_traffic1 = self.makeUrl('서울 교통 중상', page, page2)

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
                url = self.articles_crawler(i, url)
                news_url.append(url)

        # 제목, 링크, 내용 담을 리스트 생성
        news_url_1 = []

        # 1차원 리스트로 만들기(내용 제외)
        self.makeList(news_url_1, news_url)

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
        news_day = self.today_news(news_dates, news_titles)  # 1행: 날짜/ 2행: 제목

        if len(news_day):
            print(len(news_day))
            # 지역별 뉴스 고르기
            news_print = self.section_news(section, news_day)  # 각 행: 날짜-제목
        else:
            return [0]

        if len(news_print):
            print(len(news_print))
            # 추려진 뉴스 출력
            print("\n[뉴스 제목]")
            for i in range(len(news_print)):
                print(news_print[i][1])
        else:
            return [0]

        return np.array(news_print).T[1]  # 1행: 날짜/ 2행: 제목


class PlaceRecommandation():

    # SGD 기법으로 P와 Q 매트릭스를 계속 업데이트
    def get_rmse(self, R, P, Q, non_zeros):
        error = 0
        # 두개의 분해된 행렬 P와 Q.T의 내적으로 예측 R 행렬 생성
        full_pred_matrix = np.dot(P, Q.T)

        # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스 추출하여 실제 R 행렬과 예측 행렬의 RMSE 추출
        x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
        y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
        R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]

        full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

        mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
        rmse = np.sqrt(mse)

        return rmse

    # R = P X Q.T : P, Q 매트릭스 생성 메소드
    def matrix_factorization(self, R, K=30, steps=500, learning_rate=0.01, r_lambda=0.01):
        R = np.array(R)
        num_users, num_items = R.shape
        np.random.seed(0)

        P = np.random.normal(loc=0, scale=1. / K, size=(num_users, K))
        Q = np.random.normal(loc=0, scale=1. / K, size=(num_items, K))

        P = np.array(P, dtype=np.float32).round(2)
        Q = np.array(Q, dtype=np.float32).round(2)

        # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트에 저장.
        non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

        # SGD 기법으로 P와 Q 매트릭스를 계속 업데이트.
        for step in range(steps):
            for i, j, r in non_zeros:
                # 실제 값과 예측 값의 차이인 오류 값 구함
                eij = (r - np.dot(P[i, :], Q[j, :].T)).astype(np.float32)
                P[i, :] = P[i, :] + learning_rate * ((eij * Q[j, :]) - (r_lambda * P[i, :]))
                Q[j, :] = Q[j, :] + learning_rate * ((eij * P[i, :]) - (r_lambda * Q[j, :]))

            rmse = self.get_rmse(R, P, Q, non_zeros)

            # if (step % 50) == 0 :
            #     print("### iteration step : ", step," rmse : ", rmse)

        return P, Q

    # 아이템 기반의 인접 이웃 협업 필터링으로 장소 추천
    def recomm_place(self, predict_df, theme, top_n=5):
        # 예측 평점 DataFrame에서 테마에 맞는 place 추천
        # 가장 예측 평점이 높은 순으로 정렬함.
        place = predict_df.loc[:, theme].sort_values(ascending=False)[:top_n]
        return place

    def config_s3(self):
        access_key = 'AKIASWDWOABNAAXTZ3BB'
        secret_key = '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'
        return access_key, secret_key

    # 서버에 json 파일 업로드
    def upload_file(self, bucket, file_name, file):
        access_key, secret_key = self.config_s3()
        bucket = 'capstone2023-image'
        s3 = boto3.client("s3", aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key,
                          region_name='ap-northeast-2')
        # encode_file = json.dumps(file, ensure_ascii=False)
        try:
            s3.put_object(Bucket=bucket, Key=file_name, Body=file)
            # s3.upload_file(file, bucket, file_name)        
            return True
        except Exception as e:
            print(e)
            return False

    # s3 접속하기
    def get_data(self):
        access_key, secret_key = self.config_s3()
        bucket = 'capstone2023-image'
        s3 = boto3.client("s3", aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key,
                          region_name='ap-northeast-2')

        obj1 = s3.get_object(Bucket=bucket, Key='place_data.xlsx')
        # place_data = pd.read_excel('C:\\Users\\admin\\Desktop\\capstone_MachineLearning\\place_data.xlsx').fillna(0)
        place_data = pd.read_excel(io.BytesIO(obj1["Body"].read())).fillna(0)
        place_data['score'] = place_data['weather'] + (place_data['crowd'] * 2) + place_data['distance'] + (
                    place_data['like'] * 10)

        obj2 = s3.get_object(Bucket=bucket, Key='place_theme.xlsx')
        place_theme = pd.read_excel(io.BytesIO(obj2["Body"].read())).fillna(0)
        # place_theme = pd.read_excel('C:\\Users\\admin\\Desktop\\capstone_MachineLearning\\place_theme.xlsx').fillna(0)
        # place_theme = place_theme.drop(['Unnamed: 0'],axis=1)
        place_theme = place_theme.drop(['place'], axis=1)

        # 사용자가 선택한 테마 저장
        input_column = place_theme.loc[:, ['input']]
        user_theme = input_column.iloc[0, 0]
        place_theme = place_theme.drop(['input'], axis=1)

        return place_data, place_theme, user_theme

    # SCORE 계산하기: data(날씨, 이동거리, 혼잡도, 찜개수)에 따른 추천 점수('score')
    def cal_score(self, place_data, place_theme, theme_list):
        place_theme_score = pd.DataFrame(index=place_theme.index, columns=place_theme.columns)

        ### :(...index 왜이러는지 아직 제대로 못찾음... 일단은 지우고,, 다른거 먼저,,
        place_theme_score = place_theme_score.drop(['place_id'], axis=1)

        for place in place_theme.index:
            for theme in theme_list:
                if place_theme.iloc[place][theme]:
                    place_theme_score.iloc[place][theme] = place_data.iloc[place]['score'] / 60

        # 인덱스: place_id로 설정
        place_theme_score.set_index(place_data['place_id'], inplace=True)
        return place_theme_score

    # 장소 테마에 따른 예측 점수 df 만들기
    def predict(self, place_theme_score, ):
        # 행렬 분해
        P, Q = self.matrix_factorization(place_theme_score)
        pred_matrix = np.dot(P, Q.T)

        pred_matrix_place = pd.DataFrame(data=pred_matrix, index=place_theme_score.index,
                                         columns=place_theme_score.columns)
        return pred_matrix_place


@app.route('/<user_section>')
def crowling(user_section):
    crowler = news_crowling()
    data = crowler.crowling_news(user_section)
    data_df = pd.DataFrame(data)
    data_js = data_df.values.tolist()
    data = json.dumps(data, ensure_ascii=False).encode('utf8')
    print(data)
    return data


@app.route('/')
def recomm():
    Recomm = PlaceRecommandation()
    place_data, place_theme, user_theme = Recomm.get_data()
    # 테마 리스트 저장
    theme_list = list(place_theme.columns)
    place_theme_score = Recomm.cal_score(place_data, place_theme, theme_list)
    pred_matrix_place = Recomm.predict(place_theme_score)
    # 장소 추천
    recommadation = Recomm.recomm_place(pred_matrix_place, user_theme)

    # output 출력
    data_df = pd.DataFrame(recommadation.index)
    data_js = data_df.values.tolist()
    data = json.dumps(data_js)
    print(data)
    return data


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
