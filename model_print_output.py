import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import boto3
import io
import json
import joblib
from flask import Flask
app = Flask(__name__)


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

        P = np.random.normal(loc=0, scale=1./K, size=(num_users, K))
        Q = np.random.normal(loc=0, scale=1./K, size=(num_items, K))

        P = np.array(P, dtype=np.float32).round(2)
        Q = np.array(Q, dtype=np.float32).round(2)

        # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트에 저장.
        non_zeros = [ (i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0 ]

        # SGD 기법으로 P와 Q 매트릭스를 계속 업데이트.
        for step in range(steps):
            for i, j, r in non_zeros:
                # 실제 값과 예측 값의 차이인 오류 값 구함
                eij = (r - np.dot(P[i, :], Q[j, :].T)).astype(np.float32)
                P[i,:] = P[i,:] + learning_rate*((eij * Q[j, :]) - (r_lambda * P[i,:]))
                Q[j,:] = Q[j,:] + learning_rate * ((eij * P[i, :]) - (r_lambda * Q[j,:]))

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
        s3 = boto3.client("s3", aws_access_key_id = access_key, 
                            aws_secret_access_key = secret_key,
                            region_name = 'ap-northeast-2')
        # encode_file = json.dumps(file, ensure_ascii=False)
        try:
            s3.put_object(Bucket=bucket, Key=file_name, Body=file)
            # s3.upload_file(file, bucket, file_name)        
            return True
        except Exception as e : 
            print(e)
            return False
    
    # s3 접속하기    
    def get_data(self):
        access_key, secret_key = self.config_s3()
        bucket = 'capstone2023-image'
        s3 = boto3.client("s3", aws_access_key_id = access_key, 
                            aws_secret_access_key = secret_key,
                            region_name = 'ap-northeast-2')
        
        obj1 = s3.get_object(Bucket=bucket, Key='place_data.xlsx')
        # place_data = pd.read_excel('C:\\Users\\admin\\Desktop\\capstone_MachineLearning\\place_data.xlsx').fillna(0)
        place_data = pd.read_excel(io.BytesIO(obj1["Body"].read())).fillna(0)
        place_data['score'] = place_data['weather'] + (place_data['crowd']*2) + place_data['distance'] + (place_data['like']*10)

        obj2 = s3.get_object(Bucket=bucket, Key='place_theme.xlsx')
        place_theme = pd.read_excel(io.BytesIO(obj2["Body"].read())).fillna(0)
        # place_theme = pd.read_excel('C:\\Users\\admin\\Desktop\\capstone_MachineLearning\\place_theme.xlsx').fillna(0)
        # place_theme = place_theme.drop(['Unnamed: 0'],axis=1)
        place_theme = place_theme.drop(['place'],axis=1)

        #사용자가 선택한 테마 저장
        input_column = place_theme.loc[:,['input']]
        user_theme = input_column.iloc[0,0]
        place_theme = place_theme.drop(['input'],axis=1)

        return place_data, place_theme, user_theme
    
    # SCORE 계산하기: data(날씨, 이동거리, 혼잡도, 찜개수)에 따른 추천 점수('score')
    def cal_score(self, place_data, place_theme):
        place_theme_score = pd.DataFrame(index = place_theme.index, columns = place_theme.columns)
        
        ### :(...index 왜이러는지 아직 제대로 못찾음... 일단은 지우고,, 다른거 먼저,,
        place_theme_score = place_theme_score.drop(['place_id'],axis=1)

        for place in place_theme.index:
            for theme in theme_list:
                if place_theme.iloc[place][theme]:
                    place_theme_score.iloc[place][theme] = place_data.iloc[place]['score']/60

        #인덱스: place_id로 설정
        place_theme_score.set_index(place_data['place_id'], inplace=True)
        return place_theme_score
    
    # 장소 테마에 따른 예측 점수 df 만들기
    def predict(self, place_theme_score, ):
        #행렬 분해
        P,Q = self.matrix_factorization(place_theme_score)
        pred_matrix = np.dot(P, Q.T)

        pred_matrix_place = pd.DataFrame(data=pred_matrix, index= place_theme_score.index,
                                    columns = place_theme_score.columns)
        return pred_matrix_place
    
@app.route('/')
def run():

    Recomm = PlaceRecommandation()
    
    place_data, place_theme, user_theme = Recomm.get_data()

    #테마 리스트 저장
    theme_list = list(place_theme.columns)
    
    place_theme_score = Recomm.cal_score(place_data, place_theme)

    pred_matrix_place = Recomm.predict(place_theme_score)

    # 장소 추천
    recommadation = Recomm.recomm_place(pred_matrix_place, user_theme)
    
    # output 출력
    data_df = pd.DataFrame(recommadation.index)
    print(data_df.values)
    
    return data_df.values
    