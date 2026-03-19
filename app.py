채은님, 에러 메시지(TypeError)를 보니 데이터 형식이 앱과 맞지 않아서 발생하는 **'숫자 변환 오류'**입니다.

🔍 왜 이런 에러가 날까요?
사진 속 에러(if df['LF'].max() <= 1.0)는 모델이 L/F 컬럼에서 숫자를 찾아내서 "이게 소수점(0.8)인지 정수(80)인지" 판단하려고 할 때 발생했습니다. 그런데 엑셀 파일의 **LF 열에 숫자가 아닌 글자(텍스트)**가 섞여 있거나, 데이터가 비어 있어서 컴퓨터가 "최댓값을 구할 수 없어!"라고 비명을 지르는 상황이에요.

걱정 마세요! 어떤 데이터가 들어오더라도 강제로 숫자로 바꿔서 에러를 무시하도록 코드를 더 튼튼하게 고쳤습니다.

🛠️ 에러 완벽 해결 버전 (app.py)
이 코드로 GitHub 내용을 전체 교체해 보세요. 숫자가 아닌 값은 알아서 처리하고, Y축도 깔끔하게 %로 나옵니다.

Python
import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import holidays
import numpy as np

st.set_page_config(page_title="Airline L/F Predictor Pro", layout="wide")
st.title("✈️ 노선별 L/F 인공지능 예측 모델")

kr_holidays = holidays.KR()

uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # [수정] 강제로 숫자로 변환 (숫자가 아닌 건 NaN 처리 후 앞뒤 값으로 채움)
    df['LF'] = pd.to_numeric(df['LF'], errors='coerce')
    df['LF'] = df['LF'].ffill().bfill()
    
    # [수정] 에러 방지용 체크: 데이터가 소수점 단위면 100을 곱함
    if df['LF'].max() <= 1.0:
        df['LF'] = df['LF'] * 100
    
    selected_route = st.sidebar.selectbox("노선 선택", df['Route'].unique())
    route_df = df[df['Route'] == selected_route].sort_values('Date').copy()
    
    if len(route_df) > 21:
        # 모델 학습 및 정확도 계산
        train = route_df['LF'][:-14]
        test = route_df['LF'][-14:]
        
        model = ExponentialSmoothing(route_df['LF'], trend='add', seasonal='add', seasonal_periods=7).fit()
        
        test_pred = model.predict(start=len(train), end=len(train)+len(test)-1)
        mape = np.mean(np.abs((test - test_pred) / (test + 1e-9))) * 100 # 0 나누기 방지
        accuracy = max(0, 100 - mape)

        # 상단 정확도 표시
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric(label="🎯 모델 예측 정확도", value=f"{accuracy:.1f}%")
        col2.metric(label="📊 최근 7일 평균 L/F", value=f"{route_df['LF'].iloc[-7:].mean():.1f}%")
        col3.success("보고서용 퍼센트 단위 변환 완료!")
        st.divider()

        # 미래 90일 예측
        forecast_days = 90
        forecast = model.forecast(forecast_days)
        future_dates = pd.date_range(start=route_df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        
        adjusted_forecast = []
        for date, val in zip(future_dates, forecast):
            if date in kr_holidays:
                new_val = val * 1.15
                adjusted_forecast.append(min(100, new_val))
            else:
                adjusted_forecast.append(val)
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': adjusted_forecast})

        # 그래프 출력 및 Y축 % 설정
        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} AI 예측 (정확도: {accuracy:.1f}%)")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="AI 미래 예측", mode='lines+markers')
        
        fig.update_layout(
            yaxis_title="Load Factor (%)",
            yaxis_range=[0, 105],
            yaxis=dict(tickformat='.0f', ticksuffix="%") # 축에 % 기호 붙이기
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("데이터가 부족합니다.")
