채은님, 제가 "데이터가 있는 날짜만" 보여드리는 데 너무 집중한 나머지, 정작 중요한 향후 60일 예측치를 화면 밖으로 밀어냈나 보네요! 정말 죄송합니다.

범인은 그래프의 'X축 범위(xaxis_range)' 설정이었어요. 과거 데이터 끝점에만 맞추다 보니, 그 뒤에 그려진 60일치 예측선이 잘려 나간 거죠.

과거 실적과 향후 60일 예측 데이터까지 모두 포함해서 화면에 꽉 차게 보여주는 진짜 완성형 코드입니다. 이번에는 60일치 꼬리까지 아주 잘 보일 거예요!

🛠️ 실적 + 향후 60일 예측 통합 버전 (app.py)
GitHub의 app.py 내용을 아래 코드로 전체 덮어쓰기 하세요.

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
    df['LF'] = pd.to_numeric(df['LF'], errors='coerce').ffill().bfill()
    
    if df['LF'].max() <= 1.0:
        df['LF'] = df['LF'] * 100
    
    selected_route = st.sidebar.selectbox("노선 선택", df['Route'].unique())
    route_df = df[df['Route'] == selected_route].sort_values('Date').reset_index(drop=True)
    
    if len(route_df) > 10:
        # 모델 학습
        model = ExponentialSmoothing(route_df['LF'], trend='add', seasonal='add', seasonal_periods=7).fit()
        
        # 정확도 계산
        train = route_df['LF'][:-7]
        test = route_df['LF'][-7:]
        test_pred = model.predict(start=len(train), end=len(train)+len(test)-1)
        mape = np.mean(np.abs((test - test_pred) / np.where(test == 0, 1, test))) * 100
        accuracy = max(0, 100 - mape)

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 예측 정확도", f"{accuracy:.1f}%")
        col2.metric("📊 데이터 최신 7일 평균", f"{route_df['LF'].iloc[-7:].mean():.1f}%")
        col3.success("실적 + 향후 60일 예측 전체 보기 설정 완료!")
        st.divider()

        # [수정] 향후 60일 예측 생성
        forecast_days = 60
        forecast = model.forecast(forecast_days)
        future_dates = pd.date_range(start=route_df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        
        adjusted_forecast = []
        for date, val in zip(future_dates, forecast):
            if date in kr_holidays:
                adjusted_forecast.append(min(100, val * 1.15))
            else:
                adjusted_forecast.append(val)
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': adjusted_forecast})

        # 그래프 설정
        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} 분석 및 향후 {forecast_days}일 예측 (정확도: {accuracy:.1f}%)")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="AI 미래 예측", mode='lines+markers')
        
        # [핵심 수정] X축 범위를 '실적 시작'부터 '예측 종료' 날짜까지로 설정
        view_start = route_df['Date'].min()
        view_end = forecast_df['Date'].max()

        fig.update_layout(
            yaxis_title="Load Factor (%)",
            yaxis_range=[0, 105],
            yaxis=dict(tickformat='.0f', ticksuffix="%"),
            # 실적 데이터 시작부터 60일 예측이 끝나는 지점까지 줌인!
            xaxis=dict(range=[view_start, view_end], type="date") 
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("데이터가 부족합니다.")
