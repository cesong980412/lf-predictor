import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import holidays
import numpy as np

st.set_page_config(page_title="Airline L/F Predictor Pro", layout="wide")
st.title("✈️ 노선별 L/F 예측 시뮬레이터")

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
        # 데이터의 실제 마지막 7일 평균 계산
        col2.metric("📊 데이터 최신 7일 평균", f"{route_df['LF'].iloc[-7:].mean():.1f}%")
        col3.success("화면 줌인 및 단위 보정 완료!")
        st.divider()

        forecast_days = 90
        forecast = model.forecast(forecast_days)
        future_dates = pd.date_range(start=route_df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        
        adjusted_forecast = []
        for date, val in zip(future_dates, forecast):
            if date in kr_holidays:
                adjusted_forecast.append(min(100, val * 1.15))
            else:
                adjusted_forecast.append(val)
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': adjusted_forecast})

        # 그래프 설정 (X축 범위를 데이터가 있는 구간으로만 제한)
        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} 분석 결과 (정확도: {accuracy:.1f}%)")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="AI 미래 예측", mode='lines+markers')
        
        # 데이터 시작과 끝을 계산하여 줌인
        view_start = route_df['Date'].min()
        view_end = forecast_df['Date'].max()

        fig.update_layout(
            yaxis_title="Load Factor (%)",
            yaxis_range=[0, 105],
            yaxis=dict(tickformat='.0f', ticksuffix="%"),
            xaxis_range=[view_start, view_end] # 이 부분이 핵심입니다!
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("데이터가 부족합니다.")
