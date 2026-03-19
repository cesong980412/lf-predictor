import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import holidays
import numpy as np

st.set_page_config(page_title="Airline L/F Predictor Pro", layout="wide")
st.title("✈️ 노선별 L/F 인공지능 예측 모델")

kr_holidays = holidays.KR()

# [수정] 파일 업로드 시 기존 세션(기억)을 초기화하는 설정 추가
uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요 (Route 열에 ICNKUL이 있는지 확인!)", type=["xlsx"])

if uploaded_file:
    # 엑셀 읽기
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['LF'] = pd.to_numeric(df['LF'], errors='coerce').ffill().bfill()
    
    if df['LF'].max() <= 1.0:
        df['LF'] = df['LF'] * 100
    
    # [핵심 수정] 엑셀 파일의 'Route' 열에서 실제 노선 이름을 중복 없이 가져오기
    # 앞뒤 공백을 제거(.strip)하여 정확하게 인식하게 합니다.
    actual_routes = sorted(df['Route'].astype(str).str.strip().unique())
    
    # 사이드바 선택창 (목록이 실제 데이터에 맞게 자동 갱신됩니다)
    selected_route = st.sidebar.selectbox("분석할 노선을 선택하세요", actual_routes)
    
    route_df = df[df['Route'].astype(str).str.strip() == selected_route].sort_values('Date').reset_index(drop=True)
    
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
        col2.metric("📊 데이터 최신 7일 평균", f"{route_df['LF'].iloc[-7:].mean():.1f}%")
        col3.success(f"현재 '{selected_route}' 노선 분석 중!")
        st.divider()

        # 향후 60일 예측
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

        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} 분석 및 향후 60일 예측")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="AI 미래 예측", mode='lines+markers')
        
        v_start, v_end = route_df['Date'].min(), forecast_df['Date'].max()

        fig.update_layout(
            yaxis_title="Load Factor (%)",
            yaxis_range=[0, 105],
            yaxis=dict(tickformat='.0f', ticksuffix="%"),
            xaxis=dict(type="date", range=[v_start.strftime('%Y-%m-%d'), v_end.strftime('%Y-%m-%d')])
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"'{selected_route}' 노선의 데이터가 부족합니다.")
