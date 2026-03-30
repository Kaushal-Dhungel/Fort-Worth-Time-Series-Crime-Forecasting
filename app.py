from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


from utils import (
    FORECAST_STEPS,
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER,
    acf_pacf_figure,
    add_year_month_fields,
    combined_forecast_frame,
    dataset_summary,
    fit_arimax_with_temperature,
    fit_baseline_sarima,
    monthly_profile,
    prepare_analysis_data,
    residual_diagnostics_frame,
    stl_decomposition,
)


st.set_page_config(
    page_title="Fort Worth Crime Forecasting",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #dbe3ef;
        padding: 0.8rem 1rem;
        border-radius: 0.75rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.1rem;
    }
    .insight {
        background: #f5f7fb;
        border-left: 4px solid #24496b;
        padding: 0.9rem 1rem;
        border-radius: 0.4rem;
        margin: 0.6rem 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_analysis_objects():
    crime, temp, merged = prepare_analysis_data()
    summary = dataset_summary(crime, temp, merged)
    profile = monthly_profile(crime)
    seasonal_frame = add_year_month_fields(crime)
    stl_frame = stl_decomposition(crime)
    baseline = fit_baseline_sarima(merged)
    arimax, future_forecast = fit_arimax_with_temperature(merged)
    diagnostics = residual_diagnostics_frame(arimax.residuals)
    combined = combined_forecast_frame(merged, future_forecast)
    return {
        "crime": crime,
        "temp": temp,
        "merged": merged,
        "summary": summary,
        "profile": profile,
        "seasonal_frame": seasonal_frame,
        "stl": stl_frame,
        "baseline": baseline,
        "arimax": arimax,
        "future_forecast": future_forecast,
        "diagnostics": diagnostics,
        "combined": combined,
    }


def insight(text: str) -> None:
    st.markdown(f"<div class='insight'>{text}</div>", unsafe_allow_html=True)


def section_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def line_chart(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                mode="lines",
                name=column.replace("_", " "),
                line={"width": 3 if "Count" in column or column == "Observed" else 2},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        template="plotly_white",
        height=460,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    return fig


def overview_section(data: dict) -> None:
    summary = data["summary"]
    crime = data["crime"]
    temp = data["temp"]

    section_header(
        "City of Fort Worth Crime Forecasting Analysis",
        "Monthly crime trends, seasonality, and forecasting."
    )

    st.write(
        "This app summarizes monthly Fort Worth crime trends, seasonality, the relationship with temperature, and the forecasting results."
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Crime Records", f"{summary['crime_rows']}")
    metric_cols[1].metric("Temperature Records", f"{summary['temp_rows']}")
    metric_cols[2].metric(
        "Crime Span",
        f"{summary['crime_start']:%m/%Y} - {summary['crime_end']:%m/%Y}",
    )
    metric_cols[3].metric("Aligned Months", f"{summary['merged_rows']}")

    # st.subheader("Dataset Summary")
    # summary_df = pd.DataFrame(
    #     {
    #         "Dataset": ["Monthly crime counts", "Monthly average temperature"],
    #         "Source File": [
    #             "Data/Cleaned/CFW_Monthly_Crime_Count_Cleaned.csv",
    #             "Data/Cleaned/CFW_Avg_Tempr_Cleaned.csv",
    #         ],
    #         "Start": [f"{summary['crime_start']:%Y-%m}", f"{summary['temp_start']:%Y-%m}"],
    #         "End": [f"{summary['crime_end']:%Y-%m}", f"{summary['temp_end']:%Y-%m}"],
    #         "Key Variable": ["Crime_Count", "Temp"],
    #     }
    # )
    # st.dataframe(summary_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = line_chart(crime, "Monthly Crime Count", "Crimes")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        temp_fig = line_chart(temp, "Monthly Average Temperature", "Degrees Fahrenheit")
        st.plotly_chart(temp_fig, use_container_width=True)

    insight(
        f"Crime data spans {summary['crime_start']:%B %Y} through {summary['crime_end']:%B %Y}. "
        f"The aligned crime-temperature analysis covers {summary['merged_start']:%B %Y} through {summary['merged_end']:%B %Y}, "
        f"with an average monthly crime count of {summary['crime_mean']:.0f} incidents."
    )


def crime_trend_section(data: dict) -> None:
    crime = data["crime"].copy()
    crime["Rolling_3M"] = crime["Crime_Count"].rolling(3).mean()
    profile = data["profile"]
    seasonal_frame = data["seasonal_frame"]
    stl_frame = data["stl"]

    section_header(
        "Crime Trend Analysis",
        "Monthly trend, smoothed movement, seasonality, and STL decomposition.",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=crime.index, y=crime["Crime_Count"], mode="lines", name="Monthly crime", line={"color": "#1f4e79", "width": 2.5})
    )
    fig.add_trace(
        go.Scatter(x=crime.index, y=crime["Rolling_3M"], mode="lines", name="3-month rolling average", line={"color": "#d97706", "width": 3})
    )
    fig.update_layout(
        title="Monthly Crime Count With Smoothed Trend",
        xaxis_title="Date",
        yaxis_title="Crime Count",
        template="plotly_white",
        height=460,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "The rolling average smooths short-term volatility and highlights a recurring seasonal rhythm, with stronger activity in warmer parts of the year and softer periods in winter."
    )

    left, right = st.columns(2)
    with left:
        profile_fig = px.line(
            profile,
            x="Month",
            y="Average_Crime_Count",
            markers=True,
            title="Average Crime Count by Calendar Month",
        )
        profile_fig.update_traces(line={"color": "#32628d", "width": 3})
        profile_fig.update_layout(template="plotly_white", height=420, xaxis_title="", yaxis_title="Average Crime Count")
        st.plotly_chart(profile_fig, use_container_width=True)
        insight(
            "Averaging across years makes the seasonal profile easier to see: crime generally rises through spring and summer before easing in late fall and winter."
        )

    with right:
        heatmap_df = seasonal_frame.pivot(index="Year", columns="Month", values="Crime_Count")
        heatmap_fig = px.imshow(
            heatmap_df,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Seasonal Heatmap of Monthly Crime",
            labels={"color": "Crime Count"},
        )
        heatmap_fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        insight(
            "The heatmap shows that the seasonal pattern is fairly consistent across years while still revealing where individual years ran stronger or weaker than the norm."
        )

    st.subheader("STL Decomposition")
    component_order = ["Observed", "Trend", "Seasonal", "Residual"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=component_order)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for row, (component, color) in enumerate(zip(component_order, colors), start=1):
        fig.add_trace(
            go.Scatter(x=stl_frame.index, y=stl_frame[component], mode="lines", name=component, line={"color": color, "width": 2}),
            row=row,
            col=1,
        )
    fig.update_layout(
        height=820,
        template="plotly_white",
        title="STL Decomposition of Fort Worth Monthly Crime",
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )
    fig.update_yaxes(title_text="Observed", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "STL separates long-run movement from repeating seasonal effects, making it easier to see why a seasonal forecasting model is a good fit for this series."
    )


def temperature_section(data: dict) -> None:
    merged = data["merged"]
    summary = data["summary"]

    section_header(
        "Temperature vs Crime",
        "Monthly crime counts aligned with average monthly temperature.",
    )

    dual = go.Figure()
    dual.add_trace(
        go.Scatter(x=merged.index, y=merged["Crime_Count"], mode="lines", name="Crime Count", line={"color": "#1f77b4", "width": 2.5}, yaxis="y1")
    )
    dual.add_trace(
        go.Scatter(x=merged.index, y=merged["Temp"], mode="lines", name="Temperature", line={"color": "#c0392b", "width": 2.5}, yaxis="y2")
    )
    dual.update_layout(
        title="Monthly Crime Count vs Temperature",
        template="plotly_white",
        height=470,
        xaxis={"title": "Date"},
        yaxis={"title": "Crime Count"},
        yaxis2={"title": "Temperature (F)", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": 1.05, "x": 0},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    st.plotly_chart(dual, use_container_width=True)
    insight(
        "Crime and temperature often rise and fall together across the year, with the clearest alignment showing up during spring and summer."
    )

    left, right = st.columns([1.1, 1])
    with left:
        smooth = px.line(
            merged.reset_index(),
            x="Date",
            y=["Crime_3M", "Temp_3M"],
            title="Smoothed 3-Month Trends",
            labels={"value": "Smoothed Value", "variable": "Series"},
        )
        smooth.update_layout(template="plotly_white", height=420, legend_title_text="")
        st.plotly_chart(smooth, use_container_width=True)
        insight(
            "Smoothing both series makes the broader co-movement easier to see by reducing short-term monthly noise."
        )

    with right:
        scatter = px.scatter(
            merged.reset_index(),
            x="Temp",
            y="Crime_Count",
            trendline="ols",
            title="Temperature vs Crime Scatter",
            opacity=0.7,
            labels={"Temp": "Average Temperature (F)", "Crime_Count": "Monthly Crime Count"},
        )
        scatter.update_traces(marker={"size": 8, "color": "#32628d"})
        scatter.update_layout(template="plotly_white", height=420)
        st.plotly_chart(scatter, use_container_width=True)
        insight(
            f"The aligned monthly series has a correlation of {summary['correlation']:.2f}. That does not imply causation, but it supports the project's decision to test temperature as an external forecasting regressor."
        )

    stats_cols = st.columns(3)
    stats_cols[0].metric("Correlation", f"{summary['correlation']:.2f}")
    stats_cols[1].metric("Avg Crime Count", f"{merged['Crime_Count'].mean():.0f}")
    stats_cols[2].metric("Avg Temperature", f"{merged['Temp'].mean():.1f} F")


def forecasting_section(data: dict) -> None:
    merged = data["merged"]
    baseline = data["baseline"]
    arimax = data["arimax"]
    future = data["future_forecast"]
    combined = data["combined"]

    section_header(
        "Forecasting",
        "Seasonal forecasting with a SARIMA baseline and a temperature-informed ARIMAX model.",
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Train Window", "2016-02 to 2024-12")
    metric_cols[1].metric("Validation Window", "2025-01 to 2025-09")
    metric_cols[2].metric("Forecast Horizon", f"{FORECAST_STEPS} months")

    compare_df = pd.DataFrame(
        [
            {
                "Model": baseline.label,
                "Order": f"{baseline.order} x {baseline.seasonal_order}",
                "MAE": baseline.metrics["MAE"],
                "RMSE": baseline.metrics["RMSE"],
                "MAPE": baseline.metrics["MAPE"],
            },
            {
                "Model": arimax.label,
                "Order": f"{arimax.order} x {arimax.seasonal_order}",
                "MAE": arimax.metrics["MAE"],
                "RMSE": arimax.metrics["RMSE"],
                "MAPE": arimax.metrics["MAPE"],
            },
        ]
    )
    st.dataframe(compare_df.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "MAPE": "{:.2f}%"}), use_container_width=True, hide_index=True)
    insight(
        "The temperature-aware ARIMAX model performs modestly better on the 2025 holdout period, suggesting that temperature adds useful signal beyond the crime series alone."
    )

    holdout_fig = go.Figure()
    holdout_fig.add_trace(go.Scatter(x=merged.index, y=merged["Crime_Count"], mode="lines", name="Observed history", line={"color": "#111827", "width": 2}))
    holdout_fig.add_trace(go.Scatter(x=arimax.forecast.index, y=arimax.forecast["Forecast"], mode="lines", name="ARIMAX holdout forecast", line={"color": "#b91c1c", "width": 3}))
    holdout_fig.add_trace(
        go.Scatter(
            x=list(arimax.forecast.index) + list(arimax.forecast.index[::-1]),
            y=list(arimax.forecast["Upper"]) + list(arimax.forecast["Lower"][::-1]),
            fill="toself",
            fillcolor="rgba(185,28,28,0.16)",
            line={"color": "rgba(255,255,255,0)"},
            name="95% interval",
        )
    )
    holdout_fig.add_vline(x=pd.Timestamp("2025-01-01"), line_dash="dash", line_color="#6b7280")
    holdout_fig.update_layout(
        title="Actual vs Holdout Forecast",
        template="plotly_white",
        height=480,
        xaxis_title="Date",
        yaxis_title="Crime Count",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    st.plotly_chart(holdout_fig, use_container_width=True)
    insight(
        "The validation split begins in January 2025, so this chart shows how the model performed on genuinely unseen months rather than on the training data."
    )

    future_fig = go.Figure()
    future_fig.add_trace(go.Scatter(x=combined.index, y=combined["Observed"], mode="lines", name="Observed", line={"color": "#111827", "width": 2.5}))
    future_fig.add_trace(go.Scatter(x=future.index, y=future["Forecast"], mode="lines", name="12-month forecast", line={"color": "#2563eb", "width": 3}))
    future_fig.add_trace(
        go.Scatter(
            x=list(future.index) + list(future.index[::-1]),
            y=list(future["Upper"]) + list(future["Lower"][::-1]),
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="95% interval",
        )
    )
    future_fig.update_layout(
        title="Forward Crime Forecast With Temperature-Informed ARIMAX",
        template="plotly_white",
        height=480,
        xaxis_title="Date",
        yaxis_title="Crime Count",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    st.plotly_chart(future_fig, use_container_width=True)
    insight(
        "This forward view extends the crime forecast by pairing the ARIMAX model with projected monthly temperature values, producing a seasonally informed outlook and uncertainty band."
    )

    st.subheader("Forecast Table")
    forecast_table = future.reset_index().rename(columns={"index": "Date"})
    forecast_table["Date"] = forecast_table["Date"].dt.strftime("%Y-%m")
    st.dataframe(
        forecast_table.style.format(
            {"Forecast": "{:.1f}", "Lower": "{:.1f}", "Upper": "{:.1f}", "Projected_Temp": "{:.1f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def diagnostics_section(data: dict) -> None:
    arimax = data["arimax"]
    diagnostics = data["diagnostics"]

    section_header(
        "Diagnostics and Validation",
        "Residual behavior and validation outputs tied to the temperature-adjusted ARIMAX model.",
    )

    fitted_fig = go.Figure()
    fitted_fig.add_trace(go.Scatter(x=arimax.fitted.index, y=arimax.fitted, mode="lines", name="Fitted", line={"color": "#d97706", "width": 2.5}))
    fitted_fig.add_trace(go.Scatter(x=arimax.fitted.index, y=data["merged"].loc[arimax.fitted.index, "Crime_Count"], mode="lines", name="Observed", line={"color": "#1f2937", "width": 2}))
    fitted_fig.update_layout(
        title="Actual vs Fitted Values on Training Data",
        template="plotly_white",
        height=430,
        xaxis_title="Date",
        yaxis_title="Crime Count",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    st.plotly_chart(fitted_fig, use_container_width=True)
    insight(
        "The fitted-versus-observed view helps verify that the model captures the broad seasonal structure present in the training set without relying only on holdout metrics."
    )

    residual_fig = px.line(
        diagnostics.reset_index(),
        x="Date",
        y="Residual",
        title="Residual Series",
        labels={"Residual": "Residual", "Date": "Date"},
    )
    residual_fig.update_traces(line={"color": "#7c3aed", "width": 2})
    residual_fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    residual_fig.update_layout(template="plotly_white", height=380)
    st.plotly_chart(residual_fig, use_container_width=True)
    insight(
        "Residuals cluster around zero with some seasonal shocks still visible. That is consistent with a practical forecasting model rather than a perfectly white-noise residual stream."
    )

    st.subheader("Residual ACF / PACF")
    fig = acf_pacf_figure(arimax.residuals, lags=24)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    metric_cols = st.columns(3)
    metric_cols[0].metric("ARIMAX MAE", f"{arimax.metrics['MAE']:.1f}")
    metric_cols[1].metric("ARIMAX RMSE", f"{arimax.metrics['RMSE']:.1f}")
    metric_cols[2].metric("ARIMAX MAPE", f"{arimax.metrics['MAPE']:.2f}%")


def main() -> None:
    data = get_analysis_objects()

    with st.sidebar:
        st.header("Navigation")
        section = st.radio(
            "Go to section",
            [
                "Overview",
                "Crime Trend Analysis",
                "Temperature vs Crime",
                "Forecasting",
                "Diagnostics / Validation",
            ],
        )
        st.markdown("---")
        st.caption("Model Creation")
        st.write(
            f"SARIMA/ARIMAX order: `{SARIMA_ORDER}` x `{SARIMA_SEASONAL_ORDER}`"
        )
        # st.caption(
        #     "Built in Python with `statsmodels`."
        # )

    if section == "Overview":
        overview_section(data)
    elif section == "Crime Trend Analysis":
        crime_trend_section(data)
    elif section == "Temperature vs Crime":
        temperature_section(data)
    elif section == "Forecasting":
        forecasting_section(data)
    else:
        diagnostics_section(data)


if __name__ == "__main__":
    main()
