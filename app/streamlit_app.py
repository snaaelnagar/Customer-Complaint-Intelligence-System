"""
streamlit_app.py

Main Streamlit dashboard for the Machine Learning / NLP Customer
Complaint Intelligence System. Reads pre-aggregated Parquet summaries
(built by dashboard_data.py, risk_analysis.py, driver_analysis.py,
growth_analysis.py, and forecasting.py) and renders them as an
interactive, tabbed dashboard with the ML/NLP pipeline as its core.

Tabs:
    Overview                 -> KPIs, yearly trend, executive summary
    Complaint Analyzer (ML)  -> single-complaint Product/Issue/Topic
                                 prediction with confidence + human
                                 review (src/nlp_predictor.py)
    Classification            -> canonical taxonomy structure and how
                                 raw CFPB labels map to it
                                 (src/taxonomy.py, src/taxonomy_audit.py)
    Similar Complaints        -> nearest-neighbor complaint search
                                 (src/nlp_predictor.py TF-IDF vectors)
    Issue Discovery            -> unsupervised topic discovery layer
                                 (src/topic_discovery.py)
    Trends & Emerging Issues  -> growth analysis + forecasting
    Model Performance          -> classifier accuracy / Macro F1 /
                                 error analysis (src/model_metrics_reader.py)
    Business Analytics        -> Product / Issue / Company / Resolution /
                                 Consumer / Narrative / Channels /
                                 Geography / Risk / Drivers / Recommendations,
                                 nested under one parent tab
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.config_loader import config as app_config
from src.nlp_predictor import (
    analyze_complaint,
    clean_text as nlp_clean_text,
    issue_vectorizer,
    topic_words,
)
from src.taxonomy import (
    canonicalize_product,
    canonicalize_issue,
    full_taxonomy_report,
)
from src.taxonomy_audit import parse_classification_report, audit_support_counts
from src.model_metrics_reader import read_metrics_file
from src.topic_discovery import build_discovery_report, TOPIC_LABELS

DASHBOARD_DIR = ROOT_DIR / app_config["paths"]["dashboard_dir"]
MODEL_DIR = ROOT_DIR / app_config["paths"]["model_dir"]

_STREAMLIT_CONFIG = app_config.get("streamlit", {})

st.set_page_config(
    page_title=_STREAMLIT_CONFIG.get(
        "page_title", "Machine Learning / NLP Customer Complaint Intelligence System"
    ),
    page_icon=_STREAMLIT_CONFIG.get("page_icon", "📊"),
    layout=_STREAMLIT_CONFIG.get("layout", "wide"),
)


@st.cache_data(show_spinner=False)
def load_summary(file_name: str) -> pd.DataFrame:
    """
    Load a pre-built dashboard summary Parquet file.

    Cached so each summary is read from disk only once per session,
    even though every tab re-requests it on every Streamlit rerun.
    """
    path = DASHBOARD_DIR / file_name

    if not path.exists():
        st.error(f"Summary file not found: {path}")
        st.stop()

    return pd.read_parquet(path)


def format_number(value):
    """Compact display formatting: 1.2M / 850.00K / 920."""
    if pd.isna(value):
        return "0"

    value = float(value)

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value / 1_000:.2f}K"

    return str(int(value))


def shorten_text(text, max_len=35):
    """Truncate long category labels so charts stay readable."""
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."


# ==========================================================================
# Overview / Executive Summary
# ==========================================================================

def render_kpis(kpis):
    """Top-level KPI metric tiles for the Overview tab."""
    row = kpis.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Complaints", format_number(row["total_complaints"]))
    c2.metric("Companies", format_number(row["total_companies"]))
    c3.metric("Products", format_number(row["total_products"]))
    c4.metric("States", format_number(row["total_states"]))

    c5, c6, c7 = st.columns(3)
    c5.metric("Timely Response", f"{row['timely_response_pct']:.2f}%")
    c6.metric("Avg Resolution Delay", f"{row['avg_resolution_delay']:.2f} days")
    c7.metric("Narrative Available", f"{row['narrative_availability_pct']:.2f}%")


def render_yearly_trend(yearly_trend):
    """Year-over-year complaint volume line chart."""
    fig = px.line(
        yearly_trend,
        x="Year",
        y="Complaints",
        markers=True,
        title="Complaint Volume Trend Over Time",
    )
    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Complaint volume trend helps identify years with rapid growth in consumer financial complaints."
    )


def render_executive_summary(kpis):
    """Narrative executive summary block under the Overview tab."""
    row = kpis.iloc[0]

    st.subheader("Executive Summary")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Top Product", shorten_text(row["top_product"], 18))
    c2.metric("Top Company", shorten_text(row["top_company"], 18))
    c3.metric("Top Channel", row["top_channel"])
    c4.metric("Peak Year", int(row["peak_year"]))
    c5.metric("Peak Year Complaints", format_number(row["peak_year_complaints"]))

    st.success(
        f"""
        - Dataset contains **{format_number(row["total_complaints"])} consumer financial complaints**.
        - Most complained product: **{row["top_product"]}**.
        - Highest complaint company: **{row["top_company"]}**.
        - Most common issue: **{row["top_issue"]}**.
        - Most complaints are submitted through **{row["top_channel"]}**.
        - Complaint volume peaked in **{int(row["peak_year"])}** with **{format_number(row["peak_year_complaints"])} complaints**.
        - Dashboard uses aggregated Parquet summaries for fast performance.
        """
    )


def render_product_intelligence(top_products, product_share, year_product_trend):
    """Product Intelligence tab: volume, share, and top-5 trend over time."""
    st.header("Product Intelligence")

    top_products = top_products.copy()
    product_share = product_share.copy()
    year_product_trend = year_product_trend.copy()

    top_products["Product Short"] = top_products["Product"].apply(lambda x: shorten_text(x, 45))
    product_share["Product Short"] = product_share["Product"].apply(lambda x: shorten_text(x, 45))
    year_product_trend["Product Short"] = year_product_trend["Product"].apply(lambda x: shorten_text(x, 35))

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            top_products,
            x="Count",
            y="Product Short",
            orientation="h",
            text_auto=True,
            title="Top Products by Complaint Volume",
        )
        fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            product_share,
            x="Share_Pct",
            y="Product Short",
            orientation="h",
            text_auto=".2f",
            title="Product Complaint Share (%)",
        )
        fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Business Insight: **{top_products.iloc[0]['Product']}** has the highest complaint volume with **{format_number(top_products.iloc[0]['Count'])} complaints**."
    )

    # Trend line restricted to top-5 products only, otherwise the legend
    # and chart become unreadable with the full product list.
    top_5_products = top_products.head(5)["Product"].tolist()
    trend_df = year_product_trend[year_product_trend["Product"].isin(top_5_products)]

    fig = px.line(
        trend_df,
        x="Year",
        y="Complaints",
        color="Product Short",
        markers=True,
        title="Complaint Trend for Top 5 Products Over Time",
    )
    fig.update_layout(height=600, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Product-level trend reveals whether complaint growth is concentrated in credit reporting, mortgage, student loans, or other financial products."
    )


def render_issue_intelligence(top_issues, top_sub_issues, issue_product_heatmap, issue_trend):
    """Issue Intelligence tab: top issues/sub-issues, heatmap, and trend."""
    st.header("Issue Intelligence")

    top_issues = top_issues.copy()
    top_sub_issues = top_sub_issues.copy()
    issue_product_heatmap = issue_product_heatmap.copy()
    issue_trend = issue_trend.copy()

    top_issues["Issue Short"] = top_issues["Issue"].apply(lambda x: shorten_text(x, 50))
    top_sub_issues["Sub Issue Short"] = top_sub_issues["Sub Issue"].apply(lambda x: shorten_text(x, 50))
    issue_product_heatmap["Issue Short"] = issue_product_heatmap["Issue"].apply(lambda x: shorten_text(x, 45))
    issue_product_heatmap["Product Short"] = issue_product_heatmap["Product"].apply(lambda x: shorten_text(x, 35))
    issue_trend["Issue Short"] = issue_trend["Issue"].apply(lambda x: shorten_text(x, 35))

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            top_issues,
            x="Count",
            y="Issue Short",
            orientation="h",
            text_auto=True,
            title="Top Complaint Issues",
        )
        fig.update_layout(height=650, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            top_sub_issues,
            x="Count",
            y="Sub Issue Short",
            orientation="h",
            text_auto=True,
            title="Top Complaint Sub-Issues",
        )
        fig.update_layout(height=650, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Business Insight: **{top_issues.iloc[0]['Issue']}** is the most frequent customer complaint issue."
    )

    st.subheader("Issue vs Product Heatmap")

    heatmap_data = issue_product_heatmap.pivot(
        index="Issue Short",
        columns="Product Short",
        values="Count",
    ).fillna(0)

    fig = px.imshow(
        heatmap_data,
        aspect="auto",
        title="Issue vs Product Complaint Matrix",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: The heatmap shows which complaint issues are concentrated within specific financial products."
    )

    fig = px.line(
        issue_trend,
        x="Year",
        y="Complaints",
        color="Issue Short",
        markers=True,
        title="Issue Trend Over Time",
    )
    fig.update_layout(height=600, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Issue trends help identify whether specific complaint categories are increasing or declining over time."
    )


def render_company_intelligence(
    top_companies,
    company_share,
    company_timely_response,
    company_resolution,
    credit_bureau_summary,
):
    """Company Intelligence tab: volume, share, timeliness, resolution, credit bureaus."""
    st.header("Company Intelligence")

    top_companies = top_companies.copy()
    company_share = company_share.copy()
    company_timely_response = company_timely_response.copy()
    company_resolution = company_resolution.copy()

    top_companies["Company Short"] = top_companies["Company"].apply(lambda x: shorten_text(x, 45))
    company_share["Company Short"] = company_share["Company"].apply(lambda x: shorten_text(x, 45))
    company_timely_response["Company Short"] = company_timely_response["Company"].apply(lambda x: shorten_text(x, 35))
    company_resolution["Company Short"] = company_resolution["Company"].apply(lambda x: shorten_text(x, 35))

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            top_companies,
            x="Count",
            y="Company Short",
            orientation="h",
            text_auto=True,
            title="Top Companies by Complaint Volume",
        )
        fig.update_layout(height=650, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            company_share,
            x="Share_Pct",
            y="Company Short",
            orientation="h",
            text_auto=".2f",
            title="Company Complaint Share (%)",
        )
        fig.update_layout(height=650, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Complaint volume is concentrated among a small number of large financial institutions."
    )

    fig = px.bar(
        company_timely_response,
        x="Company Short",
        y="Count",
        color="Timely response?",
        barmode="group",
        title="Company vs Timely Response",
    )
    fig.update_layout(height=600, xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        company_resolution,
        x="Company Short",
        y="Count",
        color="Company response to consumer",
        title="Company vs Resolution Type",
    )
    fig.update_layout(height=650, xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    # Credit bureaus get a dedicated table since they are a high-interest
    # subset of companies that consumers and analysts ask about directly.
    st.subheader("Credit Bureau Analysis")

    st.dataframe(credit_bureau_summary, use_container_width=True)

    st.info(
        "Business Insight: Credit bureau analysis highlights complaint concentration around Experian, Equifax, and TransUnion."
    )


def render_geographic_intelligence(top_states, state_map):
    """
    Geographic Intelligence tab.

    NOTE: top_states / state_map are raw complaint counts, not
    normalized by state population. Larger states (e.g. California,
    Texas) will always rank highest here even if their per-capita
    complaint rate is lower than a smaller state's. Treat this as
    "where complaint volume is concentrated," not "where consumers
    are most affected per person."
    """
    st.header("Geographic Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            top_states,
            x="Count",
            y="State",
            orientation="h",
            text_auto=True,
            title="Top Complaint States",
        )
        fig.update_layout(
            height=550,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.choropleth(
            state_map,
            locations="State",
            locationmode="USA-states",
            color="Count",
            scope="usa",
            title="US Complaint Density Map",
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Geographic analysis highlights states with the highest concentration of consumer complaints."
    )


def render_resolution_intelligence(
    resolution_summary,
    resolution_by_year,
    delay_distribution,
    timely_response,
):
    """Resolution Intelligence tab: response types, timeliness, delay distribution."""
    st.header("Resolution Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            resolution_summary,
            x="Count",
            y="Resolution Type",
            orientation="h",
            text_auto=True,
            title="Company Response Distribution",
        )
        fig.update_layout(height=550, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            timely_response,
            names="Timely Response",
            values="Count",
            hole=0.45,
            title="Timely Response Distribution",
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Resolution distribution shows how companies are closing complaints and whether responses are delivered on time."
    )

    top_resolutions = resolution_summary.head(5)["Resolution Type"].tolist()
    trend_df = resolution_by_year[
        resolution_by_year["Company response to consumer"].isin(top_resolutions)
    ]

    fig = px.line(
        trend_df,
        x="Year",
        y="Count",
        color="Company response to consumer",
        markers=True,
        title="Resolution Type Trend Over Time",
    )
    fig.update_layout(height=600, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # delay_distribution already excludes negative and >365-day outliers
    # (clipped upstream in dashboard_data.py) so the histogram stays readable.
    if not delay_distribution.empty:
        fig = px.histogram(
            delay_distribution,
            x="Resolution_Delay",
            nbins=50,
            title="Resolution Delay Distribution",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def render_consumer_segments(consumer_group):
    """Consumer Segments tab: complaint volume by consumer tag group."""
    st.header("Consumer Segments")

    fig = px.bar(
        consumer_group,
        x="Consumer Group",
        y="Count",
        title="Consumer Segment Distribution",
        text_auto=True,
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Consumer segment analysis helps identify vulnerable groups such as older Americans and servicemembers."
    )


def render_narrative_intelligence(
    narrative_availability,
    narrative_word_count,
    narrative_product_length,
    narrative_product_availability,
    narrative_length_trend,
    narrative_complexity
    # top_narrative_words,
):
    """
    Narrative Intelligence tab: free-text complaint analysis.

    NOTE: top_narrative_words.parquet is generated by
    dashboard_data.py, but isn't included in this repository's bundled
    demo dashboard snapshot (data/processed/dashboard/). The parameter
    and the corresponding chart block below are commented out together
    so this function still runs without that file existing; uncomment
    both once the file has been (re)generated by running the full
    pipeline against the raw dataset.
    """
    st.header("Narrative Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            narrative_availability,
            names="Has Narrative",
            values="Count",
            hole=0.45,
            title="Narrative Availability",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not narrative_word_count.empty:
            fig = px.histogram(
                narrative_word_count,
                x="Narrative_Word_Count",
                nbins=50,
                title="Narrative Word Count Distribution",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Narrative word count data is not available.")

    st.info(
        "Business Insight: Narrative availability shows how often consumers provide detailed complaint descriptions."
    )

    st.subheader("Narrative Length by Product")

    product_length = narrative_product_length.copy()
    product_length["Product Short"] = product_length["Product"].apply(
        lambda x: shorten_text(x, 45)
    )

    fig = px.bar(
        product_length,
        x="Avg_Word_Count",
        y="Product Short",
        orientation="h",
        text_auto=".2f",
        title="Average Narrative Length by Product",
    )

    fig.update_layout(
        height=600,
        yaxis={"categoryorder": "total ascending"},
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Products with longer narratives usually involve more complex customer problems."
    )

    st.subheader("Narrative Availability by Product")

    product_availability = narrative_product_availability.copy()
    product_availability["Product Short"] = product_availability["Product"].apply(
        lambda x: shorten_text(x, 45)
    )

    fig = px.bar(
        product_availability,
        x="Narrative_Availability_Pct",
        y="Product Short",
        orientation="h",
        text_auto=".2f",
        title="Narrative Availability (%) by Product",
    )

    fig.update_layout(
        height=600,
        yaxis={"categoryorder": "total ascending"},
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Narrative availability by product shows where consumers are more likely to explain their issues in detail."
    )

    st.subheader("Narrative Length Trend Over Time")

    fig = px.line(
        narrative_length_trend,
        x="Year",
        y="Avg_Word_Count",
        markers=True,
        title="Average Narrative Word Count Over Time",
    )

    fig.update_layout(
        height=550,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Narrative length trend shows whether consumer complaints are becoming more detailed over time."
    )

    st.subheader("Complaint Complexity Distribution")

    fig = px.pie(
        narrative_complexity,
        names="Complexity",
        values="Count",
        hole=0.45,
        title="Complaint Complexity Distribution",
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Complaint complexity groups narratives into low, medium, and high detail based on word count."
    )

    # Top Narrative Words — dashboard_data.py generates this file, but
    # it isn't bundled in this repository's demo dashboard snapshot.
    # Re-enable this block once top_narrative_words.parquet has been
    # (re)generated by running the full pipeline against raw data.
    # st.subheader("Top Narrative Words")
    #
    # fig = px.bar(
    #     top_narrative_words.head(20),
    #     x="Count",
    #     y="Word",
    #     orientation="h",
    #     text_auto=True,
    #     title="Top 20 Words in Complaint Narratives",
    # )
    #
    # fig.update_layout(
    #     height=600,
    #     yaxis={"categoryorder": "total ascending"},
    # )
    #
    # st.plotly_chart(fig, use_container_width=True)
    #
    # st.info(
    #     "Business Insight: Top narrative words reveal the most common themes appearing in consumer complaint descriptions."
    # )


def render_channel_overview(submitted_via):
    """Submission Channel tab: how consumers file complaints."""
    st.header("Submission Channel Intelligence")

    fig = px.bar(
        submitted_via,
        x="Channel",
        y="Count",
        title="Complaint Submission Channels",
        text_auto=True,
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Submission channel analysis shows consumer preference for digital, phone, mail, or referral-based complaint submission."
    )


# ==========================================================================
# Business Analytics: Risk Intelligence (Risk Score, Drivers, Growth, Forecast)
# ==========================================================================

def render_company_risk_score(company_risk_score):
    """
    Company Risk Score tab.

    Risk_Score blends complaint volume, untimely-response %, and
    average resolution delay (see risk_analysis.py for weights).
    NOTE: there is currently no minimum complaint-count floor —
    companies with very few total complaints can land in the "High
    Risk" list purely from a small sample producing an extreme
    Untimely_Response_Pct. Treat low-volume companies in this list
    with caution until a floor is added upstream.
    """
    st.header("Company Risk Score")

    risk_df = company_risk_score.copy()
    risk_df["Company Short"] = risk_df["Company"].apply(lambda x: shorten_text(x, 45))

    top_risk = risk_df.sort_values("Risk_Score", ascending=False).head(20)

    c1, c2, c3 = st.columns(3)

    c1.metric("Highest Risk Company", shorten_text(top_risk.iloc[0]["Company"], 20))
    c2.metric("Highest Risk Score", f"{top_risk.iloc[0]['Risk_Score']:.2f}")
    c3.metric("Risk Level", top_risk.iloc[0]["Risk_Level"])

    fig = px.bar(
        top_risk,
        x="Risk_Score",
        y="Company Short",
        orientation="h",
        color="Risk_Level",
        text_auto=".2f",
        title="Top 20 High-Risk Companies",
    )

    fig.update_layout(
        height=700,
        yaxis={"categoryorder": "total ascending"},
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Risk score combines complaint volume, untimely response percentage, "
        "and average resolution delay to identify high-risk companies."
    )

    st.subheader("Risk Factor Breakdown")

    breakdown_df = top_risk[
        [
            "Company Short",
            "Complaint_Score",
            "Untimely_Score",
            "Delay_Score",
        ]
    ]

    breakdown_long = breakdown_df.melt(
        id_vars="Company Short",
        var_name="Risk Factor",
        value_name="Score",
    )

    fig = px.bar(
        breakdown_long,
        x="Company Short",
        y="Score",
        color="Risk Factor",
        title="Risk Score Component Breakdown",
    )

    fig.update_layout(
        height=650,
        xaxis_tickangle=-35,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top_risk[
            [
                "Company",
                "Complaint_Count",
                "Untimely_Response_Pct",
                "Avg_Resolution_Delay",
                "Risk_Score",
                "Risk_Level",
            ]
        ],
        use_container_width=True,
    )


def render_driver_analysis(
    driver_analysis,
    top_complaint_drivers,
    product_driver_summary,
):
    """Complaint Driver Analysis tab: root-cause Product/Issue/Sub-issue breakdown."""
    st.header("Complaint Driver Analysis")

    st.subheader("Top Complaint Drivers")

    top_drivers = top_complaint_drivers.copy()
    top_drivers["Driver"] = (
        top_drivers["Product"].apply(lambda x: shorten_text(x, 25))
        + " → "
        + top_drivers["Issue"].apply(lambda x: shorten_text(x, 35))
    )

    fig = px.bar(
        top_drivers.head(20),
        x="Complaint_Count",
        y="Driver",
        orientation="h",
        text_auto=True,
        title="Top Complaint Drivers",
    )

    fig.update_layout(
        height=700,
        yaxis={"categoryorder": "total ascending"},
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Driver analysis identifies the product and issue combinations responsible for the highest complaint volume."
    )

    st.subheader("Product → Issue → Sub-Issue Driver Tree")

    # Capped at 200 rows so the treemap stays legible and fast to render.
    tree_df = driver_analysis.head(200).copy()

    fig = px.treemap(
        tree_df,
        path=["Product", "Issue", "Sub-issue"],
        values="Complaint_Count",
        title="Complaint Driver Treemap",
    )

    fig.update_layout(height=750)

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: The treemap shows the root-cause hierarchy behind complaints: product, issue, and sub-issue."
    )

    st.subheader("Top Drivers by Product")

    st.dataframe(
        product_driver_summary[
            [
                "Product",
                "Issue",
                "Sub-issue",
                "Complaint_Count",
                "Driver_Contribution_Pct",
            ]
        ],
        use_container_width=True,
    )


def render_growth_analysis(product_growth, issue_growth, monthly_complaint_trend):
    """
    Growth Analysis tab: YoY growth for products/issues, monthly trend,
    and "New / Emerging" highlights for items with zero complaints in
    the prior year (which would otherwise be mislabeled "Stable").
    """
    st.header("Growth Analysis")

    product_growth = product_growth.copy()
    issue_growth = issue_growth.copy()

    product_growth["Product Short"] = product_growth["Product"].apply(
        lambda x: shorten_text(x, 45)
    )

    issue_growth["Issue Short"] = issue_growth["Issue"].apply(
        lambda x: shorten_text(x, 45)
    )

    # Minimum current-year volume filter avoids noisy growth % swings
    # from very low-complaint products/issues dominating the chart.
    top_product_growth = product_growth[
        product_growth["Current_Year_Complaints"] >= 1000
    ].head(15)

    top_issue_growth = issue_growth[
        issue_growth["Current_Year_Complaints"] >= 1000
    ].head(15)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            top_product_growth,
            x="YoY_Growth_Pct",
            y="Product Short",
            orientation="h",
            color="Growth_Label",
            text_auto=".2f",
            title="Fastest Growing Products",
        )

        fig.update_layout(
            height=650,
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            top_issue_growth,
            x="YoY_Growth_Pct",
            y="Issue Short",
            orientation="h",
            color="Growth_Label",
            text_auto=".2f",
            title="Fastest Growing Issues",
        )

        fig.update_layout(
            height=650,
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Growth analysis identifies products and issues where complaint volume is increasing fastest year-over-year."
    )

    monthly_plot = monthly_complaint_trend.tail(60).copy()
    fig = px.line(
        monthly_plot,
        x="Period",
        y="Complaints",
        markers=True,
        title="Monthly Complaint Trend - Last 5 Years",
    )
    st.caption(
        "Growth calculated using latest complete year vs previous complete year."
    )

    fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Monthly trend helps identify seasonality, sudden spikes, and long-term complaint growth."
    )

    # Surface "New / Emerging" items separately — these had zero
    # complaints in the prior year, so they would otherwise be hidden
    # at the top of a growth-% sorted list.
    emerging_products = product_growth[
        product_growth["Growth_Label"] == "New / Emerging"
    ]

    emerging_issues = issue_growth[
        issue_growth["Growth_Label"] == "New / Emerging"
    ]

    if not emerging_products.empty or not emerging_issues.empty:
        st.warning(
            f"New / Emerging detected: "
            f"{len(emerging_products)} products and {len(emerging_issues)} issues."
        )

    st.subheader("Product Growth Table")

    product_table = product_growth[
        [
            "Product",
            "Previous_Year_Complaints",
            "Current_Year_Complaints",
            "YoY_Growth_Pct",
            "Growth_Label",
        ]
    ].head(25).copy()

    product_table["YoY_Growth_Pct"] = product_table["YoY_Growth_Pct"].round(2)

    st.dataframe(
        product_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Issue Growth Table")

    issue_table = issue_growth[
        [
            "Issue",
            "Previous_Year_Complaints",
            "Current_Year_Complaints",
            "YoY_Growth_Pct",
            "Growth_Label",
        ]
    ].head(25).copy()

    issue_table["YoY_Growth_Pct"] = (
        issue_table["YoY_Growth_Pct"].round(2)
    )

    st.dataframe(
        issue_table,
        use_container_width=True,
        hide_index=True,
    )


def render_forecasting(complaint_forecast, forecast_summary):
    """
    Forecasting tab: Prophet-based complaint volume forecast with a
    shaded confidence band, plus validation metrics (MAPE/MAE) so the
    forecast's reliability is visible alongside the prediction itself.
    """
    st.header("Complaint Forecasting")

    summary = forecast_summary.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latest Actual Complaints",
        format_number(summary["Latest_Actual_Complaints"]),
    )

    c2.metric(
        "Next Month Forecast",
        format_number(summary["Next_Month_Forecast"]),
    )

    c3.metric(
        "Next 3 Months Forecast",
        format_number(summary["Next_3_Month_Forecast"]),
    )

    c4.metric(
        "Forecast Trend",
        summary["Forecast_Trend"],
        f"{summary['Expected_Change_Pct']:.2f}%",
    )

    mape = summary["Validation_MAPE"]

    if mape < 10:
        quality = "Excellent"
    elif mape < 20:
        quality = "Good"
    elif mape < 30:
        quality = "Acceptable"
    else:
        quality = "Poor"

    st.success(
        f"""
        Forecast Validation

        MAPE: {summary['Validation_MAPE']:.2f}%

        MAE: {summary['Validation_MAE']:.0f}

        Forecast Quality: {quality}

        Rule: MAPE below 10% is considered excellent.
        """
    )

    plot_df = complaint_forecast.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"])
    plot_df["Period"] = plot_df["Date"].dt.strftime("%Y-%m")

    actual_df = plot_df[plot_df["Type"] == "Actual"].copy()
    forecast_df = plot_df[plot_df["Type"] == "Forecast"].copy()

    # Confidence band is built from two invisible traces (upper bound,
    # then lower bound with fill="tonexty") rather than Plotly Express,
    # since px doesn't support shaded range bands directly.
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Period"],
            y=forecast_df["Upper_Bound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Period"],
            y=forecast_df["Lower_Bound"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.25)",
            line=dict(width=0),
            name="Forecast Confidence Band",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=actual_df["Period"],
            y=actual_df["Complaints"],
            mode="lines+markers",
            name="Actual Complaints",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Period"],
            y=forecast_df["Complaints"],
            mode="lines+markers",
            name="Forecast Complaints",
        )
    )

    # Vertical marker + annotation showing exactly where actuals end
    # and the forecast begins.
    if not forecast_df.empty:
        forecast_start = forecast_df.iloc[0]["Period"]

        fig.add_shape(
            type="line",
            x0=forecast_start,
            x1=forecast_start,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                dash="dash",
                width=2,
            ),
        )

        fig.add_annotation(
            x=forecast_start,
            y=1,
            xref="x",
            yref="paper",
            text="Forecast Starts",
            showarrow=False,
            yanchor="bottom",
        )

    fig.update_layout(
        title="Complaint Forecast with Confidence Band",
        height=650,
        xaxis_tickangle=-45,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Forecasting estimates future complaint volume using historical monthly complaint trends. "
        "The confidence band shows the expected forecast range, not an exact guaranteed value."
    )

    st.subheader("Forecast Data")

    st.dataframe(
        complaint_forecast.tail(12),
        use_container_width=True,
    )


def render_executive_risk_dashboard(
    company_risk_score,
    top_complaint_drivers,
    product_growth,
    issue_growth,
    forecast_summary,
):
    """
    Executive Risk Dashboard tab: single-screen rollup combining the
    highest risk company, top driver, fastest-growing product/issue,
    and forecast trend + validation metrics. Diagnostic only — this
    tab reports signals; the Recommendations tab turns them into
    suggested actions.
    """
    st.header("Executive Risk Dashboard")

    highest_risk = company_risk_score.iloc[0]
    top_driver = top_complaint_drivers.iloc[0]
    fastest_product = product_growth.iloc[0]
    fastest_issue = issue_growth.iloc[0]
    forecast = forecast_summary.iloc[0]

    model_name = (
        forecast["Model"]
        if "Model" in forecast_summary.columns
        else "Linear Trend"
    )

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    c1.metric(
        "Highest Risk Company",
        shorten_text(highest_risk["Company"], 20),
    )

    c2.metric(
        "Risk Score",
        f"{highest_risk['Risk_Score']:.2f}",
    )

    c3.metric(
        "Forecast Trend",
        forecast["Forecast_Trend"],
    )

    c4.metric(
        "Next Month Forecast",
        format_number(forecast["Next_Month_Forecast"]),
    )

    c5.metric(
        "Forecast Model",
        model_name,
    )
    c6.metric(
        "MAPE %",
        f"{forecast['Validation_MAPE']:.2f}",
    )

    c7.metric(
        "MAE",
        format_number(
            forecast["Validation_MAE"]
        ),
    )
    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.success(
            f"""
            Fastest Growing Product

            Product: {fastest_product['Product']}

            Growth: {fastest_product['YoY_Growth_Pct']:.2f}%
            """
        )

    with c2:
        st.warning(
            f"""
            Fastest Growing Issue

            Issue: {fastest_issue['Issue']}

            Growth: {fastest_issue['YoY_Growth_Pct']:.2f}%
            """
        )

    st.divider()

    st.subheader("Top Complaint Driver")

    st.info(
        f"""
        Product: {top_driver['Product']}

        Issue: {top_driver['Issue']}

        Sub-Issue: {top_driver['Sub-issue']}

        Complaints: {format_number(top_driver['Complaint_Count'])}
        """
    )

    st.divider()

    risk_counts = (
        company_risk_score["Risk_Level"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = ["Risk Level", "Count"]

    fig = px.pie(
        risk_counts,
        names="Risk Level",
        values="Count",
        hole=0.45,
        title="Risk Level Distribution",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Business Insight: Risk distribution shows how many companies fall into Low, Medium, and High Risk categories."
    )

    st.divider()

    st.subheader("Executive Summary")

    st.success(
        f"""
        • Highest Risk Company: {highest_risk['Company']}

        • Risk Score: {highest_risk['Risk_Score']:.2f}

        • Fastest Growing Product: {fastest_product['Product']}

        • Fastest Growing Issue: {fastest_issue['Issue']}

        • Top Complaint Driver: {top_driver['Issue']}

        • Forecast Trend: {forecast['Forecast_Trend']}

        • Forecast Model: {model_name}

        • Next Month Expected Complaints: {format_number(forecast['Next_Month_Forecast'])}
        """
    )


# ==========================================================================
# ML/NLP Core: Complaint Analyzer
# ==========================================================================

def render_nlp_prediction():
    """
    NLP Prediction tab: interactive complaint analyzer.

    Calls analyze_complaint() (Product classifier, Issue classifier,
    LDA topic model) on user-entered free text. Product and Issue
    predictions are canonical-taxonomy-aware (src/taxonomy.py) and
    include a confidence score, top alternatives, and an
    AUTO_ACCEPT / HUMAN_REVIEW decision (src/nlp_predictor.py). The
    raw pre-taxonomy label is also shown whenever it differs from the
    canonical one, so nothing from the underlying model is hidden.

    Topic IDs are mapped to human-readable names via the canonical
    TOPIC_LABELS dict in src/topic_discovery.py, the single source of
    truth for this mapping (notebooks/02_nlp_analysis.ipynb keeps its
    own standalone copy for notebook portability, but the application
    code has exactly one definition).
    """
    st.header("Complaint Analyzer")

    st.info(
        "Enter a consumer complaint narrative to predict Product, Issue, and Topic."
    )

    complaint_text = st.text_area(
        "Enter Complaint Narrative",
        height=220,
        placeholder="Example: Someone opened fraudulent accounts in my name and there are inquiries on my credit report..."
    )

    if st.button("Analyze Complaint"):
        if not complaint_text.strip():
            st.warning("Please enter a complaint narrative.")
            return

        try:
            with st.spinner("Analyzing complaint..."):
                result = analyze_complaint(complaint_text)
        except Exception as e:
            st.error(f"NLP prediction failed: {e}")
            return

        def render_prediction_card(title: str, prediction: dict) -> None:
            """
            Render a single Product/Issue prediction as: canonical
            label, confidence, top alternatives, and an AUTO_ACCEPT /
            HUMAN_REVIEW decision badge.
            """
            st.subheader(title)

            confidence = prediction["confidence"]
            decision = prediction["decision"]

            st.metric(
                prediction["label"],
                f"{confidence:.2%} confidence",
            )

            if decision == "AUTO_ACCEPT":
                st.success(f"Decision: ✅ AUTO-ACCEPT (≥ {prediction['confidence_threshold']:.0%} threshold)")
            else:
                st.warning(f"Decision: ⚠️ HUMAN REVIEW (< {prediction['confidence_threshold']:.0%} threshold)")

            if prediction["alternatives"]:
                alt_text = " · ".join(
                    f"{alt['label']} → {alt['confidence']:.2%}"
                    for alt in prediction["alternatives"]
                )
                st.caption(f"Alternatives: {alt_text}")

            if prediction["raw_label"] != prediction["label"]:
                st.caption(f"Raw model label (pre-taxonomy): {prediction['raw_label']}")

        c1, c2 = st.columns(2)

        with c1:
            render_prediction_card("Predicted Product", result["product"])

        with c2:
            render_prediction_card("Predicted Issue", result["issue"])

        topic = result["topic"]
        topic_id = topic["topic_id"]
        topic_name = TOPIC_LABELS.get(topic_id, f"Topic {topic_id}")

        st.subheader("Predicted Topic")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Topic",
                topic_name,
                f"{topic['confidence']:.2%} confidence",
            )

        with c2:
            st.metric(
                "Topic ID",
                topic_id,
            )

        st.write("Top Topic Keywords:")
        st.write(", ".join(topic["topic_words"]))


# ==========================================================================
# Recommendation Engine
# ==========================================================================

def render_recommendations(recommendations, executive_action_plan):
    """
    Recommendations tab: converts the diagnostic signals from Modules
    1-3 (risk scores, growth labels, forecast trend, complaint drivers)
    into a ranked Executive Action Plan plus a filterable, priority-
    coded list of individual recommendations.
    """
    st.header("Recommendation Engine")

    st.info(
        "This module converts risk, growth, forecasting, and complaint-driver signals into actionable business recommendations."
    )

    # =========================
    # Executive Action Plan
    # =========================
    st.subheader("Executive Action Plan")

    for _, row in executive_action_plan.iterrows():
        with st.container():
            st.markdown(
                f"""
                ### Priority {row['Action_Rank']}: {row['Focus_Area']}

                **Target:** {row['Target']}

                **Why it matters:** {row['Why_It_Matters']}

                **Recommended Action:**  
                {row['Recommended_Action']}
                """
            )
            st.divider()

    # =========================
    # Recommendation Summary
    # =========================
    st.subheader("Recommendation Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Recommendations",
        len(recommendations),
    )

    c2.metric(
        "High Priority",
        len(recommendations[recommendations["Priority"] == "High"]),
    )

    c3.metric(
        "Categories",
        recommendations["Category"].nunique(),
    )

    st.divider()

    # =========================
    # Priority Filter
    # =========================
    selected_priority = st.selectbox(
        "Filter by Priority",
        ["All"] + sorted(recommendations["Priority"].dropna().unique().tolist()),
    )

    filtered = recommendations.copy()

    if selected_priority != "All":
        filtered = filtered[filtered["Priority"] == selected_priority]

    # =========================
    # Category-wise Cards
    # =========================
    st.subheader("Detailed Recommendation Cards")

    for category in filtered["Category"].unique():
        category_df = filtered[filtered["Category"] == category]

        st.markdown(f"## {category}")

        # Card color follows priority: red (High) / yellow (Medium) /
        # green (Low) via Streamlit's built-in error/warning/success boxes.
        for _, row in category_df.iterrows():
            if row["Priority"] == "High":
                st.error(
                    f"""
                    **Priority:** {row['Priority']}

                    **Entity:** {row['Entity']}

                    **Signal:** {row['Signal']}

                    **Recommended Action:**  
                    {row['Recommendation']}
                    """
                )

            elif row["Priority"] == "Medium":
                st.warning(
                    f"""
                    **Priority:** {row['Priority']}

                    **Entity:** {row['Entity']}

                    **Signal:** {row['Signal']}

                    **Recommended Action:**  
                    {row['Recommendation']}
                    """
                )

            else:
                st.success(
                    f"""
                    **Priority:** {row['Priority']}

                    **Entity:** {row['Entity']}

                    **Signal:** {row['Signal']}

                    **Recommended Action:**  
                    {row['Recommendation']}
                    """
                )

    st.divider()

    # =========================
    # Raw Table
    # =========================
    with st.expander("View Raw Recommendation Table"):
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================================
# Classification tab — canonical taxonomy structure
# ==========================================================================

def render_classification() -> None:
    """
    Classification tab: shows how raw CFPB Product/Issue labels map
    into the canonical taxonomy used for training and prediction
    (src/taxonomy.py), backed by the real per-class support counts
    already saved from this project's trained classifiers
    (src/taxonomy_audit.py).
    """
    st.header("Classification: Canonical Taxonomy")

    st.info(
        "The raw CFPB Product/Issue fields have been renamed and "
        "consolidated several times since 2012. This project canonicalizes "
        "historically-fragmented raw labels into stable categories before "
        "training, without discarding the original label. "
        "Full methodology: docs/taxonomy_audit.md."
    )

    try:
        product_support = parse_classification_report(
            MODEL_DIR / "product_classifier_metrics.txt"
        )
        issue_support = parse_classification_report(
            MODEL_DIR / "issue_classifier_metrics.txt"
        )
        product_audit = audit_support_counts(product_support, canonicalize_product)
        issue_audit = audit_support_counts(issue_support, canonicalize_issue)
    except FileNotFoundError as e:
        st.warning(f"Could not load classifier metrics: {e}")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Product")
        st.metric("Raw classes", product_audit["raw_class_count"])
        st.metric("Canonical classes", product_audit["canonical_class_count"])
        st.caption(f"{product_audit['classes_removed']} classes consolidated")

    with c2:
        st.subheader("Issue")
        st.metric("Raw classes", issue_audit["raw_class_count"])
        st.metric("Canonical classes", issue_audit["canonical_class_count"])
        st.caption(f"{issue_audit['classes_removed']} classes consolidated")

    st.divider()
    st.subheader("Merged Categories")

    taxonomy_choice = st.radio(
        "Show merges for:", ["Product", "Issue"], horizontal=True
    )

    audit = product_audit if taxonomy_choice == "Product" else issue_audit

    for canonical, info in audit["merged_groups"].items():
        with st.expander(f"{canonical}  (combined support: {info['combined_support']})"):
            for raw_label, support in info["raw_components"]:
                st.write(f"- \"{raw_label}\" (support={support})")

    st.divider()
    st.subheader("Merge Rationale & Confidence")

    report = full_taxonomy_report()
    rationale_rows = report["product"] if taxonomy_choice == "Product" else report["issue"]

    st.dataframe(
        pd.DataFrame(rationale_rows)[["canonical_label", "confidence", "rationale"]],
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================================
# Similar Complaints tab — nearest-neighbor search over narrative text
# ==========================================================================

def render_similar_complaints() -> None:
    """
    Similar Complaints tab: TF-IDF cosine-similarity search over a
    corpus of complaint narratives.

    This feature needs a narrative-text corpus to search against
    (data/processed/narratives_training.parquet). That file is not
    available in this environment (see docs/taxonomy_audit.md for the
    same limitation), so this tab shows a clear message instead of
    fabricating results. The search logic itself is fully implemented
    and will work as soon as that file exists.
    """
    st.header("Similar Complaints")

    narratives_path = ROOT_DIR / "data" / "processed" / "narratives_training.parquet"

    if not narratives_path.exists():
        st.warning(
            "Similar-complaint search needs a narrative text corpus "
            f"(expected at `{narratives_path.relative_to(ROOT_DIR)}`), which "
            "is not available in this environment. The search itself is "
            "fully implemented (TF-IDF cosine similarity over the existing "
            "Issue vectorizer) and will work as soon as that file is present."
        )
        return

    from sklearn.metrics.pairwise import cosine_similarity

    @st.cache_data(show_spinner=False)
    def _load_corpus() -> pd.DataFrame:
        df = pd.read_parquet(narratives_path)
        return df[["Consumer complaint narrative", "Product", "Issue"]].dropna()

    @st.cache_resource(show_spinner=False)
    def _vectorize_corpus(_df: pd.DataFrame):
        cleaned = _df["Consumer complaint narrative"].apply(nlp_clean_text)
        return issue_vectorizer.transform(cleaned)

    corpus_df = _load_corpus()
    corpus_vectors = _vectorize_corpus(corpus_df)

    query_text = st.text_area(
        "Enter a complaint narrative to find similar complaints",
        height=160,
        placeholder="Example: My credit card was charged twice for the same purchase...",
    )

    top_n = st.slider("Number of similar complaints to show", 3, 10, 5)

    if st.button("Find Similar Complaints") and query_text.strip():
        query_vector = issue_vectorizer.transform([nlp_clean_text(query_text)])
        similarities = cosine_similarity(query_vector, corpus_vectors)[0]

        top_indices = similarities.argsort()[-top_n:][::-1]

        for rank, idx in enumerate(top_indices, start=1):
            row = corpus_df.iloc[idx]
            with st.expander(f"#{rank} — similarity {similarities[idx]:.2%} — {row['Product']} / {row['Issue']}"):
                st.write(row["Consumer complaint narrative"])


# ==========================================================================
# Issue Discovery tab — unsupervised topic discovery layer
# ==========================================================================

def render_issue_discovery() -> None:
    """
    Issue Discovery tab: unsupervised discovery layer on top of the
    trained LDA topic model (src/topic_discovery.py).

    The static topic catalog (topic keywords) is real and available
    today. The time-based "emerging issues" view needs a narrative
    corpus with timestamps, which is not available in this
    environment — that section shows a clear message instead of
    fabricated trend numbers.
    """
    st.header("Issue Discovery")

    st.info(
        "Recurring complaint themes discovered via unsupervised topic "
        "modeling (LDA), independent of the supervised Product/Issue "
        "classifiers above."
    )

    report = build_discovery_report(topic_words)

    cols = st.columns(2)
    for i, topic in enumerate(report["topics"]):
        with cols[i % 2]:
            with st.expander(f"Topic {topic.topic_id}: {topic.label}"):
                st.write(", ".join(topic.top_words))

    st.divider()
    st.subheader("Emerging Issues Over Time")
    st.warning(
        "Emerging-issue detection compares topic prevalence between two "
        "time periods and needs a narrative corpus with timestamps "
        "(data/processed/narratives_training.parquet), which is not "
        "available in this environment. The comparison logic is fully "
        "implemented in src/topic_discovery.py "
        "(compute_topic_prevalence / detect_emerging_topics) and will "
        "populate this section as soon as that data exists."
    )


# ==========================================================================
# Model Performance tab — classifier accuracy / Macro F1 / error analysis
# ==========================================================================

def render_model_performance() -> None:
    """
    Model Performance tab: headline classifier metrics and, once the
    models are retrained on canonical labels (src/nlp_model_training.py),
    Macro/Weighted F1, Top-K accuracy, and automatic error analysis
    (src/nlp_evaluation.py). Currently-shipped models were trained
    before this evaluation upgrade and are shown in their legacy
    (accuracy-only) format until they are retrained.
    """
    st.header("Model Performance")

    for title, filename in [
        ("Product Classifier", "product_classifier_metrics.txt"),
        ("Issue Classifier", "issue_classifier_metrics.txt"),
    ]:
        st.subheader(title)

        try:
            metrics = read_metrics_file(MODEL_DIR / filename)
        except FileNotFoundError:
            st.warning(f"Metrics file not found for {title}.")
            continue

        if metrics["format"] == "legacy":
            st.caption(
                "Legacy metrics format (accuracy only) — this model has not "
                "been retrained since the Macro F1 / Top-K evaluation "
                "upgrade (src/nlp_evaluation.py). See docs/evaluation.md."
            )
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            m2.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
            m3.metric("Weighted F1", f"{metrics['weighted_f1']:.2%}")
            top1 = metrics["top_k_accuracy"].get(1)
            if top1 is not None:
                m4.metric("Top-1 Accuracy", f"{top1:.2%}")

            if metrics["confused_pairs"]:
                st.write("**Where the model struggles most:**")
                for pair in metrics["confused_pairs"][:5]:
                    st.write(
                        f"- '{pair['true_label']}' → '{pair['predicted_label']}' "
                        f"(×{pair['count']})"
                    )
                    st.caption(pair["explanation"])

        with st.expander("Full per-class report"):
            st.dataframe(
                pd.DataFrame(metrics["per_class"]),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()


# ==========================================================================
# Main app: load all summaries once, then render every tab
# ==========================================================================

def main() -> None:
    st.title("📊 Machine Learning / NLP Customer Complaint Intelligence System")
    st.caption("Executive overview of consumer financial complaint data")

    # --- Overview summaries ---
    kpis = load_summary("kpis.parquet")
    yearly_trend = load_summary("yearly_trend.parquet")
    timely_response = load_summary("timely_response.parquet")
    submitted_via = load_summary("submitted_via.parquet")
    consumer_group = load_summary("consumer_group.parquet")

    top_products = load_summary("top_products.parquet")
    product_share = load_summary("product_share.parquet")
    year_product_trend = load_summary("year_product_trend.parquet")

    top_issues = load_summary("top_issues.parquet")
    top_sub_issues = load_summary("top_sub_issues.parquet")
    issue_product_heatmap = load_summary("issue_product_heatmap.parquet")
    issue_trend = load_summary("issue_trend.parquet")

    top_companies = load_summary("top_companies.parquet")
    company_share = load_summary("company_share.parquet")
    company_timely_response = load_summary("company_timely_response.parquet")
    company_resolution = load_summary("company_resolution.parquet")
    credit_bureau_summary = load_summary("credit_bureau_summary.parquet")

    resolution_summary = load_summary("resolution_summary.parquet")
    resolution_by_year = load_summary("resolution_by_year.parquet")
    delay_distribution = load_summary("delay_distribution.parquet")

    narrative_product_length = load_summary(
        "narrative_product_length.parquet"
    )

    narrative_product_availability = load_summary(
        "narrative_product_availability.parquet"
    )
    narrative_availability = load_summary("narrative_availability.parquet")
    narrative_word_count = load_summary("narrative_word_count.parquet")
    narrative_length_trend = load_summary("narrative_length_trend.parquet")
    narrative_complexity = load_summary("narrative_complexity.parquet")
    # top_narrative_words = load_summary("top_narrative_words.parquet")  # not bundled in the demo dashboard snapshot

    top_states = load_summary("top_states.parquet")
    state_map = load_summary("state_map.parquet")

    # --- Business analytics summaries ---
    company_risk_score = load_summary("company_risk_score.parquet")

    driver_analysis = load_summary("driver_analysis.parquet")
    top_complaint_drivers = load_summary("top_complaint_drivers.parquet")
    product_driver_summary = load_summary("product_driver_summary.parquet")

    product_growth = load_summary("product_growth.parquet")
    issue_growth = load_summary("issue_growth.parquet")
    monthly_complaint_trend = load_summary("monthly_complaint_trend.parquet")

    complaint_forecast = load_summary("complaint_forecast.parquet")
    forecast_summary = load_summary("forecast_summary.parquet")

    # --- Recommendation engine summaries ---
    recommendations = load_summary(
        "recommendations.parquet"
    )

    executive_action_plan = load_summary(
        "executive_action_plan.parquet"
    )

    tabs = st.tabs([
        "Overview",
        "Complaint Analyzer",
        "Classification",
        "Similar Complaints",
        "Issue Discovery",
        "Trends & Emerging Issues",
        "Model Performance",
        "Business Analytics",
    ])

    with tabs[0]:
        render_kpis(kpis)
        st.divider()
        render_yearly_trend(yearly_trend)
        st.divider()
        render_executive_summary(kpis)

    with tabs[1]:
        render_nlp_prediction()

    with tabs[2]:
        render_classification()

    with tabs[3]:
        render_similar_complaints()

    with tabs[4]:
        render_issue_discovery()

    with tabs[5]:
        st.header("Trends & Emerging Issues")
        render_growth_analysis(
            product_growth,
            issue_growth,
            monthly_complaint_trend,
        )
        st.divider()
        render_forecasting(
            complaint_forecast,
            forecast_summary,
        )

    with tabs[6]:
        render_model_performance()

    with tabs[7]:
        st.header("Business Analytics")
        st.caption(
            "Supporting business-intelligence layer: Risk, growth drivers, "
            "recommendations, and the full breakdown by Product / Issue / "
            "Company / Resolution / Consumer / Narrative / Channels / "
            "Geography. Serves the ML/NLP core above rather than being the "
            "project's main identity."
        )

        business_tabs = st.tabs([
            "Executive Risk Dashboard",
            "Risk Score",
            "Driver Analysis",
            "Recommendations",
            "Product",
            "Issue",
            "Company",
            "Resolution",
            "Consumer",
            "Narrative",
            "Channels",
            "Geography",
        ])

        with business_tabs[0]:
            render_executive_risk_dashboard(
                company_risk_score,
                top_complaint_drivers,
                product_growth,
                issue_growth,
                forecast_summary,
            )

        with business_tabs[1]:
            render_company_risk_score(company_risk_score)

        with business_tabs[2]:
            render_driver_analysis(
                driver_analysis,
                top_complaint_drivers,
                product_driver_summary,
            )

        with business_tabs[3]:
            render_recommendations(
                recommendations,
                executive_action_plan,
            )

        with business_tabs[4]:
            render_product_intelligence(top_products, product_share, year_product_trend)

        with business_tabs[5]:
            render_issue_intelligence(
                top_issues,
                top_sub_issues,
                issue_product_heatmap,
                issue_trend,
            )

        with business_tabs[6]:
            render_company_intelligence(
                top_companies,
                company_share,
                company_timely_response,
                company_resolution,
                credit_bureau_summary,
            )

        with business_tabs[7]:
            render_resolution_intelligence(
                resolution_summary,
                resolution_by_year,
                delay_distribution,
                timely_response,
            )

        with business_tabs[8]:
            render_consumer_segments(consumer_group)

        with business_tabs[9]:
            render_narrative_intelligence(
                narrative_availability,
                narrative_word_count,
                narrative_product_length,
                narrative_product_availability,
                narrative_length_trend,
                narrative_complexity,
                # top_narrative_words,
            )

        with business_tabs[10]:
            render_channel_overview(submitted_via)

        with business_tabs[11]:
            render_geographic_intelligence(top_states, state_map)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Known gaps (tracked, not yet fixed):
# 1. Geography tab uses raw complaint counts, not per-capita — add a
#    state-population reference table and a normalized metric.
# 2. Top Narrative Words is fully implemented (dashboard_data.py builds
#    top_narrative_words.parquet) but the parquet file isn't bundled in
#    this repository's demo dashboard snapshot, so the chart in
#    render_narrative_intelligence() stays commented out until the
#    full pipeline is run against the raw dataset to (re)generate it.
# ----------------------------------------------------------------------