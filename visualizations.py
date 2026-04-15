"""
visualizations.py — Plotly chart builders for analysis results.
"""
import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


def create_visualizations(analysis_data, analysis_type="complete"):
    if not analysis_data:
        return {}

    key_map = {"complete": "complete_analysis", "managed": "managed_analysis", "non_managed": "non_managed_analysis"}
    current_analysis = analysis_data.get(key_map.get(analysis_type, "complete_analysis"), {})

    if not current_analysis or current_analysis.get("metadata", {}).get("total_tickets", 0) == 0:
        return {"error": "No data available for this analysis type"}

    charts = {}

    ha = (current_analysis.get("hostname_analysis") or {}).get("chart_data", {})
    if ha.get("labels") and ha.get("values"):
        fig = go.Figure(data=[go.Bar(x=ha["labels"], y=ha["values"], marker_color='#dc3545')])
        fig.update_layout(title="Top 20 Hostnames by Ticket Count", xaxis_title="Hostname", yaxis_title="Tickets", height=400, plot_bgcolor='white', paper_bgcolor='white')
        charts["hostname"] = json.dumps(fig, cls=PlotlyJSONEncoder)

    sa = (current_analysis.get("severity_analysis") or {}).get("chart_data", {})
    if sa.get("labels") and sa.get("values"):
        colors = ['#dc3545', '#fd7e14', '#ffc107', '#6c757d']
        fig = go.Figure(data=[go.Pie(labels=sa["labels"], values=sa["values"], marker=dict(colors=colors))])
        fig.update_layout(title="Tickets by Severity", height=400, paper_bgcolor='white')
        charts["severity"] = json.dumps(fig, cls=PlotlyJSONEncoder)

    aa = (current_analysis.get("automata_analysis") or {}).get("chart_data", {})
    if aa.get("labels") and aa.get("values"):
        fig = go.Figure(data=[go.Bar(x=aa["labels"], y=aa["values"], text=[f"{p}%" for p in aa.get("percentages", [])], textposition="auto", marker_color='#dc3545')])
        fig.update_layout(title="Tickets by Suggested Automata", xaxis_title="Automata", yaxis_title="Tickets", height=400, plot_bgcolor='white', paper_bgcolor='white')
        charts["automata"] = json.dumps(fig, cls=PlotlyJSONEncoder)

    timing = current_analysis.get("timing_analysis")
    if timing and not timing.get("error", False):
        fig = go.Figure(data=go.Heatmap(z=timing["heatmap_data"], x=timing["hours"], y=timing["days"], colorscale='Reds'))
        fig.update_layout(title="Ticket Opening Heatmap (Day vs Hour)", xaxis_title="Hour of Day", yaxis_title="Day of Week", height=500, plot_bgcolor='white', paper_bgcolor='white')
        charts["timing_heatmap"] = json.dumps(fig, cls=PlotlyJSONEncoder)

    mtt = (current_analysis.get("resolution_metrics") or {}).get("monthly_ticket_trend")
    if mtt and mtt.get("months") and mtt.get("counts"):
        fig = go.Figure(data=[go.Scatter(x=mtt["months"], y=mtt["counts"], mode="lines+markers", line=dict(color='#dc3545', width=2), marker=dict(size=8))])
        fig.update_layout(title="Monthly Ticket Trend", xaxis_title="Month", yaxis_title="Tickets", height=400, plot_bgcolor='white', paper_bgcolor='white')
        charts["monthly_trend"] = json.dumps(fig, cls=PlotlyJSONEncoder)

    return charts


def create_comparison_visualizations(analysis_data):
    if not analysis_data or "comparison_data" not in analysis_data:
        return {}

    comp = analysis_data["comparison_data"]
    managed = comp.get("managed")
    non_managed = comp.get("non_managed")

    if not managed or not non_managed:
        return {"error": "Both managed and non-managed data required for comparison"}

    charts = {}
    categories = ["Managed", "Non-Managed"]

    def bar_chart(title, y_vals, ylabel, colors=None):
        colors = colors or ['#2196F3', '#dc3545']
        fig = go.Figure(data=[
            go.Bar(name=cat, x=[cat], y=[val], marker_color=col, text=[f"{val:,}"], textposition="auto")
            for cat, val, col in zip(categories, y_vals, colors)
        ])
        fig.update_layout(title=title, yaxis_title=ylabel, height=400, plot_bgcolor='white', paper_bgcolor='white', showlegend=False)
        return json.dumps(fig, cls=PlotlyJSONEncoder)

    charts["total_tickets"] = bar_chart("Total Tickets: Managed vs Non-Managed", [managed["total_tickets"], non_managed["total_tickets"]], "Tickets")
    charts["avg_mttr"] = bar_chart("Avg MTTR: Managed vs Non-Managed", [managed["avg_mttr"], non_managed["avg_mttr"]], "Hours")

    return charts
