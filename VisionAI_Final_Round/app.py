import json
import os
import re
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from google import genai

# Optional Folium (OpenStreetMap) support used if installed
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_ENABLED = True
except Exception:
    FOLIUM_ENABLED = False


# 1. PAGE CONFIGURATION & STYLING

st.set_page_config(
    page_title="Outlet Intelligence | Data Storm 7.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    #MainMenu, footer {visibility: hidden;}
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: #0f172a;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.08), transparent 30%),
            radial-gradient(circle at top right, rgba(15, 23, 42, 0.05), transparent 26%),
            linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stSidebar"] {
        background: rgba(248, 250, 252, 0.96);
        border-right: 1px solid rgba(148, 163, 184, 0.25);
    }
    .hero-shell, .section-shell, .surface-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(226, 232, 240, 1);
        border-radius: 24px;
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.06);
    }
    .hero-shell {
        padding: 1.4rem 1.5rem;
    }
    .section-shell, .surface-card {
        padding: 1rem 1.1rem;
    }
    .soft-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin: 0 0.35rem 0.35rem 0;
        border-radius: 999px;
        background: #0f172a;
        color: #ffffff;
        font-size: 0.82rem;
        line-height: 1;
    }
    .soft-chip.alt {
        background: #e2e8f0;
        color: #0f172a;
    }
    .stMetric {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        padding: 1rem;
        border: 1px solid rgba(226, 232, 240, 1);
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }
    .kpi-card {
        background: rgba(255,255,255,0.98);
        border: 1px solid rgba(226,232,240,1);
        border-radius: 14px;
        padding: 0.6rem 0.85rem;
        box-shadow: 0 10px 22px rgba(15,23,42,0.04);
        display: inline-block;
        min-width: 180px;
    }
    .kpi-card .kpi-label {
        color: #475569;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }
    .kpi-card .kpi-value {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.0;
    }
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(226, 232, 240, 1);
        border-radius: 22px;
        padding: 0.25rem 0.25rem 0.25rem 0.1rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }
    </style>
""",
    unsafe_allow_html=True,
)

ROOT_DIR = Path(__file__).resolve().parent
PROVINCE_LABELS = {
    "DIST_W": "Western",
    "DIST_C": "Central",
    "DIST_NW": "North-Western",
    "DIST_S": "Southern",
}

# 2. DATA LOADING & PROCESSING

@st.cache_data(show_spinner="Loading optimization data...")
def load_and_merge_data():
    try:
        budget_df = pd.read_csv(ROOT_DIR / "VisionAI_budget_allocations.csv")
        meta_df = pd.read_csv(ROOT_DIR / "Web_App_Data.csv")

        merged_df = pd.merge(budget_df, meta_df, on='Outlet_ID', how='inner')

        numeric_columns = [
            'Longitude',
            'Latitude',
            'Trade_Spend_Allocation_LKR',
            'Maximum_Monthly_Liters',
            'Avg_Monthly_Volume',
            'Competitor_Density',
            'Decay_POI_Score',
            'Adjusted_Cost_Per_Liter',
            'Latent_Volume',
            'Total_Investment_Cost',
            'Cooler_Count',
        ]
        for column in numeric_columns:
            if column in merged_df.columns:
                merged_df[column] = pd.to_numeric(merged_df[column], errors='coerce')

        merged_df['Province'] = merged_df['Distributor_ID'].astype(str).apply(
            lambda distributor_id: next(
                (province for prefix, province in PROVINCE_LABELS.items() if distributor_id.startswith(prefix)),
                'Unknown',
            )
        )

        merged_df['Spend_Per_Liter'] = merged_df['Trade_Spend_Allocation_LKR'] / merged_df['Maximum_Monthly_Liters'].replace(0, pd.NA)
        merged_df['Spend_Intensity_Rank'] = merged_df['Trade_Spend_Allocation_LKR'].rank(pct=True)
        merged_df['Potential_Rank'] = merged_df['Latent_Volume'].rank(pct=True)
        merged_df['Competition_Rank'] = merged_df['Competitor_Density'].rank(pct=True)
        merged_df['Growth_Gap_Liters'] = (merged_df['Latent_Volume'].fillna(0) - merged_df['Avg_Monthly_Volume'].fillna(0)).clip(lower=0)
        merged_df['Growth_Gap'] = merged_df['Growth_Gap_Liters']
        merged_df['Historical_Baseline_Liters'] = merged_df['Avg_Monthly_Volume']
        merged_df['Predicted_Maximum_Monthly_Liters'] = merged_df['Latent_Volume']
        merged_df['POI_Demand_Score'] = merged_df['Decay_POI_Score']
        merged_df['Competitor_Intensity'] = merged_df['Competitor_Density']
        merged_df['Recommended_Trade_Spend_LKR'] = merged_df['Trade_Spend_Allocation_LKR']
        merged_df['Budget_Allocated'] = merged_df['Trade_Spend_Allocation_LKR'].fillna(0) > 0

        priority_score = (
            merged_df['Potential_Rank'].fillna(0)
            + (1 - merged_df['Competition_Rank'].fillna(0))
            + (1 - merged_df['Spend_Intensity_Rank'].fillna(0))
        )
        merged_df['Priority_Score'] = priority_score
        high_cut = priority_score.quantile(0.67) if len(merged_df) else 0
        low_cut = priority_score.quantile(0.33) if len(merged_df) else 0

        def classify_priority(score):
            if pd.isna(score):
                return 'Medium'
            if score >= high_cut:
                return 'High'
            if score <= low_cut:
                return 'Low'
            return 'Medium'

        merged_df['Priority_Level'] = merged_df['Priority_Score'].apply(classify_priority)

        merged_df = merged_df.dropna(subset=['Longitude', 'Latitude'])
        return merged_df
    except FileNotFoundError:
        st.error("⚠️ CSV files not found. Ensure 'VisionAI_budget_allocations.csv' and 'Web_App_Data.csv' are in the root directory.")
        return pd.DataFrame()


def get_llm_api_key():
    try:
        secrets = st.secrets
        if 'GEMINI_API_KEY' in secrets:
            return secrets['GEMINI_API_KEY']
        if 'GOOGLE_API_KEY' in secrets:
            return secrets['GOOGLE_API_KEY']
    except Exception:
        pass

    return os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or ''


def parse_llm_payload(raw_text):
    if not raw_text:
        return {}

    cleaned_text = re.sub(r"```(?:json)?|```", "", raw_text).strip()
    match = re.search(r"\{.*\}", cleaned_text, flags=re.S)
    candidate = match.group(0) if match else cleaned_text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {
            'headline': 'Business briefing',
            'summary': cleaned_text,
            'drivers': [],
            'risks': [],
            'decision': '',
            'next_step': '',
        }


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def build_xai_prompt(outlet_data):
    return f"""
You are a retail strategy analyst speaking to non-technical business users.
Explain why this outlet received its allocation in simple, confident business language.

Outlet facts:
- Outlet ID: {outlet_data['Outlet_ID']}
- Province: {outlet_data.get('Province', 'Unknown')}
- Distributor: {outlet_data.get('Distributor_ID', 'N/A')}
- Outlet type: {outlet_data.get('Outlet_Type', 'N/A')}
- Outlet size: {outlet_data.get('Outlet_Size', 'N/A')}
- Cooler count: {outlet_data.get('Cooler_Count', 'N/A')}
- Trade spend allocation (LKR): {outlet_data.get('Trade_Spend_Allocation_LKR', 'N/A')}
- Latent volume: {outlet_data.get('Latent_Volume', 'N/A')}
- Maximum monthly liters: {outlet_data.get('Maximum_Monthly_Liters', 'N/A')}
- Average monthly volume: {outlet_data.get('Avg_Monthly_Volume', 'N/A')}
- Competitor density: {outlet_data.get('Competitor_Density', 'N/A')}
- Decay POI score: {outlet_data.get('Decay_POI_Score', 'N/A')}
- Adjusted cost per liter: {outlet_data.get('Adjusted_Cost_Per_Liter', 'N/A')}

Return valid JSON only with this exact structure:
{{
  "headline": "short punchy title",
  "summary": "exactly two concise sentences",
  "drivers": [
    {{"factor": "name", "direction": "positive or negative", "strength": 0, "reason": "short business explanation"}}
  ],
  "decision": "one sentence explaining the recommended commercial action",
  "risks": ["risk 1", "risk 2"],
  "next_step": "one sentence for the sales team"
}}

Use 3 to 4 drivers. Keep the language practical and readable. Do not mention that you are an AI model.
"""


def generate_xai_explanation(outlet_data, api_key):
    if not api_key:
        return {
            'headline': 'LLM not configured',
            'summary': 'The app is ready for server-side Gemini configuration, but no API key is currently loaded.',
            'drivers': [],
            'risks': [],
            'decision': 'Configure GEMINI_API_KEY on the server to enable business justifications.',
            'next_step': 'Add the key to Streamlit secrets or an environment variable so the chat briefing can run.',
        }

    prompt = build_xai_prompt(outlet_data)

    try:
        with st.spinner('Generating the outlet briefing...'):
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return parse_llm_payload(response.text)
    except Exception as error:
        return {
            'headline': 'Briefing unavailable',
            'summary': f'Gemini could not generate an explanation right now: {error}',
            'drivers': [],
            'risks': [],
            'decision': 'Try again after verifying the server-side Gemini configuration.',
            'next_step': 'Refresh the page and generate the outlet briefing again.',
        }


def apply_filters(df):
    with st.sidebar:
        st.markdown("### Studio Controls")
        st.caption('The Gemini key is loaded server-side. No API key field is shown to users.')

        api_status = 'Connected' if get_llm_api_key() else 'Not configured'
        st.markdown(f'<span class="soft-chip">Gemini: {api_status}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="soft-chip alt">White-theme executive view</span>', unsafe_allow_html=True)

        st.markdown('---')
        st.markdown('### Filters')

        provinces = sorted([province for province in df['Province'].dropna().unique().tolist() if province != 'Unknown'])
        distributors = sorted(df['Distributor_ID'].dropna().astype(str).unique().tolist())
        outlet_types = sorted(df['Outlet_Type'].dropna().astype(str).unique().tolist())
        priorities = ['High', 'Medium', 'Low']

        selected_provinces = st.multiselect('Province', options=provinces, default=provinces)
        selected_distributors = st.multiselect('Distributor', options=distributors, default=distributors)
        selected_types = st.multiselect('Outlet type', options=outlet_types, default=outlet_types)
        selected_priorities = st.multiselect('Priority level', options=priorities, default=priorities)
        budget_status = st.selectbox('Budget allocated?', options=['Either', 'Allocated', 'Not allocated'])

        spend_min = float(df['Trade_Spend_Allocation_LKR'].min())
        spend_max = float(df['Trade_Spend_Allocation_LKR'].max())
        spend_range = st.slider('Trade spend range (LKR)', spend_min, spend_max, (spend_min, spend_max))
        outlet_search = st.text_input('Search outlet ID')

    filtered_df = df.copy()

    if selected_provinces:
        filtered_df = filtered_df[filtered_df['Province'].isin(selected_provinces)]
    if selected_distributors:
        filtered_df = filtered_df[filtered_df['Distributor_ID'].isin(selected_distributors)]
    if selected_types:
        filtered_df = filtered_df[filtered_df['Outlet_Type'].isin(selected_types)]
    if selected_priorities:
        filtered_df = filtered_df[filtered_df['Priority_Level'].isin(selected_priorities)]

    filtered_df = filtered_df[
        (filtered_df['Trade_Spend_Allocation_LKR'] >= spend_range[0])
        & (filtered_df['Trade_Spend_Allocation_LKR'] <= spend_range[1])
    ]

    if budget_status == 'Allocated':
        filtered_df = filtered_df[filtered_df['Budget_Allocated']]
    elif budget_status == 'Not allocated':
        filtered_df = filtered_df[~filtered_df['Budget_Allocated']]

    if outlet_search:
        filtered_df = filtered_df[
            filtered_df['Outlet_ID'].astype(str).str.contains(outlet_search, case=False, na=False)
        ]

    return filtered_df


def render_hero(filtered_df):
    total_spend = filtered_df['Trade_Spend_Allocation_LKR'].sum()
    outlet_count = len(filtered_df)
    avg_spend = total_spend / outlet_count if outlet_count else 0
    avg_potential = filtered_df['Latent_Volume'].mean() if 'Latent_Volume' in filtered_df.columns else 0

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="soft-chip">Data Storm 7.0</div>
            <div class="soft-chip alt">Decision support for non-technical business users</div>
            <h1 style="margin: 0.6rem 0 0.35rem 0; font-size: 2.1rem; line-height: 1.08; color: #0f172a;">Outlet Intelligence & Optimization Engine</h1>
            <p style="margin: 0; color: #475569; font-size: 1rem; max-width: 900px;">
                A clean white executive workspace for exploring outlet potential, trade spend allocation, and AI-generated commercial reasoning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write('')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total budget allocated', f'LKR {total_spend:,.0f}')
    col2.metric('Outlets in view', f'{outlet_count:,}')
    col3.metric('Avg spend per outlet', f'LKR {avg_spend:,.0f}')
    col4.metric('Avg latent volume', f'{avg_potential:,.0f}' if pd.notna(avg_potential) else 'N/A')


def render_overview(filtered_df):
    left_col, right_col = st.columns([1.7, 1], gap='large')

    with left_col:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        st.markdown('### Spatial allocation map')

        map_df = filtered_df.copy()
        max_spend = map_df['Trade_Spend_Allocation_LKR'].max() if not map_df.empty else 1
        map_df['Bubble_Radius'] = (map_df['Trade_Spend_Allocation_LKR'] / max_spend * 32000 + 5000).fillna(8000)

        # If Folium + streamlit_folium are available, use OpenStreetMap tiles (no API key required)
        if FOLIUM_ENABLED:
            try:
                center_lat = float(map_df['Latitude'].mean()) if not map_df.empty else 6.9271
                center_lon = float(map_df['Longitude'].mean()) if not map_df.empty else 79.8612
                m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')

                for _, r in map_df.iterrows():
                    lat = r.get('Latitude')
                    lon = r.get('Longitude')
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    spend = safe_float(r.get('Trade_Spend_Allocation_LKR', 0))
                    popup = folium.Popup(f"Outlet: {r.get('Outlet_ID')}<br>Province: {r.get('Province')}<br>Distributor: {r.get('Distributor_ID')}<br>Spend: LKR {spend:,.0f}", max_width=300)
                    radius = max(4, int((spend / max_spend) * 18))
                    folium.CircleMarker([lat, lon], radius=radius, color='#0ea5e9', fill=True, fill_opacity=0.65, popup=popup).add_to(m)

                st_folium(m, width='100%', height=520)
            except Exception:
                st.info('Map rendering with Folium failed — falling back to pydeck.')
                view_state = pdk.ViewState(
                    latitude=float(map_df['Latitude'].mean()) if not map_df.empty else 6.9271,
                    longitude=float(map_df['Longitude'].mean()) if not map_df.empty else 79.8612,
                    zoom=9.8,
                    pitch=30,
                )

                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=map_df,
                    get_position='[Longitude, Latitude]',
                    get_radius='Bubble_Radius',
                    radius_scale=1,
                    radius_min_pixels=6,
                    radius_max_pixels=42,
                    get_fill_color='[14, 165, 233, 175]',
                    get_line_color='[15, 23, 42, 140]',
                    line_width_min_pixels=1,
                    pickable=True,
                    auto_highlight=True,
                )

                st.pydeck_chart(
                    pdk.Deck(
                        map_style='mapbox://styles/mapbox/light-v11',
                        initial_view_state=view_state,
                        layers=[layer],
                        tooltip={
                            'text': 'Outlet: {Outlet_ID}\nProvince: {Province}\nDistributor: {Distributor_ID}\nSpend: LKR {Trade_Spend_Allocation_LKR}'
                        },
                    ),
                    use_container_width=True,
                )
        else:
            view_state = pdk.ViewState(
                latitude=float(map_df['Latitude'].mean()) if not map_df.empty else 6.9271,
                longitude=float(map_df['Longitude'].mean()) if not map_df.empty else 79.8612,
                zoom=9.8,
                pitch=30,
            )

            layer = pdk.Layer(
                'ScatterplotLayer',
                data=map_df,
                get_position='[Longitude, Latitude]',
                get_radius='Bubble_Radius',
                radius_scale=1,
                radius_min_pixels=6,
                radius_max_pixels=42,
                get_fill_color='[14, 165, 233, 175]',
                get_line_color='[15, 23, 42, 140]',
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True,
            )

            st.pydeck_chart(
                pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v11',
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={
                        'text': 'Outlet: {Outlet_ID}\nProvince: {Province}\nDistributor: {Distributor_ID}\nSpend: LKR {Trade_Spend_Allocation_LKR}'
                    },
                ),
                use_container_width=True,
            )
        st.caption('Bubble size reflects allocated spend. Hover any point to inspect the outlet level detail.')
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        st.markdown('### Allocation leaders')

        top_outlets = filtered_df.nlargest(min(10, len(filtered_df)), 'Trade_Spend_Allocation_LKR').copy()
        if not top_outlets.empty:
            chart_df = top_outlets[['Outlet_ID', 'Trade_Spend_Allocation_LKR', 'Province']].sort_values(
                'Trade_Spend_Allocation_LKR'
            )
            st.vega_lite_chart(
                chart_df,
                {
                    'mark': {'type': 'bar', 'cornerRadiusEnd': 5},
                    'encoding': {
                        'y': {'field': 'Outlet_ID', 'type': 'nominal', 'sort': '-x', 'title': ''},
                        'x': {'field': 'Trade_Spend_Allocation_LKR', 'type': 'quantitative', 'title': 'Spend (LKR)'},
                        'color': {
                            'field': 'Province',
                            'type': 'nominal',
                            'scale': {'scheme': 'teals'},
                            'legend': {'title': 'Province'},
                        },
                        'tooltip': [
                            {'field': 'Outlet_ID', 'type': 'nominal'},
                            {'field': 'Province', 'type': 'nominal'},
                            {'field': 'Trade_Spend_Allocation_LKR', 'type': 'quantitative', 'format': ',.0f'},
                        ],
                    },
                    'height': 340,
                },
                use_container_width=True,
            )

            top_row = top_outlets.iloc[0]
            st.markdown(f"<span class='soft-chip'>Top outlet: {top_row['Outlet_ID']}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='soft-chip alt'>Province: {top_row['Province']}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='soft-chip alt'>Spend: LKR {top_row['Trade_Spend_Allocation_LKR']:,.0f}</span>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def render_browse(filtered_df):
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('### Outlet browser')
    st.caption('Browse the filtered portfolio, inspect the strategic drivers, and export the current view.')

    display_columns = [
        'Outlet_ID',
        'Province',
        'Distributor_ID',
        'Outlet_Type',
        'Outlet_Size',
        'Trade_Spend_Allocation_LKR',
        'Latent_Volume',
        'Maximum_Monthly_Liters',
        'Competitor_Density',
        'Decay_POI_Score',
        'Adjusted_Cost_Per_Liter',
    ]
    available_columns = [column for column in display_columns if column in filtered_df.columns]

    styled_df = filtered_df[available_columns].copy()
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Trade_Spend_Allocation_LKR': st.column_config.NumberColumn('Trade Spend (LKR)', format='LKR %.0f'),
            'Latent_Volume': st.column_config.NumberColumn('Latent Volume', format='%.0f'),
            'Maximum_Monthly_Liters': st.column_config.NumberColumn('Max Monthly Liters', format='%.0f'),
            'Competitor_Density': st.column_config.NumberColumn('Competitor Density', format='%.2f'),
            'Decay_POI_Score': st.column_config.NumberColumn('Decay POI Score', format='%.2f'),
            'Adjusted_Cost_Per_Liter': st.column_config.NumberColumn('Adjusted Cost/Liter', format='%.2f'),
        },
    )

    export_csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        'Download filtered portfolio',
        data=export_csv,
        file_name='filtered_outlet_portfolio.csv',
        mime='text/csv',
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_xai_chat(filtered_df, api_key):
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('### Explainable AI chat')
    st.caption('This view behaves like a modern chat assistant for sales teams: choose an outlet and ask for a business-friendly briefing.')

    if 'xai_history' not in st.session_state:
        st.session_state.xai_history = []

    selected_outlet_id = st.selectbox('Select outlet for briefing', options=filtered_df['Outlet_ID'].tolist())
    outlet_data = filtered_df[filtered_df['Outlet_ID'] == selected_outlet_id].iloc[0].to_dict()

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric('Allocation', f"LKR {safe_float(outlet_data.get('Trade_Spend_Allocation_LKR', 0)):,.0f}")
    summary_col2.metric('Latent volume', f"{safe_float(outlet_data.get('Latent_Volume', 0)):,.0f}")
    summary_col3.metric('Competition', f"{safe_float(outlet_data.get('Competitor_Density', 0)):,.2f}")

    user_prompt = f"Why did Outlet {selected_outlet_id} get this budget?"
    if st.button('Generate WhatsApp-style briefing', type='primary', use_container_width=True):
        explanation = generate_xai_explanation(outlet_data, api_key)
        st.session_state.xai_history.append(
            {
                'outlet_id': selected_outlet_id,
                'prompt': user_prompt,
                'response': explanation,
            }
        )

    if not st.session_state.xai_history:
        with st.chat_message('assistant'):
            st.markdown('Pick an outlet and generate a briefing. The response will appear as a chat thread with business-language drivers and a visual factor chart.')

    for message in reversed(st.session_state.xai_history[-5:]):
        with st.chat_message('user'):
            st.markdown(message['prompt'])

        with st.chat_message('assistant'):
            payload = message['response']
            st.markdown(f"**{payload.get('headline', 'Outlet briefing')}**")
            st.markdown(payload.get('summary', ''))

            drivers = payload.get('drivers', [])
            if drivers:
                driver_df = pd.DataFrame(drivers)
                driver_df['strength'] = pd.to_numeric(driver_df.get('strength', 0), errors='coerce').fillna(0)
                driver_df['magnitude'] = driver_df['strength'].abs()
                st.vega_lite_chart(
                    driver_df,
                    {
                        'mark': {'type': 'bar', 'cornerRadiusEnd': 5},
                        'encoding': {
                            'y': {'field': 'factor', 'type': 'nominal', 'sort': '-x', 'title': ''},
                            'x': {'field': 'magnitude', 'type': 'quantitative', 'title': 'Driver strength'},
                            'color': {
                                'field': 'direction',
                                'type': 'nominal',
                                'scale': {
                                    'domain': ['positive', 'negative'],
                                    'range': ['#0f766e', '#dc2626'],
                                },
                                'legend': {'title': ''},
                            },
                            'tooltip': [
                                {'field': 'factor', 'type': 'nominal'},
                                {'field': 'direction', 'type': 'nominal'},
                                {'field': 'strength', 'type': 'quantitative'},
                                {'field': 'reason', 'type': 'nominal'},
                            ],
                        },
                        'height': 220,
                    },
                    use_container_width=True,
                )

                for driver in drivers:
                    tone = '🟢' if str(driver.get('direction', '')).lower().startswith('p') else '🔴'
                    st.markdown(
                        f"<span class='soft-chip'>{tone} {driver.get('factor', 'Driver')}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(driver.get('reason', ''))

            if payload.get('decision'):
                st.info(payload['decision'])
            if payload.get('risks'):
                st.warning(' | '.join(payload['risks']))
            if payload.get('next_step'):
                st.success(payload['next_step'])

    st.markdown('</div>', unsafe_allow_html=True)


def render_data_summary(filtered_df):
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('### Data snapshot')
    st.caption('A compact operational view of the currently filtered dataset.')

    summary = filtered_df.describe(include='all').T
    st.dataframe(summary, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def load_xai_explanations():
    xai_path = ROOT_DIR / 'xai_explanations.csv'
    if not xai_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(xai_path)
    except Exception:
        return pd.DataFrame()


def render_spatial_map(df, color_mode='Priority_Level', size_mode='Predicted_Maximum_Monthly_Liters'):
    map_df = df.copy()
    if map_df.empty:
        st.info('No outlets available for the map.')
        return

    color_field = color_mode if color_mode in map_df.columns else 'Priority_Level'
    size_field = size_mode if size_mode in map_df.columns else 'Predicted_Maximum_Monthly_Liters'

    if FOLIUM_ENABLED:
        try:
            center_lat = float(map_df['Latitude'].mean())
            center_lon = float(map_df['Longitude'].mean())
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles='OpenStreetMap')
            palette = {'High': '#dc2626', 'Medium': '#f59e0b', 'Low': '#0ea5e9'}
            for _, row in map_df.iterrows():
                lat = row.get('Latitude')
                lon = row.get('Longitude')
                if pd.isna(lat) or pd.isna(lon):
                    continue
                color_value = row.get(color_field)
                if color_field in ['Growth_Gap', 'Growth_Gap_Liters']:
                    color = '#0f766e' if safe_float(color_value) > 0 else '#94a3b8'
                else:
                    color = palette.get(str(color_value), '#64748b')
                size_value = safe_float(row.get(size_field, 0))
                radius = max(4, min(18, int(size_value / 300)))
                popup_text = (
                    f"Outlet: {row.get('Outlet_ID')}<br>"
                    f"Province: {row.get('Province')}<br>"
                    f"Distributor: {row.get('Distributor_ID')}<br>"
                    f"Priority: {row.get('Priority_Level')}<br>"
                    f"Potential: {safe_float(row.get('Predicted_Maximum_Monthly_Liters', 0)):,.0f}"
                )
                folium.CircleMarker(
                    [lat, lon],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_opacity=0.68,
                    popup=folium.Popup(popup_text, max_width=320),
                ).add_to(m)
            st_folium(m, width='100%', height=520)
            return
        except Exception:
            st.info('OpenStreetMap rendering failed, using a fallback map.')

    view_state = pdk.ViewState(
        latitude=float(map_df['Latitude'].mean()),
        longitude=float(map_df['Longitude'].mean()),
        zoom=9,
        pitch=30,
    )
    bubble_field = size_field if size_field in map_df.columns else 'Predicted_Maximum_Monthly_Liters'
    map_df['Bubble_Radius'] = pd.to_numeric(map_df[bubble_field], errors='coerce').fillna(0)
    map_df['Bubble_Radius'] = (map_df['Bubble_Radius'] / max(map_df['Bubble_Radius'].max(), 1) * 40000 + 5000)

    if color_field in ['Priority_Level']:
        color_map = {'High': [220, 38, 38, 180], 'Medium': [245, 158, 11, 180], 'Low': [14, 165, 233, 180]}
        map_df['Fill_Color'] = map_df[color_field].map(color_map).apply(lambda x: x if isinstance(x, list) else [100, 116, 139, 180])
        map_df['Fill_R'] = map_df['Fill_Color'].apply(lambda x: x[0])
        map_df['Fill_G'] = map_df['Fill_Color'].apply(lambda x: x[1])
        map_df['Fill_B'] = map_df['Fill_Color'].apply(lambda x: x[2])
        map_df['Fill_A'] = map_df['Fill_Color'].apply(lambda x: x[3])
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_df,
            get_position='[Longitude, Latitude]',
            get_radius='Bubble_Radius',
            get_fill_color='[Fill_R, Fill_G, Fill_B, Fill_A]',
            pickable=True,
            auto_highlight=True,
        )
    else:
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_df,
            get_position='[Longitude, Latitude]',
            get_radius='Bubble_Radius',
            get_fill_color='[14, 165, 233, 180]',
            pickable=True,
            auto_highlight=True,
        )

    st.pydeck_chart(
        pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v11',
            initial_view_state=view_state,
            layers=[layer],
        ),
        use_container_width=True,
    )


def render_executive_overview(df):
    data = df.copy()
    total_potential = safe_float(data['Predicted_Maximum_Monthly_Liters'].sum())
    total_baseline = safe_float(data['Historical_Baseline_Liters'].sum())
    total_gap = safe_float(data['Growth_Gap_Liters'].sum())
    west_budget = 5_000_000
    high_priority = int((data['Priority_Level'] == 'High').sum())
    west_budget_df = allocate_budget_western(df, west_budget)
    expected_incremental = safe_float(west_budget_df['Expected_Incremental_Liters'].sum()) if not west_budget_df.empty else 0

    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('## Executive Overview')
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Predicted Potential Liters</div><div class="kpi-value">{total_potential:,.0f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Historical Baseline Liters</div><div class="kpi-value">{total_baseline:,.0f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Growth Gap Liters</div><div class="kpi-value">{total_gap:,.0f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-label">Western Province Budget</div><div class="kpi-value">LKR 5,000,000</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="kpi-card"><div class="kpi-label">Expected Incremental Liters from Budget</div><div class="kpi-value">{expected_incremental:,.0f}</div></div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="kpi-card"><div class="kpi-label">Number of High Priority Outlets</div><div class="kpi-value">{high_priority:,}</div></div>', unsafe_allow_html=True)

    prov_df = data.groupby('Province', dropna=True)['Predicted_Maximum_Monthly_Liters'].sum().reset_index()
    gap_df = data.groupby('Distributor_ID', dropna=True)['Growth_Gap_Liters'].sum().reset_index().nlargest(12, 'Growth_Gap_Liters')
    left, right = st.columns(2)
    with left:
        st.markdown('### Potential by province')
        st.vega_lite_chart(
            prov_df,
            {
                'mark': {'type': 'bar', 'cornerRadiusEnd': 5},
                'encoding': {
                    'x': {'field': 'Province', 'type': 'nominal', 'title': ''},
                    'y': {'field': 'Predicted_Maximum_Monthly_Liters', 'type': 'quantitative', 'title': 'Predicted potential liters'},
                    'color': {'field': 'Province', 'type': 'nominal', 'scale': {'scheme': 'teals'}},
                },
                'height': 320,
            },
            use_container_width=True,
        )
    with right:
        st.markdown('### Growth gap by distributor')
        st.vega_lite_chart(
            gap_df,
            {
                'mark': {'type': 'bar', 'cornerRadiusEnd': 5},
                'encoding': {
                    'x': {'field': 'Growth_Gap_Liters', 'type': 'quantitative', 'title': 'Growth gap liters'},
                    'y': {'field': 'Distributor_ID', 'type': 'nominal', 'sort': '-x', 'title': ''},
                    'color': {'field': 'Distributor_ID', 'type': 'nominal', 'legend': None},
                },
                'height': 320,
            },
            use_container_width=True,
        )

    st.markdown('### Outlet locations colored by priority')
    render_spatial_map(data, color_mode='Priority_Level', size_mode='Predicted_Maximum_Monthly_Liters')
    st.markdown('</div>', unsafe_allow_html=True)


def render_outlet_explorer(df):
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('## Outlet Explorer')
    st.caption('Use filters to browse predictions and trade spend across the outlet network.')

    display_columns = [
        'Outlet_ID', 'Province', 'Distributor_ID', 'Outlet_Type',
        'Historical_Baseline_Liters', 'Predicted_Maximum_Monthly_Liters',
        'Growth_Gap_Liters', 'POI_Demand_Score', 'Competitor_Intensity',
        'Recommended_Trade_Spend_LKR', 'Priority_Level', 'Budget_Allocated',
    ]
    available_columns = [column for column in display_columns if column in df.columns]
    table_df = df[available_columns].copy()
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_outlet_drilldown(df, api_key):
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('## Outlet Drilldown')
    xai_df = load_xai_explanations()
    outlet_id = st.selectbox('Select Outlet_ID', options=df['Outlet_ID'].tolist())
    row = df[df['Outlet_ID'] == outlet_id].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Historical Baseline', f"{safe_float(row.get('Historical_Baseline_Liters', 0)):,.0f}")
    c2.metric('Predicted Potential', f"{safe_float(row.get('Predicted_Maximum_Monthly_Liters', 0)):,.0f}")
    c3.metric('Growth Gap', f"{safe_float(row.get('Growth_Gap_Liters', 0)):,.0f}")
    c4.metric('Recommended Spend', f"LKR {safe_float(row.get('Recommended_Trade_Spend_LKR', 0)):,.0f}")
    c5.metric('Priority Level', row.get('Priority_Level', 'N/A'))

    st.markdown('### Why this outlet got this score')
    positive = []
    negative = []
    if safe_float(row.get('Predicted_Maximum_Monthly_Liters', 0)) > safe_float(row.get('Historical_Baseline_Liters', 0)):
        positive.append('Predicted potential is higher than the historical baseline, indicating growth headroom.')
    if safe_float(row.get('POI_Demand_Score', 0)) > 0:
        positive.append('Local POI influence is present and supports demand capture.')
    if safe_float(row.get('Competitor_Intensity', 0)) < 10:
        positive.append('Competitor pressure is manageable, so spend should convert more efficiently.')
    if safe_float(row.get('Cooler_Count', 0)) <= 0:
        negative.append('No cooler/capacity support is recorded, so execution may need operational follow-up.')
    if safe_float(row.get('Historical_Baseline_Liters', 0)) > safe_float(row.get('Predicted_Maximum_Monthly_Liters', 0)):
        negative.append('Historical sales are already above the latent ceiling, limiting immediate upside.')

    explanation = None
    if not xai_df.empty and 'Outlet_ID' in xai_df.columns:
        matched = xai_df[xai_df['Outlet_ID'] == outlet_id]
        if not matched.empty:
            explanation = matched.iloc[0].to_dict()

    if explanation is None:
        if st.button('Generate Business Explanation', use_container_width=True):
            explanation = generate_xai_explanation(row.to_dict(), api_key)
            st.session_state['last_xai_explanation'] = explanation
    else:
        st.info('Using precomputed explanation from xai_explanations.csv')

    st.markdown('**Top positive drivers**')
    for item in positive or ['No strong positive driver detected from the current features.']:
        st.write(f'- {item}')
    st.markdown('**Top negative drivers**')
    for item in negative or ['No strong negative driver detected from the current features.']:
        st.write(f'- {item}')
    st.markdown(f"**Local POI influence:** {safe_float(row.get('POI_Demand_Score', 0)):,.2f}")
    st.markdown(f"**Competitor pressure:** {safe_float(row.get('Competitor_Intensity', 0)):,.2f}")
    st.markdown(f"**Cooler / capacity constraint:** {safe_float(row.get('Cooler_Count', 0)):,.0f}")
    st.markdown(f"**Historical sales pattern:** {safe_float(row.get('Historical_Baseline_Liters', 0)):,.0f}")
    st.success('Recommended action: align merchandising support, monitor replenishment, and prioritize visit cadence for high-priority outlets.')

    payload = explanation or st.session_state.get('last_xai_explanation')
    if payload:
        st.markdown('### Business explanation')
        st.markdown(f"**{payload.get('headline', 'Outlet briefing')}**")
        st.markdown(payload.get('summary', ''))
        drivers = payload.get('drivers', [])
        if drivers:
            driver_df = pd.DataFrame(drivers)
            driver_df['strength'] = pd.to_numeric(driver_df.get('strength', 0), errors='coerce').fillna(0)
            driver_df['magnitude'] = driver_df['strength'].abs()
            st.vega_lite_chart(
                driver_df,
                {
                    'mark': {'type': 'bar', 'cornerRadiusEnd': 5},
                    'encoding': {
                        'y': {'field': 'factor', 'type': 'nominal', 'sort': '-x', 'title': ''},
                        'x': {'field': 'magnitude', 'type': 'quantitative', 'title': 'Driver strength'},
                        'color': {'field': 'direction', 'type': 'nominal', 'scale': {'domain': ['positive', 'negative'], 'range': ['#0f766e', '#dc2626']}},
                    },
                    'height': 220,
                },
                use_container_width=True,
            )
        if payload.get('decision'):
            st.info(payload['decision'])
        if payload.get('next_step'):
            st.success(payload['next_step'])

    st.markdown('</div>', unsafe_allow_html=True)


def allocate_budget_western(df, budget=5_000_000):
    west = df[df['Province'] == 'Western'].copy()
    if west.empty:
        return pd.DataFrame()

    west['Budget_Score'] = (
        west['Growth_Gap_Liters'].fillna(0)
        + west['Predicted_Maximum_Monthly_Liters'].fillna(0) * 0.1
        + (1 - west['Competition_Rank'].fillna(0)) * 10
    )
    total_score = west['Budget_Score'].sum()
    if total_score <= 0:
        west['Budget_Score'] = 1
        total_score = west['Budget_Score'].sum()

    west['Recommended_Spend_LKR'] = (west['Budget_Score'] / total_score) * budget
    west['Expected_Incremental_Liters'] = (west['Growth_Gap_Liters'].fillna(0) * west['Recommended_Spend_LKR']) / max(budget, 1)
    west['ROI_Score'] = west['Expected_Incremental_Liters'] / west['Recommended_Spend_LKR'].replace(0, pd.NA)
    return west.sort_values('Recommended_Spend_LKR', ascending=False)


def render_budget_optimizer(df):
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown('## Budget Optimizer')
    st.caption('Allocate the Western Province budget across outlets using the current growth-gap and priority logic.')

    budget = st.number_input('Total budget available (LKR)', min_value=1000000, max_value=50000000, value=5000000, step=100000)
    west = allocate_budget_western(df, budget=budget)

    if west.empty:
        st.warning('No Western Province outlets are available in the current dataset.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    total_allocated = float(west['Recommended_Spend_LKR'].sum())
    remaining_budget = max(budget - total_allocated, 0)
    funded_outlets = int((west['Recommended_Spend_LKR'] > 0).sum())
    expected_incremental = float(west['Expected_Incremental_Liters'].sum())
    liters_per_lkr = expected_incremental / total_allocated if total_allocated else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Budget Available</div><div class="kpi-value">LKR {budget:,.0f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Budget Allocated</div><div class="kpi-value">LKR {total_allocated:,.0f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-label">Remaining Budget</div><div class="kpi-value">LKR {remaining_budget:,.0f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-label">Number of Funded Outlets</div><div class="kpi-value">{funded_outlets:,}</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="kpi-card"><div class="kpi-label">Expected Incremental Liters</div><div class="kpi-value">{expected_incremental:,.0f}</div></div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="kpi-card"><div class="kpi-label">Expected Liters per LKR</div><div class="kpi-value">{liters_per_lkr:.4f}</div></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown('### Budget allocation by distributor')
        distributor_df = west.groupby('Distributor_ID', dropna=True)['Recommended_Spend_LKR'].sum().reset_index().sort_values('Recommended_Spend_LKR', ascending=False)
        st.vega_lite_chart(
            distributor_df,
            {
                'mark': {'type': 'bar', 'cornerRadiusEnd': 5},
                'encoding': {
                    'x': {'field': 'Distributor_ID', 'type': 'nominal', 'sort': '-y', 'title': ''},
                    'y': {'field': 'Recommended_Spend_LKR', 'type': 'quantitative', 'title': 'Recommended spend (LKR)'},
                    'color': {'field': 'Distributor_ID', 'type': 'nominal', 'legend': None, 'scale': {'scheme': 'teals'}},
                    'tooltip': [
                        {'field': 'Distributor_ID', 'type': 'nominal'},
                        {'field': 'Recommended_Spend_LKR', 'type': 'quantitative', 'format': ',.0f'},
                    ],
                },
                'height': 300,
            },
            use_container_width=True,
        )

    with right_col:
        st.markdown('### Spend vs expected incremental liters')
        st.vega_lite_chart(
            west,
            {
                'mark': {'type': 'circle', 'size': 120, 'opacity': 0.75},
                'encoding': {
                    'x': {'field': 'Recommended_Spend_LKR', 'type': 'quantitative', 'title': 'Spend (LKR)'},
                    'y': {'field': 'Expected_Incremental_Liters', 'type': 'quantitative', 'title': 'Expected incremental liters'},
                    'color': {'field': 'Priority_Level', 'type': 'nominal', 'scale': {'scheme': 'teals'}},
                    'tooltip': [
                        {'field': 'Outlet_ID', 'type': 'nominal'},
                        {'field': 'Distributor_ID', 'type': 'nominal'},
                        {'field': 'Recommended_Spend_LKR', 'type': 'quantitative', 'format': ',.0f'},
                        {'field': 'Expected_Incremental_Liters', 'type': 'quantitative', 'format': ',.0f'},
                        {'field': 'ROI_Score', 'type': 'quantitative', 'format': '.4f'},
                    ],
                },
                'height': 300,
            },
            use_container_width=True,
        )

    st.markdown('### Top 20 funded outlets')
    funded_columns = [
        'Outlet_ID',
        'Distributor_ID',
        'Historical_Baseline_Liters',
        'Predicted_Maximum_Monthly_Liters',
        'Growth_Gap_Liters',
        'Recommended_Spend_LKR',
        'Expected_Incremental_Liters',
        'ROI_Score',
        'Priority_Level',
    ]
    available_columns = [column for column in funded_columns if column in west.columns]
    st.dataframe(
        west[available_columns].head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            'Historical_Baseline_Liters': st.column_config.NumberColumn('Baseline Liters', format='%.0f'),
            'Predicted_Maximum_Monthly_Liters': st.column_config.NumberColumn('Predicted Potential', format='%.0f'),
            'Growth_Gap_Liters': st.column_config.NumberColumn('Growth Gap', format='%.0f'),
            'Recommended_Spend_LKR': st.column_config.NumberColumn('Recommended Spend (LKR)', format='LKR %.0f'),
            'Expected_Incremental_Liters': st.column_config.NumberColumn('Expected Incremental Liters', format='%.0f'),
            'ROI_Score': st.column_config.NumberColumn('ROI Score', format='%.4f'),
        },
    )

    st.download_button(
        'Download Western budget allocation CSV',
        data=west.to_csv(index=False).encode('utf-8'),
        file_name='western_budget_allocation.csv',
        mime='text/csv',
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 3. MAIN APP LAYOUT
# ==========================================
def main():
    df = load_and_merge_data()

    if df.empty:
        st.warning("Awaiting data connection to initialize dashboard.")
        st.stop()

    st.sidebar.markdown('### VisionAI Outlet Intelligence Engine')
    page = st.sidebar.radio(
        'Pages',
        [
            'Executive Overview',
            'Outlet Explorer',
            'Budget Optimizer',
            'Outlet Drilldown',
        ],
        index=0,
    )

    filtered_df = apply_filters(df)
    if filtered_df.empty:
        st.warning('No outlets match the current filters. Widen the selection to continue.')
        st.stop()

    render_hero(filtered_df)

    if page == 'Executive Overview':
        render_executive_overview(filtered_df)
    elif page == 'Outlet Explorer':
        render_outlet_explorer(filtered_df)
    elif page == 'Budget Optimizer':
        render_budget_optimizer(df)
    elif page == 'Outlet Drilldown':
        render_outlet_drilldown(filtered_df, get_llm_api_key())

if __name__ == "__main__":
    main()