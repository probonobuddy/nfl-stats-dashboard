import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

# Set page layout to wide
st.set_page_config(layout="wide")

# REMOVED: DATA_URL (gameStats.csv)
QB_DATA_URL = r"C:\NFL Data\nflscrapr_games.csv"

# --- 1. DATA LOADING FUNCTIONS ---
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Ensure numeric columns for matching
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'], errors='coerce')
    if 'season' in df.columns:
        df['season'] = pd.to_numeric(df['season'], errors='coerce')
    return df

@st.cache_data
def load_pbp_data(season):
    pbp_data_url = rf"C:\NFL Data\pbp_{season}.csv"
    try:
        pbp_df = pd.read_csv(pbp_data_url, low_memory=False)
        cols_to_numeric = ['epa', 'success', 'rush', 'pass', 'week']
        for c in cols_to_numeric:
            if c in pbp_df.columns:
                pbp_df[c] = pd.to_numeric(pbp_df[c], errors='coerce')
        return pbp_df
    except FileNotFoundError:
        st.error(f"Play-by-play data file not found for season {season} at {pbp_data_url}")
        return None

@st.cache_data
def get_game_metadata(qb_df, season, week, posteam):
    # Filter for the specific game
    game_row = qb_df[(qb_df['season'] == season) &
                     (qb_df['week'] == week) &
                     ((qb_df['home_team'] == posteam) |
                      (qb_df['away_team'] == posteam))]

    if not game_row.empty:
        row = game_row.iloc[0]
        
        home_score = row.get('home_score', np.nan)
        away_score = row.get('away_score', np.nan)
        
        # Get Betting Info (nflscrapr usually has 'spread_line_x' as Away Spread)
        raw_spread = row.get('spread_line_x', np.nan) 
        total_val = row.get('total_line_x', np.nan)
        
        # Determine Home/Away status and format spread
        if row['home_team'] == posteam:
            # User is Home Team. 
            # If Raw (Away) Spread is 7.0 (Away is +7), Home is -7.
            # So Home Spread = -1 * Raw Spread
            spread_val = -1 * raw_spread if pd.notna(raw_spread) else "N/A"
            
            return (row.get('home_qb_name', 'N/A'), 
                    row.get('away_qb_name', 'N/A'), 
                    home_score, 
                    away_score, 
                    True,
                    spread_val,
                    total_val)
                    
        elif row['away_team'] == posteam:
            # User is Away Team.
            # Raw Spread is Away Spread.
            spread_val = raw_spread if pd.notna(raw_spread) else "N/A"
            
            return (row.get('away_qb_name', 'N/A'), 
                    row.get('home_qb_name', 'N/A'), 
                    home_score, 
                    away_score, 
                    False,
                    spread_val,
                    total_val)
            
    # Default return if no game found
    return "N/A", "N/A", np.nan, np.nan, None, "N/A", "N/A"

# --- 2. ADVANCED METRICS CALCULATION FUNCTIONS ---

def get_recency_weights(df, current_week):
    weeks_ago = (current_week - 1) - df['week']
    decay_factor = 0.10 
    min_weight = 0.50
    w = np.where(weeks_ago < 2, 1.0, 1.0 - ((weeks_ago - 1) * decay_factor))
    w = np.maximum(w, min_weight)
    return w

def calculate_weekly_oe_details(subset, current_week):
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    subset = subset.copy()
    subset['weight_col'] = get_recency_weights(subset, current_week)

    run_mask = ((subset['rush'] == 1) & (subset['play_type_nfl'] != 'PAT2'))
    pass_mask = ((subset['pass'] == 1) & (subset['play_type_nfl'] != 'PAT2'))
    
    def_avgs = {}
    off_avgs = {}
    
    subset['w_epa'] = subset['epa'] * subset['weight_col']

    for mask, name in [(run_mask, 'run'), (pass_mask, 'pass')]:
        masked_data = subset[mask]
        
        def_simple = masked_data.groupby('defteam')[['epa', 'success']].mean().rename(
            columns={'epa': f'def_{name}_epa', 'success': f'def_{name}_sr'}
        )
        off_simple = masked_data.groupby('posteam')[['epa', 'success']].mean().rename(
            columns={'epa': f'off_{name}_epa', 'success': f'off_{name}_sr'}
        )
        
        def calc_weighted_baseline_fast(df, group_col):
            g = df.groupby(group_col)[['w_epa', 'weight_col']].sum()
            prefix = 'off' if group_col == 'posteam' else 'def'
            return (g['w_epa'] / g['weight_col'].replace(0, np.nan)).rename(f'{prefix}_{name}_epa_recency')

        def_wgt = calc_weighted_baseline_fast(masked_data, 'defteam')
        off_wgt = calc_weighted_baseline_fast(masked_data, 'posteam')
        
        def_avgs[name] = def_simple.join(def_wgt, how='outer')
        off_avgs[name] = off_simple.join(off_wgt, how='outer')

    def_baselines = def_avgs['run'].join(def_avgs['pass'], how='outer')
    off_baselines = off_avgs['run'].join(off_avgs['pass'], how='outer')

    # Offense Results
    off_results = []
    for mask, name in [(run_mask, 'run'), (pass_mask, 'pass')]:
        w = subset[mask].groupby(['posteam', 'defteam', 'week']).agg(
            epa=('epa', 'mean'),
            count=('play_type', 'count')
        ).reset_index().rename(columns={'count': f'{name}_count'})
        
        w = w.merge(def_baselines[[f'def_{name}_epa', f'def_{name}_epa_recency']], 
                    left_on='defteam', right_index=True, how='left')
        
        w[f'{name}_epa_oe'] = w['epa'] - w[f'def_{name}_epa']
        w[f'{name}_epa_oe_recency'] = w['epa'] - w[f'def_{name}_epa_recency']
        off_results.append(w[['posteam', 'week', 'defteam', f'{name}_epa_oe', f'{name}_epa_oe_recency', f'{name}_count']])
        
    off_base = off_results[0].merge(off_results[1], on=['posteam','week','defteam'], how='outer').fillna(0)
    off_base['total_count'] = off_base['run_count'] + off_base['pass_count']
    
    def get_total_oe(df, suffix=''):
        num = (df[f'run_epa_oe{suffix}'] * df['run_count']) + (df[f'pass_epa_oe{suffix}'] * df['pass_count'])
        den = df['total_count'].replace(0, np.nan)
        return (num / den).fillna(0)

    off_base['total_epa_oe'] = get_total_oe(off_base, '')
    off_base['total_epa_oe_recency'] = get_total_oe(off_base, '_recency')
    
    # Defense Results
    def_results = []
    for mask, name in [(run_mask, 'run'), (pass_mask, 'pass')]:
        w = subset[mask].groupby(['defteam', 'posteam', 'week']).agg(
            epa=('epa', 'mean'),
            count=('play_type', 'count')
        ).reset_index().rename(columns={'count': f'{name}_count'})
        
        w = w.merge(off_baselines[[f'off_{name}_epa', f'off_{name}_epa_recency']], 
                    left_on='posteam', right_index=True, how='left')
        
        w[f'{name}_epa_oe'] = w[f'off_{name}_epa'] - w['epa']
        w[f'{name}_epa_oe_recency'] = w[f'off_{name}_epa_recency'] - w['epa']
        def_results.append(w[['defteam', 'week', 'posteam', f'{name}_epa_oe', f'{name}_epa_oe_recency', f'{name}_count']])
        
    def_base = def_results[0].merge(def_results[1], on=['defteam','week','posteam'], how='outer').fillna(0)
    def_base['total_count'] = def_base['run_count'] + def_base['pass_count']
    def_base['total_epa_oe'] = get_total_oe(def_base, '')
    def_base['total_epa_oe_recency'] = get_total_oe(def_base, '_recency')

    return off_base, def_base

def aggregate_season_stats(weekly_df, team_col, current_week):
    if weekly_df.empty:
        return pd.DataFrame()

    df = weekly_df.copy()
    df['agg_weight'] = get_recency_weights(df, current_week)
    
    metrics = ['total_epa_oe', 'run_epa_oe', 'pass_epa_oe']
    counts = ['total_count', 'run_count', 'pass_count']
    
    for m, c in zip(metrics, counts):
        df[f'{m}_prod'] = df[m] * df[c]
    
    grouped = df.groupby(team_col)
    sums = grouped[[f'{m}_prod' for m in metrics] + counts].sum()
    
    stats = pd.DataFrame(index=sums.index)
    
    for m, c in zip(metrics, counts):
        stats[f'{m}_avg'] = sums[f'{m}_prod'] / sums[c].replace(0, np.nan)
        stats[f'{m}_avg'] = stats[f'{m}_avg'].fillna(0)

    for m, c in zip(metrics, counts):
        target_col = f"{m}_recency"
        weight_col = f"{c}_w_agg"
        prod_col = f"{m}_rec_prod"
        
        df[weight_col] = df[c] * df['agg_weight']
        df[prod_col] = df[target_col] * df[weight_col]
        
    sums_rec = grouped[[f'{m}_rec_prod' for m in metrics] + [f'{c}_w_agg' for c in counts]].sum()
    
    for m, c in zip(metrics, counts):
        stats[f'{m}_wgt'] = sums_rec[f'{m}_rec_prod'] / sums_rec[f'{c}_w_agg'].replace(0, np.nan)
        stats[f'{m}_wgt'] = stats[f'{m}_wgt'].fillna(0)
        
    return stats

@st.cache_data
def generate_power_rankings_data(pbp_data, excluded_games_list):
    if pbp_data.empty: return pd.DataFrame()
    
    subset = pbp_data.copy()
    if excluded_games_list:
        mask = pd.Series(False, index=subset.index)
        for t, w in excluded_games_list:
            game_mask = (subset['week'] == w) & ((subset['posteam'] == t) | (subset['defteam'] == t))
            mask = mask | game_mask
        subset = subset[~mask]

    if subset.empty: return pd.DataFrame()

    max_week = subset['week'].max()
    current_calc_week = max_week + 1

    off_weekly, def_weekly = calculate_weekly_oe_details(subset, current_week=current_calc_week)
    off_entering = aggregate_season_stats(off_weekly, 'posteam', current_calc_week)
    def_entering = aggregate_season_stats(def_weekly, 'defteam', current_calc_week)

    off_entering.columns = ['off_' + c for c in off_entering.columns]
    def_entering.columns = ['def_' + c for c in def_entering.columns]

    combined = off_entering.join(def_entering, how='outer').reset_index().rename(columns={'index':'team', 'posteam':'team'})
    
    offense_defense_split = 0.58
    combined['ovr_total_epa_oe_wgt'] = (offense_defense_split * combined['off_total_epa_oe_wgt']) + \
                                       ((1 - offense_defense_split) * combined['def_total_epa_oe_wgt'])
    combined['week'] = max_week
    return combined

@st.cache_data
def generate_team_dashboard_data(pbp_data, excluded_games_list):
    if pbp_data.empty: return pd.DataFrame()
    
    # 1. Apply exclusions
    subset = pbp_data.copy()
    if excluded_games_list:
        mask = pd.Series(False, index=subset.index)
        for t, w in excluded_games_list:
            game_mask = (subset['week'] == w) & ((subset['posteam'] == t) | (subset['defteam'] == t))
            mask = mask | game_mask
        subset = subset[~mask]
            
    if subset.empty: return pd.DataFrame()
    subset = subset.sort_values('week')
    
    # 2. Calculate FULL SEASON Actuals (Bars)
    max_week = subset['week'].max()
    full_season_off, full_season_def = calculate_weekly_oe_details(subset, current_week=max_week + 1)
    
    # 3. Iterate for Trend Stats (Line)
    dashboard_rows = []
    all_weeks = sorted(subset['week'].unique())
    
    for w in all_weeks:
        # A. TREND LINE
        off_slice = full_season_off[full_season_off['week'] <= w].copy()
        if not off_slice.empty:
            cols_to_fix = ['total_epa_oe', 'pass_epa_oe', 'run_epa_oe']
            for c in cols_to_fix:
                if c in off_slice.columns:
                    off_slice[f'{c}_recency'] = off_slice[c]
            
            off_trend_agg = aggregate_season_stats(off_slice, 'posteam', current_week=w+1)
            off_trend_agg.columns = ['off_' + c for c in off_trend_agg.columns]
        else:
            off_trend_agg = pd.DataFrame()

        def_slice = full_season_def[full_season_def['week'] <= w].copy()
        if not def_slice.empty:
            cols_to_fix = ['total_epa_oe', 'pass_epa_oe', 'run_epa_oe']
            for c in cols_to_fix:
                if c in def_slice.columns:
                    def_slice[f'{c}_recency'] = def_slice[c]

            def_trend_agg = aggregate_season_stats(def_slice, 'defteam', current_week=w+1)
            def_trend_agg.columns = ['def_' + c for c in def_trend_agg.columns]
            
        # B. ACTUALS (Bars)
        off_actuals_row = full_season_off[full_season_off['week'] == w].copy()
        def_actuals_row = full_season_def[full_season_def['week'] == w].copy()

        if off_actuals_row.empty: continue

        week_df = off_actuals_row.rename(columns={
            'posteam': 'team', 'defteam': 'opponent',
            'total_epa_oe': 'off_total_epa_oe_actual', 'pass_epa_oe': 'off_pass_epa_oe_actual', 'run_epa_oe': 'off_run_epa_oe_actual'
        })[['team', 'week', 'opponent', 'off_total_epa_oe_actual', 'off_pass_epa_oe_actual', 'off_run_epa_oe_actual']]
        
        if not off_trend_agg.empty:
            week_df = week_df.merge(off_trend_agg, left_on='team', right_index=True, how='left')
        else:
            for c in ['off_total_epa_oe_wgt', 'off_pass_epa_oe_wgt', 'off_run_epa_oe_wgt']:
                week_df[c] = 0.0
                
        def_week_df = def_actuals_row.rename(columns={
            'defteam': 'team', 'posteam': 'opponent',
            'total_epa_oe': 'def_total_epa_oe_actual', 'pass_epa_oe': 'def_pass_epa_oe_actual', 'run_epa_oe': 'def_run_epa_oe_actual'
        })[['team', 'week', 'opponent', 'def_total_epa_oe_actual', 'def_pass_epa_oe_actual', 'def_run_epa_oe_actual']]
        
        if not def_trend_agg.empty:
            def_week_df = def_week_df.merge(def_trend_agg, left_on='team', right_index=True, how='left')
        else:
             for c in ['def_total_epa_oe_wgt', 'def_pass_epa_oe_wgt', 'def_run_epa_oe_wgt']:
                def_week_df[c] = 0.0
        
        combined_week = pd.merge(week_df, def_week_df, on=['team', 'week', 'opponent'], how='outer')
        dashboard_rows.append(combined_week)
        
    if not dashboard_rows:
        return pd.DataFrame()
        
    full_df = pd.concat(dashboard_rows, ignore_index=True)
    full_df['is_home'] = 1 
    
    return full_df

# --- 3. HEATMAP CALCULATION FUNCTIONS ---
@st.cache_data
def pass_offense_zover_average_combined(plays, target_team=None, target_receiver="Team"):
    passes = plays[(plays.pass_attempt==1)&(plays.sack==0)].copy()
    passes['pass_defensed'] = passes.pass_defense_1_player_id.notna() * 1

    depth_names = ['blos', 'short', 'intermediate', 'deep']
    loc_names = ['left', 'middle', 'right']
    
    passes['depth'] = pd.cut(passes['air_yards'], bins=[-np.inf,0,10,20,np.inf], labels=depth_names)
    passes['pass_location'] = pd.Categorical(passes['pass_location'], categories=loc_names, ordered=False)
    passes['depth'] = pd.Categorical(passes['depth'], categories=depth_names, ordered=True)
    passes['attempts'] = passes['complete_pass'] + passes['incomplete_pass']

    all_depth_locations = pd.MultiIndex.from_product([depth_names, loc_names], names=['depth', 'pass_location'])

    team_grouper = passes.groupby(['posteam', 'depth', 'pass_location'], observed=False)['attempts'].sum().reset_index()
    team_totals = passes.groupby('posteam', observed=False)['attempts'].sum().reset_index(name='attempts_total')
    pass_dep_loc_all = team_grouper.merge(team_totals, on='posteam', how='left')
    pass_dep_loc_all['perc_attempts'] = pass_dep_loc_all['attempts'] / pass_dep_loc_all['attempts_total']

    zone_stats = pass_dep_loc_all.groupby(['depth', 'pass_location'], observed=False)['perc_attempts'].mean().reset_index()
    zone_stats.columns = ['depth', 'pass_location', 'zone_mean']

    is_specific_player = (target_receiver != "Team") and (target_receiver != target_team) and (target_team is not None)

    if is_specific_player:
        receiver_plays = passes[(passes['posteam'] == target_team) & (passes['receiver'] == target_receiver)]
        if receiver_plays.empty: return pd.DataFrame()
        rec_grouper = receiver_plays.groupby(['depth', 'pass_location'], observed=False)['attempts'].sum()
        rec_grouper = rec_grouper.reindex(all_depth_locations, fill_value=0).reset_index()
        rec_total = receiver_plays['attempts'].sum()
        pass_dep_loc = rec_grouper
        pass_dep_loc['attempts_total'] = rec_total
        pass_dep_loc['posteam'] = target_team 
        pass_dep_loc['perc_attempts'] = pass_dep_loc['attempts'] / rec_total
    else:
        pass_dep_loc = pass_dep_loc_all

    pass_dep_loc = pass_dep_loc.merge(zone_stats, on=['depth', 'pass_location'], how='left')
    pass_dep_loc['zone_mean'] = pass_dep_loc['zone_mean'].replace(0, np.nan)
    pass_dep_loc['pct_of_avg'] = (pass_dep_loc['perc_attempts'] / pass_dep_loc['zone_mean']) * 100
    pass_dep_loc['pct_of_avg'] = pass_dep_loc['pct_of_avg'].fillna(0) 

    offense_grid = pd.pivot_table(pass_dep_loc, values='pct_of_avg', index=['posteam', 'depth'], columns='pass_location', aggfunc='mean', fill_value=0)
    return offense_grid

@st.cache_data
def pass_defense_zover_average_combined(plays):
    passes = plays[(plays.pass_attempt==1)&(plays.sack==0)].copy()
    passes['pass_defensed'] = passes.pass_defense_1_player_id.notna() * 1

    depth_names = ['blos', 'short', 'intermediate', 'deep']
    loc_names = ['left', 'middle', 'right']

    passes['depth'] = pd.cut(passes['air_yards'], bins=[-np.inf,0,10,20,np.inf], labels=depth_names)
    passes['pass_location'] = pd.Categorical(passes['pass_location'], categories=loc_names, ordered=False)
    passes['depth'] = pd.Categorical(passes['depth'], categories=depth_names, ordered=True)
    passes['attempts'] = passes['complete_pass'] + passes['incomplete_pass']

    pass_dep_loc = passes.groupby(['defteam', 'depth', 'pass_location'], observed=False)['success'].mean().reset_index()
    zone_stats = pass_dep_loc.groupby(['depth', 'pass_location'], observed=False)['success'].mean().reset_index()
    zone_stats.columns = ['depth', 'pass_location', 'zone_mean']

    pass_dep_loc = pass_dep_loc.merge(zone_stats, on=['depth', 'pass_location'], how='left')
    pass_dep_loc['zone_mean'] = pass_dep_loc['zone_mean'].replace(0, np.nan)
    pass_dep_loc['pct_of_avg'] = (pass_dep_loc['success'] / pass_dep_loc['zone_mean']) * 100
    pass_dep_loc['pct_of_avg'] = pass_dep_loc['pct_of_avg'].fillna(0)

    defense_grid = pd.pivot_table(pass_dep_loc, values='pct_of_avg', index=['defteam', 'depth'], columns='pass_location', aggfunc='mean', fill_value=0)
    return defense_grid

# --- PLOTLY HELPERS ---
def create_plotly_dashboard(subset, latest_stats, team_name):
    def get_4wk_delta(series):
        if len(series) < 5: return 0.0
        return series.iloc[-1] - series.iloc[-5]

    metrics_map = [
        ('off', 'total', f"{team_name} Total Offense"),
        ('off', 'pass', "Pass Offense"),
        ('off', 'run', "Run Offense"),
        ('def', 'total', f"{team_name} Total Defense"),
        ('def', 'pass', "Pass Defense"),
        ('def', 'run', "Run Defense")
    ]
    
    titles = []
    
    for side, metric, base_title in metrics_map:
        wgt_col = f'{side}_{metric}_epa_oe_wgt'
        delta = get_4wk_delta(subset[wgt_col])
        trend_str = f"{delta:+.2f} vs 4w ago"
        full_title = f"<b>{base_title} ({trend_str})</b>"
        titles.append(full_title)

    # 2. CREATE SUBPLOTS
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"rowspan": 2}, {}],
            [None, {}],
            [{"rowspan": 2}, {}],
            [None, {}]
        ],
        subplot_titles=titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.05
    )

    # 3. ADD TRACES
    grid_pos = [
        ('off', 'total', 1, 1), ('off', 'pass', 1, 2), ('off', 'run', 2, 2),
        ('def', 'total', 3, 1), ('def', 'pass', 3, 2), ('def', 'run', 4, 2)
    ]

    present_weeks = sorted(subset['week'].unique())
    tick_vals = [w for w in present_weeks if w > 0] 
    
    for (side, metric, row, col) in grid_pos:
        actual_col = f'{side}_{metric}_epa_oe_actual'
        wgt_col = f'{side}_{metric}_epa_oe_wgt'
        
        colors = ['#2ECC40' if x > 0 else '#FF4136' for x in subset[actual_col]]
        fig.add_trace(
            go.Bar(
                x=subset['week'], y=subset[actual_col],
                marker_color=colors, opacity=0.7,
                showlegend=False,
                text=subset['opponent'],
                hovertemplate="Week %{x}<br>Opp: %{text}<br>EPA OE: %{y:.3f}<extra></extra>"
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=subset['week'], y=subset[wgt_col],
                mode='lines+markers',
                line=dict(color='white', width=3),
                marker=dict(size=6, color='white'),
                showlegend=False,
                hovertemplate="Trend: %{y:.3f}<extra></extra>"
            ),
            row=row, col=col
        )

        base_wgt_col = f'{side}_{metric}_epa_oe_wgt'
        sorted_vals = latest_stats.sort_values(base_wgt_col, ascending=False).reset_index(drop=True)
        top5 = sorted_vals.iloc[4][base_wgt_col] if len(sorted_vals) > 4 else 0
        bot5 = sorted_vals.iloc[27][base_wgt_col] if len(sorted_vals) > 27 else 0

        fig.add_hline(y=top5, line_dash="dot", line_color="cyan", row=row, col=col, annotation_text="Top 5", annotation_position="top left")
        fig.add_hline(y=bot5, line_dash="dot", line_color="orange", row=row, col=col, annotation_text="Btm 5", annotation_position="bottom left")
        fig.add_hline(y=0, line_color="gray", row=row, col=col)
        
        fig.update_xaxes(tickvals=tick_vals, row=row, col=col)

    # 4. UPDATE LAYOUT & STYLING
    fig.update_layout(
        template="plotly_dark",
        height=1000, 
        title_text=f"<b>{team_name} Performance Dashboard (Opponent Adjusted EPA/Play)</b>",
        title_font_size=20,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    fig.update_xaxes(showline=False, linewidth=1, linecolor='white', mirror=True, ticks='outside')
    fig.update_yaxes(showline=False, linewidth=1, linecolor='white', mirror=True, ticks='outside')
            
    return fig

def create_heatmap_figure(df, title):
    df = df.set_index('Depth').reindex(['deep', 'intermediate', 'short', 'blos'])
    z_values = df[['Left', 'Middle', 'Right']].values
    text_values = df[['Left', 'Middle', 'Right']].applymap(lambda x: f"<b>{x:.0f}%</b>").values 
    x_labels = ['Left', 'Middle', 'Right']
    y_labels = ['Deep', 'Intermediate', 'Short', 'BLOS'] 
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values, x=x_labels, y=y_labels,
        text=text_values, texttemplate="%{text}",
        textfont={"color": "black", "size": 14, "family": "Arial Black"}, 
        colorscale="RdBu_r", zmid=100, showscale=False
    ))
    
    fig.update_layout(
        title=title, template="plotly_dark",
        width=500, height=400,
        yaxis=dict(autorange='reversed', scaleanchor="x", scaleratio=1, title_font=dict(color="white"), tickfont=dict(color="white", size=12)),
        xaxis=dict(side="top", title_font=dict(color="white"), tickfont=dict(color="white", size=12)),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# --- MAIN APP LOGIC ---

# LOAD ONLY QB DF (NFLSCRAPR)
qb_df = load_data(QB_DATA_URL)

seasons = sorted(qb_df['season'].unique())
# Teams derived from all home/away
teams = sorted(pd.concat([qb_df['home_team'], qb_df['away_team']]).unique())

default_season_index = len(seasons) - 1 if seasons else 0
selected_season = st.selectbox('Select Season', seasons, index=default_season_index)
selected_team = st.selectbox('Select Team (posteam)', teams)

# --- STATE MANAGEMENT FOR EXCLUDED GAMES ---
if 'excluded_games' not in st.session_state:
    st.session_state.excluded_games = set()
if 'editor_key_version' not in st.session_state:
    st.session_state.editor_key_version = 0

if 'last_selected_season' not in st.session_state:
    st.session_state.last_selected_season = selected_season

if selected_season != st.session_state.last_selected_season:
    st.session_state.excluded_games = set()
    st.session_state.editor_key_version += 1 
    st.session_state.last_selected_season = selected_season

plays = load_pbp_data(selected_season)
if plays is None: st.stop()

# Determine available weeks from PBP data
team_plays = plays[(plays['posteam'] == selected_team) | (plays['defteam'] == selected_team)]
if not team_plays.empty:
    available_weeks = sorted(team_plays['week'].unique())
else:
    # Fallback to games file logic if PBP missing
    available_weeks = sorted(qb_df[(qb_df['season'] == selected_season) & 
                                   ((qb_df['home_team'] == selected_team) | 
                                    (qb_df['away_team'] == selected_team))]['week'].unique())

st.markdown("---")

# --- DASHBOARD & TABLE LAYOUT ---
col1, col2 = st.columns([3, 1])

# --- SCHEDULE COLUMN (RIGHT SIDE) ---
with col2:
    sched_header_col1, sched_header_col2, sched_header_col3 = st.columns([2, 1, 1])
    
    default_sched_index = teams.index(selected_team) if selected_team in teams else 0
    schedule_team = st.selectbox(
        "Schedule Team", 
        teams, 
        index=default_sched_index, 
        key="schedule_team_selector"
    )

    sched_team_plays = plays[(plays['posteam'] == schedule_team) | (plays['defteam'] == schedule_team)]
    if not sched_team_plays.empty:
        sched_weeks = sorted(sched_team_plays['week'].unique())
    else:
        sched_weeks = []

    sched_table_data = []
    
    for week in sched_weeks: 
        # Using ONLY qb_df (nflscrapr)
        starter, opponent_qb, home_score, away_score, is_home_game, spread_val, total_val = get_game_metadata(qb_df, selected_season, week, schedule_team)
        
        if isinstance(spread_val, (int, float)):
            spread = f"{spread_val:+.1f}"
        else:
            spread = str(spread_val)
        
        total = str(total_val)
        
        if pd.notna(home_score) and pd.notna(away_score) and is_home_game is not None:
            if is_home_game:
                team_score, opp_score = int(home_score), int(away_score)
            else:
                team_score, opp_score = int(away_score), int(home_score)
                
            if team_score > opp_score: result_str = f"W {team_score}-{opp_score}"
            elif team_score < opp_score: result_str = f"L {team_score}-{opp_score}"
            else: result_str = f"T {team_score}-{opp_score}"
        else:
            result_str = "N/A" 

        is_selected_initial = (schedule_team, week) not in st.session_state.excluded_games
        
        suffix = "(H)" if is_home_game else "(A)"
        
        week_num_str = f" {week}" if week < 10 else f"{week}"
        week_display = f"{week_num_str} {suffix}"

        sched_table_data.append({
            'Selected': is_selected_initial,
            'Week': week_display,
            'Week_Num': week, # Hidden column
            'Starting QB': starter, 
            'Opponent QB': opponent_qb, 
            'Result': result_str, 
            'Spread': spread, 
            'Total': total
        })

    sched_table_df = pd.DataFrame(sched_table_data)
    
    if not sched_table_df.empty:
        sched_table_df = sched_table_df.sort_values('Week_Num')

    changes_pending = False
    new_exclusions_from_editor = set()

    if not sched_table_df.empty:
        with sched_header_col1:
            st.subheader("Game Schedule")

        edited_schedule = st.data_editor(
            sched_table_df,
            column_config={
                "Selected": st.column_config.CheckboxColumn(
                    "Select",
                    help="Uncheck to exclude this game. Click 'Apply' to update stats.",
                    default=True,
                ),
                "Week": st.column_config.TextColumn(
                    "Week",
                    help="Game Week (Home/Away)",
                    width="small"
                ),
                "Week_Num": None 
            },
            disabled=["Week", "Starting QB", "Opponent QB", "Result", "Spread", "Total"],
            hide_index=True,
            use_container_width=True,
            height=1000,
            key=f"schedule_editor_{schedule_team}_{st.session_state.editor_key_version}" 
        )
        
        unchecked_rows = edited_schedule[edited_schedule['Selected'] == False]
        unchecked_weeks = set(unchecked_rows['Week_Num'])
        
        current_global_exclusions = st.session_state.excluded_games
        current_team_weeks = set(sched_table_df['Week_Num'])
        
        for w in current_team_weeks:
            item = (schedule_team, w)
            is_excluded_in_editor = w in unchecked_weeks
            is_excluded_in_saved = item in current_global_exclusions
            
            if is_excluded_in_editor != is_excluded_in_saved:
                changes_pending = True
                break
        
        if changes_pending:
            new_global_exclusions = set([x for x in current_global_exclusions if x[0] != schedule_team])
            for w in unchecked_weeks:
                new_global_exclusions.add((schedule_team, w))
        else:
            new_global_exclusions = current_global_exclusions

    else:
        st.info("No schedule data.")

    with sched_header_col2:
        if changes_pending:
            if st.button("Apply", type="primary", help="Update stats with changes"):
                st.session_state.excluded_games = new_global_exclusions
                st.rerun()
        
    with sched_header_col3:
        if st.button("Reset", help="Select all games"):
            st.session_state.excluded_games = set()
            st.session_state.editor_key_version += 1 
            st.rerun()

# --- DASHBOARD COLUMN (LEFT SIDE) ---
excluded_games_list = list(st.session_state.excluded_games)
latest_stats = generate_power_rankings_data(plays, excluded_games_list)
team_dashboard_df = generate_team_dashboard_data(plays, excluded_games_list)

with col1:
    if not team_dashboard_df.empty:
        subset = team_dashboard_df[team_dashboard_df['team'] == selected_team].sort_values('week').copy()
        if not subset.empty:
            fig_dashboard = create_plotly_dashboard(subset, latest_stats, selected_team)
            st.plotly_chart(fig_dashboard, use_container_width=True)
        else:
            st.warning("No dashboard data available.")
    else:
        st.warning("Dashboard data generation failed.")

# --- POWER METRICS ---
st.markdown("---")
if not team_dashboard_df.empty:
    valid_weeks = team_dashboard_df[team_dashboard_df['team'] == selected_team]['week']
    if not valid_weeks.empty:
        min_week, max_week = valid_weeks.min(), valid_weeks.max()
    else:
        min_week, max_week = 0, 0
    
st.subheader(f"NFL Power Metrics (Weeks {min_week} - {max_week})")

if not latest_stats.empty:
    tier_col1, tier_col2 = st.columns([2, 1])
    with tier_col1:
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(10, 7), dpi=1000) 
            x = latest_stats['off_total_epa_oe_wgt']
            y = latest_stats['def_total_epa_oe_wgt']
            teams_scatter = latest_stats['team']
            ax.scatter(x, y, s=100, alpha=0.8, c='dodgerblue', edgecolors='white', linewidth=0.8)
            for i, team in enumerate(teams_scatter):
                ax.text(x[i], y[i]+0.002, team, fontsize=9, ha='center', va='bottom', fontweight='bold', color='white')
            ax.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(0, color='white', linestyle='--', alpha=0.5, linewidth=1)
            x_max, x_min = latest_stats['off_total_epa_oe_wgt'].max(), latest_stats['off_total_epa_oe_wgt'].min()
            y_max, y_min = latest_stats['def_total_epa_oe_wgt'].max(), latest_stats['def_total_epa_oe_wgt'].min()
            ax.text(x_max, y_max, 'Good Offense\nGood Defense', ha='right', va='top', fontsize=10, color='lime', alpha=0.8, weight='bold')
            ax.text(x_max, y_min, 'Good Offense\nBad Defense', ha='right', va='bottom', fontsize=10, color='orange', alpha=0.8, weight='bold')
            ax.text(x_min, y_max, 'Bad Offense\nGood Defense', ha='left', va='top', fontsize=10, color='orange', alpha=0.8, weight='bold')
            ax.text(x_min, y_min, 'Bad Offense\nBad Defense', ha='left', va='bottom', fontsize=10, color='red', alpha=0.8, weight='bold')
            ax.set_title('Recency Weighted EPA Over Expectation', fontsize=14, fontweight='bold', color='white')
            ax.set_xlabel('Offense: Weighted EPA OE', fontsize=10, fontweight='bold', color='white')
            ax.set_ylabel('Defense: Weighted EPA OE (Positive = Good)', fontsize=10, fontweight='bold', color='white')
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_edgecolor('white')
            st.pyplot(fig)

    with tier_col2:
        display_cols = ['team','ovr_total_epa_oe_wgt','off_total_epa_oe_wgt','off_pass_epa_oe_wgt', 'off_run_epa_oe_wgt','def_total_epa_oe_wgt','def_pass_epa_oe_wgt', 'def_run_epa_oe_wgt']
        rank_df = latest_stats[display_cols].sort_values('ovr_total_epa_oe_wgt', ascending=False).head(32).reset_index(drop=True)
        rank_df.index = rank_df.index + 1
        rank_df.columns = ['Team', 'Overall', 'Off Total', 'Off Pass', 'Off Run', 'Def Total', 'Def Pass', 'Def Run']
        table_height = (len(rank_df) + 1) * 35 + 3
        st.dataframe(
            rank_df.style.background_gradient(cmap='coolwarm', subset=['Overall', 'Off Total', 'Off Pass', 'Off Run', 'Def Total', 'Def Pass', 'Def Run'])
                   .format('{:,.3f}', subset=['Overall', 'Off Total', 'Off Pass', 'Off Run', 'Def Total', 'Def Pass', 'Def Run']),
            use_container_width=True, height=table_height
        )
else:
    st.info("Insufficient data for Power Rankings.")

# --- GRIDS ---
if plays is not None:
    plays_filtered = plays.copy()
    if excluded_games_list:
        mask = pd.Series(False, index=plays_filtered.index)
        for t, w in excluded_games_list:
            game_mask = (plays_filtered['week'] == w) & ((plays_filtered['posteam'] == t) | (plays_filtered['defteam'] == t))
            mask = mask | game_mask
        plays_filtered = plays_filtered[~mask]
else:
    plays_filtered = pd.DataFrame()

st.markdown("---")
st.header("Passing")
grid_c1, grid_c2 = st.columns(2)

with grid_c1:
    st.subheader("Offense Target Rates (% of Avg)")
    if not plays_filtered.empty:
        valid_pass_plays = plays_filtered[(plays_filtered['posteam'] == selected_team) & (plays_filtered['pass_attempt'] == 1) & (plays_filtered['sack'] == 0)]
        if not valid_pass_plays.empty:
            receiver_stats = valid_pass_plays.groupby('receiver')['yards_gained'].sum().sort_values(ascending=False)
            receiver_list = [selected_team] + receiver_stats.index.tolist()
            selected_receiver = st.selectbox("Select Target (Ranked by Yds):", receiver_list)
            offense_grid_df = pass_offense_zover_average_combined(plays_filtered, selected_team, selected_receiver)
        else:
             offense_grid_df = pd.DataFrame()
             selected_receiver = selected_team
    else:
        offense_grid_df = pd.DataFrame()
        selected_receiver = selected_team

    if not offense_grid_df.empty and selected_team in offense_grid_df.index.get_level_values('posteam'):
        team_offense_grid = offense_grid_df.xs(selected_team, level='posteam')
        team_offense_grid_display = team_offense_grid[['left', 'middle', 'right']].reset_index()
        team_offense_grid_display.columns = ['Depth', 'Left', 'Middle', 'Right']
        fig_off_grid = create_heatmap_figure(team_offense_grid_display, f"{selected_receiver} vs League Avg")
        st.plotly_chart(fig_off_grid, use_container_width=False)
    else:
        st.warning("Could not display Offense Grid.")

with grid_c2:
    st.subheader("Defense Success Allowed (% of Avg)")
    defense_teams = sorted(teams)
    default_def_index = defense_teams.index(selected_team) if selected_team in defense_teams else 0
    selected_defense_team = st.selectbox("Select Defense:", defense_teams, index=default_def_index)
    
    if not plays_filtered.empty:
        defense_grid_df = pass_defense_zover_average_combined(plays_filtered)
    else:
        defense_grid_df = pd.DataFrame()
    
    if not defense_grid_df.empty and selected_defense_team in defense_grid_df.index.get_level_values('defteam'):
        team_defense_grid = defense_grid_df.xs(selected_defense_team, level='defteam')
        team_defense_grid_display = team_defense_grid[['left', 'middle', 'right']].reset_index()
        team_defense_grid_display.columns = ['Depth', 'Left', 'Middle', 'Right']
        fig_def_grid = create_heatmap_figure(team_defense_grid_display, f"{selected_defense_team} Defense vs League Avg")
        st.plotly_chart(fig_def_grid, use_container_width=False)
    else:
        st.warning("Could not display Defense Grid.")

with st.expander("How to read the Passing Grids"):
    st.write("""
    **100%:** League Average.
    **> 100% (Red):** Higher than league average. (Offense: Targets more often / Defense: Allows more success)
    **< 100% (Blue):** Lower than league average.
    """)