from pathlib import Path
import sys
import subprocess
import copy
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.gridspec import GridSpec
import uuid

st.set_page_config(page_title="Metapop Admissions Explorer", layout="wide")

# ----- Torch guard -----
try:
    import torch
except Exception as e:
    st.error(
        "PyTorch failed to import. Ensure `requirements.txt` pins `torch==2.9.0`, "
        "then Manage app → Clear cache & reboot.\n\n"
        f"Import error: {e}"
    )
    st.stop()

APP_DIR = Path(__file__).parent.resolve()
CLT_DIR = APP_DIR / "CLT_BaseModel"

def ensure_clt_repo():
    if CLT_DIR.exists():
        return True
    try:
        st.info("CLT_BaseModel not found. Cloning…")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/LP-relaxation/CLT_BaseModel.git", str(CLT_DIR)],
            check=True, capture_output=True, text=True
        )
        st.success("CLT_BaseModel cloned.")
        return True
    except Exception as e:
        st.error(
            "Failed to clone CLT_BaseModel. Vendor the folder at `CLT_BaseModel/` and redeploy.\n\n"
            f"Details: {e}"
        )
        return False

if not ensure_clt_repo():
    st.stop()

if str(CLT_DIR) not in sys.path:
    sys.path.insert(0, str(CLT_DIR))

try:
    import clt_toolkit as clt
    import flu_core as flu
except Exception as e:
    st.error(
        "Could not import `clt_toolkit`/`flu_core` from CLT_BaseModel. "
        "Verify the repo structure and try again.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# ===== Helpers =====
def as_like(val, like_tensor):
    return torch.as_tensor(val, dtype=like_tensor.dtype, device=like_tensor.device)

def set_beta_by_location(p, beta_vec):
    L, A, R = p.beta_baseline.shape
    vec = np.asarray(beta_vec, dtype=float)
    if len(vec) != L:
        raise ValueError(f"beta_by_location length must be {L}, got {len(vec)}")
    new_beta = torch.as_tensor(vec, dtype=p.beta_baseline.dtype, device=p.beta_baseline.device)\
                    .view(L, 1, 1).expand(L, A, R)
    p.beta_baseline = new_beta

def apply_rate_multiplier(p, field_name, mult):
    cur = getattr(p, field_name)
    setattr(p, field_name, cur * as_like(mult, cur))

def apply_prob_multiplier_clip01(p, field_name, mult):
    cur = getattr(p, field_name)
    t = torch.as_tensor(cur, dtype=torch.as_tensor(cur).dtype)
    new_val = (t * float(mult)).clamp(0.0, 1.0)
    like = torch.as_tensor(cur, dtype=new_val.dtype)
    new_val = new_val.to(like.dtype)
    setattr(p, field_name, new_val)

def extract_compartment_data(state, params, precomputed, schedules, T, tpd, compartment):
    """Extract time series for a specific compartment from simulation."""
    with torch.no_grad():
        # Run full simulation
        all_states = flu.torch_simulate_full_trajectory(state, params, precomputed, schedules, T, tpd)
        
        # Extract the specific compartment
        compartment_tensor = getattr(all_states, compartment)
        return compartment_tensor.cpu().numpy()

# ===== Load model inputs =====
@st.cache_resource(show_spinner=True)
def load_model_inputs():
    import pandas as pd
    T = 180
    timesteps_per_day = 4

    texas_files_path = CLT_DIR / "flu_instances" / "texas_input_files"
    calibration_files_path = CLT_DIR / "flu_instances" / "calibration_research_input_files"

    subpopA_init_vals_fp = calibration_files_path / "subpopA_init_vals.json"
    subpopB_init_vals_fp = calibration_files_path / "subpopB_init_vals.json"
    subpopC_init_vals_fp = calibration_files_path / "subpopC_init_vals.json"
    common_subpop_params_fp = texas_files_path / "common_subpop_params.json"
    mixing_params_fp = calibration_files_path / "ABC_mixing_params.json"
    simulation_settings_fp = texas_files_path / "simulation_settings.json"

    calendar_df = pd.read_csv(texas_files_path / "school_work_calendar.csv", index_col=0)
    humidity_df = pd.read_csv(texas_files_path / "absolute_humidity_austin_2023_2024.csv", index_col=0)
    vaccines_df = pd.read_csv(texas_files_path / "daily_vaccines_constant.csv", index_col=0)

    schedules_info = flu.FluSubpopSchedules(
        absolute_humidity=humidity_df,
        flu_contact_matrix=calendar_df,
        daily_vaccines=vaccines_df,
    )

    subpopA_init_vals = clt.make_dataclass_from_json(subpopA_init_vals_fp, flu.FluSubpopState)
    subpopB_init_vals = clt.make_dataclass_from_json(subpopB_init_vals_fp, flu.FluSubpopState)
    subpopC_init_vals = clt.make_dataclass_from_json(subpopC_init_vals_fp, flu.FluSubpopState)

    common_subpop_params = clt.make_dataclass_from_json(common_subpop_params_fp, flu.FluSubpopParams)
    mixing_params = clt.make_dataclass_from_json(mixing_params_fp, flu.FluMixingParams)
    simulation_settings = clt.make_dataclass_from_json(simulation_settings_fp, flu.SimulationSettings)

    simulation_settings = clt.updated_dataclass(simulation_settings, {"timesteps_per_day": timesteps_per_day})

    subpopA_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 1.5})
    subpopB_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.5})
    subpopC_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.2})

    subpopA = flu.FluSubpopModel(subpopA_init_vals, subpopA_params, simulation_settings,
                                 np.random.Generator(np.random.MT19937(111)), schedules_info, name="subpopA")
    subpopB = flu.FluSubpopModel(subpopB_init_vals, subpopB_params, simulation_settings,
                                 np.random.Generator(np.random.MT19937(222)), schedules_info, name="subpopB")
    subpopC = flu.FluSubpopModel(subpopC_init_vals, subpopC_params, simulation_settings,
                                 np.random.Generator(np.random.MT19937(333)), schedules_info, name="subpopC")

    flu_demo_model = flu.FluMetapopModel([subpopA, subpopB, subpopC], mixing_params)
    d = flu_demo_model.get_flu_torch_inputs()

    return dict(
        base_state=d["state_tensors"],
        base_params=d["params_tensors"],
        base_schedules=d["schedule_tensors"],
        base_precomputed=d["precomputed"],
        T=T,
        timesteps_per_day=timesteps_per_day,
    )

ctx = load_model_inputs()
base_state = ctx["base_state"]
base_params = ctx["base_params"]
base_schedules = ctx["base_schedules"]
base_precomputed = ctx["base_precomputed"]
T = ctx["T"]
timesteps_per_day = ctx["timesteps_per_day"]

L, A, R = base_params.beta_baseline.shape

# ===== Initialize session state =====
if 'figures' not in st.session_state:
    st.session_state.figures = []

# ===== Color palette =====
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# ===== Available compartments =====
COMPARTMENTS = ['S', 'E', 'IP', 'ISR', 'ISH', 'IA', 'HR', 'HD', 'R', 'D', 'M', 'MV']

# ===== Layout =====
col_left, col_right = st.columns([1, 2])

# ===== LEFT PANE: Controls =====
with col_left:
    st.title("Controls")
    
    if st.button("➕ Create New Figure", use_container_width=True):
        new_fig = {
            'id': str(uuid.uuid4()),
            'metric': 'ISH',
            'view': 'Aggregate',
            'series': [],
            'advanced': {
                'm_EI': 1.0, 'm_IP': 1.0, 'm_ISR': 1.0, 'm_IAR': 1.0,
                'm_ISH': 1.0, 'm_HR': 1.0, 'm_HD': 1.0,
                'm_EIAp': 1.0, 'm_IPinf': 1.0, 'm_IAinf': 1.0,
                'm_RS': 1.0, 'm_wane_inf': 1.0, 'm_wane_vax': 1.0,
                'm_inf_inf': 1.0, 'm_vax_inf': 1.0, 'm_inf_hosp': 1.0,
                'm_inf_death': 1.0, 'm_vax_hosp': 1.0, 'm_vax_death': 1.0
            }
        }
        st.session_state.figures.append(new_fig)
        st.rerun()
    
    st.markdown("---")
    
    # Display each figure's controls
    for fig_idx, fig in enumerate(st.session_state.figures):
        with st.container():
            st.markdown(f"### 📊 Figure {fig_idx + 1}")
            
            # Metric selection
            fig['metric'] = st.selectbox(
                "Compartment",
                COMPARTMENTS,
                index=COMPARTMENTS.index(fig['metric']),
                key=f"metric_{fig['id']}"
            )
            
            # View selection
            fig['view'] = st.selectbox(
                "View Type",
                ['Aggregate', 'By Location', 'By Location×Age'],
                index=['Aggregate', 'By Location', 'By Location×Age'].index(fig['view']),
                key=f"view_{fig['id']}"
            )
            
            # Add data series button
            if st.button(f"➕ Add Data Series", key=f"add_series_{fig['id']}", use_container_width=True):
                color_idx = len(fig['series']) % len(COLOR_PALETTE)
                new_series = {
                    'id': str(uuid.uuid4()),
                    'name': f"Series {len(fig['series']) + 1}",
                    'color': COLOR_PALETTE[color_idx],
                    'beta': [0.0005, 0.0005, 0.0005],
                    'data': None,
                    'status': 'Not run'
                }
                fig['series'].append(new_series)
                st.rerun()
            
            # Display series
            for series_idx, series in enumerate(fig['series']):
                with st.expander(f"📈 {series['name']}", expanded=True):
                    cols = st.columns([3, 1])
                    with cols[0]:
                        series['name'] = st.text_input(
                            "Series Name",
                            series['name'],
                            key=f"name_{series['id']}"
                        )
                    with cols[1]:
                        series['color'] = st.color_picker(
                            "Color",
                            series['color'],
                            key=f"color_{series['id']}"
                        )
                    
                    # Beta inputs
                    beta_cols = st.columns(3)
                    for i in range(3):
                        with beta_cols[i]:
                            series['beta'][i] = st.number_input(
                                f"β L{i}",
                                min_value=0.0,
                                max_value=0.2,
                                value=float(series['beta'][i]),
                                step=0.0001,
                                format="%.4f",
                                key=f"beta_{i}_{series['id']}"
                            )
                    
                    # Run simulation button
                    if st.button("▶️ Run Simulation", key=f"run_{series['id']}", use_container_width=True):
                        with st.spinner("Running simulation..."):
                            # Build params with beta and advanced settings
                            p = copy.deepcopy(base_params)
                            set_beta_by_location(p, series['beta'])
                            
                            # Apply advanced settings
                            adv = fig['advanced']
                            apply_rate_multiplier(p, "E_to_I_rate", adv['m_EI'])
                            apply_rate_multiplier(p, "IP_to_IS_rate", adv['m_IP'])
                            apply_rate_multiplier(p, "ISR_to_R_rate", adv['m_ISR'])
                            apply_rate_multiplier(p, "IA_to_R_rate", adv['m_IAR'])
                            apply_rate_multiplier(p, "ISH_to_H_rate", adv['m_ISH'])
                            apply_rate_multiplier(p, "HR_to_R_rate", adv['m_HR'])
                            apply_rate_multiplier(p, "HD_to_D_rate", adv['m_HD'])
                            
                            apply_prob_multiplier_clip01(p, "E_to_IA_prop", adv['m_EIAp'])
                            p.IP_relative_inf = p.IP_relative_inf * as_like(adv['m_IPinf'], torch.as_tensor(p.IP_relative_inf))
                            p.IA_relative_inf = p.IA_relative_inf * as_like(adv['m_IAinf'], torch.as_tensor(p.IA_relative_inf))
                            
                            apply_rate_multiplier(p, "R_to_S_rate", adv['m_RS'])
                            p.inf_induced_immune_wane = p.inf_induced_immune_wane * as_like(adv['m_wane_inf'], torch.as_tensor(p.inf_induced_immune_wane))
                            p.vax_induced_immune_wane = p.vax_induced_immune_wane * as_like(adv['m_wane_vax'], torch.as_tensor(p.vax_induced_immune_wane))
                            
                            apply_prob_multiplier_clip01(p, "inf_induced_inf_risk_reduce", adv['m_inf_inf'])
                            apply_prob_multiplier_clip01(p, "vax_induced_inf_risk_reduce", adv['m_vax_inf'])
                            apply_prob_multiplier_clip01(p, "inf_induced_hosp_risk_reduce", adv['m_inf_hosp'])
                            apply_prob_multiplier_clip01(p, "inf_induced_death_risk_reduce", adv['m_inf_death'])
                            apply_prob_multiplier_clip01(p, "vax_induced_hosp_risk_reduce", adv['m_vax_hosp'])
                            apply_prob_multiplier_clip01(p, "vax_induced_death_risk_reduce", adv['m_vax_death'])
                            
                            # Extract compartment data
                            data = extract_compartment_data(
                                base_state, p, base_precomputed, base_schedules, T, timesteps_per_day, fig['metric']
                            )
                            series['data'] = data
                            series['status'] = '✓ Complete'
                        st.success("Simulation complete!")
                        st.rerun()
                    
                    st.caption(f"Status: {series['status']}")
                    
                    # Delete series button
                    if st.button("🗑️ Delete Series", key=f"del_series_{series['id']}", use_container_width=True):
                        fig['series'].remove(series)
                        st.rerun()
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings", expanded=False):
                st.markdown("**Rate Multipliers**")
                adv = fig['advanced']
                
                cols = st.columns(2)
                with cols[0]:
                    adv['m_EI'] = st.slider("E→I ×", 0.0, 5.0, adv['m_EI'], 0.01, key=f"adv_m_EI_{fig['id']}")
                    adv['m_IP'] = st.slider("IP→IS ×", 0.0, 5.0, adv['m_IP'], 0.01, key=f"adv_m_IP_{fig['id']}")
                    adv['m_ISR'] = st.slider("IS→R ×", 0.0, 5.0, adv['m_ISR'], 0.01, key=f"adv_m_ISR_{fig['id']}")
                    adv['m_IAR'] = st.slider("IA→R ×", 0.0, 5.0, adv['m_IAR'], 0.01, key=f"adv_m_IAR_{fig['id']}")
                with cols[1]:
                    adv['m_ISH'] = st.slider("IS→H ×", 0.0, 5.0, adv['m_ISH'], 0.01, key=f"adv_m_ISH_{fig['id']}")
                    adv['m_HR'] = st.slider("H→R ×", 0.0, 5.0, adv['m_HR'], 0.01, key=f"adv_m_HR_{fig['id']}")
                    adv['m_HD'] = st.slider("H→D ×", 0.0, 5.0, adv['m_HD'], 0.01, key=f"adv_m_HD_{fig['id']}")
                
                st.markdown("**Split / Infectiousness**")
                cols = st.columns(2)
                with cols[0]:
                    adv['m_EIAp'] = st.slider("E→IA prop ×", 0.0, 5.0, adv['m_EIAp'], 0.01, key=f"adv_m_EIAp_{fig['id']}")
                    adv['m_IPinf'] = st.slider("IP rel inf ×", 0.0, 5.0, adv['m_IPinf'], 0.01, key=f"adv_m_IPinf_{fig['id']}")
                with cols[1]:
                    adv['m_IAinf'] = st.slider("IA rel inf ×", 0.0, 5.0, adv['m_IAinf'], 0.01, key=f"adv_m_IAinf_{fig['id']}")
                
                st.markdown("**Immunity**")
                cols = st.columns(2)
                with cols[0]:
                    adv['m_RS'] = st.slider("R→S ×", 0.0, 5.0, adv['m_RS'], 0.01, key=f"adv_m_RS_{fig['id']}")
                    adv['m_wane_inf'] = st.slider("inf wane ×", 0.0, 5.0, adv['m_wane_inf'], 0.01, key=f"adv_m_wane_inf_{fig['id']}")
                with cols[1]:
                    adv['m_wane_vax'] = st.slider("vax wane ×", 0.0, 5.0, adv['m_wane_vax'], 0.01, key=f"adv_m_wane_vax_{fig['id']}")
                
                st.markdown("**Risk Reductions**")
                cols = st.columns(2)
                with cols[0]:
                    adv['m_inf_inf'] = st.slider("inf risk ×", 0.0, 5.0, adv['m_inf_inf'], 0.01, key=f"adv_m_inf_inf_{fig['id']}")
                    adv['m_vax_inf'] = st.slider("vax risk ×", 0.0, 5.0, adv['m_vax_inf'], 0.01, key=f"adv_m_vax_inf_{fig['id']}")
                    adv['m_inf_hosp'] = st.slider("inf hosp ×", 0.0, 5.0, adv['m_inf_hosp'], 0.01, key=f"adv_m_inf_hosp_{fig['id']}")
                with cols[1]:
                    adv['m_inf_death'] = st.slider("inf death ×", 0.0, 5.0, adv['m_inf_death'], 0.01, key=f"adv_m_inf_death_{fig['id']}")
                    adv['m_vax_hosp'] = st.slider("vax hosp ×", 0.0, 5.0, adv['m_vax_hosp'], 0.01, key=f"adv_m_vax_hosp_{fig['id']}")
                    adv['m_vax_death'] = st.slider("vax death ×", 0.0, 5.0, adv['m_vax_death'], 0.01, key=f"adv_m_vax_death_{fig['id']}")
            
            # Delete figure button
            if st.button("🗑️ Delete Figure", key=f"del_fig_{fig['id']}", use_container_width=True):
                st.session_state.figures.remove(fig)
                st.rerun()
            
            st.markdown("---")

# ===== RIGHT PANE: Figures =====
with col_right:
    st.title("Visualizations")
    
    if len(st.session_state.figures) == 0:
        st.info("Create a figure to get started!")
    
    for fig_idx, fig in enumerate(st.session_state.figures):
        st.markdown(f"## Figure {fig_idx + 1}: {fig['metric']}")
        
        # Filter series with data
        series_with_data = [s for s in fig['series'] if s['data'] is not None]
        
        if len(series_with_data) == 0:
            st.warning("No simulation data yet. Add a series and run simulation.")
            continue
        
        # Create figure based on view type
        if fig['view'] == 'Aggregate':
            # Single plot with all series
            fig_plot, ax = plt.subplots(figsize=(10, 5))
            xs = np.arange(T)
            
            for series in series_with_data:
                # Aggregate over L, A, R
                y_agg = np.sum(series['data'], axis=(0, 1, 2))
                ax.plot(xs, y_agg, label=series['name'], color=series['color'], linewidth=2)
            
            ax.set_xlabel("Time (days)")
            ax.set_ylabel(f"{fig['metric']} Count")
            ax.set_title(f"{fig['metric']} - Aggregate")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            st.pyplot(fig_plot, use_container_width=True)
            
        elif fig['view'] == 'By Location':
            # 3 subplots for locations
            fig_plot, axes = plt.subplots(1, 3, figsize=(15, 4))
            xs = np.arange(T)
            
            for loc in range(L):
                for series in series_with_data:
                    # Aggregate over A, R
                    y_loc = np.sum(series['data'][loc, :, :, :], axis=(0, 1))
                    axes[loc].plot(xs, y_loc, label=series['name'], color=series['color'], linewidth=2)
                
                axes[loc].set_xlabel("Time (days)")
                axes[loc].set_ylabel(f"{fig['metric']} Count")
                axes[loc].set_title(f"Location {loc}")
                axes[loc].legend()
                axes[loc].grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_plot, use_container_width=True)
            
        elif fig['view'] == 'By Location×Age':
            # Grid of subplots for L×A
            fig_plot = plt.figure(figsize=(15, 4 * L))
            gs = GridSpec(L, A, figure=fig_plot)
            xs = np.arange(T)
            
            for loc in range(L):
                for age in range(A):
                    ax = fig_plot.add_subplot(gs[loc, age])
                    
                    for series in series_with_data:
                        # Aggregate over R only
                        y_loc_age = np.sum(series['data'][loc, age, :, :], axis=0)
                        ax.plot(xs, y_loc_age, label=series['name'], color=series['color'], linewidth=1.5)
                    
                    ax.set_xlabel("Time (days)", fontsize=8)
                    ax.set_ylabel(f"{fig['metric']}", fontsize=8)
                    ax.set_title(f"L{loc}, A{age}", fontsize=9)
                    if loc == 0 and age == 0:
                        ax.legend(fontsize=7)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.tick_params(labelsize=7)
            
            plt.tight_layout()
            st.pyplot(fig_plot, use_container_width=True)
        
        st.markdown("---")
