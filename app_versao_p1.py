import streamlit as st
import numpy as np
import pandas as pd
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import io
from itertools import product
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

from scipy.stats import shapiro



# (se tiver scipy / pyDOE, ficam aqui também)

# =============================================
# Configuração DA PÁGINA  (TEM QUE SER A PRIMEIRA COISA do Streamlit)
# =============================================
st.set_page_config(page_title="TaguchiApp", layout="wide")

logo = Image.open("assets/logo_taguchiapp.png")

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.image(logo, width=250)

with col2:
    # Ajuste fino: diminuí o padding-top para 3px
    st.markdown(
        """
        <div style="float: right; text-align: right; max-width: 400px; padding-top: 20px;">
            <h3 style="font-size: 20px; font-weight: bold; 
                       margin: 0 0 1px 0; line-height: 0; color: #1f3a5e;">
                Planejamento e Análise Experimental Taguchi
            </h3>
            <p style="font-size: 14px; margin: 0; line-height: 0; color: #555; letter-spacing: 0.5px;">
                Versão 26.prv08
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
# Linha separadora (opcional)
st.markdown("---")


# aqui embaixo vêm as suas funções: oa_from_name, built_in_catalog, section_factors_and_oa, section_results, etc.


# ============================
# Imports opcionais
# ============================
try:
    from scipy.stats import f as f_dist, t as t_dist
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    f_dist = None
    t_dist = None

try:
    from pyDOE3 import get_orthogonal_array
    HAS_PYDOE3 = True
except Exception:
    HAS_PYDOE3 = False
    get_orthogonal_array = None


# ---------------------------------------------
# Variável de interesse
# ---------------------------------------------
var_label = st.text_input(
    "Variável de interesse (ex.: Produção de H₂)",
    "Produção de H₂",
    help="Digite o nome da variável de interesse. Tecle ENTER ao finalizar!"
)

if var_label:
    st.session_state["var_label"] = var_label
    st.success(f"✅ **Variável definida:** {var_label}")
else:
    st.session_state["var_label"] = "Produção de H₂"
    st.write("**Variável definida:** Produção de H₂")


# ============================
# Utilitários de OA / catálogo
# ============================
def built_in_catalog():
    return {
        "L4(2^3)"     : {"cols2": 3,  "cols3": 0,  "n": 4},
        "L8(2^7)"     : {"cols2": 7,  "cols3": 0,  "n": 8},
        "L9(3^4)"     : {"cols2": 0,  "cols3": 4,  "n": 9},
        "L12(2^11)"   : {"cols2": 11, "cols3": 0,  "n": 12},
        "L16(2^15)"   : {"cols2": 15, "cols3": 0,  "n": 16},
        "L18(2^1 3^7)": {"cols2": 1,  "cols3": 7,  "n": 18},
        "L27(3^13)"   : {"cols2": 0,  "cols3": 13, "n": 27},
    }


PYDOE3_NAME_MAP = {
    "L18(2^1 3^7)": "L18(6^1 3^6)",
    "L27(3^13)":    "L27(2^1 3^12)",
}


def oa_from_name(name: str) -> np.ndarray:
    # 1) Tenta pyDOE3
    if HAS_PYDOE3 and get_orthogonal_array is not None:
        try:
            lookup = PYDOE3_NAME_MAP.get(name, name)
            arr = np.asarray(get_orthogonal_array(lookup), dtype=int)
            # se vier 1/2/3, convertemos para 0/1/2:
            if arr.min() == 1:
                arr = arr - 1
            return arr
        except Exception:
            # se der erro no pyDOE3, cai pros fallbacks internos
            pass

    # 2) Fallbacks internos (0-based)
    import numpy as np

    if name == "L4(2^3)":
        return np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=int)

    if name == "L8(2^7)":
        return np.array([
            [0,0,0,0,0,0,0],
            [0,0,0,1,1,1,1],
            [0,1,1,0,0,1,1],
            [0,1,1,1,1,0,0],
            [1,0,1,0,1,0,1],
            [1,0,1,1,0,1,0],
            [1,1,0,0,1,1,0],
            [1,1,0,1,0,0,1],
        ], dtype=int)

    if name == "L9(3^4)":
        return np.array([
            [0,0,0,0],
            [0,1,1,1],
            [0,2,2,2],
            [1,0,1,2],
            [1,1,2,0],
            [1,2,0,1],
            [2,0,2,1],
            [2,1,0,2],
            [2,2,1,0],
        ], dtype=int)

    if name == "L16(2^15)":
        arr12 = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2],
            [1,1,1,2,2,2,2,1,1,1,1,2,2,2,2],
            [1,1,1,2,2,2,2,2,2,2,2,1,1,1,1],
            [1,2,2,1,1,2,2,1,1,2,2,1,1,2,2],
            [1,2,2,1,1,2,2,2,2,1,1,2,2,1,1],
            [1,2,2,2,2,1,1,1,1,2,2,2,2,1,1],
            [1,2,2,2,2,1,1,2,2,1,1,1,1,2,2],
            [2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [2,1,2,1,2,1,2,2,1,2,1,2,1,2,1],
            [2,1,2,2,1,2,1,1,2,1,2,2,1,2,1],
            [2,1,2,2,1,2,1,2,1,2,1,1,2,1,2],
            [2,2,1,1,2,2,1,1,2,2,1,1,2,2,1],
            [2,2,1,1,2,2,1,2,1,1,2,2,1,1,2],
            [2,2,1,2,1,1,2,1,2,2,1,2,1,1,2],
            [2,2,1,2,1,1,2,2,1,1,2,1,2,2,1],
        ], dtype=int)
        return arr12 - 1

    if name == "L18(2^1 3^7)":
        part1 = np.array([
            [1,1,1,1,1,1,1],[1,1,2,2,2,2,2],[1,1,3,3,3,3,3],
            [1,2,1,1,2,2,3],[1,2,2,2,3,3,1],[1,2,3,3,1,1,2],
            [1,3,1,2,1,3,2],[1,3,2,3,2,1,3],[1,3,3,1,3,2,1],
            [2,1,1,3,3,2,2],[2,1,2,1,1,3,3],[2,1,3,2,2,1,1],
            [2,2,1,2,3,1,3],[2,2,2,3,1,2,1],[2,2,3,1,2,3,2],
            [2,3,1,3,2,3,1],[2,3,2,1,3,1,2],[2,3,3,2,1,2,3],
        ], dtype=int)
        col8 = np.array([
            [1],[2],[3],[3],[1],[2],[3],[1],[2],
            [1],[2],[3],[2],[3],[1],[2],[3],[1],
        ], dtype=int)
        return np.hstack([(part1[:, 0:1] - 1), (part1[:, 1:] - 1), (col8 - 1)])

    if name == "L27(3^13)":
        arr27 = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,3,3,3,3,3,3,3,3,3],
            [1,2,2,2,1,1,1,2,2,2,3,3,3],
            [1,2,2,2,2,2,2,3,3,3,1,1,1],
            [1,2,2,2,3,3,3,1,1,1,2,2,2],
            [1,3,3,3,1,1,1,3,3,3,2,2,2],
            [1,3,3,3,2,2,2,1,1,1,3,3,3],
            [1,3,3,3,3,3,3,2,2,2,1,1,1],
            [2,1,2,3,1,2,3,1,2,3,1,2,3],
            [2,1,2,3,2,3,1,2,3,1,2,3,1],
            [2,1,2,3,3,1,2,3,1,2,3,1,2],
            [2,2,3,1,1,2,3,2,3,1,3,1,2],
            [2,2,3,1,2,3,1,3,1,2,1,2,3],
            [2,2,3,1,3,1,2,1,2,3,2,3,1],
            [2,3,1,2,1,2,3,3,1,2,2,3,1],
            [2,3,1,2,2,3,1,1,2,3,3,1,2],
            [2,3,1,2,3,1,2,2,3,1,1,2,3],
        ], dtype=int)
        return arr27 - 1

    # se nada casou:
    raise RuntimeError(f"OA '{name}' não disponível.")

def to_latex_var(name: str) -> str:
    """
    Converte nomes do tipo R_M em R_{M} para uso em LaTeX.
    Casos com múltiplos '_' viram subscrito agrupado.
    """
    if "_" in name:
        base, *subs = name.split("_")
        sub = "_".join(subs)
        return rf"{base}_{{{sub}}}"
    return name



def full_factorial_runs(levels_by_factor: list[int]) -> int:
    runs = 1
    for n in levels_by_factor:
        runs *= int(n)
    return runs


# ============================
# Configuração da página
# ============================
def configure_page():
    st.set_page_config(page_title="Taguchi App", layout="wide")
    st.title("Taguchi App")
    st.caption(
        """
        <div style="font-size:16px; font-weight:bold;">
            Taguchi App — Planejamento e Análise Experimental Taguchi — Versão v25.01<br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # estado do fluxo (wizard)
    if "step" not in st.session_state:
        st.session_state["step"] = "start"


# ============================
# Entrada da variável de interesse
# ============================
def input_var_label() -> str:
    var_label = st.text_input(
        "Variável de interesse (ex.: Produção de H₂)",
        "Produção de H₂",
        help="Digite o nome da variável de interesse. Tecle ENTER ao finalizar!",
    )

    if var_label:
        st.success(f"✅ **Variável definida:** {var_label}")
    else:
        st.write("**Variável definida:** Produção de H₂")

    # salva no session_state para uso em outras seções
    st.session_state["var_label"] = var_label or "Produção de H₂"
    return st.session_state["var_label"]


# ---------------------------------------------
# Upload de fatores (em função)
# ---------------------------------------------
def section_factors_and_oa():
    with st.container():
        upl = st.file_uploader(
            "**Carregar arquivo de fatores**",
            type=["xlsx"],
            key="fatores_upl",
            help="Selecione o arquivo Excel com a configuração dos fatores (aba 'Fatores')."
        )

        # Se nada foi enviado, apenas sai da função
        if not upl:
            return

        try:
            df_fatores = pd.read_excel(upl, sheet_name='Fatores')
            if 'Factor' not in df_fatores.columns:
                st.error("❌ Coluna 'Factor' não encontrada no arquivo.")
                return

            st.success("✅ Arquivo carregado com sucesso!")
            st.dataframe(df_fatores, use_container_width=True, hide_index=True)  # <- sem índice

            st.markdown("---")
            st.subheader("🔍 Análise Automática dos Fatores")
            fatores = df_fatores['Factor'].astype(str).tolist()
            num_fatores = len(fatores)

            level_cols = [col for col in df_fatores.columns if col.startswith('Level')]
            niveis_por_fator, niveis_rotulos = [], []
            for _, row in df_fatores.iterrows():
                lvls = [row[col] for col in level_cols if pd.notna(row[col])]
                niveis_por_fator.append(len(lvls))
                niveis_rotulos.append([str(x) for x in lvls])

            niveis_unicos = list(set(niveis_por_fator))
            mesmo_numero_niveis = len(niveis_unicos) == 1
            dof_necessario = sum([n - 1 for n in niveis_por_fator])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Número de Fatores", num_fatores)
            c2.metric(
                "Níveis por Fator",
                f"{niveis_unicos[0]}" if mesmo_numero_niveis else f"misto: {min(niveis_por_fator)}–{max(niveis_por_fator)}"
            )
            c3.metric("Graus de Liberdade Necessários", dof_necessario)
            c4.metric("Experimentos no Fatorial Completo", full_factorial_runs(niveis_por_fator))

            st.markdown("---")
            st.subheader("🎯 Matrizes Ortogonais Recomendadas")
            catalog = built_in_catalog()
            matrizes_candidatas = []

            for nome, specs in catalog.items():
                if specs['n'] - 1 < dof_necessario:
                    continue

                if mesmo_numero_niveis:
                    if niveis_unicos[0] == 2 and specs['cols2'] >= num_fatores:
                        matrizes_candidatas.append((nome, specs))
                    elif niveis_unicos[0] == 3 and specs['cols3'] >= num_fatores:
                        matrizes_candidatas.append((nome, specs))
                else:
                    f2 = sum(1 for n in niveis_por_fator if n == 2)
                    f3 = sum(1 for n in niveis_por_fator if n == 3)
                    if specs['cols2'] >= f2 and specs['cols3'] >= f3:
                        matrizes_candidatas.append((nome, specs))

            matrizes_candidatas.sort(key=lambda x: x[1]['n'])

            if not matrizes_candidatas:
                st.warning("⚠️ Nenhuma matriz ortogonal padrão adequada foi encontrada.")
                return

            total_full = full_factorial_runs(niveis_por_fator)
            linhas = []
            for nome, specs in matrizes_candidatas:
                eficiencia = (1 - specs['n'] / total_full) * 100 if total_full > 0 else 0.0
                linhas.append({
                    "Matriz": nome,
                    "Experimentos (n)": specs['n'],
                    "Colunas (2 níveis)": specs['cols2'],
                    "Colunas (3 níveis)": specs['cols3'],
                    "Economia de corridas (%)": f"{eficiencia:.1f}%"
                })

            df_recomendacoes = pd.DataFrame(linhas)
            st.dataframe(df_recomendacoes, use_container_width=True, hide_index=True)  # <- sem índice
            st.caption("ℹ️ Economia de corridas em relação ao fatorial completo")

            st.markdown("---")
            st.subheader("🎛️ Seleção da Matriz Ortogonal")
            matriz_opcoes = [m[0] for m in matrizes_candidatas]
            matriz_selecionada = st.selectbox(
                "Escolha a matriz para gerar o experimento:",
                options=matriz_opcoes,
                index=0,
            )

            if st.button("🔄 Gerar Matriz Experimental", type="primary"):
                try:
                    matriz_oa = oa_from_name(matriz_selecionada)
                    if matriz_oa.shape[1] < num_fatores:
                        st.error("❌ A OA selecionada tem menos colunas do que o número de fatores.")
                        return

                    matriz_oa = matriz_oa[:, :num_fatores]
                    df_codificada = pd.DataFrame(matriz_oa, columns=fatores)

                    df_niveis = pd.DataFrame(index=df_codificada.index)
                    for j, fator in enumerate(fatores):
                        rotulos = niveis_rotulos[j]
                        max_code = matriz_oa[:, j].max()
                        if max_code >= len(rotulos):
                            st.warning(
                                f"⚠️ Fator **{fator}** tem {len(rotulos)} níveis, "
                                f"mas a OA possui código até {int(max_code)}. Revise."
                            )
                        df_niveis[fator] = [
                            rotulos[c] if c < len(rotulos) else f"lvl{c+1}"
                            for c in matriz_oa[:, j]
                        ]

                    df_niveis.insert(0, "Experimento", range(1, len(df_niveis) + 1))

                    # 🔴 Aqui salvamos tudo no session_state
                    st.session_state['matriz_selecionada'] = matriz_selecionada
                    st.session_state['matriz_oa'] = matriz_oa
                    st.session_state['df_fatores'] = df_fatores
                    st.session_state['df_experimentos_cod'] = df_codificada
                    st.session_state['df_experimentos'] = df_niveis
                    st.session_state['var_label'] = st.session_state.get('var_label', 'Variável de Interesse')
                    st.session_state['step'] = 'results'

                    st.success(f"✅ Matriz {matriz_selecionada} gerada com sucesso!")
                
                except Exception as e:
                    st.error(f"❌ Erro ao gerar a matriz: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
        st.markdown("---")

def compute_anova_sn(df_effects: pd.DataFrame, factor_cols: list[str], sn_col: str):
    # Vetor de S/N por ensaio
    y_sn = df_effects[sn_col].to_numpy(dtype=float)
    N = len(y_sn)

    if N <= 1:
        return None, {"error": "Número insuficiente de ensaios para calcular ANOVA."}

    grand_mean_sn = float(np.nanmean(y_sn))
    ss_total = float(np.nansum((y_sn - grand_mean_sn) ** 2))
    df_total = N - 1

    # Soma de quadrados por fator
    factor_entries = []
    ss_factors_sum = 0.0
    df_factors_sum = 0

    for fac in factor_cols:
        g = df_effects.groupby(df_effects[fac].astype(str))[sn_col]
        means = g.mean()
        counts = g.size()

        ss_fac = float(np.nansum(counts * (means - grand_mean_sn) ** 2))
        df_fac = len(means) - 1

        ss_factors_sum += ss_fac
        df_factors_sum += df_fac

        factor_entries.append({"Fonte": fac, "gl": df_fac, "SQ": ss_fac})

    # Erro bruto
    ss_error_raw = ss_total - ss_factors_sum
    if ss_error_raw < 0 and abs(ss_error_raw) < 1e-10:
        ss_error_raw = 0.0
    df_error_raw = df_total - df_factors_sum

    # Contribuições originais
    for ent in factor_entries:
        ent["Contrib_orig"] = (100.0 * ent["SQ"] / ss_total) if ss_total > 0 else np.nan

    used_pooling = False
    pooled_names = []
    kept_entries = factor_entries.copy()
    ss_error = ss_error_raw
    df_error = df_error_raw

    # Pooling automático se não houver GL de erro
    if df_error_raw <= 0:
        used_pooling = True

        sorted_entries = sorted(
            factor_entries,
            key=lambda e: (np.inf if np.isnan(e["Contrib_orig"]) else e["Contrib_orig"])
        )

        candidates = [
            e for e in sorted_entries
            if (not np.isnan(e["Contrib_orig"])) and (e["Contrib_orig"] < 5.0)
        ]

        if not candidates and len(sorted_entries) > 1:
            candidates = [sorted_entries[0]]

        if len(candidates) >= len(sorted_entries):
            candidates = candidates[:-1]

        ss_pool = 0.0
        df_pool = 0
        pooled_names = [ent["Fonte"] for ent in candidates]

        for ent in candidates:
            ss_pool += ent["SQ"]
            df_pool += ent["gl"]

        ss_error = max(0.0, ss_error_raw) + ss_pool
        df_error = max(0, df_error_raw) + df_pool

        kept_entries = [ent for ent in factor_entries if ent["Fonte"] not in pooled_names]

        if df_error <= 0 or len(kept_entries) == 0:
            used_pooling = False
            pooled_names = []
            kept_entries = factor_entries
            ss_error = ss_error_raw
            df_error = df_error_raw

    rows_anova = []

    # Caso sem GL de erro (nem com pooling)
    if df_error <= 0:
        for ent in factor_entries:
            rows_anova.append({
                "Fonte": ent["Fonte"],
                "GL": ent["gl"],
                "SQ": ent["SQ"],
                "QM": np.nan,
                "F": np.nan,
                "p-valor": np.nan,
                "Contribuição (%)": ent["Contrib_orig"],
                "Significativo (5%)": "n/d",
            })

        rows_anova.append({
            "Fonte": "Erro",
            "GL": 0,
            "SQ": ss_error_raw,
            "QM": np.nan,
            "F": np.nan,
            "p-valor": np.nan,
            "Contribuição (%)": np.nan,
            "Significativo (5%)": "n/d",
        })

        rows_anova.append({
            "Fonte": "Total",
            "GL": df_total,
            "SQ": ss_total,
            "QM": np.nan,
            "F": np.nan,
            "p-valor": np.nan,
            "Contribuição (%)": (100.0 if ss_total > 0 else np.nan),
            "Significativo (5%)": "n/d",
        })

        anova_df = pd.DataFrame(rows_anova)
        for col in ["SQ", "QM", "F", "p-valor", "Contribuição (%)"]:
            if col in anova_df.columns:
                anova_df[col] = pd.to_numeric(anova_df[col], errors="coerce").round(4)

        meta = {
            "used_pooling": used_pooling,
            "pooled_names": pooled_names,
            "factor_entries": factor_entries,
            "kept_entries": kept_entries,
            "ss_total": ss_total,
        }
        return anova_df, meta

    # Caso com GL de erro (normal ou via pooling)
    ms_error = ss_error / df_error if df_error > 0 else np.nan

    def contrib_final_ss(ss_part):
        return 100.0 * ss_part / ss_total if ss_total > 0 else np.nan

    for ent in kept_entries:
        gl_k = ent["gl"]
        ss_k = ent["SQ"]
        ms_k = ss_k / gl_k if gl_k > 0 else np.nan
        F_k = ms_k / ms_error if (gl_k > 0 and ms_error > 0) else np.nan

        if HAS_SCIPY and f_dist is not None and gl_k > 0 and df_error > 0 and not np.isnan(F_k):
            try:
                p_k = float(f_dist.sf(F_k, gl_k, df_error))
            except Exception:
                p_k = np.nan
        else:
            p_k = np.nan

        signif = ("Sim (p < 0,05)" if (not np.isnan(p_k) and p_k < 0.05) else "Não") if not np.isnan(p_k) else "n/d"

        rows_anova.append({
            "Fonte": ent["Fonte"],
            "GL": gl_k,
            "SQ": ss_k,
            "QM": ms_k,
            "F": F_k,
            "p-valor": p_k,
            "Contribuição (%)": contrib_final_ss(ss_k),
            "Significativo (5%)": signif,
        })

    rows_anova.append({
        "Fonte": "Erro" + (" (com pooling)" if used_pooling else ""),
        "GL": df_error,
        "SQ": ss_error,
        "QM": ms_error,
        "F": np.nan,
        "p-valor": np.nan,
        "Contribuição (%)": contrib_final_ss(ss_error),
        "Significativo (5%)": "n/d",
    })

    rows_anova.append({
        "Fonte": "Total",
        "GL": df_total,
        "SQ": ss_total,
        "QM": np.nan,
        "F": np.nan,
        "p-valor": np.nan,
        "Contribuição (%)": (100.0 if ss_total > 0 else np.nan),
        "Significativo (5%)": "n/d",
    })

    anova_df = pd.DataFrame(rows_anova)
    for col in ["SQ", "QM", "F", "p-valor", "Contribuição (%)"]:
        if col in anova_df.columns:
            anova_df[col] = pd.to_numeric(anova_df[col], errors="coerce").round(4)

    meta = {
        "used_pooling": used_pooling,
        "pooled_names": pooled_names,
        "factor_entries": factor_entries,
        "kept_entries": kept_entries,
        "ss_total": ss_total,
    }
    return anova_df, meta

##
def _build_X_row_from_levels(levels: dict, factor_cols: list, X_dum_cols: list):
    """
    levels: {"A": "1", "B": "2", ...} com valores como strings
    X_dum_cols: colunas do X_dum do ajuste (inclui 'Constante' + dummies)
    """
    # cria DF com 1 linha
    row_df = pd.DataFrame([{f: str(levels[f]) for f in factor_cols}])

    # dummies iguais às do treino (drop_first=True)
    row_dum = pd.get_dummies(row_df, drop_first=True, dtype=float)

    # garante colunas esperadas (sem a Constante ainda)
    cols_no_const = [c for c in X_dum_cols if c != "Constante"]
    for c in cols_no_const:
        if c not in row_dum.columns:
            row_dum[c] = 0.0

    # remove colunas extras (se aparecerem)
    row_dum = row_dum[cols_no_const]

    # insere Constante
    row_dum.insert(0, "Constante", 1.0)

    X_new = row_dum.to_numpy(dtype=float)  # shape (1, p1)
    return X_new


def _as_float_scalar(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Valor vazio não pode ser convertido para escalar.")
    return float(arr[0])

def predicao_usuario_regressao(
    df_plan, factor_cols, var_label,
    beta_hat, XtX_inv, sigma2_hat, df_res, t_dist,
    X_dum_cols,
    key_prefix="reg_pred"   # <- evita conflito de keys em outras abas
):
    st.markdown("---")
    st.subheader("🧮 Predição (Regressão) para qualquer combinação")

    # ----------------- Entrada do usuário -----------------
    levels = {}
    for f in factor_cols:
        lvls = sorted(
            df_plan[f].astype(str).unique(),
            key=lambda z: int(z) if str(z).isdigit() else str(z)
        )
        levels[f] = st.selectbox(
            f"Nível para {f}",
            lvls,
            key=f"{key_prefix}_lvl_{var_label}_{f}",
        )

    # ----------------- Predição do ponto (ŷ) -----------------
    X_new = _build_X_row_from_levels(levels, factor_cols, X_dum_cols)
    y_hat_new = _as_float_scalar(X_new @ beta_hat)

    # ----------------- IC 95% da média -----------------
    alpha = 0.05
    t_crit = t_dist.ppf(1 - alpha/2, df=df_res) if df_res > 0 else np.nan

    v_mean = _as_float_scalar(sigma2_hat * (X_new @ XtX_inv @ X_new.T))
    se_mean = np.sqrt(max(v_mean, 0.0))

    ic_low = y_hat_new - t_crit * se_mean if np.isfinite(t_crit) else np.nan
    ic_high = y_hat_new + t_crit * se_mean if np.isfinite(t_crit) else np.nan

    # ======================================================
    # 🔍 Resultados em tela (SEU PADRÃO)
    # ======================================================
    st.markdown("🔍 **Resultados da predição (regressão)**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Predição para {var_label}", f"{y_hat_new:.4f}")

    with col2:
        ic_txt = f"[{ic_low:.4f}; {ic_high:.4f}]" if np.isfinite(ic_low) else "n/d"
        st.markdown(f"- **IC 95% (média)**: {ic_txt}")

    st.caption(
        "IC 95% (média): intervalo onde se espera que esteja a média verdadeira da resposta "
        "para a combinação de níveis selecionada."
    )

    # ======================================================
    # 📥 Exportações
    # ======================================================
    st.markdown("### 📥 Exportações de predição")

    # 1) CSV — ensaio (predição atual)
    df_one = pd.DataFrame([{
        **{f: str(levels[f]) for f in factor_cols},
        f"{var_label}_pred": y_hat_new,
        "IC95_low": ic_low,
        "IC95_high": ic_high,
    }])

    buf_one = io.StringIO()
    df_one.to_csv(buf_one, index=False)
    fname_one = f"predicao_ponto_regressao_{var_label}.csv"

    # 2) CSV — matriz fatorial completa (opcional)
    gerar_full = st.checkbox(
        "Gerar matriz fatorial completa (recomendado apenas para número moderado de fatores/níveis)",
        value=False,
        key=f"{key_prefix}_chk_full_{var_label}",
    )

    buf_full = None
    fname_full = f"predicao_regressao_todas_combinacoes_{var_label}.csv"

    if gerar_full:
        combos = _all_factorial_combinations(df_plan, factor_cols)

        # trava de segurança (ajuste como quiser)
        max_combos = 200_000
        if len(combos) > max_combos:
            st.warning(
                f"⚠️ A matriz teria {len(combos):,} combinações (limite: {max_combos:,}). "
                "Reduza níveis/fatores ou use Top-N."
            )
        else:
            rows = []
            for levels_i in combos:
                X_i = _build_X_row_from_levels(levels_i, factor_cols, X_dum_cols)
                y_i = _as_float_scalar(X_i @ beta_hat)

                v_i = _as_float_scalar(sigma2_hat * (X_i @ XtX_inv @ X_i.T))
                se_i = np.sqrt(max(v_i, 0.0))

                ic_l = y_i - t_crit * se_i if np.isfinite(t_crit) else np.nan
                ic_h = y_i + t_crit * se_i if np.isfinite(t_crit) else np.nan

                rows.append({
                    **levels_i,
                    f"{var_label}_pred": y_i,
                    "IC95_low": ic_l,
                    "IC95_high": ic_h,
                })

            df_full = pd.DataFrame(rows).sort_values(f"{var_label}_pred", ascending=False).reset_index(drop=True)

            buf_full = io.StringIO()
            df_full.to_csv(buf_full, index=False)

    # -------------------------
    # Downloads em 2 colunas
    # -------------------------
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.download_button(
            "📥 Baixar ensaio (predição atual)",
            data=buf_one.getvalue().encode("utf-8"),
            file_name=fname_one,
            mime="text/csv",
            key=f"{key_prefix}_dl_pred_one_{var_label}",
        )

    with col_b2:
        if buf_full is None:
            st.download_button(
                "📥 Baixar matriz fatorial completa (predições)",
                data="Arquivo indisponível (marque a opção para gerar).".encode("utf-8"),
                file_name=fname_full,
                mime="text/plain",
                key=f"{key_prefix}_dl_pred_full_disabled_{var_label}",
                disabled=True,
            )
        else:
            st.download_button(
                "📥 Baixar matriz fatorial completa (predições)",
                data=buf_full.getvalue().encode("utf-8"),
                file_name=fname_full,
                mime="text/csv",
                key=f"{key_prefix}_dl_pred_full_{var_label}",
            )

    return levels, y_hat_new



def _all_factorial_combinations(df_plan, factor_cols):
    """Retorna lista de dicts com todas combinações de níveis existentes em df_plan."""
    levels_by_factor = {}
    for f in factor_cols:
        lvls = sorted(
            df_plan[f].astype(str).unique(),
            key=lambda z: int(z) if str(z).isdigit() else str(z)
        )
        levels_by_factor[f] = lvls

    combos = itertools.product(*[levels_by_factor[f] for f in factor_cols])

    out = []
    for combo in combos:
        out.append({f: str(v) for f, v in zip(factor_cols, combo)})
    return out



def ponto_otimo_regressao(
    df_plan,
    factor_cols,
    var_label,
    beta_hat,
    XtX_inv,
    sigma2_hat,
    df_res,
    t_dist,
    X_dum_cols,
    per_factor_tables,
    alpha=0.05,
):
    st.markdown("---")
    st.subheader("⭐ Ponto ótimo do Taguchi + Predição (Regressão) com IC")

    Y_hat_taguchi = st.session_state.get("Y_hat_taguchi_opt", np.nan)
    
    # 1) Pega o ponto ótimo do Taguchi (via S/N médio por nível)
    opt_levels = _opt_levels_from_tables(
        factor_cols=factor_cols,
        df_plan=df_plan,
        per_factor_tables=per_factor_tables,
    )

    st.markdown("🔍 **Ponto ótimo (Taguchi)**")
    chips_html = "<div style='display:flex; flex-wrap:wrap; gap:8px;'>"
    for fac in factor_cols:
        chips_html += f"""
            <div style="padding:6px 12px; background:#ecfdf5;
                        border-radius:999px; font-size:13px; color:#064e3b;
                        box-shadow:0 2px 6px rgba(0,0,0,0.08);">
                <span style="font-weight:600; color:#065f46;">{fac}:</span> {opt_levels.get(fac, "-")}
            </div>"""
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    # 2) Predição via regressão nesse ponto ótimo
    X_new = _build_X_row_from_levels(opt_levels, factor_cols, X_dum_cols)  # shape (1,p)
    y_hat = _as_float_scalar(X_new @ beta_hat)

    st.session_state["Y_hat_reg_opt"] = float(np.asarray(y_hat).squeeze())

    # 3) IC 95% da média e IP 95% individual
    t_crit = t_dist.ppf(1 - alpha/2, df=df_res) if (df_res > 0) else np.nan

    v_mean = _as_float_scalar(sigma2_hat * (X_new @ XtX_inv @ X_new.T))
    se_mean = float(np.sqrt(max(v_mean, 0.0)))

    ic_low = y_hat - t_crit * se_mean if np.isfinite(t_crit) else np.nan
    ic_high = y_hat + t_crit * se_mean if np.isfinite(t_crit) else np.nan

    st.divider()
    st.markdown("🔍 **Predição no ponto ótimo (Regressão)**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="text-align:center; margin: 14px 0 8px;">
              <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                          border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                  Valor previsto (Regressão) — {var_label}
                </div>
                <div style="font-size:26px; font-weight:700; color:#064e3b;">
                  {("n/d" if np.isnan(y_hat) else f"{y_hat:.4f}")}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        ic_txt = f"[{ic_low:.4f}; {ic_high:.4f}]" if np.isfinite(ic_low) else "n/d"
        st.markdown(f"- **IC {int((1-alpha)*100)}% (média)**: {ic_txt}")


    # 4) Exportação CSV
    row = {fac: str(opt_levels.get(fac, "-")) for fac in factor_cols}
    row.update({
        f"{var_label}_pred_reg": (np.nan if np.isnan(y_hat) else round(y_hat, 6)),
        "IC95_low": (np.nan if np.isnan(ic_low) else round(ic_low, 6)),
        "IC95_high": (np.nan if np.isnan(ic_high) else round(ic_high, 6)),
    })


    df_out = pd.DataFrame([row])
    buf = io.StringIO()
    df_out.to_csv(buf, index=False)

    st.download_button(
        "📥 Baixar ponto ótimo (Taguchi) + predição regressão + intervalos (CSV)",
        data=buf.getvalue().encode("utf-8"),
        file_name=f"ponto_otimo_taguchi_pred_reg_{var_label}.csv",
        mime="text/csv",
        key="dl_opt_taguchi_reg",
    )
    st.markdown("---")

def render_predicoes_otimo_reg_vs_taguchi_sem_ic(
    var_label,
):


    # lê valores já calculados (sem recalcular)
    y_hat_reg = st.session_state.get("Y_hat_reg_opt", np.nan)
    y_hat_taguchi = st.session_state.get("Y_hat_taguchi_opt", np.nan)

    # garante escalar float
    try:
        y_hat_reg = float(np.asarray(y_hat_reg).squeeze())
    except Exception:
        y_hat_reg = np.nan

    try:
        y_hat_taguchi = float(np.asarray(y_hat_taguchi).squeeze())
    except Exception:
        y_hat_taguchi = np.nan

    st.markdown("🔍 **Predições no ponto ótimo**")

    col1, col2 = st.columns(2)

    # --------- REGRESSÃO ---------
    with col1:
        st.markdown(
            f"""
            <div style="text-align:center; margin: 14px 0 8px;">
              <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                          border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                  Valor previsto (Regressão) — {var_label}
                </div>
                <div style="font-size:26px; font-weight:700; color:#064e3b;">
                  {("n/d" if np.isnan(y_hat_reg) else f"{y_hat_reg:.4f}")}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------- TAGUCHI ---------
    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; margin: 14px 0 8px;">
              <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                          border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                  Valor previsto (Taguchi) — {var_label}
                </div>
                <div style="font-size:26px; font-weight:700; color:#064e3b;">
                  {("n/d" if np.isnan(y_hat_taguchi) else f"{y_hat_taguchi:.4f}")}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

def render_confirmacao_regressao_sem_upload(
    factor_cols,
    df_plan,
    per_factor_tables,
    var_label,
    beta_hat,
    X_dum_cols,
):
    st.subheader("🧪 Ensaio de confirmação (Regressão)")

    st.caption(
        "Esta seção reutiliza o **mesmo ensaio de confirmação carregado na aba Taguchi** "
        "e compara a **média observada** com o **valor previsto pela regressão** no mesmo ponto."
    )

    st.markdown("---")
    st.markdown("**1️⃣ Escolha o ponto de análise do ensaio de confirmação**")
    
    if "modo_conf_prev_conf_reg" not in st.session_state:
        st.session_state["modo_conf_prev_conf_reg"] = "Ponto ótimo (recomendado)"
    
    modo_conf = st.radio(
        "Selecione a combinação de níveis a ser utilizada:",
        ("Ponto ótimo (recomendado)", "Outra combinação de níveis"),
        index=0,
        key="modo_conf_conf_reg",
    )


    
    # ao mudar o modo, limpa estados locais desta seção (apenas chaves confreg_)
    if st.session_state["modo_conf_prev_conf_reg"] != modo_conf:
        for k in list(st.session_state.keys()):
            if k.startswith("confreg_"):
                st.session_state.pop(k, None)
        st.session_state["modo_conf_prev_conf_reg"] = modo_conf


    # valores observados do ensaio (sempre vêm do Taguchi; sem upload aqui)
    y_conf_vals = st.session_state.get(f"conf_y_vals__{var_label}", None)
    
    if y_conf_vals is None:
        st.info(
            "⏳ Nenhum ensaio de confirmação encontrado para esta variável.\n\n"
            "Vá na aba **Taguchi → 🧪 Ensaios de confirmação**, faça o upload uma vez, e volte aqui."
        )
        return

    
    # escolhe o ponto (níveis) conforme o modo
    conf_levels = {}
    
    if modo_conf == "Ponto ótimo (recomendado)":
        opt_levels = _opt_levels_from_tables(factor_cols, df_plan, per_factor_tables)
    
        lista_niveis = []
        for fac in factor_cols:
            default_lvl = str(sorted(df_plan[fac].astype(str).unique())[0])
            conf_levels[fac] = str(opt_levels.get(fac, default_lvl))
        st.markdown("\n".join(lista_niveis))
    
    else:
        st.markdown("Selecione manualmente os níveis utilizados no ensaio de confirmação:")
        for fac in factor_cols:
            niveis = sorted(
                df_plan[fac].astype(str).unique(),
                key=lambda z: int(z) if str(z).isdigit() else str(z),
            )
            conf_levels[fac] = st.selectbox(
                f"Nível para {fac} no ensaio de confirmação:",
                niveis,
                key=f"confreg_{fac}",
            )



    y_conf_vals = np.asarray(y_conf_vals, dtype=float)
    y_conf_vals = y_conf_vals[~np.isnan(y_conf_vals)]

    if y_conf_vals.size == 0:
        st.warning("⚠️ O ensaio de confirmação salvo não contém valores numéricos válidos.")
        return

    y_obs = float(np.mean(y_conf_vals))

    st.markdown("---")
    st.markdown("**2️⃣ Estatísticas descritivas do ensaio de confirmação**")

    st.caption(
        f"Réplicas: **{y_conf_vals.size}** | "
        f"Mín: **{np.min(y_conf_vals):.4f}** | "
        f"Máx: **{np.max(y_conf_vals):.4f}** | "
        f"Desvio-padrão: **{np.std(y_conf_vals, ddof=1):.4f}**"
    )
    
    st.markdown("---")
    st.markdown("**3️⃣ Comparação entre observado e previsto (Regressão)**")

    if any(str(conf_levels.get(f, "-")) == "-" for f in factor_cols):
        st.warning("⚠️ Há fator(es) sem nível definido. Ajuste os níveis para calcular a predição.")
        return


    # predição da regressão no mesmo ponto do ensaio (conf_levels)
    try:
        X_new = _build_X_row_from_levels(conf_levels, factor_cols, X_dum_cols)
        y_hat_reg = _as_float_scalar(X_new @ beta_hat)
    except Exception as e:
        y_hat_reg = float("nan")
        st.warning(f"Não foi possível calcular a predição da regressão no ponto do ensaio: {e}")

    err_abs = abs(y_obs - y_hat_reg) if np.isfinite(y_hat_reg) else np.nan
    den = abs(y_hat_reg)
    err_rel = (100.0 * err_abs / den) if (np.isfinite(err_abs) and np.isfinite(den) and den > 1e-12) else np.nan

    
    titulo_ponto = "Ponto ótimo (recomendado)" if modo_conf == "Ponto ótimo (recomendado)" else "Ponto manual (outra combinação)"
    st.markdown(f"🔎 **{titulo_ponto}**")
    
    chips_html = "<div style='display:flex; flex-wrap:wrap; gap:8px;'>"
    for fac in factor_cols:
        chips_html += f"""
            <div style="padding:6px 12px; background:#ecfdf5;
                        border-radius:999px; font-size:13px; color:#064e3b;
                        box-shadow:0 2px 6px rgba(0,0,0,0.08);">
                <span style="font-weight:600; color:#065f46;">{fac}:</span> {conf_levels.get(fac, "-")}
            </div>"""
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)   # pequeno 



    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="text-align:center; margin: 14px 0 8px;">
              <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                          border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                  Valor previsto (Regressão) — {var_label}
                </div>
                <div style="font-size:26px; font-weight:700; color:#064e3b;">
                  {("n/d" if np.isnan(y_hat_reg) else f"{y_hat_reg:.4f}")}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; margin: 14px 0 8px;">
              <div style="display:inline-block; padding:12px 22px; background:#eff6ff;
                          border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                <div style="font-size:14px; color:#1d4ed8; font-weight:600; margin-bottom:4px;">
                  Valor observado (média) — {var_label}
                </div>
                <div style="font-size:26px; font-weight:700; color:#1e3a8a;">
                  {f"{y_obs:.4f}"}
                </div>
                <div style="margin-top:10px; font-size:13px; color:#111827;">
                  Erro absoluto: <b>{("n/d" if np.isnan(err_abs) else f"{err_abs:.4f}")}</b><br/>
                  Erro relativo: <b>{("n/d" if np.isnan(err_rel) else f"{err_rel:.2f}%")}</b>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")
##





def _predict_combo(level_dict, mean_y, df_effects, sn_col, per_factor_tables, factor_cols, df_plan):
    # Y
    y_by_run = np.asarray(mean_y, dtype=float)
    Y_bar = float(np.nanmean(y_by_run))
    efeitos_y = []

    for fac in factor_cols:
        nivel = str(level_dict[fac])
        mask = (df_plan[fac].astype(str) == nivel).values
        media_nivel = float(np.nanmean(y_by_run[mask])) if mask.any() else np.nan
        efeitos_y.append(media_nivel - Y_bar)

    y_pred = float(Y_bar + np.nansum(efeitos_y))

    # S/N
    sn_bar = float(df_effects[sn_col].mean())
    efeitos_sn = []

    for fac in factor_cols:
        nivel = str(level_dict[fac])

        fac_df = per_factor_tables.get(fac, pd.DataFrame())
        media_sn = np.nan

        if (not fac_df.empty) and {"Nível", "S/N médio (dB)"}.issubset(fac_df.columns):
            media_sn = fac_df.loc[fac_df["Nível"].astype(str) == nivel, "S/N médio (dB)"].mean()

        if pd.isna(media_sn):
            mask = (df_plan[fac].astype(str) == nivel)
            media_sn = float(df_effects.loc[mask, sn_col].mean())

        efeitos_sn.append(media_sn - sn_bar)

    eta_pred = float(sn_bar + np.nansum(efeitos_sn))

    return y_pred, eta_pred


def render_exportacoes_predicao(
    user_levels,
    Y_hat,
    eta_hat,
    var_label,
    mean_y,
    df_effects,
    sn_col,
    per_factor_tables,
    factor_cols,
    df_plan,
):
    # ---------- (1) Ensaio atual ----------
    row_dict = {fac: user_levels[fac] for fac in factor_cols}

    if np.isfinite(Y_hat) and np.isfinite(eta_hat):
        y_pred_one, eta_pred_one = (Y_hat, eta_hat)
    else:
        y_pred_one, eta_pred_one = _predict_combo(
            row_dict, mean_y, df_effects, sn_col, per_factor_tables, factor_cols, df_plan
        )

    df_pred_one = pd.DataFrame([{
        **row_dict,
        f"Previsão {var_label}": (np.nan if not np.isfinite(y_pred_one) else round(y_pred_one, 6)),
        "Previsão S/N (dB)": (np.nan if not np.isfinite(eta_pred_one) else round(eta_pred_one, 6)),
    }])

    buf_one = io.StringIO()
    df_pred_one.to_csv(buf_one, index=False)
    fname_one = f"ensaio_predito_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # ---------- (2) Matriz fatorial completa ----------
    levels_map = {fac: sorted(df_plan[fac].astype(str).unique()) for fac in factor_cols}

    rows = []
    for combo in product(*[levels_map[fac] for fac in factor_cols]):
        combo_dict = {fac: level for fac, level in zip(factor_cols, combo)}
        y_pred, eta_pred = _predict_combo(
            combo_dict, mean_y, df_effects, sn_col, per_factor_tables, factor_cols, df_plan
        )
        rows.append({
            **combo_dict,
            f"Previsão {var_label}": (np.nan if not np.isfinite(y_pred) else round(y_pred, 6)),
            "Previsão S/N (dB)": (np.nan if not np.isfinite(eta_pred) else round(eta_pred, 6)),
        })

    df_full = pd.DataFrame(rows)
    buf_full = io.StringIO()
    df_full.to_csv(buf_full, index=False)
    fname_full = f"matriz_fatorial_predicoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    st.markdown("<br><br>", unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.download_button(
            "📥 Baixar ensaio (predição atual)",
            buf_one.getvalue().encode("utf-8"),
            file_name=fname_one,
            mime="text/csv",
            key="dl_pred_one",
        )

    with col_b2:
        st.download_button(
            "📥 Baixar matriz fatorial completa (predições)",
            buf_full.getvalue().encode("utf-8"),
            file_name=fname_full,
            mime="text/csv",
            key="dl_pred_full",
        )


def estimativas_ponto_otimo(
    factor_cols,
    df_plan,
    per_factor_tables,
    var_label,
    mean_y,
    df_effects,
    sn_col,
    opt_table=None,   # opcional: se você quiser reaproveitar a tabela do "nível ótimo"
):
    st.markdown("---")
    st.subheader("🎯 Estimativa de valores no ponto ótimo")

    # ==============================
    # 🔹 Resumo do ponto ótimo
    # ==============================
    opt_levels = {}
    selected_level_means = []  # S/N médio (dB) no nível ótimo por fator

    for fac in factor_cols:
        fac_df = per_factor_tables.get(fac, pd.DataFrame())

        if (not fac_df.empty) and {"Nível", "S/N médio (dB)"}.issubset(fac_df.columns) and (not fac_df["S/N médio (dB)"].isna().all()):
            vmax = float(fac_df["S/N médio (dB)"].max())
            best_levels = fac_df.loc[fac_df["S/N médio (dB)"] == vmax, "Nível"].astype(str).tolist()

            opt_levels[fac] = best_levels[0] if best_levels else "-"
            selected_level_means.append(vmax)
        else:
            opt_levels[fac] = "-"
            selected_level_means.append(np.nan)

    st.markdown("🔍 **Ponto ótimo**")

    chips_html = "<div style='display:flex; flex-wrap:wrap; gap:8px;'>"
    for fac in factor_cols:
        chips_html += f"""
            <div style="padding:6px 12px; background:#ecfdf5;
                        border-radius:999px; font-size:13px; color:#064e3b;
                        box-shadow:0 2px 6px rgba(0,0,0,0.08);">
                <span style="font-weight:600; color:#065f46;">{fac}:</span> {opt_levels[fac]}
            </div>"""
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("🔍 **Resultados das predições no ponto ótimo**")

    # ==============================
    # 🔹 Caixas (verde)
    # ==============================
    colY, colSN = st.columns(2)

    # --------- COLUNA Y ---------
    with colY:
        try:
            col_media_otimo = f"Média de {var_label} no Nível Ótimo"

            # Preferência: reaproveitar opt_table (se foi fornecida e tem a coluna)
            if (opt_table is not None) and (col_media_otimo in opt_table.columns):
                Y_best_means = opt_table[col_media_otimo].to_numpy(dtype=float)
                k = len(Y_best_means)
            else:
                # Fallback: calcula média de Y no nível ótimo direto do plano usando mean_y (média por ensaio)
                y_by_run = np.asarray(mean_y, dtype=float)
                Y_best_means = []

                for fac in factor_cols:
                    nivel = str(opt_levels[fac])
                    if nivel == "-":
                        Y_best_means.append(np.nan)
                    else:
                        mask = (df_plan[fac].astype(str) == nivel).values
                        Y_best_means.append(float(np.nanmean(y_by_run[mask])) if mask.any() else np.nan)

                k = len(factor_cols)

            Y_bar = float(np.nanmean(np.asarray(mean_y, dtype=float)))
            if (k > 0) and np.isfinite(Y_bar) and (not np.isnan(np.array(Y_best_means, dtype=float)).any()):
                Y_hat_taguchi = float(np.sum(Y_best_means) - (k - 1) * Y_bar)
            else:
                Y_hat_taguchi = float("nan")

            # --- guardar para reuso na aba regressão (sem recalcular) ---
            st.session_state["Y_hat_taguchi_opt"] = Y_hat_taguchi
            st.session_state["opt_levels_taguchi"] = opt_levels


        except Exception as e:
            st.warning(f"Não foi possível calcular a previsão de {var_label}: {e}")
            Y_hat_taguchi = float("nan")

        st.markdown(
            f"""
            <div style="text-align:center; margin: 14px 0 8px;">
              <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                          border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                  Valor previsto (Taguchi) — {var_label}
                </div>
                <div style="font-size:26px; font-weight:700; color:#064e3b;">
                  {("n/d" if np.isnan(Y_hat_taguchi) else f"{Y_hat_taguchi:.3f}")}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------- COLUNA S/N ---------
    with colSN:
        try:
            grand_mean = float(df_effects[sn_col].mean())
        except Exception:
            grand_mean = float("nan")

        k = len(factor_cols)
        best_means_sn = np.array(selected_level_means, dtype=float)

        if (k > 0) and np.isfinite(grand_mean) and (not np.isnan(best_means_sn).any()):
            eta_hat_taguchi = float(best_means_sn.sum() - (k - 1) * grand_mean)
        else:
            eta_hat_taguchi = float("nan")

        st.markdown(
            f"""
            <div style="text-align:center; margin: 14px 0 8px;">
              <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                          border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                  S/N previsto (Taguchi Aditivo) — {var_label}
                </div>
                <div style="font-size:26px; font-weight:700; color:#064e3b;">
                  {("n/d" if np.isnan(eta_hat_taguchi) else f"{eta_hat_taguchi:.3f} dB")}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ==============================
    # 🔹 Baixar ponto ótimo com estimativas
    # ==============================
    row = {fac: opt_levels[fac] for fac in factor_cols}
    row.update({
        f"Previsão {var_label}": (np.nan if np.isnan(Y_hat_taguchi) else round(Y_hat_taguchi, 6)),
        "Previsão S/N (dB)": (np.nan if np.isnan(eta_hat_taguchi) else round(eta_hat_taguchi, 6)),
    })

    df_opt_export = pd.DataFrame([row])
    buf_opt = io.StringIO()
    df_opt_export.to_csv(buf_opt, index=False)
    fname_opt = f"ponto_otimo_estimada_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    st.download_button(
        "📥 Baixar ponto ótimo com estimativas",
        data=buf_opt.getvalue().encode("utf-8"),
        file_name=fname_opt,
        mime="text/csv",
        key="dl_opt_estimates",
    )

    st.markdown("---")

def compute_snr(vals, sn_tipo, nominal_target=None):
    """Calcula S/N (dB) para um vetor 1D de réplicas (Taguchi)."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan

    if sn_tipo == "Maior é melhor":
        return float(-10.0 * np.log10(np.mean(1.0 / (vals ** 2))))
    if sn_tipo == "Menor é melhor":
        return float(-10.0 * np.log10(np.mean(vals ** 2)))

    # Nominal é melhor
    target = nominal_target
    if (target is None) or (not np.isfinite(target)) or (target == 0.0):
        target = float(np.mean(vals))

    if vals.size < 2:
        return np.nan

    var = float(np.var(vals, ddof=1))
    if var <= 0:
        return np.nan
    return float(10.0 * np.log10((target ** 2) / var))


def _opt_levels_from_tables(factor_cols, df_plan, per_factor_tables):
    """Extrai níveis ótimos (1 por fator) a partir de per_factor_tables (S/N médio por nível)."""
    opt_levels = {}
    for fac in factor_cols:
        fac_df = per_factor_tables.get(fac, pd.DataFrame())
        lvl_fallback = str(sorted(df_plan[fac].astype(str).unique())[0])

        if (not fac_df.empty) and {"Nível", "S/N médio (dB)"}.issubset(set(fac_df.columns)) and (not fac_df["S/N médio (dB)"].isna().all()):
            vmax = float(fac_df["S/N médio (dB)"].max())
            best_levels = fac_df.loc[fac_df["S/N médio (dB)"] == vmax, "Nível"].astype(str).tolist()
            opt_levels[fac] = best_levels[0] if best_levels else lvl_fallback
        else:
            opt_levels[fac] = lvl_fallback

    return opt_levels


def render_ensaio_confirmacao(
    factor_cols,
    df_plan,
    var_label,
    mean_y,
    df_effects,
    sn_col,
    per_factor_tables,
    sn_tipo,
    nominal_target,
):
    """UI + cálculo do Ensaio de Confirmação (upload de matriz de réplicas)."""
    st.subheader("🧪 Ensaios de confirmação")

    st.caption(
        "Use esta seção para comparar os resultados de um ensaio de confirmação "
        "com os valores preditos pelo modelo aditivo de efeitos principais."
    )

    st.markdown("---")    
    st.markdown("**1️⃣ Escolha o ponto de análise do ensaio de confirmação**")

    if "modo_conf_prev_conf" not in st.session_state:
        st.session_state["modo_conf_prev_conf"] = "Ponto ótimo (recomendado)"


    modo_conf = st.radio(
        "Selecione a combinação de níveis a ser utilizada:",
        ("Ponto ótimo (recomendado)", "Outra combinação de níveis"),
        index=0,
        key="modo_conf_conf",
    )


    # ao mudar o modo, limpa estados do ensaio
    if st.session_state["modo_conf_prev_conf"] != modo_conf:
        for k in list(st.session_state.keys()):
            if k.startswith("conf_") or k.startswith("y_conf_") or k.startswith("sn_conf_") or k == "conf_upl":
                st.session_state.pop(k, None)
        st.session_state["modo_conf_prev_conf"] = modo_conf

    conf_levels = {}

    if modo_conf == "Ponto ótimo (recomendado)":
        opt_levels = _opt_levels_from_tables(factor_cols, df_plan, per_factor_tables)
        st.markdown("Usando os níveis ótimos encontrados na análise anterior:")
        lista_niveis = []
        for fac in factor_cols:
            nivel_esc = str(opt_levels.get(fac, str(sorted(df_plan[fac].astype(str).unique())[0])))
            conf_levels[fac] = nivel_esc
            lista_niveis.append(f"- **{fac}**: nível `{nivel_esc}`")
        st.markdown("\n".join(lista_niveis))

    else:
        st.markdown("Selecione manualmente os níveis utilizados no ensaio de confirmação:")
        for fac in factor_cols:
            niveis = sorted(df_plan[fac].astype(str).unique())
            conf_levels[fac] = st.selectbox(
                f"Nível para {fac} no ensaio de confirmação:",
                niveis,
                key=f"conf_{fac}",
            )

    # ----------------- Passo 2 -----------------
    st.markdown("---")
    st.markdown("**2️⃣ Carregue os resultados do ensaio de confirmação**")

    st.markdown(
        "Faça o upload de uma **matriz de repetições** do ensaio de confirmação. "
        "O arquivo pode ser `.xlsx` ou `.csv`. Todas as colunas numéricas serão "
        "usadas como valores reais de "
        f"**{var_label}** nas repetições (todas as linhas)."
    )

    y_conf_vals = np.array([], dtype=float)

    conf_upl = st.file_uploader(
        "📤 Carregar matriz de repetições do ensaio de confirmação",
        type=["xlsx", "csv"],
        key="conf_upl",
    )

    if conf_upl is not None:
        try:
            if conf_upl.name.endswith(".csv"):
                df_conf = pd.read_csv(conf_upl, sep=";")
            else:
                df_conf = pd.read_excel(conf_upl)

            num_cols = [c for c in df_conf.columns if pd.api.types.is_numeric_dtype(df_conf[c])]

            if len(num_cols) == 0:
                st.error("❌ Nenhuma coluna numérica encontrada no arquivo de confirmação.")
            else:
                vals = df_conf[num_cols].to_numpy(dtype=float).ravel()
                vals = vals[~np.isnan(vals)]

                if vals.size == 0:
                    st.error("❌ Não há valores numéricos válidos na matriz de confirmação.")
                else:
                    y_conf_vals = vals
                    st.success(f"✅ {len(y_conf_vals)} valores de {var_label} carregados para o ensaio de confirmação.")
                    st.dataframe(pd.DataFrame({var_label: y_conf_vals}), use_container_width=True, hide_index=True)
                    st.info(f"A razão S/N do ensaio de confirmação será calculada com o mesmo tipo: **{sn_tipo}**.")

                    # ✅ salvar para ser reutilizado na aba Regressão (sem novo upload)
                    st.session_state["conf_last_var_label"] = var_label
                    st.session_state[f"conf_y_vals__{var_label}"] = y_conf_vals
                    st.session_state[f"conf_levels__{var_label}"] = conf_levels


        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo de confirmação: {e}")
            y_conf_vals = np.array([], dtype=float)

    # ----------------- Passo 3 -----------------
    st.markdown("---")
    st.markdown("**3️⃣ Comparação entre médias observadas e valores preditos**")

    if y_conf_vals.size == 0:
        st.info("⏳ Primeiro carregue a matriz do ensaio de confirmação no **Passo 2**.")
        return

    y_conf_mean = float(np.nanmean(y_conf_vals))
    try:
        sn_conf_mean = float(compute_snr(y_conf_vals, sn_tipo, nominal_target))
    except Exception:
        sn_conf_mean = float("nan")

    # Usa a sua função _predict_combo (ela já existe no seu código)
    try:
        y_pred, sn_pred = _predict_combo(
            conf_levels,
            mean_y=mean_y,
            df_effects=df_effects,
            sn_col=sn_col,
            per_factor_tables=per_factor_tables,
            factor_cols=factor_cols,
            df_plan=df_plan,
        )
        Y_hat_conf = float(y_pred)
        eta_hat_conf = float(sn_pred)
    except Exception as e:
        Y_hat_conf = float("nan")
        eta_hat_conf = float("nan")
        st.warning(f"Não foi possível calcular as previsões do ensaio de confirmação: {e}")

    err_y = abs(y_conf_mean - Y_hat_conf) if np.isfinite(Y_hat_conf) else float("nan")
    err_rel_y = (100.0 * err_y / abs(Y_hat_conf)) if (np.isfinite(err_y) and np.isfinite(Y_hat_conf) and Y_hat_conf != 0.0) else float("nan")

    err_sn = abs(sn_conf_mean - eta_hat_conf) if np.isfinite(eta_hat_conf) else float("nan")
    err_rel_sn = (100.0 * err_sn / abs(eta_hat_conf)) if (np.isfinite(err_sn) and np.isfinite(eta_hat_conf) and eta_hat_conf != 0.0) else float("nan")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="text-align:center; margin: 14px 0 8px;">
              <div style="display:inline-block; padding:16px 26px; background:#eff6ff;
                          border-radius:12px; box-shadow:0 3px 12px rgba(0,0,0,0.14);">
                <div style="font-size:17px; color:#1d4ed8; font-weight:700; margin-bottom:6px;">
                  {var_label}: Média observada vs Média Predita
                </div>
                <div style="font-size:15px; color:#1f2937; margin-bottom:6px; line-height:1.35;">
                  Média observada: <strong style="font-size:17px;">{("n/d" if np.isnan(y_conf_mean) else f"{y_conf_mean:.4f}")}</strong><br/>
                  Predito: <strong style="font-size:17px;">{("n/d" if np.isnan(Y_hat_conf) else f"{Y_hat_conf:.4f}")}</strong>
                </div>
                <div style="font-size:15px; color:#374151; line-height:1.35;">
                  Erro absoluto: <strong style="font-size:17px;">{("n/d" if np.isnan(err_y) else f"{err_y:.4f}")}</strong><br/>
                  Erro relativo: <strong style="font-size:17px;">{("n/d" if np.isnan(err_rel_y) else f"{err_rel_y:.2f}%")}</strong>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; margin: 14px 0 8px;">
              <div style="display:inline-block; padding:16px 26px; background:#eff6ff;
                          border-radius:12px; box-shadow:0 3px 12px rgba(0,0,0,0.14);">
                <div style="font-size:17px; color:#1d4ed8; font-weight:700; margin-bottom:6px;">
                  S/N (dB) observado vs S/N Predito
                </div>
                <div style="font-size:15px; color:#1f2937; margin-bottom:6px; line-height:1.35;">
                  S/N observado: <strong style="font-size:17px;">{("n/d" if np.isnan(sn_conf_mean) else f"{sn_conf_mean:.4f} dB")}</strong><br/>
                  Predito: <strong style="font-size:17px;">{("n/d" if np.isnan(eta_hat_conf) else f"{eta_hat_conf:.4f} dB")}</strong>
                </div>
                <div style="font-size:15px; color:#374151; line-height:1.35;">
                  Erro absoluto: <strong style="font-size:17px;">{("n/d" if np.isnan(err_sn) else f"{err_sn:.4f} dB")}</strong><br/>
                  Erro relativo: <strong style="font-size:17px;">{("n/d" if np.isnan(err_rel_sn) else f"{err_rel_sn:.2f}%")}</strong>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")


# =========================
# Seção persistente de Resultados (compacta e modular)
# =========================
def section_results():

    # Se ainda não existe matriz experimental, sai da função
    if st.session_state.get("df_experimentos") is None:
        return

    df_plan = st.session_state["df_experimentos"]
    df_cod = st.session_state.get("df_experimentos_cod")
    var_label = st.session_state.get("var_label", "Variável")
    matriz_selecionada = st.session_state.get("matriz_selecionada", "OA")

    # ======================================================
    # DOWNLOAD DA MATRIZ
    # ======================================================
    st.subheader("📊 Matriz Experimental Gerada")
    st.dataframe(df_plan, use_container_width=True, hide_index=True)

    @st.cache_data
    def convert_df(_df):
        return _df.to_csv(index=False, sep=";").encode("utf-8")

    st.download_button(
        "📥 Baixar Matriz Experimental (CSV)",
        data=convert_df(df_plan),
        file_name=f"matriz_experimental_{matriz_selecionada}.csv",
        mime="text/csv",
    )

    # =============================================
    # 🔤 Idioma para os rótulos dos gráficos
    # =============================================
    lang = st.radio(
        "Idioma / Language para os rótulos dos gráficos:",
        options=["Português", "English"],
        index=0,
        horizontal=True,
        key="lang_taguchi_plots",
    )

    if lang == "Português":
        main_x_tmpl = "Níveis de {fator}"
        main_y_default = "S/N médio (dB)"
        inter_x_tmpl = "Níveis de {fac_x}"
        inter_y_default = "S/N médio (dB)"
        surf_x_tmpl = "Níveis de {fx}"
        surf_y_tmpl = "Níveis de {fy}"
        surf_z_tmpl = "Média de {var_label}"
    else:
        main_x_tmpl = "Levels of {fator}"
        main_y_default = "Average S/N (dB)"
        inter_x_tmpl = "Levels of {fac_x}"
        inter_y_default = "Average S/N (dB)"
        surf_x_tmpl = "Levels of {fx}"
        surf_y_tmpl = "Levels of {fy}"
        surf_z_tmpl = "Mean of {var_label}"

    
    st.markdown("---")

        # ======================================================
    # Função 1 — Upload dos Resultados
    # ======================================================
    def upload_resultados():
        st.subheader("📤 Upload de Resultados Experimentais (Réplicas/triplicatas)")
        var_label_local = st.session_state.get("var_label", "Variável de Interesse")

        # Tipo de razão S/N (igual ao app_regressao)
        sn_tipo = st.selectbox(
            "Tipo de razão Sinal-Ruído (S/N) (Taguchi)",
            options=["Maior é melhor", "Menor é melhor", "Nominal é melhor"],
            index=0,
        )
        alvo_nominal = None
        if sn_tipo == "Nominal é melhor":
            alvo_nominal = st.number_input("Alvo (m)", value=0.0, help="Para Nominal é melhor")

        # Fórmula em LaTeX (igual ao app_regressao)
        sn_formulas = {
            "Maior é melhor":  r"S/N = -10 \log_{10} \left( \dfrac{1}{n} \sum_{i=1}^{n} \dfrac{1}{y_i^{2}} \right)",
            "Menor é melhor":  r"S/N = -10 \log_{10} \left( \dfrac{1}{n} \sum_{i=1}^{n} y_i^{2} \right)",
            "Nominal é melhor": r"S/N = 10 \log_{10} \left( \dfrac{m^{2}}{s^{2}} \right) \quad (m = \bar{y} \text{ se alvo não informado})"
        }
        st.markdown("**Fórmula da Razão Sinal-Ruído (S/N) selecionada:**")
        st.latex(sn_formulas[sn_tipo])

        # Upload do arquivo de resultados
        upl = st.file_uploader(
            "**Carregar arquivo de resultados (réplicas do experimento)**",
            type=["xlsx", "csv"],
            key="resultados_upl",
        )
        if not upl:
            return None, None, sn_tipo, alvo_nominal

        # --------- Leitura ---------
        if upl.name.endswith(".csv"):
            df_resultados = pd.read_csv(upl, sep=";")
        else:
            df_resultados = pd.read_excel(upl)

        # Padroniza nome da coluna "Experimento"
        exp_col = None
        for c in df_resultados.columns:
            if str(c).strip().lower() in {"experimento", "experiments", "exp", "run"}:
                exp_col = c
                break

        if exp_col is None:
            st.error("❌ O arquivo de resultados precisa ter a coluna 'Experimento'.")
            return None, None, sn_tipo, alvo_nominal

        df_res = df_resultados.copy()
        df_res.rename(columns={exp_col: "Experimento"}, inplace=True)

        # Colunas numéricas (réplicas)
        num_cols = [
            c for c in df_res.columns
            if c != "Experimento" and pd.api.types.is_numeric_dtype(df_res[c])
        ]
        if len(num_cols) == 0:
            st.error("❌ Nenhuma coluna numérica de resposta encontrada.")
            return None, None, sn_tipo, alvo_nominal

        # ======= Validações =======
        n_exp_plan = len(df_plan)
        n_exp_res = df_res["Experimento"].nunique()
        dups = df_res["Experimento"][df_res["Experimento"].duplicated()].unique()
        if len(dups) > 0:
            st.error(f"❌ Há experimentos repetidos: {sorted(dups)}")
            return None, None, sn_tipo, alvo_nominal

        if n_exp_res != n_exp_plan:
            st.error(
                f"❌ Resultados possuem {n_exp_res} experimentos; plano tem {n_exp_plan}."
            )
            return None, None, sn_tipo, alvo_nominal

        esperados = set(range(1, n_exp_plan + 1))
        presentes = set(df_res["Experimento"])
        faltando = sorted(esperados - presentes)
        if faltando:
            st.error(f"❌ Faltando experimentos: {faltando}")
            return None, None, sn_tipo, alvo_nominal

        # Mensagens de sucesso + mostra a matriz de resultados carregada
        st.success("✅ Número de experimentos confere com a matriz experimental!")
        st.success("✅ Arquivo de resultados carregado com sucesso!")
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        st.markdown("---")

        # Join com o plano (df_plan) — matriz combinada
        df_join = df_plan.merge(df_res, on="Experimento", how="left")

        return df_join, num_cols, sn_tipo, alvo_nominal

    # 🔹 Aqui era onde estava faltando a chamada:
    df_join, num_cols, sn_tipo, alvo_nominal = upload_resultados()

    # Se o usuário ainda não fez upload, para por aqui
    if df_join is None:
        return

    # ======================================================
    # Função 2 — Cálculo de médias e S/N
    # ======================================================
    def calcular_sn():
        reps = df_join[num_cols].to_numpy(dtype=float)

        # Médias Y
        mean_y = np.nanmean(reps, axis=1)

        # Desvios
        std_y = np.nanstd(reps, axis=1, ddof=1)

        # --- Funções S/N ---
        def sn_larger(vals):
            return -10 * np.log10(np.mean(1.0 / (vals**2)))

        def sn_smaller(vals):
            return -10 * np.log10(np.mean(vals**2))

        def sn_nominal(vals, target):
            if len(vals) < 2:
                return np.nan
            return 10 * np.log10((target**2) / np.var(vals, ddof=1))

        SNR = []
        for row in reps:
            vals = row[~np.isnan(row)]
            if sn_tipo == "Maior é melhor":
                SNR.append(sn_larger(vals))
            elif sn_tipo == "Menor é melhor":
                SNR.append(sn_smaller(vals))
            else:
                SNR.append(sn_nominal(vals, alvo_nominal))

        df_local = df_join.copy()
        df_local["_Ymean"] = mean_y
        df_local["_SN"] = SNR

        return df_local, mean_y, SNR

    df_join, mean_y, SNR = calcular_sn()



    # Nada a fazer ainda
    if df_join is None:
        return

    # ======================================================
    # Função 2 — Cálculo de médias e S/N
    # ======================================================
    def calcular_sn():
        # usa o df_join do escopo externo apenas para leitura
        reps = df_join[num_cols].to_numpy(dtype=float)

        # Médias Y
        mean_y = np.nanmean(reps, axis=1)

        # Desvios
        std_y = np.nanstd(reps, axis=1, ddof=1)

        # --- Funções S/N ---
        def sn_larger(vals):
            return -10 * np.log10(np.mean(1.0 / (vals**2)))

        def sn_smaller(vals):
            return -10 * np.log10(np.mean(vals**2))

        def sn_nominal(vals, target):
            if len(vals) < 2:
                return np.nan
            return 10 * np.log10((target**2) / np.var(vals, ddof=1))

        SNR = []
        for row in reps:
            vals = row[~np.isnan(row)]
            if sn_tipo == "Maior é melhor":
                SNR.append(sn_larger(vals))
            elif sn_tipo == "Menor é melhor":
                SNR.append(sn_smaller(vals))
            else:
                SNR.append(sn_nominal(vals, alvo_nominal))

        # 🔹 trabalha em uma cópia, não no df_join “de fora”
        df_local = df_join.copy()
        df_local["_Ymean"] = mean_y
        df_local["_SN"] = SNR

        return df_local, mean_y, SNR


    df_join, mean_y, SNR = calcular_sn()

        # ======================================================
    # Resumo: resultado por ensaio + médias globais
    # (comportamento similar ao app_regressao)
    # ======================================================
    st.markdown("### 📊 Resultado por ensaio")

    sn_table = pd.DataFrame({
        "Experimento": df_plan["Experimento"],
        f"Média de {var_label}": mean_y.astype(float),
        f"S/N das réplicas ({var_label}) [dB]": SNR,
    })

    st.dataframe(sn_table, use_container_width=True, hide_index=True)

    # Médias globais
    Y_bar = float(np.nanmean(mean_y))
    SN_bar = float(np.nanmean(SNR))

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            label=f"Média global de {var_label}",
            value=f"{Y_bar:.3f}",
        )
    with c2:
        st.metric(
            label="Média global de S/N (réplicas)",
            value=f"{SN_bar:.3f} dB",
        )

    st.markdown("---")


    # ======================================================
    # Pré-cálculo global para todas as funções da aba
    # ======================================================
    factor_cols = [c for c in df_plan.columns if c != "Experimento"]
    
    per_factor = {}     # S/N médio por nível
    per_factor_Y = {}   # Y médio por nível
    grand_mean = float(np.mean(SNR))
    
    for f in factor_cols:
        df_tmp = df_join.copy()
        df_tmp[f] = df_tmp[f].astype(str)
    
        # S/N MÉDIO POR NÍVEL
        g_sn = df_tmp.groupby(f)["_SN"].mean()
        per_factor[f] = g_sn.to_frame("S/N médio")
    
        # Y MÉDIO POR NÍVEL
        g_y = df_tmp.groupby(f)["_Ymean"].mean()
        per_factor_Y[f] = g_y.to_dict()


    # ======================================================
    # Função 3 — Efeitos + Tabelas (SEM gráficos aqui)
    # ======================================================
    def mostrar_efeitos_e_graficos(lang, main_x_tmpl, main_y_default):
        st.subheader("📈 Efeitos principais na razão S/N (médias por nível)")

        # 🔀 Toggle com a explicação do efeito (igual ao app_regressao)
        if st.toggle("🔴🔴🔴 O que é o 'efeito'? (clique para ver) 🔴🔴🔴",
                     value=False,
                     key="show_efeito"):
            st.markdown(
                r"""
                O **efeito** de um fator $k$ no nível $\ell$ é definido como o desvio
                da resposta média da razão Sinal-Ruído (S/N), obtida nesse nível específico,
                em relação à média global do experimento. Em outros termos, para cada
                **fator** denotado por $k$ e cada **nível** $\ell$ desse fator,
                define-se o efeito como a diferença entre a média de S/N nesse nível
                e a média global:
                """
            )
            st.latex(
                r"\text{Efeito}(k,\ell)=\overline{\mathrm{S/N}}_{k,\ell}"
                r"-\overline{\mathrm{S/N}}_{\text{global}}"
            )
            st.markdown(
                r"""
                **em que,**  
                • $k \in \{1,\dots,K\}$ é o índice do fator (ex.: Temperatura, Pressão, ...),
                  sendo $K$ o número total de fatores.  

                • $\ell \in \{1,\dots,L_k\}$ representa o índice do nível do fator $k$,
                  sendo $L_k$ o número de níveis do respectivo fator. 

                • $\overline{\mathrm{S/N}}_{k,\ell}$: média da razão Sinal-Ruído considerando
                  apenas os ensaios em que o fator $k$ foi fixado no nível $\ell$.

                • $\overline{\mathrm{S/N}}_{\text{global}}$: média da razão Sinal-Ruído
                  considerando todos os ensaios do experimento.
                """
            )

        st.markdown("---")

        # Vamos usar o df_join com a coluna "_SN" (S/N das réplicas)
        df_effects = df_join.copy()
        sn_col = "_SN"

        # Tabelas formatadas por fator (para exibição)
        per_factor_tables = {}

        for fac in factor_cols:
            # níveis como string, ordenados naturalmente (1,2,3,...)
            lvls_in_plan = df_plan[fac].astype(str).unique().tolist()
            try:
                order_nat = sorted(lvls_in_plan, key=lambda s: int(s))
            except Exception:
                order_nat = sorted(lvls_in_plan)

            tmp = df_effects.copy()
            tmp[fac] = tmp[fac].astype(str)

            g = (
                tmp
                .groupby(fac, as_index=True)[sn_col]
                .mean()
                .reindex(order_nat)
            )

            fac_df = (
                pd.DataFrame({"Nível": g.index, "S/N médio (dB)": g.values})
                .reset_index(drop=True)
            )

            # Garante numérico e calcula Efeito (dB)
            fac_df["S/N médio (dB)"] = pd.to_numeric(
                fac_df["S/N médio (dB)"], errors="coerce"
            )
            fac_df["Efeito (dB)"] = fac_df["S/N médio (dB)"] - float(grand_mean)
            fac_df[["S/N médio (dB)", "Efeito (dB)"]] = fac_df[
                ["S/N médio (dB)", "Efeito (dB)"]
            ].round(3)

            per_factor_tables[fac] = fac_df

        st.markdown("🔍 Tabelas por fator (S/N médio por nível)")
        # Renderiza as tabelas — até 4 fatores por linha
        COLS_PER_ROW = 4
        for i in range(0, len(factor_cols), COLS_PER_ROW):
            bloco = factor_cols[i:i + COLS_PER_ROW]
            cols = st.columns(len(bloco))
            for j, fac in enumerate(bloco):
                with cols[j]:
                    st.markdown(f"**Fator: {fac}**")
                    st.dataframe(
                        per_factor_tables[fac],
                        use_container_width=True,
                        hide_index=True,
                    )

        # ============================
        # 📥 Baixar tabelas por fator (CSV único)
        # ============================
        if per_factor_tables:
            df_emp = pd.concat(
                [df.assign(**{"Fator": f}) for f, df in per_factor_tables.items()],
                ignore_index=True
            )
            st.download_button(
                "📥 Baixar tabelas por fator (CSV)",
                data=df_emp.to_csv(index=False).encode("utf-8"),
                file_name=f"efeitos_sn_por_fator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_efeitos_fator_csv",
            )

        st.markdown("---")

        # Mantém o mesmo retorno de antes (para compatibilidade)
        return per_factor, grand_mean, factor_cols


    # ======================================================
    # Gráficos de efeitos médios por fator (para aba 2D)
    # ======================================================


    def mostrar_interacoes(lang, inter_x_tmpl, inter_y_default):
        # Precisa de pelo menos 2 fatores
        if len(factor_cols) < 2:
            st.info("São necessários pelo menos dois fatores para visualizar interações.")
            return

        # -------------------------------------------------
        # 📈 Efeitos médios — gráficos por fator (estilo Minitab)
        # -------------------------------------------------
        df_effects = df_join.copy()
        sn_col = "_SN"

        # média global do S/N das réplicas
        grand_mean = df_effects[sn_col].mean()

        # Tabelas por fator apenas para alimentar os gráficos
        per_factor_tables = {}
        for fac in factor_cols:
            lvls_in_plan = df_plan[fac].astype(str).unique().tolist()
            try:
                order_nat = sorted(lvls_in_plan, key=lambda s: int(s))
            except Exception:
                order_nat = sorted(lvls_in_plan)

            tmp = df_effects.copy()
            tmp[fac] = tmp[fac].astype(str)

            g = (
                tmp
                .groupby(fac, as_index=True)[sn_col]
                .mean()
                .reindex(order_nat)
            )

            fac_df = (
                pd.DataFrame({"Nível": g.index, "S/N médio (dB)": g.values})
                .reset_index(drop=True)
            )
            per_factor_tables[fac] = fac_df

        # Rótulos em função do idioma
                # -------------------------------------------------
        # Escolha de idioma SÓ para os gráficos de efeitos médios
        # (independente do idioma usado nos gráficos de interação)
        # -------------------------------------------------
        lang_effects = st.radio(
            "Idioma / Language (efeitos médios):",
            ["Português", "English"],
            index=0,
            horizontal=True,
            key="lang_effects_2d",
        )

        if lang_effects == "Português":
            y_label_factors = "S/N médio (dB)"
            x_label_factors = "Níveis dos parâmetros"
            global_mean_label = "Média global"
            hover_template = "Nível=%{x}<br>S/N=%{y:.3f} dB<extra></extra>"
        else:
            y_label_factors = "Average S/N (dB)"
            x_label_factors = "Parameter levels"
            global_mean_label = "Overall mean"
            hover_template = "Level=%{x}<br>S/N=%{y:.3f} dB<extra></extra>"


        st.subheader("📊 Efeitos médios — gráficos por fator")

        # Até 4 gráficos por linha
        MAX_COLS = 4
        cols = MAX_COLS if len(factor_cols) >= MAX_COLS else (len(factor_cols) if len(factor_cols) > 0 else 1)
        rows = math.ceil(len(factor_cols) / cols) if len(factor_cols) > 0 else 1
        fig_all = make_subplots(rows=rows, cols=cols, subplot_titles=factor_cols)

        # Mesma escala Y em todos os subplots (inclui a média global)
        all_y = []
        for _fac in factor_cols:
            _df = per_factor_tables[_fac].copy().reset_index(drop=True)
            all_y.extend(_df["S/N médio (dB)"].astype(float).tolist())
        if not math.isnan(grand_mean):
            all_y.append(float(grand_mean))

        if len(all_y) > 0:
            ymin, ymax = min(all_y), max(all_y)
            pad = 0.1 * (ymax - ymin if ymax > ymin else (abs(ymax) if ymax != 0 else 1.0))
            y_range = [ymin - pad, ymax + pad]
        else:
            y_range = None

        r, c = 1, 1
        for fac in factor_cols:
            fac_df = per_factor_tables[fac].copy().reset_index(drop=True)

            num_levels = len(fac_df)
            x_cat = [str(i) for i in range(1, num_levels + 1)]
            y_vals = fac_df["S/N médio (dB)"].astype(float).tolist()

            # Curva do fator
            fig_all.add_trace(
                go.Scatter(
                    x=x_cat,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{fac}",
                    showlegend=False,
                    hovertemplate=hover_template,
                ),
                row=r, col=c
            )


            # Linha da média global em TODOS os subplots
            if not math.isnan(grand_mean):
                fig_all.add_trace(
                    go.Scatter(
                        x=x_cat,
                        y=[grand_mean] * len(x_cat),
                        mode="lines",
                        name=global_mean_label,
                        line=dict(dash="dash"),
                        showlegend=(r == 1 and c == 1),
                        hovertemplate=f"{global_mean_label}=%{{y:.3f}} dB<extra></extra>",
                    ),
                    row=r, col=c
                )

            # Eixo Y só com rótulo no 1º subplot; todos com mesmo range
            if r == 1 and c == 1:
                fig_all.update_yaxes(title_text=y_label_factors, range=y_range, row=r, col=c)
            else:
                fig_all.update_yaxes(title_text=None, range=y_range, row=r, col=c)

            # X categórico e título
            fig_all.update_xaxes(
                title_text=x_label_factors,
                type="category",
                tickmode="array",
                tickvals=x_cat,
                ticktext=x_cat,
                categoryorder="category ascending",
                row=r, col=c
            )

            # avança colunas
            c += 1
            if c > cols:
                c = 1
                r += 1

        fig_all.update_layout(height=280 * rows, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_all, use_container_width=True)

        # -----------------------------
        # 📥 Downloads (cores / P&B)
        # -----------------------------
        st.markdown("📄 Baixar figura")
        color_mode = st.radio(
            "Modo de cores para exportação:",
            ["Cores (original)", "Preto e branco"],
            index=0,
            help="A visualização na tela permanece em cores. A opção afeta apenas os arquivos baixados.",
            key="color_mode_2d_effects",
        )

        # Cópia para exportação
        fig_exp = go.Figure(fig_all.to_dict())

        rows = math.ceil(len(factor_cols) / cols) if len(factor_cols) > 0 else 1
        export_width = 1100
        export_height = 320 * rows + 80

        fig_exp.update_layout(
            width=export_width,
            height=export_height,
            margin=dict(l=70, r=30, t=60, b=70),
            paper_bgcolor="white",
            plot_bgcolor="white",
            template="plotly_white",
        )

        if color_mode == "Preto e branco":
            dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
            t_idx = 0
            for tr in fig_exp.data:
                if isinstance(tr, go.Scatter):
                    is_global_mean = (getattr(tr, "name", "") == global_mean_label) or (
                        hasattr(tr, "hovertemplate") and global_mean_label in str(tr.hovertemplate)
                    )
                    tr.update(
                        line=dict(
                            color="black",
                            width=2,
                            dash=("dot" if is_global_mean else dash_cycle[t_idx % len(dash_cycle)]),
                        ),
                        marker=dict(color="black", size=7),
                    )
                    if not is_global_mean:
                        t_idx += 1

        def _export_bytes(fmt: str):
            try:
                return fig_exp.to_image(
                    format=fmt,
                    scale=2,
                    width=export_width,
                    height=export_height,
                )
            except Exception:
                st.warning(
                    "Para exportar imagens, é necessário o pacote **kaleido**.\n\n"
                    "Instale com:\n\n"
                    "`pip install -U kaleido`\n\n"
                    "ou\n\n"
                    "`conda install -c conda-forge python-kaleido -y`"
                )
                raise

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📥 Gerar PNG"):
                try:
                    png_bytes = _export_bytes("png")
                    st.download_button(
                        "Baixar PNG",
                        data=png_bytes,
                        file_name="efeitos_medios_todos_fatores.png",
                        mime="image/png",
                    )
                except Exception:
                    pass

        with col2:
            if st.button("📥 Gerar SVG"):
                try:
                    svg_bytes = _export_bytes("svg")
                    st.download_button(
                        "Baixar SVG",
                        data=svg_bytes,
                        file_name="efeitos_medios_todos_fatores.svg",
                        mime="image/svg+xml",
                    )
                except Exception:
                    pass

        
        with col3:
            if st.button("📥 Gerar PDF"):
                try:
                    pdf_bytes = _export_bytes("pdf")
                    st.download_button(
                        "Baixar PDF",
                        data=pdf_bytes,
                        file_name="efeitos_medios_todos_fatores.pdf",
                        mime="application/pdf",
                    )
                except Exception:
                    pass

        with col4:
            if st.button("📥 Gerar HTML interativo"):
                html_bytes = pio.to_html(
                    fig_all, include_plotlyjs="cdn", full_html=False
                ).encode("utf-8")
                st.download_button(
                    "Baixar HTML",
                    data=html_bytes,
                    file_name="efeitos_medios_todos_fatores.html",
                    mime="text/html",
                )


        # -------------------------------------------------
        # 🔗 Gráficos de interação entre fatores (S/N)
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("🔗 Gráficos de interação entre fatores (S/N)")

        # Idioma específico desta seção
        lang_int = st.radio(
            "Idioma / Language (interações):",
            ["Português", "English"],
            index=0,
            horizontal=True,
            key="lang_inter_2d",
        )

        if lang_int == "Português":
            cap_text = (
                "Selecione um fator para o eixo X e outro para formar as curvas. "
                "O gráfico mostra a S/N média para cada combinação de níveis."
            )
            x_label_tpl = "Níveis de {fac}"
            y_label_default = "S/N médio (dB)"
            global_mean_label = "Média global (S/N)"
            hover_sn = "S/N médio=%{y:.3f} dB"
        else:
            cap_text = (
                "Select a factor for the X-axis and another to form the curves. "
                "The plot shows the mean S/N for each combination of factor levels."
            )
            x_label_tpl = "Levels of {fac}"
            y_label_default = "Average S/N (dB)"
            global_mean_label = "Overall mean (S/N)"
            hover_sn = "Mean S/N=%{y:.3f} dB"

        if len(factor_cols) < 2:
            st.info("São necessários pelo menos dois fatores para visualizar interações.")
            return

        st.caption(cap_text)

        # Escolha dos fatores
        fac_x = st.selectbox(
            "Fator no eixo X / X-axis factor:",
            factor_cols,
            index=0,
            key="inter_x",
        )

        fac_lines = st.selectbox(
            "Fator para as curvas / Line factor:",
            [f for f in factor_cols if f != fac_x],
            index=0,
            key="inter_lines",
        )

        # Prepara dados (garante string)
        tmp_int = df_join.copy()
        tmp_int[fac_x] = tmp_int[fac_x].astype(str)
        tmp_int[fac_lines] = tmp_int[fac_lines].astype(str)

        # S/N médio por combinação (fac_x, fac_lines)
        mean_inter = (
            tmp_int
            .groupby([fac_x, fac_lines])["_SN"]
            .mean()
            .reset_index()
        )

        # Média global de S/N na interação
        grand_mean_int = float(mean_inter["_SN"].mean())

        # Ordenação "natural" 1,2,3,...
        def _nat_sort(vals):
            try:
                return sorted(vals, key=lambda v: int(v))
            except Exception:
                return sorted(vals)

        x_levels = _nat_sort(mean_inter[fac_x].unique().tolist())
        line_levels = _nat_sort(mean_inter[fac_lines].unique().tolist())

        # Figura de interação
        fig_int = go.Figure()

        # 🔹 Símbolos para distinguir curvas (colorblind-friendly)
        marker_symbols = [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-up",
            "triangle-down",
            "triangle-left",
            "triangle-right",
            "star",
            "hexagon",
            "hexagon2",
            "pentagon",
        ]

        for idx, lvl in enumerate(line_levels):
            df_line = mean_inter[mean_inter[fac_lines] == lvl]
            y_vals = []
            for xv in x_levels:
                sub = df_line[df_line[fac_x] == xv]
                y_vals.append(
                    float(sub["_SN"].iloc[0]) if not sub.empty else float("nan")
                )

            symbol = marker_symbols[idx % len(marker_symbols)]

            fig_int.add_trace(
                go.Scatter(
                    x=x_levels,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{fac_lines} = {lvl}",
                    marker=dict(
                        symbol=symbol,
                        size=9,
                        line=dict(width=1),
                    ),
                    line=dict(width=2),
                    hovertemplate=(
                        f"{fac_x}=%{{x}}<br>"
                        f"{fac_lines}={lvl}<br>"
                        f"{hover_sn}<extra></extra>"
                    ),
                )
            )

        # 👉 Reta da média global
        if not math.isnan(grand_mean_int):
            fig_int.add_trace(
                go.Scatter(
                    x=x_levels,
                    y=[grand_mean_int] * len(x_levels),
                    mode="lines",
                    name=global_mean_label,
                    line=dict(dash="dash"),
                    hovertemplate=f"{global_mean_label}=%{{y:.3f}} dB<extra></extra>",
                )
            )

        # Rótulos padrão (podem ser editados abaixo)
        default_x_label = x_label_tpl.format(fac=fac_x)
        default_y_label = y_label_default

        c_ax1, c_ax2 = st.columns(2)
        with c_ax1:
            x_label = st.text_input(
                "Rótulo eixo X (interação) / X-axis label (interaction):",
                default_x_label,
                key="x_label_inter",
            )
        with c_ax2:
            y_label = st.text_input(
                "Rótulo eixo Y (interação) / Y-axis label (interaction):",
                default_y_label,
                key="y_label_inter",
            )

        fig_int.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            hovermode="x",
            margin=dict(l=10, r=10, t=40, b=40),
        )

        st.plotly_chart(fig_int, use_container_width=True)

        # =============================================
        # 📄 Baixar gráfico de interação (leve)
        # =============================================
        st.markdown("### 📄 Baixar gráfico de interação")

        color_mode_int = st.radio(
            "Modo de cores para exportação:",
            ["Cores (original)", "Preto e branco"],
            index=0,
            key="color_mode_interaction",
            help="A visualização na tela permanece em cores. A opção afeta apenas os arquivos baixados.",
        )

        # Cópia para exportação
        fig_exp_int = go.Figure(fig_int.to_dict())

        export_width_int = 900
        export_height_int = 600

        fig_exp_int.update_layout(
            width=export_width_int,
            height=export_height_int,
            margin=dict(l=70, r=40, t=60, b=70),
            paper_bgcolor="white",
            plot_bgcolor="white",
            template="plotly_white",
        )

        # Preto e branco (opcional)
        if color_mode_int == "Preto e branco":
            dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
            t_idx = 0
            for tr in fig_exp_int.data:
                if isinstance(tr, go.Scatter):
                    tr.update(
                        line=dict(
                            color="black",
                            width=2,
                            dash=dash_cycle[t_idx % len(dash_cycle)],
                        ),
                        marker=dict(color="black", size=7),
                    )
                    t_idx += 1

        # Função auxiliar para exportar (gerada SÓ quando o usuário clicar)
        def _export_bytes_int(fmt: str):
            try:
                return fig_exp_int.to_image(
                    format=fmt,
                    scale=2,
                    width=export_width_int,
                    height=export_height_int,
                )
            except Exception:
                st.warning(
                    "Para exportar imagens, é necessário o pacote **kaleido**.\n\n"
                    "Instale com:\n\n"
                    "`pip install -U kaleido`\n\n"
                    "ou\n\n"
                    "`conda install -c conda-forge python-kaleido -y`"
                )
                raise

        col_i1, col_i2, col_i3, col_i4 = st.columns(4)

        with col_i1:
            if st.button("📥 Gerar PNG", key="btn_int_png"):
                try:
                    png_bytes_int = _export_bytes_int("png")
                    st.download_button(
                        "Baixar PNG",
                        data=png_bytes_int,
                        file_name="grafico_interacao_SN.png",
                        mime="image/png",
                        key="dl_int_png",
                    )
                except Exception:
                    pass

        with col_i2:
            if st.button("📥 Gerar SVG", key="btn_int_svg"):
                try:
                    svg_bytes_int = _export_bytes_int("svg")
                    st.download_button(
                        "Baixar SVG (vetorial)",
                        data=svg_bytes_int,
                        file_name="grafico_interacao_SN.svg",
                        mime="image/svg+xml",
                        key="dl_int_svg",
                    )
                except Exception:
                    pass

        with col_i3:
            if st.button("📥 Gerar PDF", key="btn_int_pdf"):
                try:
                    pdf_bytes_int = _export_bytes_int("pdf")
                    st.download_button(
                        "Baixar PDF",
                        data=pdf_bytes_int,
                        file_name="grafico_interacao_SN.pdf",
                        mime="application/pdf",
                        key="dl_int_pdf",
                    )
                except Exception:
                    pass

        with col_i4:
            if st.button("📥 Gerar HTML", key="btn_int_html"):
                html_bytes_int = pio.to_html(
                    fig_int, include_plotlyjs="cdn", full_html=False
                ).encode("utf-8")
                st.download_button(
                    "Baixar HTML (interativo)",
                    data=html_bytes_int,
                    file_name="grafico_interacao_SN.html",
                    mime="text/html",
                    key="dl_int_html",
                )



    def mostrar_superficie_3d():
        if len(factor_cols) < 2:
            return

        st.subheader("🌐 Superfície de interação — média da razão S/N")


        st.caption( "O eixo Z representa a média da razão sinal-ruído (S/N) "
                    "para cada combinação dos níveis dos dois fatores.")


        fx = st.selectbox(
            "Fator — eixo X:",
            factor_cols
        )
        
        fy = st.selectbox(
            "Fator — eixo Y:",
            [f for f in factor_cols if f != fx]
        )


        df_tmp = df_join.copy()
        df_tmp[fx] = df_tmp[fx].astype(str)
        df_tmp[fy] = df_tmp[fy].astype(str)

        grid = df_tmp.groupby([fx, fy])["_Ymean"].mean().reset_index()

        xs = sorted(grid[fx].unique(), key=lambda z: int(z))
        ys = sorted(grid[fy].unique(), key=lambda z: int(z))

        Z = np.zeros((len(ys), len(xs)))
        for i, yv in enumerate(ys):
            for j, xv in enumerate(xs):
                val = grid[(grid[fx] == xv) & (grid[fy] == yv)]["_Ymean"]
                Z[i, j] = float(val)

        fig = go.Figure(data=[go.Surface(
            x=list(range(len(xs))),
            y=list(range(len(ys))),
            z=Z,
            colorscale="turbo",  # mais nítida que Viridis em muitos monitores
            opacity=0.9,
            contours=dict(z=dict(show=True, project_z=False))  # linhas de nível
        )])

        
        xv = np.arange(len(xs))
        yv = np.arange(len(ys))
        X, Y = np.meshgrid(xv, yv)
        
        z0 = float(np.nanmin(Z))  # plano XY
        n_levels = 8
        
        levels = np.linspace(np.nanmin(Z), np.nanmax(Z), n_levels)
        
        cs = plt.contour(X, Y, Z, levels=levels)
        plt.close()
        
        cmin = float(np.nanmin(Z))
        cmax = float(np.nanmax(Z))
        cmid = float(np.nanmean(Z))   # ou np.nanmedian(Z)
        
        fig.add_trace(go.Surface(
            x=xv,
            y=yv,
            z=Z,
            colorscale="turbo",
            cmin=cmin,
            cmax=cmax,
            cmid=cmid,
            opacity=0.95,
            showscale=True
        ))

        
        # Curvas de nível no plano XY
        for level_segs in cs.allsegs:
            for seg in level_segs:
                fig.add_trace(go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=np.full(seg.shape[0], z0),
                    mode="lines",
                    line=dict(color="darkgreen", width=3, dash="dash"), 
                    showlegend=False
                ))


        lang_surface_3d = st.radio(
            "Idioma / Language (superfície 3D):",
            ["Português", "English"],
            index=0,
            horizontal=True,
            key="lang_surface_3d",
        )
        
        if lang_surface_3d == "Português":
            zaxis_label = "Média da razão S/N"
        else:
            zaxis_label = "Mean S/N ratio"
        
                        
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=fx,
                    ticktext=xs,
                    tickvals=list(range(len(xs)))
                ),
                yaxis=dict(
                    title=fy,
                    ticktext=ys,
                    tickvals=list(range(len(ys)))
                ),
                zaxis_title=zaxis_label,   # <-- aqui
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        st.plotly_chart(fig, use_container_width=True)


        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        
        with col_i1:
            if st.button("📥 Gerar PNG", key="btn_surf3d_png"):
                try:
                    png_bytes = _export_bytes_int(fig, "png")
                    st.download_button(
                        "Baixar PNG",
                        data=png_bytes,
                        file_name="superficie_3d_interacao_SN.png",
                        mime="image/png",
                        key="dl_surf3d_png",
                    )
                except Exception:
                    pass
        
        with col_i2:
            if st.button("📥 Gerar SVG", key="btn_surf3d_svg"):
                try:
                    svg_bytes = _export_bytes_int(fig, "svg")
                    st.download_button(
                        "Baixar SVG (vetorial)",
                        data=svg_bytes,
                        file_name="superficie_3d_interacao_SN.svg",
                        mime="image/svg+xml",
                        key="dl_surf3d_svg",
                    )
                except Exception:
                    pass
        
        with col_i3:
            if st.button("📥 Gerar PDF", key="btn_surf3d_pdf"):
                try:
                    pdf_bytes = _export_bytes_int(fig, "pdf")
                    st.download_button(
                        "Baixar PDF",
                        data=pdf_bytes,
                        file_name="superficie_3d_interacao_SN.pdf",
                        mime="application/pdf",
                        key="dl_surf3d_pdf",
                    )
                except Exception:
                    pass
        
        with col_i4:
            if st.button("📥 Gerar HTML", key="btn_surf3d_html"):
                html_bytes = pio.to_html(
                    fig, include_plotlyjs="cdn", full_html=False
                ).encode("utf-8")
                st.download_button(
                    "Baixar HTML (interativo)",
                    data=html_bytes,
                    file_name="superficie_3d_interacao_SN.html",
                    mime="text/html",
                    key="dl_surf3d_html",
                )


    

    def mostrar_regra_delta():
        st.subheader("📐 A regra Delta por fator")

        # 🔽 Toggle explicativo (igual ao app_regressao)
        if st.toggle("🔴🔴🔴 O que é o 'Delta'? (clique para ver) 🔴🔴🔴",
                     value=False,
                     key="show_delta"):
            st.markdown(r"""
Em linha gerais, o valor de $\Delta$ fornece uma medida comparativa de influência de cada fator sobre a resposta do problema, sendo que fatores com maiores valores de $\Delta$ são considerados mais relevantes, pois produzem maior variação na razão sinal-ruído média entre seus níveis. Especificamente, para cada fator $k$, o **Delta** $(\Delta_k)$ é dado pela **amplitude** entre a maior e a menor **S/N média** dos seus níveis:
""")
            st.latex(r"\Delta_k = \max_{\ell} \, \overline{\mathrm{S/N}}_{k,\ell} - \min_{\ell} \, \overline{\mathrm{S/N}}_{k,\ell}")

            st.markdown(r"""
**Procedimento de cálculo (passos):**
1. Agrupe a $\mathrm{S/N}$ por **nível** do fator $k$.
2. Calcule a **$\mathrm{S/N}$ média** em cada nível.
3. Identifique **máximo** e **mínimo** dessas médias.
4. Faça $\Delta = \textrm{máx} - \textrm{mín}$ (em dB).

**Interpretação.**
- Valor de $\Delta_k$ grande $\implies$ o fator $k$ **altera fortemente** a resposta (maior influência).
- Valor de $\Delta_k \approx 0$ $\implies$ pouca ou nenhuma influência detectável via $\mathrm{S/N}$.

**Observações rápidas:**
- Válido para qualquer tipo de S/N (maior-melhor, menor-melhor, nominal-melhor).
- Em **empates** de S/N média entre níveis, adote uma regra estável (p.ex., a **ordem natural** dos níveis).
- Ordena fatores por influência (***regra Delta***), porém **não** testa significância.
- Para **significância estatística**, use **ANOVA sobre S/N** em complemento à regra delta.
""")

        st.markdown("---")
        st.markdown("🔍 Tabela de cálculo da regra delta por fator")

        # Reconstrói pequenas tabelas por fator a partir de per_factor
        rows = []
        for fac, g in per_factor.items():
            # g: DataFrame com índice = níveis, coluna "S/N médio"
            s = pd.to_numeric(g["S/N médio"], errors="coerce")
            if s.isna().all():
                rows.append({
                    "Fator": fac,
                    "S/N médio máx. (dB)": float("nan"),
                    "S/N médio mín. (dB)": float("nan"),
                    "Δ (dB)": float("nan"),
                })
                continue

            vmax = float(s.max())
            vmin = float(s.min())
            rows.append({
                "Fator": fac,
                "S/N médio máx. (dB)": round(vmax, 3),
                "S/N médio mín. (dB)": round(vmin, 3),
                "Δ (dB)": round(vmax - vmin, 3),
            })

        delta_simple_df = (
            pd.DataFrame(rows)
            .sort_values("Δ (dB)", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        delta_simple_df["Rank (Δ)"] = np.arange(1, len(delta_simple_df) + 1)

        st.dataframe(delta_simple_df, use_container_width=True, hide_index=True)

        # 📥 Download CSV da regra delta por fator
        buf = io.StringIO()
        delta_simple_df.to_csv(buf, index=False)
        st.download_button(
            "📥 Baixar delta por fator (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"delta_por_fator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="dl_delta_por_fator_csv",
        )


    def tabelas_observado_predito():
        st.subheader("📊 Observado × Predito")

        Y = df_join["_Ymean"].values
        SN = df_join["_SN"].values

        factor_cols_local = [c for c in df_plan.columns if c != "Experimento"]

        Y_bar = np.mean(Y)
        SN_bar = np.mean(SN)

        predY = []
        predSN = []

        # ATENÇÃO: aqui ainda falta você definir per_factor_Y em algum lugar
        for i in range(len(df_plan)):
            somaY = 0
            somaSN = 0
            for fac in factor_cols_local:
                lvl = str(df_join.loc[i, fac])
                somaY += per_factor_Y[fac][lvl]
                somaSN += per_factor[fac].loc[lvl, "S/N médio"]
            predY.append(somaY - (len(factor_cols_local) - 1) * Y_bar)
            predSN.append(somaSN - (len(factor_cols_local) - 1) * SN_bar)

        df_pred = pd.DataFrame({
            "Y_obs": Y,
            "Y_pred": predY,
            "SN_obs": SN,
            "SN_pred": predSN
        })

        st.dataframe(df_pred.round(3))

    def predicao_usuario():
        st.markdown("---")
        st.subheader("🧮 Predição para qualquer combinação")
    
        # ----------------- Entrada do usuário -----------------
        levels = {}
        for f in factor_cols:
            lvls = sorted(df_plan[f].astype(str).unique(), key=lambda z: int(z))
            levels[f] = st.selectbox(f"Nível para {f}", lvls)

        mean_y = df_join["_Ymean"].values  # garanta isso no escopo
    
        # ----------------- Predição S/N (modelo aditivo clássico) -----------------
        SN_bar = np.mean(SNR)
        somaSN = 0.0
    
        for f, lvl in levels.items():
            somaSN += per_factor[f].loc[lvl, "S/N médio"]
    
        pred_sn = somaSN - (len(factor_cols) - 1) * SN_bar
    
        # ----------------- Predição da resposta (se não existir, fica n/d) -----------------
        Y_hat = np.nan          # mantém compatibilidade com o layout
        eta_hat = pred_sn      # apenas alias lógico, sem mudar o cálculo

        # ----------------- Predição da variável de interesse (modelo aditivo) -----------------
        Y_bar = df_join["_Ymean"].mean()  # ou df_plan["_Ymean"].mean(), dependendo de onde está a coluna
        
        somaY = 0.0
        for f, lvl in levels.items():
            # média de Y no nível escolhido (marginalizando os demais fatores)
            somaY += df_join[df_join[f].astype(str) == str(lvl)]["_Ymean"].mean()
        
        Y_hat = somaY - (len(factor_cols) - 1) * Y_bar

    
        # ----------------- Resultados -----------------
        st.markdown("🔍 **Resultados das predições no ponto fornecido pelo usuário**")
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown(
                f"""
                <div style="text-align:center; margin: 14px 0 8px;">
                  <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                              border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                    <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                      Previsão para {var_label}
                    </div>
                    <div style="font-size:26px; font-weight:700; color:#064e3b;">
                      {("n/d" if np.isnan(Y_hat) else f"{Y_hat:.3f}")}
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
        with col2:
            st.markdown(
                f"""
                <div style="text-align:center; margin: 14px 0 8px;">
                  <div style="display:inline-block; padding:12px 22px; background:#ecfdf5;
                              border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,0.12);">
                    <div style="font-size:14px; color:#065f46; font-weight:600; margin-bottom:4px;">
                      Previsão para S/N (dB)
                    </div>
                    <div style="font-size:26px; font-weight:700; color:#064e3b;">
                      {("n/d" if np.isnan(pred_sn) else f"{pred_sn:.3f} dB")}
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

                # =========================
        # 📥 Exportações de predição
        # =========================
        
        # df_effects e sn_col (os mesmos padrões do resto do app)
        df_effects = df_join.copy()
        sn_col = "_SN"
        
        # monta per_factor_tables no formato esperado por _predict_combo()
        per_factor_tables = {}
        for fac in factor_cols:
            tmp = df_effects.copy()
            tmp[fac] = tmp[fac].astype(str)
        
            g = tmp.groupby(fac, as_index=True)[sn_col].mean()
        
            # ordenação natural (1,2,3,...) quando possível
            try:
                order_nat = sorted(g.index.tolist(), key=lambda s: int(s))
            except Exception:
                order_nat = sorted(g.index.tolist())
        
            g = g.reindex(order_nat)
            per_factor_tables[fac] = pd.DataFrame({
                "Nível": g.index.astype(str),
                "S/N médio (dB)": g.values
            })
    
        render_exportacoes_predicao(
            user_levels=levels,
            Y_hat=Y_hat,
            eta_hat=eta_hat,
            var_label=var_label,
            mean_y=mean_y,
            df_effects=df_effects,
            sn_col=sn_col,
            per_factor_tables=per_factor_tables,
            factor_cols=factor_cols,
            df_plan=df_plan,
        )

        
        
        estimativas_ponto_otimo(
            factor_cols=factor_cols,
            df_plan=df_plan,
            per_factor_tables=per_factor_tables,
            var_label=var_label,
            mean_y=mean_y,
            df_effects=df_effects,
            sn_col=sn_col,
            opt_table=opt_table if "opt_table" in globals() else None,
        )


            # ================================================================
        # 🧪 Ensaio de confirmação (comparação Observado × Predito)
        # ================================================================
        nominal_target = alvo_nominal if sn_tipo == "Nominal é melhor" else None


    def anova_taguchi():
        """
        Aba ANOVA – Análise de Variância do Planejamento Experimental (Taguchi).
        Implementação será feita posteriormente.
        """
        st.subheader("📊 ANOVA sobre a razão S/N")

        st.caption(
            "Esta ANOVA é baseada na razão S/N por ensaio, usando apenas efeitos principais. "
            "Ela decompõe a variação total de S/N em parcelas atribuídas a cada fator e ao erro."
        )

        if st.toggle("🔴🔴🔴 O que é esta ANOVA? (clique para ver) 🔴🔴🔴", value=False, key="show_anova_help"):
            st.markdown(r"""
            A ANOVA (Análise de Variância) aqui considera a **razão S/N de cada ensaio** como resposta
            e decompõe a soma de quadrados total em:

            - **Soma de Quadrados do Fator** ($SQ_k$): quanto cada fator $k$ contribui para a variação de S/N;  
            - **Soma de Quadrados de Erro** $(SQ_{\textrm{erro}})$: variação não explicada pelos efeitos principais;  
            - **Soma de Quadrados Total** $(SQ_{\textrm{total}})$: variação total da S/N em torno da média global.

            Como o planejamento é ortogonal, a contribuição de cada fator é calculada por:
            """)
            st.latex(r"""
            SQ_k \;=\; \sum_{\ell} n_{k,\ell}\,\bigl(\overline{\mathrm{S/N}}_{k,\ell}
            - \overline{\mathrm{S/N}}_{\text{global}}\bigr)^2
            """)
            st.markdown(r"""
            em que $\overline{\mathrm{S/N}}_{k,\ell}$ é a média de S/N no nível $\ell$ do fator $k$
            e $n_{k,\ell}$ é o número de ensaios nesse nível.
            """)

            st.markdown(r"""
            A **soma de quadrados total** é dada por:
            """)    
            st.latex(r"""
            SQ_{\text{total}} = \sum_{i=1}^{N} \left(\mathrm{S/N}_i -\overline{\mathrm{S/N}}_{\text{global}}\right)^2
            """)
            
            st.markdown(r"""
            e a **soma de quadrados do erro** é obtida por diferença:
            """)    
            st.latex(r"""
                SQ_{\text{erro}}
                =
                SQ_{\text{total}}
                \;-\;
                \sum_k SQ_k
            """)
            
                
            st.markdown(r"""
                ### 📐 Termos usados na tabela ANOVA
                A tabela exibida pelo aplicativo contém as seguintes colunas: 
                - **GL (graus de liberdade)**  
                  Para um fator com $L$ níveis: $$GL = L - 1$$  
                  Para o erro:  $$GL_{\text{erro}} = GL_{\text{total}} - \sum_k GL_k$$

               - **SQ (Soma de quadrados)**  Quantidade de variação explicada por cada fonte.  

               - **QM (Quadrado Médio)**  É a variância média explicada pela fonte:  
                  $$QM_k = \dfrac{SQ_k}{GL_k}\ \ $$   e   $$\ \ QM_{\text{erro}} = \dfrac{SQ_{\text{erro}}}{GL_{\text{erro}}}$$  

               - **F (estatística F de Fisher)**  Mede o quanto a variância explicada pelo fator excede a variância residual:  $$F_k = \dfrac{QM_k}{QM_{\text{erro}}}$$  

               - **p-valor**  Probabilidade de observar um valor de $F_k$ tão grande assumindo hipótese nula:   $$p_k = \mathbb{P}\left[F_{GL_k,\,GL_{\text{erro}}} \ge F_k\right]$$  
                  Fatores com $p_k < 0{,}05$ são considerados **estatisticamente significativos**.

               - **Contribuição (%)**  Mede a importância relativa do fator na variação total:  
            """)
            st.latex(r"""
                \text{Contribuição}_k(\%) \;=\;
                100 \cdot \frac{SQ_k}{SQ_{\text{total}}}
            """)


            st.markdown(r"""
                ---
                
                ### ⚠️ Quando não existe $SQ_{\textrm{erro}}$?
                
                Em matrizes como **L9 com 4 fatores**, os fatores consomem todos os GL:
                
                $$GL_{\text{erro}} = 0$$
                
                Nesse caso **não é possível calcular**:
                - $QM_{\text{erro}}$  
                - $F$  
                - p-valores  
                
                A ANOVA mostra apenas **SQ** e **GL**, sem testes estatísticos.
            """) 

            st.markdown(r"""
                ### 🔁 Pooling (Agrupamento no Erro)
                Quando $GL_{\text{erro}} = 0$, a ANOVA torna-se estatisticamente indeterminada, pois não é possível calcular $QM_{\text{erro}}$. Para contornar esse problema em planejamentos ortogonais saturados (como L9 com 4 fatores), aplica-se o procedimento conhecido como **pooling**.
                
                Nesse procedimento, fatores cuja contribuição é considerada pequena são tratados como fontes de variação não sistemática. Assim, seus termos são incorporados ao termo de erro, redefinindo:
                
                $$
                SQ_{\text{erro}}^{(\text{pool})}
                = SQ_{\text{erro}}^{(\text{bruto})}
                + \sum_{k \in \mathcal{P}} SQ_k,
                $$
                
                $$
                GL_{\text{erro}}^{(\text{pool})}
                = GL_{\text{erro}}^{(\text{bruto})}
                + \sum_{k \in \mathcal{P}} GL_k,
                $$
                
                onde $\mathcal{P}$ denota o conjunto de fatores agrupados no erro.
                
                Essa redefinição produz um termo de erro com
                $GL_{\text{erro}}^{(\text{pool})} > 0$, permitindo calcular:
                
                $$
                QM_{\text{erro}}^{(\text{pool})}
                = \frac{SQ_{\text{erro}}^{(\text{pool})}}
                {GL_{\text{erro}}^{(\text{pool})}},
                $$
                
                e, consequentemente, as estatísticas $F_k$ e respectivos p-valores.
            """)


            st.markdown(r"""
                ### 🤖 Estratégia de pooling usada pelo aplicativo
                    
                Quando $GL_{\text{erro}} \le 0$:
                    
                1. O app ordena os fatores pela **contribuição (%)**.  
                2. Fatores com contribuição $<5\%$ são candidatos naturais.  
                3. Se nenhum tiver $<5\%$, o app escolhe o **menor** $SQ$.  
                4. O app nunca agrupa **todos** os fatores.  
                5. Uma vez criado o erro com $GL > 0$, calcula $F$, p-valor e contribuições.  
                6. O app exibe quais fatores foram agrupados.
                    
                Assim, a ANOVA fica estatisticamente válida com interpretação completa.
            """)

        st.markdown("---")





        # =========================================
        # ANOVA (S/N como resposta) - aqui df_effects existe
        # =========================================
        df_effects = df_join.copy()
        sn_col = "_SN"  # porque você criou df_join["_SN"] no calcular_sn()
        
        anova_df, meta = compute_anova_sn(df_effects=df_effects, factor_cols=factor_cols, sn_col=sn_col)
        
        if anova_df is None:
            st.error("❌ " + meta.get("error", "Erro ao calcular ANOVA."))
        else:
            st.markdown("🔍 **Tabela ANOVA (razão S/N como resposta)**")
            st.dataframe(anova_df, use_container_width=True, hide_index=True)
    
            st.caption(
                "ℹ️ **Significativo (5%)**: "
                "`Sim` = p < 0,05; "
                "`Não` = p ≥ 0,05; "
                "`n/d` = não determinado (sem GL de erro ou sem cálculo de p-valor)."
            )
            
            # Mensagem de pooling (se aplicável)
            if meta.get("used_pooling") and meta.get("pooled_names"):
                pooled_str = ", ".join(meta["pooled_names"])
                st.info(
                    "🔁 **Pooling automático aplicado**: "
                    "fatores com contribuição percentual reduzida foram incorporados ao termo de erro "
                    f"para viabilizar a estimativa estatística. "
                    f"Fator(es) agrupado(s): **{pooled_str}**."
                )

        
            # (opcional) tabela dos fatores poolados com contribuição original
            if meta.get("used_pooling") and meta.get("pooled_names"):
                pooled_rows = []
                for ent in meta.get("factor_entries", []):
                    if ent["Fonte"] in meta["pooled_names"]:
                        pooled_rows.append({
                            "Fator poolado": ent["Fonte"],
                            "SQ (original)": ent["SQ"],
                            "Contribuição original (%)": ent["Contrib_orig"],
                        })
                if pooled_rows:
                    pooled_df = pd.DataFrame(pooled_rows)
                    for col in ["SQ (original)", "Contribuição original (%)"]:
                        pooled_df[col] = pd.to_numeric(pooled_df[col], errors="coerce").round(4)
                    st.markdown("📌 **Fatores agrupados no erro (pooling)**")
                    st.dataframe(pooled_df, use_container_width=True, hide_index=True)
            
    
            # Download CSV
            buf_anova = io.StringIO()
            anova_df.to_csv(buf_anova, index=False)
            st.download_button(
                "📥 Baixar tabela ANOVA (CSV)",
                data=buf_anova.getvalue().encode("utf-8"),
                file_name=f"anova_SN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_anova_sn",
            )

            st.markdown("---")
            # =========================
            # Relatório automático — ANOVA (Taguchi)
            # =========================
            if anova_df is not None:
                st.subheader("🧾 Relatório automático — ANOVA Taguchi (S/N)")
            
                dfA = anova_df.copy()
            
                # Normaliza nomes de colunas (tolerante)
                cols = {c.lower(): c for c in dfA.columns}
            
                # Possíveis colunas esperadas
                col_fonte = cols.get("fonte", None) or cols.get("source", None)
                col_p = cols.get("p", None) or cols.get("p-valor", None) or cols.get("pvalor", None) or cols.get("p-value", None)
                col_sig = cols.get("significativo", None) or cols.get("significant", None)
                col_contrib = (cols.get("contribuição (%)", None) or cols.get("contribuicao (%)", None) or
                               cols.get("contrib (%)", None) or cols.get("contribution (%)", None))
            
                # Se não achar "Fonte", tenta a primeira coluna
                if col_fonte is None and len(dfA.columns) > 0:
                    col_fonte = dfA.columns[0]
            
                # Funções auxiliares
                def _to_num(s):
                    return pd.to_numeric(s, errors="coerce")
            
                def _fmt_p(pv):
                    return f"{pv:.3f}".replace(".", ",") if np.isfinite(pv) else "—"
            
                bullets = []
            
                # ---------
                # Seleciona apenas fatores (exclui Total/Erro/Resíduo se existirem)
                # ---------
                fonte_vals = dfA[col_fonte].astype(str)
            
                mask_fatores = ~fonte_vals.str.lower().isin(["total", "erro", "error", "resíduo", "residuo", "residual", "pure error"])
                dfF = dfA.loc[mask_fatores].copy()
            
                # ---------
                # Significância por p-valor (preferencial) ou coluna "Significativo"
                # ---------
                if col_p is not None:
                    dfF["p_num"] = _to_num(dfF[col_p])
            
                    sig = dfF.loc[dfF["p_num"] < 0.05, [col_fonte, "p_num"]].sort_values("p_num")
                    ns = dfF.loc[dfF["p_num"] >= 0.05, [col_fonte, "p_num"]].sort_values("p_num")
                    nd = dfF.loc[~np.isfinite(dfF["p_num"]), [col_fonte]].copy()
            
                    if len(sig) > 0:
                        parts = [f"{r[col_fonte]} (P={_fmt_p(r['p_num'])})" for _, r in sig.iterrows()]
                        bullets.append("**Fatores significativos (5%)**: " + ", ".join(parts) + ".")
            
                    if len(ns) > 0:
                        parts = [f"{r[col_fonte]} (P={_fmt_p(r['p_num'])})" for _, r in ns.iterrows()]
                        bullets.append("**Fatores não significativos (5%)**: " + ", ".join(parts) + ".")
            
                    if len(nd) > 0:
                        parts = [str(r[col_fonte]) for _, r in nd.iterrows()]
                        bullets.append("**p-valor não determinado (n/d)** para: " + ", ".join(parts) + ".")
                else:
                    # fallback: usa coluna "Significativo" se existir (Sim/Não/n/d)
                    if col_sig is not None:
                        sig = dfF[dfF[col_sig].astype(str).str.lower().isin(["sim", "yes", "y", "true"])]
                        ns = dfF[dfF[col_sig].astype(str).str.lower().isin(["não", "nao", "no", "n", "false"])]
                        nd = dfF[~dfF.index.isin(sig.index) & ~dfF.index.isin(ns.index)]
            
                        if len(sig) > 0:
                            bullets.append("**Fatores significativos (5%)**: " + ", ".join(sig[col_fonte].astype(str).tolist()) + ".")
                        if len(ns) > 0:
                            bullets.append("**Fatores não significativos (5%)**: " + ", ".join(ns[col_fonte].astype(str).tolist()) + ".")
                        if len(nd) > 0:
                            bullets.append("**Significância não determinada (n/d)** para: " + ", ".join(nd[col_fonte].astype(str).tolist()) + ".")
                    else:
                        bullets.append("Não foi possível determinar significância (colunas de p-valor/Significativo não encontradas).")
            
                # ---------
                # Maior contribuição (%)
                # ---------
                if col_contrib is not None:
                    dfF["contrib_num"] = _to_num(dfF[col_contrib])
                    if dfF["contrib_num"].notna().any():
                        idx = dfF["contrib_num"].idxmax()
                        top_name = str(dfF.loc[idx, col_fonte])
                        top_val = float(dfF.loc[idx, "contrib_num"])
                        bullets.append(f"👉 **{top_name}** é o fator com maior contribuição (**{str(f'{top_val:.2f}').replace('.', ',')}%**).")
            
                # ---------
                # Pooling (se aplicável)
                # ---------
                if meta.get("used_pooling") and meta.get("pooled_names"):
                    pooled_str = ", ".join(meta["pooled_names"])
                    bullets.append(
                        "🔁 **Pooling automático aplicado**: "
                        f"fatores com baixa contribuição foram agrupados no erro para permitir a estimativa estatística "
                        f"({pooled_str})."
                    )
            
                # Exibição
                for b in bullets:
                    st.markdown(f"- {b}")

        
        if not HAS_SCIPY:
            st.info(
                "ℹ️ Os p-valores não foram calculados porque o pacote **SciPy** não está disponível.\n"
                "Se desejar p-valores, instale SciPy no ambiente de execução:\n\n"
                "`pip install scipy`"
            )

        
        st.markdown("---")
    

    
    def regressao_multipla(per_factor_tables):
        st.subheader("📉 Regressão múltipla")
 
        st.caption(
            "Esta Regressão Múltipla é ajustada diretamente aos **dados do problema** (valores observados da resposta), "
            "e **não** à razão S/N. O objetivo é modelar a variável resposta como uma combinação linear dos fatores "
            "(e, opcionalmente, de interações), permitindo estimar efeitos, prever valores e avaliar significância."
        )

        if st.toggle("🔴🔴🔴 O que é esta Regressão Múltipla? (clique para ver) 🔴🔴🔴", value=False, key="show_mr_help"):
            st.markdown(r"""
                A **Regressão Linear Múltipla** modela a resposta observada $\mathbf{y} \in \R^{n \times 1}$ como uma função linear de vários preditores
            (fatores) $\mathbf{x}_1,\mathbf{x}_2, \dots,\mathbf{x}_p \in \R^{n \times 1}$. De forma geral, o modelo pode ser escrito como
            """)
            st.latex(r" \mathbf{y} = \beta_0 \mathbf{x}_0 + \sum_{i=1}^{p}\beta_i \mathbf{x}_i + \boldsymbol{\varepsilon}")
            st.markdown(r"""
            onde 

            - **$\mathbf{y}\in\mathbb{R}^{n \times 1}$** é o vetor de observações da resposta; 
            - $\mathbf{x}_0 \in \R^{n \times 1}$ é um vetor constante  de entradas unitárias, isto é, $\mathbf{x}_0 = [1\ 1\ \cdots\ 1]^T = \mathbf{1}_n \in \R^{n \times 1}$, o qual é responsável pelo processo de intercepto;
            - $\mathbf{x}_i \in \R^{n \times 1}$ é o vetor de observações do fator $i \in \{0,1,\cdots, p\}$; 
            -  $\beta_i$ representa o coeficiente associado ao fator $i \in \{0,1,\cdots, p\}$;
            - $\boldsymbol{\varepsilon}  \in \R^{n \times 1}$ é denominado de vetor de erros, o qual representa a parte da resposta não explicada pelo modelo e assumida de natureza aleatória. Em geral, assume-se que: $\mathbb{E}(\boldsymbol{\varepsilon})=\mathbf{0}_n$ (média nula) e $\operatorname{Var}(\boldsymbol{\varepsilon})=\sigma^2\mathbf{I}_n$ (matriz de covariância do vetor de erros é proporcional à matriz identidade de ordem $n$). 
            """)
            st.markdown(r"""
            O problema de Regressão Linear Múltipla consiste em estimar o vetor de coeficientes
            $\boldsymbol{\beta}=(\beta_0,\beta_1,\ldots,\beta_p)^\top \in \mathbb{R}^{p+1}$
            a partir dos dados observados, de modo a obter a melhor aproximação linear da resposta.
            
            Para isso, adota-se o critério dos **Mínimos Quadrados Ordinários (MQO)**,
            que consiste em minimizar a soma dos quadrados dos resíduos, isto é,
            resolver o problema de otimização
            """)
            
            st.latex(r"""
            \min_{\boldsymbol{\beta}\in\mathbb{R}^{p+1}}
            \; \|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|^2,
            """)
            
            st.markdown(r"""
            em que $\mathbf{X}=[\,\mathbf{x}_0\ \mathbf{x}_1\ \cdots\ \mathbf{x}_p\,]\in\mathbb{R}^{n\times(p+1)}$
            é a matriz de projeto do modelo.
            
            Sob a hipótese de posto completo da matriz $\mathbf{X}$,
            a solução do problema é única e é dada por
            """)
            
            st.latex(r"""
            \hat{\boldsymbol{\beta}}
            =
            (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}.
            """)
            
            st.markdown(r"""
            O vetor $\hat{\boldsymbol{\beta}}$ é denominado **estimador de mínimos quadrados**
            e fornece a equação estimada do modelo,
            $\hat{\mathbf{y}}=\mathbf{X}\hat{\boldsymbol{\beta}}$,
            bem como o vetor de resíduos
            $\hat{\boldsymbol{\varepsilon}}=\mathbf{y}-\hat{\mathbf{y}}$.
            
            Geometricamente, o vetor $\hat{\mathbf{y}}$ corresponde à projeção ortogonal
            de $\mathbf{y}$ sobre o espaço coluna de $\mathbf{X}$,
            enquanto o vetor de resíduos é ortogonal a esse espaço.
            """)

            st.markdown(r"""
            Sob as hipóteses usuais do modelo de regressão linear,
            em particular $\mathbb{E}(\boldsymbol{\varepsilon})=\mathbf{0}_n$
            e $\operatorname{Var}(\boldsymbol{\varepsilon})=\sigma^2\mathbf{I}_n$,
            o estimador de mínimos quadrados apresenta propriedades estatísticas fundamentais.
            Em particular, tem-se que:
            """)
            
            st.latex(r"""
            \mathbb{E}(\hat{\boldsymbol{\beta}})=\boldsymbol{\beta},
            \qquad
            \operatorname{Var}(\hat{\boldsymbol{\beta}})
            =
            \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}.
            """)
            
            st.markdown(r"""
            A primeira igualdade indica que $\hat{\boldsymbol{\beta}}$ é um **estimador não viesado**
            do vetor de coeficientes $\boldsymbol{\beta}$.
            A segunda expressa a **matriz de covariância do estimador**,
            mostrando que a precisão das estimativas depende da variância do erro
            e da estrutura da matriz de projeto $\mathbf{X}$.
            """)
            
            st.markdown(r"""
            Os resultados anteriores permitem definir e interpretar as principais
            **métricas de avaliação** do modelo de Regressão Linear Múltipla,
            as quais podem ser organizadas em quatro grupos:
            qualidade do ajuste, erro preditivo, significância estatística
            e diagnóstico do modelo.
            """)


            st.markdown(r"## Avaliação do modelo de regressão")

            st.markdown(r"""
            Nesta seção, conectamos as propriedades do estimador de Mínimos Quadrados Ordinários (MQO)
            com as métricas usuais utilizadas na avaliação de modelos de Regressão Linear Múltipla.
            Utiliza-se a norma euclidiana $\|\cdot\|$ em $\mathbb{R}^n$, para a qual
            $\|\mathbf{v}\|^2=\mathbf{v}^\top\mathbf{v}$.
            """)
            
            
            # -------------------------------------------------
            # Definições básicas
            # -------------------------------------------------
            st.markdown(r"### Definições básicas")
            
            st.latex(r"""
            \hat{\mathbf{y}}=\mathbf{X}\hat{\boldsymbol{\beta}}, \qquad
            \hat{\boldsymbol{\varepsilon}}=\mathbf{y}-\hat{\mathbf{y}},
            """)
            
            st.latex(r"""
            \bar y=\frac{1}{n}\mathbf{1}_n^\top\mathbf{y}, \qquad
            \mathbf{y}_c=\mathbf{y}-\bar y\,\mathbf{1}_n.
            """)
            
            st.latex(r"""
            \mathbf{H}
            =
            \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top,
            \qquad
            h_{ii}=(\mathbf{H})_{ii}.
            """)
            
            st.markdown(r"""
            A matriz $\mathbf{H}$ é denominada **matriz chapéu**.
            Seus elementos diagonais $h_{ii}$ medem a alavancagem das observações
            e são utilizados na definição de métricas preditivas baseadas em validação cruzada.
            """)

            
            # -------------------------------------------------
            # 1. Qualidade do ajuste
            # -------------------------------------------------
            st.markdown(r"### 1. Qualidade do ajuste")
            
            st.markdown(r"""
            A qualidade do ajuste é avaliada por meio da decomposição da variabilidade da resposta.
            Definem-se as seguintes somas de quadrados:
            """)
            
            st.latex(r"""
            \text{SQ}_{\mathrm{res}}=\|\hat{\boldsymbol{\varepsilon}}\|^2, \qquad
            \text{SQ}_{\mathrm{tot}}=\|\mathbf{y}_c\|^2 \qquad \textrm{e} \qquad \text{SQ}_{\mathrm{reg}}=\|\hat{\mathbf{y}}-\bar y\,\mathbf{1}_n\|^2.
            """)
            
            st.markdown(r"""
            Essas quantidades satisfazem a decomposição
            $\text{SQ}_{\mathrm{tot}}=\text{SQ}_{\mathrm{reg}}+\text{SQ}_{\mathrm{res}}$.
            O coeficiente de determinação é então definido por:
            """)
            
            st.latex(r"""
            R^2
            =
            1-\frac{\text{SQ}_{\mathrm{res}}}{\text{SQ}_{\mathrm{tot}}}
            =
            \frac{\text{SQ}_{\mathrm{reg}}}{\text{SQ}_{\mathrm{tot}}}.
            """)
            
            st.markdown(r"""
            Em modelos com múltiplos preditores, utiliza-se frequentemente o coeficiente de determinação ajustado:
            """)
            
            st.latex(r"""
            R^2_{\mathrm{aj}}
            =
            1-\frac{\text{SQ}_{\mathrm{res}}/(n-p-1)}{\text{SQ}_{\mathrm{tot}}/(n-1)}.
            """)
            
            # -------------------------------------------------
            # 2. Erro preditivo
            # -------------------------------------------------
            st.markdown(r"### 2. Erro preditivo")
            
            st.markdown(r"""
            O erro preditivo do modelo é quantificado a partir do vetor de resíduos.
            Uma estimativa não viesada da variância do erro é dada por:
            """)
            
            st.latex(r"""
            \hat{\sigma}^2
            =
            \frac{\|\hat{\boldsymbol{\varepsilon}}\|^2}{n-p-1}.
            """)
            
            st.markdown(r"""
            O estimador $\hat{\sigma}=\sqrt{\hat{\sigma}^2}$ fornece uma medida da escala média do erro
            e é frequentemente utilizado como **RMSE (Root Mean Squared Error)** do modelo:
            """)
            
            st.latex(r"""
            \text{RMSE}
            =
            \hat{\sigma}
            =
            \sqrt{\frac{\|\hat{\boldsymbol{\varepsilon}}\|^2}{n-p-1}}.
            """)

            st.markdown(r"""
            No **Sumário do Modelo**, essa quantidade é usualmente denotada por **$S$**
            e corresponde à estimativa do desvio-padrão do erro do modelo
            (quadrado médio do erro).
            """)

            
            st.markdown(r"""
            Como medida alternativa, mais robusta à presença de valores extremos,
            pode-se utilizar o erro absoluto médio:
            """)
            
            st.latex(r"""
            \text{MAE}
            =
            \frac{1}{n}\|\hat{\boldsymbol{\varepsilon}}\|_1,
            """)

            st.markdown(r"""
            em que a norma $\|\mathbf{\cdot}\|_1$ é definida por:  $\|\mathbf{v}\|_1=\sum_{i=1}^n |v_i|$. 
            """)

            st.markdown(r"""
            Além das métricas baseadas nos resíduos ajustados,
            avaliam-se métricas de caráter preditivo,
            baseadas na validação cruzada do tipo leave-one-out.
            """)

            st.latex(r"""
            \text{PRESS}
            =
            \sum_{i=1}^n
            \left(
            \frac{\hat{\varepsilon}_i}{1-h_{ii}}
            \right)^2
            =
            \left\|
            \left(\mathbf{I}_n-\operatorname{diag}(\mathbf{H})\right)^{-1}
            \hat{\boldsymbol{\varepsilon}}
            \right\|^2.
            """)


            st.markdown(r"""
            A estatística PRESS (Prediction Sum of Squares)
            mede o erro de previsão do modelo em validação cruzada leave-one-out, em que 
            valores menores indicam melhor capacidade preditiva. Por sua vez, O coeficiente de determinação preditivo $R^2_{\mathrm{pred}}$ quantifica a proporção da variabilidade da resposta explicada em termos de previsão fora da amostra.
            """)

            st.latex(r"""
            R^2_{\mathrm{pred}}
            =
            1-\frac{\text{PRESS}}{\text{SQ}_{\mathrm{tot}}}.
            """)

            # -------------------------------------------------
            # 3. Significância estatística
            # -------------------------------------------------
            st.markdown(r"### 3. Significância estatística")
            
            st.markdown(r"""
            Sob as hipóteses $\mathbb{E}(\boldsymbol{\varepsilon})=\mathbf{0}_n$ e
            $\operatorname{Var}(\boldsymbol{\varepsilon})=\sigma^2\mathbf{I}_n$, o estimador de MQO satisfaz:
            """)
            
            st.latex(r"""
            \mathbb{E}(\hat{\boldsymbol{\beta}})=\boldsymbol{\beta},
            \qquad
            \operatorname{Var}(\hat{\boldsymbol{\beta}})
            =
            \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}.
            """)
            
            st.markdown(r"""
            Na prática, substitui-se $\sigma^2$ por $\hat{\sigma}^2$, obtendo-se a matriz de covariância estimada:
            """)
            
            st.latex(r"""
            \widehat{\operatorname{Var}}(\hat{\boldsymbol{\beta}})
            =
            \hat{\sigma}^2(\mathbf{X}^\top\mathbf{X})^{-1}.
            """)
            
            st.markdown(r"""
            O erro-padrão associado ao coeficiente $\hat\beta_i$ é então:
            """)
            
            st.latex(r"""
            \operatorname{EP}(\hat\beta_i)
            =
            \sqrt{\hat{\sigma}^2\left[(\mathbf{X}^\top\mathbf{X})^{-1}\right]_{ii}}.
            """)
            
            st.markdown(r"""
            Essas quantidades fundamentam os testes de hipóteses do tipo \(t\) para coeficientes individuais
            e o teste \(F\) para a significância global do modelo, definido por:
            """)
            
            st.latex(r"""
            F
            =
            \frac{\text{SQ}_{\mathrm{reg}}/p}{\text{SQ}_{\mathrm{res}}/(n-p-1)}.
            """)

            # -------------------------------------------------
            # 4. Diagnóstico do modelo
            # -------------------------------------------------
            st.markdown(r"### 4. Diagnóstico do modelo")
            
            st.markdown(r"""
            O diagnóstico do modelo baseia-se na análise dos resíduos e na estrutura geométrica do ajuste.
            Define-se a matriz de projeção (matriz chapéu):
            """)
            
            st.latex(r"""
            \mathbf{H}
            =
            \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top,
            \qquad
            \hat{\mathbf{y}}=\mathbf{H}\mathbf{y}.
            """)
            
            st.markdown(r"""
            Os elementos diagonais $h_{ii}$ medem a alavancagem das observações,
            sendo utilizados na identificação de pontos influentes.
            Problemas de multicolinearidade estão associados à estrutura de
            $(\mathbf{X}^\top\mathbf{X})^{-1}$, motivando o uso de medidas como o VIF (Fator de Inflação da Variância).
            """)

            st.markdown(r"""
            O **Fator de Inflação da Variância (VIF)** quantifica o impacto da multicolinearidade
            sobre a variância dos estimadores de mínimos quadrados.
            Para o coeficiente associado ao preditor $\mathbf{x}_j$, o VIF é definido por
            """)
            
            st.latex(r"""
            \mathrm{VIF}_j
            =
            \frac{1}{1-R_j^2},
            """)
            
            st.markdown(r"""
            onde $R_j^2$ é o coeficiente de determinação obtido ao se regressar
            o preditor $\mathbf{x}_j$ em função dos demais preditores do modelo.
            """)
            
            st.markdown(r"""
            Do ponto de vista matricial, o VIF está diretamente relacionado
            aos elementos diagonais da matriz $(\mathbf{X}^\top\mathbf{X})^{-1}$,
            que governam a variância dos estimadores $\hat{\boldsymbol{\beta}}$.
            A interpretação do Fator de Inflação da Variância (VIF) é usualmente feita da seguinte forma:

- $\mathrm{VIF}_j = 1$: ausência de multicolinearidade;
- $1 < \mathrm{VIF}_j \lesssim 5$: colinearidade moderada;
- $\mathrm{VIF}_j \gtrsim 10$: colinearidade severa.

Valores elevados de VIF indicam instabilidade nos coeficientes estimados,
com inflacionamento dos erros-padrão e redução da confiabilidade dos testes de significância.
            """)

            st.markdown(r"""
            Além das métricas baseadas em resíduos, alavancagem e multicolinearidade,
            podem-se utilizar **critérios de informação** para a comparação entre modelos.
            Esses critérios equilibram a qualidade do ajuste com a complexidade do modelo,
            penalizando a inclusão excessiva de parâmetros.
            """)
            
            st.markdown(r"""
            Os critérios mais utilizados são o **AIC (Akaike Information Criterion)** e o
            **BIC (Bayesian Information Criterion)**, definidos, respectivamente, por:
            """)
            
            st.latex(r"""
            \mathrm{AIC}
            =
            n\ln\!\left(\frac{\text{SQ}_{\mathrm{res}}}{n}\right)
            +2(p+1),
            """)
            
            st.latex(r"""
            \mathrm{BIC}
            =
            n\ln\!\left(\frac{\text{SQ}_{\mathrm{res}}}{n}\right)
            +(p+1)\ln n.
            """)
            
            st.markdown(r"""
            Os critérios AIC e BIC penalizam a complexidade do modelo
            e são utilizados principalmente para a **comparação entre modelos concorrentes**.
            Em ambos os casos, **valores menores indicam modelos preferíveis**.
            """)
        st.markdown("---")

        # =========================
        # Coeficientes da Regressão
        # =========================
        st.markdown("### Coeficientes da Regressão")
        
        # y (resposta): valores do problema (ex.: média por ensaio)
        y = np.asarray(mean_y, dtype=float).reshape(-1, 1)
        n = y.shape[0]
        
        # X: fatores (dummies para categóricos), com intercepto
        X_raw = df_plan[factor_cols].copy()
        X_num = X_raw.apply(pd.to_numeric, errors="ignore")
        X_dum = pd.get_dummies(X_num, drop_first=True, dtype=float)
        X_dum.insert(0, "Constante", 1.0)
        
        X = X_dum.to_numpy(dtype=float)
        terms = list(X_dum.columns)
        
        p1 = X.shape[1]        # (p+1)
        p = p1 - 1
        df_res = n - p1
        
        # MQO
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)  # robusto
        beta_hat = XtX_inv @ (X.T @ y)
        beta = beta_hat.ravel()    
        y_hat = X @ beta_hat
        eps_hat = y - y_hat
        
        SQ_res = float((eps_hat.T @ eps_hat)[0, 0])
        sigma2_hat = SQ_res / df_res if df_res > 0 else np.nan
        
        # Variância/EP dos coeficientes
        Var_beta = sigma2_hat * XtX_inv
        se = np.sqrt(np.maximum(np.diag(Var_beta), 0.0)).reshape(-1, 1)
        
        # Estatística t e p-valor (bicaudal)
        t_vals = (beta_hat / se).ravel()
        
        # se você já tem t_dist do scipy.stats:
        #   from scipy.stats import t as t_dist
        p_vals = np.array([
            2 * (1 - t_dist.cdf(abs(tv), df=df_res)) if (df_res > 0 and np.isfinite(tv)) else np.nan
            for tv in t_vals
        ], dtype=float)
        
        # IC 95%
        alpha = 0.05
        t_crit = t_dist.ppf(1 - alpha/2, df=df_res) if df_res > 0 else np.nan
        ci_low = (beta_hat - t_crit * se).ravel()
        ci_high = (beta_hat + t_crit * se).ravel()
        
        def fmt_ci(a, b, nd=1):
            if not (np.isfinite(a) and np.isfinite(b)):
                return ""
            return f"({a:.{nd}f}; {b:.{nd}f})"
        
        # VIF (para cada preditor exceto intercepto)
        vif = [""] * p1
        if p >= 1:
            for j in range(1, p1):  # pula intercepto
                xj = X[:, [j]]
                X_others = np.delete(X, j, axis=1)
                # regressão auxiliar: xj ~ X_others
                XtXo = X_others.T @ X_others
                XtXo_inv = np.linalg.pinv(XtXo)
                bj = XtXo_inv @ (X_others.T @ xj)
                xj_hat = X_others @ bj
        
                num = float(((xj - xj_hat).T @ (xj - xj_hat))[0, 0])
                den = float(((xj - np.mean(xj)).T @ (xj - np.mean(xj)))[0, 0])
                R2j = 1 - (num/den) if den > 0 else 0.0
        
                vif_j = 1.0 / (1.0 - R2j) if (1.0 - R2j) > 1e-12 else np.inf
                vif[j] = f"{vif_j:.2f}" if np.isfinite(vif_j) else "Inf"
        
        # Monta tabela final (mesmo cabeçalho do seu relatório)
        df_coef = pd.DataFrame({
            "Termo": terms,
            "Coef": beta_hat.ravel(),
            "EP de Coef": se.ravel(),
            "IC de 95%": [fmt_ci(a, b, nd=1) for a, b in zip(ci_low, ci_high)],
            "Valor-T": t_vals,
            "Valor-P": p_vals,
            "VIF": vif
        })
        
        # Formatação numérica (sem perder tipo no dataframe)
        df_show = df_coef.copy()
        df_show["Coef"] = df_show["Coef"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
        df_show["EP de Coef"] = df_show["EP de Coef"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
        df_show["Valor-T"] = df_show["Valor-T"].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "")
        df_show["Valor-P"] = df_show["Valor-P"].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "")
        
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.caption(
    "EP de Coef (Erro-Padrão do Coeficiente) mede a incerteza associada à estimativa "
    "de cada coeficiente de regressão e é utilizado no cálculo do valor-t, do valor-p "
    "e dos intervalos de confiança."
)
        st.caption(
    "VIF (Variance Inflation Factor – Fator de Inflação da Variância) mede o grau de "
    "multicolinearidade entre um preditor e os demais. Valores elevados de VIF indicam "
    "coeficientes instáveis e aumento do erro-padrão das estimativas."
    )

                # -----------------------------
        # Download — Coeficientes
        # -----------------------------
        buf_coef = io.StringIO()
        df_show.to_csv(buf_coef, index=False)
        
        st.download_button(
            "📥 Baixar coeficientes da regressão",
            buf_coef.getvalue().encode("utf-8"),
            file_name=f"coeficientes_regressao_{var_label}.csv",
            mime="text/csv",
            key="dl_coef_regressao",
        )

        st.markdown("---")

        st.markdown("### 🧾 Equação de regressão do modelo")

        # 1) versão LaTeX bonita:
        # y_hat = beta0 + beta1*x1 + ... + betap*xp
        terms_latex = []
        for i, name in enumerate(["Constante"] + factor_cols):
            b = float(beta[i])
            if i == 0:
                terms_latex.append(f"{b:.4f}")
            else:
                sign = "+" if b >= 0 else "-"
                coef_abs = abs(b)
                # nome do regressor em LaTeX (escape simples)
                var = to_latex_var(name)
                terms_latex.append(f"{sign}\; {coef_abs:.4f}\,{var}")
        
        eq_latex = r"\hat{y} = " + " ".join(terms_latex)
        st.latex(eq_latex)
        
        st.caption(f"Modelo ajustado para {var_label}, com termos lineares nos fatores codificados.")

        st.markdown("---")

        st.markdown("### 📌 Sumário do Modelo")

        # -------------------------------
        # Quantidades globais da regressão
        # -------------------------------
        
        # Dimensões
        n = X.shape[0]
        p = X.shape[1] - 1   # número de preditores (exclui intercepto)
        
        # Graus de liberdade
        gl_res = n - p - 1

        # 1) Coeficientes
        beta_hat = XtX_inv @ X.T @ y
        
        # 2) Ajuste e resíduos  <<< ESTE BLOCO É OBRIGATÓRIO
        y_hat = X @ beta_hat
        residuals = y - y_hat
        
        # Somas de quadrados
        SQ_res = np.linalg.norm(residuals)**2
        SQ_tot = np.linalg.norm(y - y.mean())**2
        SQ_reg = SQ_tot - SQ_res
        
        # Coeficientes de determinação
        R2 = 1 - SQ_res / SQ_tot
        R2_aj = 1 - (SQ_res / gl_res) / (SQ_tot / (n - 1))
        
        # Estatísticas do erro
        sigma2_hat = SQ_res / gl_res
        RMSE = np.sqrt(sigma2_hat)   # S (Minitab)


        # =========================
        # PRESS e R²(pred)
        # =========================
        
        # Matriz chapéu e alavancagens
        H = X @ XtX_inv @ X.T
        h = np.clip(np.diag(H), 0.0, 0.999999)  # evita divisão por zero
        
        # PRESS (LOOCV)
        press_res = (residuals.ravel() / (1.0 - h))
        PRESS = float(np.sum(press_res**2))
        
        # R² preditivo
        R2_pred = 1 - PRESS / SQ_tot if SQ_tot > 0 else np.nan


        # =========================
        # AICc e BIC (seleção de modelos)
        # =========================
        
        k = X.shape[1]  # número de parâmetros (inclui constante)
        
        if (SQ_res > 0) and (n > 0):
            AIC = n * np.log(SQ_res / n) + 2 * k
        
            # AIC corrigido (AICc), quando possível
            if (n - k - 1) > 0:
                AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)
            else:
                AICc = AIC
        
            BIC = n * np.log(SQ_res / n) + k * np.log(n)
        else:
            AICc = np.nan
            BIC = np.nan

        
        df_summary = pd.DataFrame({
            "S": [RMSE],
            "R²": [100 * R2],
            "R² (aj)": [100 * R2_aj],
            "PRESS": [PRESS],
            "R² (pred)": [100 * R2_pred],
            "AICc": [AICc],
            "BIC": [BIC],
        })
        
        df_summary_fmt = df_summary.copy()
        for c in df_summary_fmt.columns:
            df_summary_fmt[c] = df_summary_fmt[c].map(
                lambda x: f"{x:.2f}" if np.isfinite(x) else ""
            )
        
        st.dataframe(df_summary_fmt, use_container_width=True, hide_index=True)
        # -----------------------------
        # Download — Sumário do Modelo
        # -----------------------------
        buf_summary = io.StringIO()
        df_summary_fmt.to_csv(buf_summary, index=False)
        
        st.download_button(
            "📥 Baixar sumário do modelo",
            buf_summary.getvalue().encode("utf-8"),
            file_name=f"sumario_modelo_{var_label}.csv",
            mime="text/csv",
            key="dl_sumario_modelo",
        )


        st.markdown("---")
        st.subheader("📋 Tabela de resíduos (diagnóstico)")
        
        # Vetores 1D
        y_obs = np.asarray(y, dtype=float).ravel()
        y_fit = np.asarray(y_hat, dtype=float).ravel()
        resid = np.asarray(residuals, dtype=float).ravel()
        
        # Leverage (h_ii) — você já calcula H e h no seu bloco do PRESS
        # Se ainda não calculou H/h acima, use este fallback:
        try:
            h_ii = np.asarray(h, dtype=float).ravel()
        except Exception:
            H_local = X @ XtX_inv @ X.T
            h_ii = np.clip(np.diag(H_local), 0.0, 0.999999)
        
        # S (RMSE / sigma_hat) — você já tem RMSE (S do Minitab)
        S = float(RMSE) if np.isfinite(RMSE) else float("nan")
        
        # Resíduo padronizado: e_i / S
        std_resid = resid / S if (np.isfinite(S) and S > 0) else np.full_like(resid, np.nan)
        
        # Resíduo studentizado interno: e_i / (S * sqrt(1 - h_ii))
        den = S * np.sqrt(np.maximum(1.0 - h_ii, 1e-12)) if (np.isfinite(S) and S > 0) else np.full_like(resid, np.nan)
        stud_resid = resid / den
        
        df_resid = pd.DataFrame({
            "Experimento": df_plan["Experimento"].values if "Experimento" in df_plan.columns else np.arange(1, len(y_obs) + 1),
            f"{var_label} (observado)": y_obs,
            f"{var_label} (predito)": y_fit,
            "Resíduo": resid
        })
        
        # Formatação leve (não destrói os dados originais se você quiser exportar depois)
        df_show = df_resid.copy()
        for c in df_show.columns:
            if c != "Experimento":
                df_show[c] = pd.to_numeric(df_show[c], errors="coerce").map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
        
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        
        st.caption(
            "Resíduo = observado − ajustado. "
            "Resíduo padronizado usa S (RMSE). "
            "Resíduo studentizado ajusta também pela alavancagem (hᵢᵢ)."
        )

        # -----------------------------
        # Download — Tabela de resíduos
        # -----------------------------
        buf_resid = io.StringIO()
        df_show.to_csv(buf_resid, index=False)
        
        st.download_button(
            "📥 Baixar tabela de resíduos (normalidade)",
            data=buf_resid.getvalue().encode("utf-8"),
            file_name=f"residuos_normalidade_{var_label}.csv",
            mime="text/csv",
            key="dl_residuos_normalidade",
        )

        
        st.markdown("---")

        st.subheader("🔎 Diagnóstico do Modelo — Normalidade dos Resíduos")
        
        st.caption(
            "A normalidade dos resíduos é uma suposição importante da regressão linear, "
            "especialmente para a validade de testes de hipóteses e intervalos de confiança."
        )

        resid = residuals.ravel()  # ou residuals.squeeze()
        W, p_shapiro = shapiro(resid)

        c1, c2 = st.columns(2)
        
        with c1:
            st.metric("Estatística W (Shapiro–Wilk)", f"{W:.4f}")
        
        with c2:
            st.metric("p-valor", f"{p_shapiro:.4g}")

        if p_shapiro < 0.05:
            st.warning(
                "⚠️ O teste de Shapiro–Wilk rejeita a hipótese de normalidade dos resíduos "
                "(p < 0,05). Avalie transformações ou modelos alternativos."
            )
        else:
            st.success(
                "✅ Não há evidências estatísticas contra a normalidade dos resíduos "
                "(p ≥ 0,05)."
            )

        st.markdown("---")
        st.subheader("📈 QQ-plot dos Resíduos")
        
        # Idioma (mesmo padrão do app)
        lang_diag = st.radio(
            "Idioma / Language (QQ-plot):",
            ["Português", "English"],
            index=0,
            horizontal=True,
            key="lang_qqplot_diag",
        )
        
        if lang_diag == "Português":
            caption_txt = (
                "O QQ-plot (Quantile–Quantile) compara os quantis dos resíduos com os quantis "
                "de uma distribuição Normal padrão. Alinhamento próximo à reta indica "
                "normalidade aproximada."
            )
            title_txt = "QQ-plot dos resíduos"
            x_label = "Quantis teóricos (Normal)"
            y_label = "Quantis observados (Resíduos)"
            name_points = "Resíduos"
            name_line = "Referência Normal"
            hover_tmpl = "Quantil teórico: %{x:.3f}<br>Resíduo: %{y:.3f}<extra></extra>"
        else:
            caption_txt = (
                "The QQ-plot (Quantile–Quantile) compares residual quantiles with the quantiles "
                "of a standard Normal distribution. Points close to the reference line indicate "
                "approximately normal residuals."
            )
            title_txt = "Residual QQ-plot"
            x_label = "Theoretical quantiles (Normal)"
            y_label = "Observed quantiles (Residuals)"
            name_points = "Residuals"
            name_line = "Normal reference"
            hover_tmpl = "Theoretical quantile: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>"
        
        st.caption(caption_txt)
        
        # ----------------------------
        # Resíduos (garante vetor 1D)
        # ----------------------------
        resid = np.asarray(residuals, dtype=float).ravel()
        resid = resid[np.isfinite(resid)]
        
        if not HAS_SCIPY or len(resid) < 3:
            st.info("QQ-plot indisponível (SciPy não instalado) ou amostra insuficiente."
                    if lang_diag == "Português"
                    else "QQ-plot unavailable (SciPy not installed) or insufficient sample size.")
        else:
            from scipy.stats import norm
            import plotly.graph_objects as go
        
            resid_sorted = np.sort(resid)
            n = len(resid_sorted)
        
            probs = (np.arange(1, n + 1) - 0.5) / n
            q_theoretical = norm.ppf(probs)
        
            mu = resid_sorted.mean()
            sigma = resid_sorted.std(ddof=1)
        
            fig = go.Figure()
        
            fig.add_trace(go.Scatter(
                x=q_theoretical,
                y=resid_sorted,
                mode="markers",
                name=name_points,
                marker=dict(size=8, opacity=0.75),
                hovertemplate=hover_tmpl,
            ))
        
            fig.add_trace(go.Scatter(
                x=q_theoretical,
                y=mu + sigma * q_theoretical,
                mode="lines",
                name=name_line,
                hoverinfo="skip",
            ))
        
            fig.update_layout(
                title=title_txt,
                xaxis_title=x_label,
                yaxis_title=y_label,
                template="simple_white",
                height=450,
                legend=dict(
                    orientation="v",
                    x=1.05,          # empurra para fora do gráfico (direita)
                    xanchor="left",
                    y=1,
                    yanchor="top"
                ),
                margin=dict(r=120)   # espaço extra à direita para a legenda
            )

        
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("📄 **Baixar figura**")
        
        color_mode = st.radio(
            "Modo de cores para exportação:",
            ["Cores (original)", "Preto e branco"],
            index=0,
            help="A visualização na tela permanece em cores. A opção afeta apenas os arquivos baixados.",
            key="color_mode_qqplot",
        )
        
        # Dimensões de exportação
        export_width = 900
        export_height = 520
        
        
        def _make_export_fig():
            # Cópia para exportação (não altera o fig exibido)
            fig_exp = go.Figure(fig.to_dict())
        
            # Espaço extra à direita para legenda externa
            fig_exp.update_layout(
                width=export_width,
                height=export_height,
                margin=dict(l=70, r=160, t=70, b=70),
                paper_bgcolor="white",
                plot_bgcolor="white",
                template="plotly_white",
            )
        
            # Preto e branco apenas no arquivo exportado
            if color_mode == "Preto e branco":
                for tr in fig_exp.data:
                    if isinstance(tr, go.Scatter):
                        tr.update(
                            marker=dict(color="black"),
                            line=dict(color="black", width=2),
                        )
        
            return fig_exp
        
        
        def _export_bytes(fmt: str):
            fig_exp = _make_export_fig()
            try:
                return fig_exp.to_image(
                    format=fmt,
                    scale=2,
                    width=export_width,
                    height=export_height,
                )
            except Exception:
                st.warning(
                    "Para exportar imagens, é necessário o pacote **kaleido**.\n\n"
                    "Instale com:\n\n"
                    "`pip install -U kaleido`\n\n"
                    "ou\n\n"
                    "`conda install -c conda-forge python-kaleido -y`"
                )
                raise
        
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📥 Gerar PNG", key="btn_png_qqplot"):
                try:
                    png_bytes = _export_bytes("png")
                    st.download_button(
                        "Baixar PNG",
                        data=png_bytes,
                        file_name="qqplot_residuos.png",
                        mime="image/png",
                        key="dl_png_qqplot",
                    )
                except Exception:
                    pass
        
        with col2:
            if st.button("📥 Gerar SVG", key="btn_svg_qqplot"):
                try:
                    svg_bytes = _export_bytes("svg")
                    st.download_button(
                        "Baixar SVG",
                        data=svg_bytes,
                        file_name="qqplot_residuos.svg",
                        mime="image/svg+xml",
                        key="dl_svg_qqplot",
                    )
                except Exception:
                    pass
        
        with col3:
            if st.button("📥 Gerar PDF", key="btn_pdf_qqplot"):
                try:
                    pdf_bytes = _export_bytes("pdf")
                    st.download_button(
                        "Baixar PDF",
                        data=pdf_bytes,
                        file_name="qqplot_residuos.pdf",
                        mime="application/pdf",
                        key="dl_pdf_qqplot",
                    )
                except Exception:
                    pass
        
        with col4:
            if st.button("📥 Gerar HTML interativo", key="btn_html_qqplot"):
                html_bytes = pio.to_html(
                    fig, include_plotlyjs="cdn", full_html=False
                ).encode("utf-8")
                st.download_button(
                    "Baixar HTML",
                    data=html_bytes,
                    file_name="qqplot_residuos.html",
                    mime="text/html",
                    key="dl_html_qqplot",
                )

        st.markdown("---")


        st.subheader("🧾 Relatório automático — Regressão múltipla")
        
        alpha = 0.05
        alpha_marg = 0.10  # faixa "marginal" (0,05 < P <= 0,10)
        
        # ---------
        # df_coef -> preparar p-valor e VIF numéricos
        # ---------
        df_rep = df_coef.copy()
        
        # remove intercepto
        df_rep = df_rep[df_rep["Termo"].astype(str).str.lower() != "constante"].copy()
        
        df_rep["p"] = pd.to_numeric(df_rep["Valor-P"], errors="coerce")
        df_rep["vif_num"] = pd.to_numeric(df_rep["VIF"], errors="coerce")
        
        def fmt_p(pval):
            return f"{pval:.3f}".replace(".", ",") if np.isfinite(pval) else "—"
        
        def join_terms(df_tp):
            # Ex.: "Com (P=0,037), MR (P=0,062)"
            parts = []
            for _, r in df_tp.iterrows():
                parts.append(f"{r['Termo']} (P={fmt_p(r['p'])})")
            return ", ".join(parts)
        
        # ---------
        # Classificação por significância
        # ---------
        sig = df_rep.loc[df_rep["p"] <= alpha, ["Termo", "p"]].sort_values("p")
        marg = df_rep.loc[(df_rep["p"] > alpha) & (df_rep["p"] <= alpha_marg), ["Termo", "p"]].sort_values("p")
        ns = df_rep.loc[df_rep["p"] > alpha_marg, ["Termo", "p"]].sort_values("p")
        
        bullets = []
        
        if len(sig) > 0:
            bullets.append(f"**Significativos (P ≤ 0,05):** {join_terms(sig)}.")
        if len(marg) > 0:
            bullets.append(f"**Marginalmente significativos (0,05 < P ≤ 0,10):** {join_terms(marg)}.")
        if len(ns) > 0:
            bullets.append(f"**Não significativos (P > 0,10):** {join_terms(ns)}.")
        
        # ---------
        # Multicolinearidade (VIF)
        # ---------
        vif_vals = df_rep["vif_num"].dropna().values
        if len(vif_vals) > 0:
            vmax = float(np.max(vif_vals))
            if vmax <= 1.05:
                bullets.append("**VIF ≈ 1** para todos os termos → sem evidência de multicolinearidade.")
            elif vmax <= 5:
                bullets.append(f"**VIF máximo = {vmax:.2f}** → multicolinearidade baixa/moderada.")
            elif vmax <= 10:
                bullets.append(f"**VIF máximo = {vmax:.2f}** → atenção: multicolinearidade relevante.")
            else:
                bullets.append(f"**VIF máximo = {vmax:.2f}** → multicolinearidade alta (coeficientes podem estar instáveis).")
        else:
            bullets.append("VIF não disponível para todos os termos.")
        
        
        # ---------
        # Maior contribuição (%), se existir
        # ---------
        dict_contrib = locals().get("dict_contrib", None)
        if isinstance(dict_contrib, dict) and len(dict_contrib) > 0:
            k_top = max(dict_contrib, key=dict_contrib.get)
            v_top = dict_contrib[k_top]
            if np.isfinite(v_top):
                bullets.append(f"👉 **{k_top}** é o que mais contribui (**{str(f'{v_top:.2f}').replace('.', ',')}%**).")
        
        # ---------
        # Exibição
        # ---------
        for b in bullets:
            st.markdown(f"- {b}")

        st.markdown("---")
        st.subheader("🧾 Relatório automático — Diagnóstico dos resíduos")
        
        alpha = 0.05
        bul_res = []
        
        # Garanta que resid é 1D (use o que você já tem: residuals ou resid)
        # Se você já tem "resid" definido antes, pode pular estas duas linhas.
        try:
            resid_1d = np.asarray(resid, dtype=float).ravel()
        except NameError:
            resid_1d = np.asarray(residuals, dtype=float).ravel()
        
        resid_1d = resid_1d[np.isfinite(resid_1d)]
        
        # --- Shapiro–Wilk (teste formal) ---
        if np.isfinite(p_shapiro):
            if p_shapiro >= alpha:
                bul_res.append(
                    f"Não há evidências estatísticas contra a normalidade dos resíduos "
                    f"(Shapiro–Wilk, P={str(f'{p_shapiro:.3f}').replace('.', ',')})."
                )
            else:
                bul_res.append(
                    f"O teste de Shapiro–Wilk rejeita a hipótese de normalidade dos resíduos "
                    f"(P={str(f'{p_shapiro:.3f}').replace('.', ',')}). "
                    "Considere transformação (log/Box-Cox) ou métodos mais robustos, se necessário."
                )
        else:
            bul_res.append("O teste de Shapiro–Wilk não pôde ser avaliado para os resíduos.")
        
        # --- Complemento gráfico (QQ-plot) ---
        # Como o QQ-plot é visual, usamos uma heurística leve para comentar “caudas”:
        if len(resid_1d) >= 8:
            q1, q99 = np.quantile(resid_1d, [0.01, 0.99])
            q25, q75 = np.quantile(resid_1d, [0.25, 0.75])
            iqr = q75 - q25
        
            # regra simples: caudas “pesadas” se extremos são muito grandes vs IQR
            tail_score = (abs(q1) + abs(q99)) / (iqr + 1e-12)
        
            if tail_score <= 6:
                bul_res.append("O QQ-plot sugere boa aderência à Normal, sem desvios fortes nas caudas.")
            else:
                bul_res.append("O QQ-plot sugere possíveis desvios nas caudas (valores extremos mais pronunciados).")
        else:
            bul_res.append("O QQ-plot foi gerado; para amostras pequenas, priorize a interpretação visual.")
        
        # Exibição
        for r in bul_res:
            st.markdown(f"- {r}")

        
        # 2) Predição para qualquer combinação
        X_dum_cols = list(X_dum.columns)  # IMPORTANTÍSSIMO (mesmo layout do treino)
        
        levels_user, yhat_user = predicao_usuario_regressao(
            df_plan=df_plan,
            factor_cols=factor_cols,
            var_label=var_label,
            beta_hat=beta_hat,
            XtX_inv=XtX_inv,
            sigma2_hat=sigma2_hat,
            df_res=df_res,
            t_dist=t_dist,
            X_dum_cols=X_dum_cols,
        )
        
        # 3) Ponto ótimo via regressão
        ponto_otimo_regressao(
            df_plan=df_plan,
            factor_cols=factor_cols,
            var_label=var_label,
            beta_hat=beta_hat,
            XtX_inv=XtX_inv,
            sigma2_hat=sigma2_hat,
            df_res=df_res,
            t_dist=t_dist,
            X_dum_cols=X_dum_cols,
            per_factor_tables=per_factor_tables,
        )

        render_predicoes_otimo_reg_vs_taguchi_sem_ic(var_label)

        render_confirmacao_regressao_sem_upload(
            factor_cols=factor_cols,
            df_plan=df_plan,
            per_factor_tables=per_factor_tables,
            var_label=var_label,
            beta_hat=beta_hat,
            X_dum_cols=X_dum_cols,
        )


        
    


    # =============================================
    # 🔖 Abas de resultados
    # =============================================

    st.markdown("""
    <style>

    /* TODAS as abas (ativa e inativas) */
    button[role="tab"] {
        font-size: 24px !important;      /* fonte bem maior */
        font-weight: 900 !important;     /* negrito forte */
        padding: 14px 26px !important;   /* mais área clicável */
        margin-right: 10px !important;
        border-radius: 12px 12px 0 0 !important;
        border: none !important;
        background: #e3e9f7 !important;  /* cinza-azulado claro */
        color: #0a2d5c !important;       /* azul escuro */
    }

    /* ABA ATIVA */
    button[role="tab"][aria-selected="true"] {
        background: #ffffff !important;  /* fundo branco */
        color: #000000 !important;       /* texto preto */
        font-size: 28px !important;      /* ainda maior */
        font-weight: 900 !important;     /* super negrito */
    }

    </style>
    """, unsafe_allow_html=True)



    tab_efeitos, tab_inter, tab_3d, tab_pred, tab_conf, tab_anova, tab_reg = st.tabs(
        ["Efeitos e Delta", 
         "Interações (2D)", 
         "Interações (3D)", 
         "Predições",
         "Ensaios de confirmação", 
         "ANOVA", 
         "Regressão múltipla"]
    )

    with tab_efeitos:
        per_factor, grand_mean, factor_cols = mostrar_efeitos_e_graficos(
            lang, main_x_tmpl, main_y_default
        )
        mostrar_regra_delta()

    with tab_inter:

        # 2) Gráficos de interação entre fatores (S/N)
        mostrar_interacoes(lang, inter_x_tmpl, inter_y_default)


    with tab_3d:
        mostrar_superficie_3d()

    with tab_pred:
        tabelas_observado_predito()
        predicao_usuario()

    with tab_conf:
        render_ensaio_confirmacao(
        factor_cols=factor_cols,
        df_plan=df_plan,
        var_label=var_label,
        mean_y=mean_y,
        df_effects=df_join,   # aqui é o df com _SN por ensaio
        sn_col="_SN",
        per_factor_tables=per_factor,
        sn_tipo=sn_tipo,
        nominal_target=None,
    )

    with tab_anova:
        anova_taguchi()

    with tab_reg:
        regressao_multipla(per_factor_tables=per_factor)



def main():
    # A parte de configuração da página e var_label já está no topo do arquivo,
    # então aqui só chamamos os blocos principais.
    section_factors_and_oa()
    section_results()


if __name__ == "__main__":
    main()

