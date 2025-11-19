from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from volare_model.serving import LocalModelService


DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "Cleaned_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model_pipeline.joblib"
DEFAULT_API_URL = "http://localhost:8000"


@st.cache_resource(show_spinner=False)
def load_model_service():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {MODEL_PATH}. "
            "Ejecutá el notebook tp3_modelado_vuelos.ipynb para generar el artefacto."
        )
    return LocalModelService(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("No se encontró el dataset base para poblar opciones del formulario.")
    return pd.read_csv(DATA_PATH)


def get_unique_sorted(df: pd.DataFrame, column: str) -> list[str]:
    return sorted(df[column].dropna().unique().tolist())


def main():
    st.set_page_config(page_title="Estimador de tarifas aéreas", layout="wide")
    st.title("Estimador interactivo de tarifas de vuelos domésticos")
    st.write(
        """Esta app utiliza un modelo de regresión entrenado localmente para estimar tarifas
        de vuelos domésticos. Podés explorar cómo varían las predicciones según ruta, clase,
        días de anticipación, horarios y escalas."""
    )

    df = load_reference_data()
    model_service = load_model_service()

    with st.sidebar:
        st.header("Configuración de inferencia")
        inference_mode = st.radio(
            "Modo", ["Pipeline local", "API FastAPI"], index=0, help="La API debe estar corriendo en el puerto definido."
        )
        api_base_url = st.text_input("URL base de la API", value=DEFAULT_API_URL)
        st.caption(f"Artefacto cargado: {MODEL_PATH.name}")

    tab_pred, tab_whatif, tab_cases, tab_manual, tab_analysis = st.tabs(
        ["Predicción", "What-if", "Casos típicos", "Predicción completa", "Análisis del dataset"]
    )

    # --- Pestaña 1: Predicción principal con comparaciones enriquecidas ---
    with tab_pred:
        col1, col2 = st.columns(2)

        with col1:
            journey_date = st.date_input("Fecha del vuelo", value=date.today())
            source_options = get_unique_sorted(df, "Source")
            dest_options = get_unique_sorted(df, "Destination")
            try:
                default_source_idx = source_options.index("Delhi")
            except ValueError:
                default_source_idx = 0
            try:
                default_dest_idx = dest_options.index("Mumbai")
            except ValueError:
                default_dest_idx = 0
            source = st.selectbox("Origen", options=source_options, index=default_source_idx)
            destination = st.selectbox("Destino", options=dest_options, index=default_dest_idx)

        with col2:
            class_options = get_unique_sorted(df, "Class")
            try:
                default_class_idx = class_options.index("Economy")
            except ValueError:
                default_class_idx = 0
            flight_class = st.selectbox("Clase", options=class_options, index=default_class_idx)
            st.caption("El resto de parámetros se evalúan automáticamente con promedios/rangos.")

        # Valores base derivados del dataset
        base_mask = (df["Source"] == source) & (df["Destination"] == destination) & (df["Class"] == flight_class)
        base_subset = df[base_mask] if not base_mask.empty else df

        base_airline = base_subset["Airline"].mode().iloc[0] if not base_subset.empty else df["Airline"].mode().iloc[0]
        base_duration = float(base_subset["Duration_in_hours"].median()) if not base_subset.empty else float(
            df["Duration_in_hours"].median()
        )
        days_left_default = int(base_subset["Days_left"].median()) if not base_subset.empty else int(
            df["Days_left"].median()
        )

        departure_slots = get_unique_sorted(df, "Departure")
        arrival_slots = get_unique_sorted(df, "Arrival")
        stop_options = get_unique_sorted(df, "Total_stops")

        def predict_with_payload(payload: dict) -> float | None:
            try:
                if inference_mode == "Pipeline local":
                    return model_service.predict(payload)
                response = httpx.post(
                    f"{api_base_url.rstrip('/')}/predict",
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                return float(data.get("fare"))
            except Exception as exc:  # pragma: no cover
                st.error(f"Error al generar la predicción: {exc}")
                return None

        if st.button("Calcular rangos de predicción", type="primary"):
            st.success(
                f"Usando aerolínea más frecuente para la ruta ({base_airline}), duración ~{base_duration:.1f}h y "
                f"{days_left_default} días de anticipación como base."
            )

            # Rango por franja de salida
            dep_results = []
            for dep in departure_slots:
                payload = {
                    "Date_of_journey": journey_date.isoformat(),
                    "Journey_day": journey_date.strftime("%A"),
                    "Airline": base_airline,
                    "Flight_code": None,
                    "Class": flight_class,
                    "Source": source,
                    "Departure": dep,
                    "Total_stops": stop_options[0],
                    "Arrival": arrival_slots[0],
                    "Destination": destination,
                    "Duration_in_hours": base_duration,
                    "Days_left": days_left_default,
                }
                pred = predict_with_payload(payload)
                if pred is not None:
                    dep_results.append({"Salida": dep, "Tarifa (INR)": pred})
            if dep_results:
                st.subheader("Rango por franja horaria de salida")
                st.table(
                    pd.DataFrame(dep_results)
                    .sort_values("Tarifa (INR)")
                    .style.format({"Tarifa (INR)": "{:,.0f}"})
                )

            # Rango por franja de llegada
            arr_results = []
            for arr in arrival_slots:
                payload = {
                    "Date_of_journey": journey_date.isoformat(),
                    "Journey_day": journey_date.strftime("%A"),
                    "Airline": base_airline,
                    "Flight_code": None,
                    "Class": flight_class,
                    "Source": source,
                    "Departure": departure_slots[0],
                    "Total_stops": stop_options[0],
                    "Arrival": arr,
                    "Destination": destination,
                    "Duration_in_hours": base_duration,
                    "Days_left": days_left_default,
                }
                pred = predict_with_payload(payload)
                if pred is not None:
                    arr_results.append({"Llegada": arr, "Tarifa (INR)": pred})
            if arr_results:
                st.subheader("Rango por franja horaria de llegada")
                st.table(
                    pd.DataFrame(arr_results)
                    .sort_values("Tarifa (INR)")
                    .style.format({"Tarifa (INR)": "{:,.0f}"})
                )

            # Rango por cantidad de escalas
            stop_results = []
            for stop in stop_options:
                payload = {
                    "Date_of_journey": journey_date.isoformat(),
                    "Journey_day": journey_date.strftime("%A"),
                    "Airline": base_airline,
                    "Flight_code": None,
                    "Class": flight_class,
                    "Source": source,
                    "Departure": departure_slots[0],
                    "Total_stops": stop,
                    "Arrival": arrival_slots[0],
                    "Destination": destination,
                    "Duration_in_hours": base_duration,
                    "Days_left": days_left_default,
                }
                pred = predict_with_payload(payload)
                if pred is not None:
                    stop_results.append({"Escalas": stop, "Tarifa (INR)": pred})
            if stop_results:
                st.subheader("Rango por cantidad de escalas")
                st.table(
                    pd.DataFrame(stop_results)
                    .sort_values("Tarifa (INR)")
                    .style.format({"Tarifa (INR)": "{:,.0f}"})
                )

            # Curva por días de anticipación
            days_range = np.linspace(df["Days_left"].min(), df["Days_left"].max(), num=12, dtype=int)
            curve_results = []
            for dleft in days_range:
                payload = {
                    "Date_of_journey": journey_date.isoformat(),
                    "Journey_day": journey_date.strftime("%A"),
                    "Airline": base_airline,
                    "Flight_code": None,
                    "Class": flight_class,
                    "Source": source,
                    "Departure": departure_slots[0],
                    "Total_stops": stop_options[0],
                    "Arrival": arrival_slots[0],
                    "Destination": destination,
                    "Duration_in_hours": base_duration,
                    "Days_left": int(dleft),
                }
                pred = predict_with_payload(payload)
                if pred is not None:
                    curve_results.append({"Days_left": int(dleft), "Tarifa (INR)": pred})
            if curve_results:
                st.subheader("Cómo cambia la predicción según los días de anticipación")
                curve_df = pd.DataFrame(curve_results)
                st.line_chart(curve_df.set_index("Days_left"))

            # Comparación por aerolínea
            airline_results = []
            airline_options = get_unique_sorted(df, "Airline")
            for air in airline_options:
                payload = {
                    "Date_of_journey": journey_date.isoformat(),
                    "Journey_day": journey_date.strftime("%A"),
                    "Airline": air,
                    "Flight_code": None,
                    "Class": flight_class,
                    "Source": source,
                    "Departure": departure_slots[0],
                    "Total_stops": stop_options[0],
                    "Arrival": arrival_slots[0],
                    "Destination": destination,
                    "Duration_in_hours": base_duration,
                    "Days_left": days_left_default,
                }
                pred = predict_with_payload(payload)
                if pred is not None:
                    airline_results.append({"Airline": air, "Tarifa (INR)": pred})

            if airline_results:
                st.subheader("Cómo cambia la predicción según la aerolínea")
                airline_df = pd.DataFrame(airline_results).sort_values("Tarifa (INR)")
                fig_air, ax_air = plt.subplots(figsize=(8, 4))
                sns.barplot(data=airline_df, x="Airline", y="Tarifa (INR)", ax=ax_air)
                ax_air.set_ylabel("Tarifa estimada (INR)")
                ax_air.set_xlabel("Aerolínea")
                ax_air.tick_params(axis="x", rotation=45)
                st.pyplot(fig_air)

    # --- Pestaña 2: What-if analysis ---
    with tab_whatif:
        st.subheader("Exploración de escenarios (what-if)")
        st.caption(
            "Partimos de un set mínimo de parámetros (día, origen, destino, clase) y exploramos el efecto "
            "de días de anticipación, aerolínea predominante y escalas."
        )

        col1, col2 = st.columns(2)

        with col1:
            base_date = st.date_input("Fecha base del vuelo", value=date.today(), key="whatif_date")
            source_options_w = get_unique_sorted(df, "Source")
            dest_options_w = get_unique_sorted(df, "Destination")
            class_options_w = get_unique_sorted(df, "Class")
            try:
                default_source_idx_w = source_options_w.index("Delhi")
            except ValueError:
                default_source_idx_w = 0
            try:
                default_dest_idx_w = dest_options_w.index("Mumbai")
            except ValueError:
                default_dest_idx_w = 0
            try:
                default_class_idx_w = class_options_w.index("Economy")
            except ValueError:
                default_class_idx_w = 0
            base_source = st.selectbox(
                "Origen", options=source_options_w, index=default_source_idx_w, key="whatif_source"
            )
            base_destination = st.selectbox(
                "Destino", options=dest_options_w, index=default_dest_idx_w, key="whatif_destination"
            )
            base_class = st.selectbox(
                "Clase", options=class_options_w, index=default_class_idx_w, key="whatif_class"
            )
        with col2:
            st.caption("Se usan promedios de duración y aerolínea predominante para la ruta.")

        base_mask_w = (
            (df["Source"] == base_source) & (df["Destination"] == base_destination) & (df["Class"] == base_class)
        )
        base_subset_w = df[base_mask_w] if not base_mask_w.empty else df
        base_airline_w = (
            base_subset_w["Airline"].mode().iloc[0] if not base_subset_w.empty else df["Airline"].mode().iloc[0]
        )
        base_duration_w = (
            float(base_subset_w["Duration_in_hours"].median())
            if not base_subset_w.empty
            else float(df["Duration_in_hours"].median())
        )
        base_days_left_w = (
            int(base_subset_w["Days_left"].median()) if not base_subset_w.empty else int(df["Days_left"].median())
        )

        min_days_w = int(df["Days_left"].min())
        max_days_w = int(df["Days_left"].max())
        interactive_days = st.slider(
            "¿Qué pasa si compro X días antes/después? (ajusta los días de anticipación)",
            min_value=min_days_w,
            max_value=max_days_w,
            value=base_days_left_w,
            key="whatif_days_interactive",
        )

        stop_options_w = get_unique_sorted(df, "Total_stops")
        departure_slots_w = get_unique_sorted(df, "Departure")
        arrival_slots_w = get_unique_sorted(df, "Arrival")

        if st.button("Calcular escenarios what-if"):
            scenarios = []

            def build_payload(total_stops: str, dep: str, arr: str, days_left_val: int, label: str) -> dict:
                return {
                    "label": label,
                    "payload": {
                        "Date_of_journey": base_date.isoformat(),
                        "Journey_day": base_date.strftime("%A"),
                        "Airline": base_airline_w,
                        "Flight_code": None,
                        "Class": base_class,
                        "Source": base_source,
                        "Departure": dep,
                        "Total_stops": total_stops,
                        "Arrival": arr,
                        "Destination": base_destination,
                        "Duration_in_hours": base_duration_w,
                        "Days_left": int(days_left_val),
                    },
                }

            # Escenarios: base, cambiar días_left, vuelo directo, vuelo con escalas, distintas franjas
            scenarios.append(build_payload(stop_options_w[0], departure_slots_w[0], arrival_slots_w[0], base_days_left_w, "Base"))
            scenarios.append(build_payload(stop_options_w[0], departure_slots_w[0], arrival_slots_w[0], interactive_days, f"Comprar con {interactive_days} días de anticipación"))
            if len(stop_options_w) > 1:
                scenarios.append(build_payload(stop_options_w[-1], departure_slots_w[0], arrival_slots_w[0], base_days_left_w, f"Escalas: {stop_options_w[-1]}"))
            if len(departure_slots_w) > 1:
                scenarios.append(build_payload(stop_options_w[0], departure_slots_w[-1], arrival_slots_w[0], base_days_left_w, f"Salida: {departure_slots_w[-1]}"))
            if len(arrival_slots_w) > 1:
                scenarios.append(build_payload(stop_options_w[0], departure_slots_w[0], arrival_slots_w[-1], base_days_left_w, f"Llegada: {arrival_slots_w[-1]}"))

            results = []
            for scenario in scenarios:
                try:
                    pred_val = predict_with_payload(scenario["payload"])
                    if pred_val is None:
                        continue
                    results.append(
                        {
                            "Escenario": scenario["label"],
                            "Escalas": scenario["payload"]["Total_stops"],
                            "Salida": scenario["payload"]["Departure"],
                            "Llegada": scenario["payload"]["Arrival"],
                            "Días anticipación": scenario["payload"]["Days_left"],
                            "Duración (h)": scenario["payload"]["Duration_in_hours"],
                            "Tarifa estimada (INR)": pred_val,
                        }
                    )
                except Exception as exc:  # pragma: no cover
                    st.error(f"Error al obtener predicción para '{scenario['label']}': {exc}")

            if results:
                results_df = pd.DataFrame(results)

                # Calcular diferencias contra el escenario base
                base_mask = results_df["Escenario"] == "Base"
                if base_mask.any():
                    base_price = results_df.loc[base_mask, "Tarifa estimada (INR)"].iloc[0]
                    results_df["Δ vs base (INR)"] = results_df["Tarifa estimada (INR)"] - base_price
                    results_df["Δ vs base (%)"] = (
                        results_df["Δ vs base (INR)"] / base_price * 100.0 if base_price != 0 else 0
                    )
                else:
                    results_df["Δ vs base (INR)"] = 0.0
                    results_df["Δ vs base (%)"] = 0.0

                # Ordenar por precio (más barato primero)
                results_df = results_df.sort_values("Tarifa estimada (INR)").reset_index(drop=True)

                # Identificar mejor escenario (más barato)
                best_row = results_df.iloc[0]
                best_label = best_row["Escenario"]
                best_price = best_row["Tarifa estimada (INR)"]
                best_delta_pct = best_row["Δ vs base (%)"]

                st.subheader("Comparación de escenarios")
                styled = results_df.style.format(
                    {
                        "Tarifa estimada (INR)": "{:,.0f}",
                        "Δ vs base (INR)": "{:+,.0f}",
                        "Δ vs base (%)": "{:+.1f}%",
                        "Duración (h)": "{:.1f}",
                    }
                )
                # Resaltar el escenario más barato
                def highlight_best(row):
                    return ["background-color: #d5f5e3" if row["Escenario"] == best_label else "" for _ in row]

                styled = styled.apply(highlight_best, axis=1)
                st.dataframe(styled)

                # Resumen textual
                if base_mask.any():
                    if best_label == "Base":
                        st.info(
                            "El escenario base ya es la opción más económica entre los analizados "
                            f"(tarifa estimada ~₹{best_price:,.0f})."
                        )
                    else:
                        st.success(
                            f"La opción más económica es **{best_label}**, con una tarifa estimada de "
                            f"~₹{best_price:,.0f}, lo que implica un cambio de "
                            f"{best_delta_pct:+.1f}% respecto al escenario base."
                        )

    # --- Pestaña 3: Casos típicos ---
    with tab_cases:
        st.subheader("Casos típicos de viaje")
        st.caption(
            "Seleccioná un tipo de viaje para ver escenarios típicos sugeridos para la ruta Delhi-Mumbai."
        )

        preset_type = st.selectbox(
            "Tipo de viaje",
            options=[
                "Viaje de negocios (poca anticipación)",
                "Viaje planificado (anticipación alta)",
                "Escapada de fin de semana",
            ],
        )

        # Parámetros fijos de ruta/clase
        preset_source = "Delhi" if "Delhi" in get_unique_sorted(df, "Source") else df["Source"].mode().iloc[0]
        preset_dest = "Mumbai" if "Mumbai" in get_unique_sorted(df, "Destination") else df["Destination"].mode().iloc[0]
        preset_class = "Economy" if "Economy" in get_unique_sorted(df, "Class") else df["Class"].mode().iloc[0]

        # Configurar days_left típico según tipo de viaje
        if "negocios" in preset_type:
            days_left_options = [1, 3, 7]
            description = "Viajes con muy poca anticipación, fechas flexibles y foco en minimizar tiempo."
        elif "planificado" in preset_type:
            days_left_options = [15, 30, 60]
            description = "Viajes comprados con bastante tiempo de anticipación, buscando mejor precio."
        else:
            days_left_options = [7, 14, 21]
            description = "Escapadas cortas, normalmente de fin de semana o feriados."

        st.info(description)

        # Variables base para la ruta
        base_mask_case = (
            (df["Source"] == preset_source) & (df["Destination"] == preset_dest) & (df["Class"] == preset_class)
        )
        base_subset_case = df[base_mask_case] if not base_mask_case.empty else df
        base_airline_case = (
            base_subset_case["Airline"].mode().iloc[0]
            if not base_subset_case.empty
            else df["Airline"].mode().iloc[0]
        )
        base_duration_case = (
            float(base_subset_case["Duration_in_hours"].median())
            if not base_subset_case.empty
            else float(df["Duration_in_hours"].median())
        )

        departure_slots_case = get_unique_sorted(df, "Departure")
        arrival_slots_case = get_unique_sorted(df, "Arrival")
        stop_options_case = get_unique_sorted(df, "Total_stops")

        if st.button("Generar casos típicos"):
            today = date.today()
            day_name = today.strftime("%A")

            cases = []
            for dleft in days_left_options:
                payload = {
                    "Date_of_journey": today.isoformat(),
                    "Journey_day": day_name,
                    "Airline": base_airline_case,
                    "Flight_code": None,
                    "Class": preset_class,
                    "Source": preset_source,
                    "Departure": departure_slots_case[0],
                    "Total_stops": stop_options_case[0],
                    "Arrival": arrival_slots_case[0],
                    "Destination": preset_dest,
                    "Duration_in_hours": base_duration_case,
                    "Days_left": dleft,
                }
                pred = predict_with_payload(payload)
                if pred is not None:
                    cases.append(
                        {
                            "Tipo de viaje": preset_type,
                            "Días anticipación": dleft,
                            "Origen": preset_source,
                            "Destino": preset_dest,
                            "Clase": preset_class,
                            "Aerolínea": base_airline_case,
                            "Tarifa estimada (INR)": pred,
                        }
                    )

            if cases:
                cases_df = pd.DataFrame(cases).sort_values("Días anticipación")
                st.dataframe(
                    cases_df.style.format(
                        {
                            "Tarifa estimada (INR)": "{:,.0f}",
                        }
                    )
                )

    # --- Pestaña 4: Predicción completa con todos los parámetros ---
    with tab_manual:
        st.subheader("Predicción completa (todas las variables)")
        st.caption(
            "Configurá manualmente todas las variables del vuelo para obtener una predicción precisa."
        )

        col1, col2 = st.columns(2)

        with col1:
            manual_date = st.date_input("Fecha del vuelo (manual)", value=date.today(), key="manual_date")
            manual_day = manual_date.strftime("%A")
            manual_airline = st.selectbox("Aerolínea", options=get_unique_sorted(df, "Airline"), key="manual_airline")
            manual_class = st.selectbox("Clase", options=get_unique_sorted(df, "Class"), key="manual_class")
            manual_source = st.selectbox("Origen", options=get_unique_sorted(df, "Source"), key="manual_source")
            manual_destination = st.selectbox(
                "Destino", options=get_unique_sorted(df, "Destination"), key="manual_destination"
            )
            manual_flight_code = st.text_input("Código de vuelo (opcional)", key="manual_flight_code")

        with col2:
            manual_departure = st.selectbox(
                "Franja horaria de salida", options=get_unique_sorted(df, "Departure"), key="manual_departure"
            )
            manual_arrival = st.selectbox(
                "Franja horaria de llegada", options=get_unique_sorted(df, "Arrival"), key="manual_arrival"
            )
            manual_stops = st.selectbox(
                "Cantidad de escalas", options=get_unique_sorted(df, "Total_stops"), key="manual_stops"
            )
            manual_days_left = st.slider(
                "Días de anticipación (Days_left)",
                min_value=int(df["Days_left"].min()),
                max_value=int(df["Days_left"].max()),
                value=int(df["Days_left"].median()),
                key="manual_days_left",
            )
            manual_duration = st.slider(
                "Duración estimada (horas)",
                min_value=float(np.floor(df["Duration_in_hours"].min())),
                max_value=float(np.ceil(df["Duration_in_hours"].max())),
                value=float(df["Duration_in_hours"].median()),
                step=0.25,
                key="manual_duration",
            )

        if st.button("Calcular predicción manual", type="primary"):
            manual_payload = {
                "Date_of_journey": manual_date.isoformat(),
                "Journey_day": manual_day,
                "Airline": manual_airline,
                "Flight_code": manual_flight_code or None,
                "Class": manual_class,
                "Source": manual_source,
                "Departure": manual_departure,
                "Total_stops": manual_stops,
                "Arrival": manual_arrival,
                "Destination": manual_destination,
                "Duration_in_hours": manual_duration,
                "Days_left": manual_days_left,
            }
            pred = predict_with_payload(manual_payload)
            if pred is not None:
                st.success(f"Tarifa estimada para este vuelo: ₹{pred:,.0f} (INR)")

    # --- Pestaña 3: Análisis exploratorio del dataset ---
    with tab_analysis:
        st.subheader("Análisis exploratorio del dataset de vuelos")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribución de tarifas (Fare)**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df["Fare"], bins=40, kde=True, ax=ax)
            ax.set_xlabel("Fare (INR)")
            st.pyplot(fig)

            st.markdown("**Top 10 rutas por cantidad de vuelos**")
            routes = df["Source"] + "-" + df["Destination"]
            top_routes = routes.value_counts().head(10)
            st.bar_chart(top_routes)

        with col2:
            st.markdown("**Aerolíneas con más vuelos**")
            airline_counts = df["Airline"].value_counts().head(10)
            st.bar_chart(airline_counts)

            st.markdown("**Distribución de días de anticipación (Days_left)**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(df["Days_left"], bins=40, kde=False, ax=ax2)
            ax2.set_xlabel("Days_left")
            st.pyplot(fig2)

        st.markdown("**Relación entre duración y tarifa**")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            data=df.sample(min(2000, len(df)), random_state=42),
            x="Duration_in_hours",
            y="Fare",
            hue="Total_stops",
            alpha=0.4,
            ax=ax3,
        )
        ax3.set_xlabel("Duración (horas)")
        ax3.set_ylabel("Fare (INR)")
        ax3.legend(title="Escalas", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig3)

        st.markdown(
            "Este panel resume las tendencias principales: distribución de precios, rutas frecuentes, "
            "aerolíneas con más vuelos y cómo se relacionan duración, anticipación y tarifa."
        )


if __name__ == "__main__":
    main()
