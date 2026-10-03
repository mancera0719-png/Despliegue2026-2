import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de Aprobación de Curso", layout="wide")

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación permite realizar predicciones de la nota final utilizando un modelo de Bagging pre-entrenado, ya sea ingresando datos manualmente o subiendo un archivo Excel.")

# Función para cargar artefactos
@st.cache_resource
def cargar_artefactos():
    try:
        one_hot_cols = joblib.load('one_hot_columns.joblib')
        scaler = joblib.load('min_max_scaler.joblib')
        model = joblib.load('bagging_optimizado.joblib')
        return one_hot_cols, scaler, model
    except Exception as e:
        st.error(f"Error al cargar los artefactos del modelo: {e}")
        return None, None, None

one_hot_transformer, scaler, model = cargar_artefactos()

if one_hot_transformer and scaler and model:
    # Definir las columnas de Felder que espera el modelo
    if isinstance(one_hot_transformer, list):
        si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
    else:
        si_columnas_one_hot = ['Felder_equilibrio', 'Felder_intuitivo', 'Felder_reflexivo', 'Felder_secuencial', 'Felder_sensorial', 'Felder_verbal', 'Felder_visual']

    # Menú de navegación lateral
    opcion = st.sidebar.selectbox("Selecciona el modo de entrada:", ["Predicción Individual", "Carga Masiva (Excel)"])

    if opcion == "Predicción Individual":
        st.header("Predicción Individual (Datos de Entrada)")
        opciones_felder = [col.replace('Felder_', '') for col in si_columnas_one_hot]
        if not opciones_felder:
            opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']

        felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder)
        examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)

        if st.button("Realizar Predicción"):
            try:
                # Crear DataFrame de una fila
                df_input = pd.DataFrame([{'Felder': felder_input, 'Examen_admisión': examen_input}])
                
                # Procesamiento
                df_procesado = df_input.copy()
                for col_name in si_columnas_one_hot:
                    valor_esperado = col_name.replace('Felder_', '')
                    df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(float)
                
                df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
                df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])[0][0]
                df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')

                columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
                df_procesado = df_procesado[columnas_ordenadas]

                # Predicción
                prediccion = model.predict(df_procesado)[0]
                st.success(f"### La predicción del modelo (Nota Final Estimada) es: {prediccion:.4f}")
                
                with st.expander("Ver variables procesadas enviadas al modelo"):
                    st.dataframe(df_procesado)
            except Exception as e:
                st.error(f"Ocurrió un error durante el procesamiento o la predicción: {e}")

    elif opcion == "Carga Masiva (Excel)":
        st.header("Predicción Masiva desde Archivo Excel")
        st.write("Sube un archivo de Excel (.xlsx) que contenga las variables `Felder` y `Examen_admisión`.")

        uploaded_file = st.file_uploader("Selecciona un archivo Excel", type=["xlsx"])

        if uploaded_file is not None:
            try:
                df_excel = pd.read_excel(uploaded_file)
                st.subheader("Datos Originales Subidos")
                st.dataframe(df_excel.head())

                # Validar columnas requeridas
                columnas_requeridas = ['Felder', 'Examen_admisión']
                faltantes = [col for col in columnas_requeridas if col not in df_excel.columns]

                if faltantes:
                    st.error(f"El archivo no contiene las siguientes columnas requeridas: {faltantes}")
                else:
                    if st.button("Procesar y Predecir Archivo"):
                        df_procesado = df_excel.copy()

                        # Aplicar codificación One-Hot para Felder de forma manual
                        for col_name in si_columnas_one_hot:
                            valor_esperado = col_name.replace('Felder_', '')
                            df_procesado[col_name] = (df_procesado['Felder'].astype(str).str.strip().str.lower() == valor_esperado.lower()).astype(float)

                        # Eliminar columnas irrelevantes para el modelo que estén en el dataset original
                        columnas_a_eliminar = ['ID', 'Año - Semestre', 'Felder', 'Nota_final', 'Aprobo']
                        df_procesado = df_procesado.drop(columns=[c for c in columnas_a_eliminar if c in df_procesado.columns], errors='ignore')

                        # Normalizar Examen_admisión
                        # El scaler requiere una matriz de 2D, transformamos toda la columna
                        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
                        df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')

                        # Reordenar columnas
                        columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
                        df_procesado = df_procesado[columnas_ordenadas]

                        # Realizar predicciones masivas
                        predicciones = model.predict(df_procesado)

                        # Agregar predicciones al DataFrame original para su descarga
                        df_resultado = df_excel.copy()
                        df_resultado['Nota_final_estimada'] = predicciones

                        st.success("¡Predicciones realizadas con éxito!")
                        st.subheader("Resultados Estimados")
                        st.dataframe(df_resultado)

                        # Permitir descargar el archivo resultante
                        @st.cache_data
                        def convert_df(df_to_download):
                            import io
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df_to_download.to_excel(writer, index=False, sheet_name='Predicciones')
                            return output.getvalue()

                        excel_data = convert_df(df_resultado)
                        st.download_button(
                            label="Descargar Excel con Predicciones",
                            data=excel_data,
                            file_name="predicciones_aprobacion.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            except Exception as e:
                st.error(f"Ocurrió un error al procesar el archivo Excel: {e}")
else:
    st.warning("Por favor, asegúrate de que los archivos 'one_hot_columns.joblib', 'min_max_scaler.joblib' y 'bagging_optimizado.joblib' se encuentren en la ruta correcta.")
