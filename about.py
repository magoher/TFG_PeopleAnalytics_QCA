import streamlit as st

def show():
    # ======= PAGE CONFIG =======
    st.title("About This Platform")
    st.markdown("### Trabajo Final de Graduación · LEAD University")
    st.markdown("---")

    # ======= INTRO =======
    st.header("Propósito del Proyecto")
    st.write(
        """
        Esta plataforma fue desarrollada como parte del **Trabajo Final de Graduación (TFG)** 
        en el área de **Analítica Cualitativa Comparada (QCA)** y **People Analytics**.
        Su objetivo principal es brindar una herramienta interactiva, robusta y reproducible 
        para ejecutar, visualizar y documentar análisis QCA desde una interfaz moderna e intuitiva.
        """
    )

    # ======= APPROACH =======
    st.header("Enfoque Metodológico")
    st.write(
        """
        El sistema está basado en la metodología **Qualitative Comparative Analysis (QCA)**, 
        incorporando elementos de:
        - *Calibration Theory*  
        - *Truth Table Construction*  
        - *Consistency & Coverage Analysis*  
        - *Prime Implicant Minimization*  
        - *Robustness Checking*  
        - *Causal Pathway Visualization*
        
        Lo anterior permite analizar configuraciones causales complejas mediante lógica booleana 
        y relaciones condicionales, en vez de depender únicamente de técnicas estadísticas tradicionales.
        """
    )

    # ======= SYSTEM ARCHITECTURE =======
    st.header("Arquitectura del Sistema")
    st.write(
        """
        La plataforma está estructurada en módulos especializados:

        **1. Calibration Module (`calibration.py`)**  
        Para transformar variables crudas en conjuntos difusos o nítidos.

        **2. Truth Table Engine (`truth_table.py`)**  
        Construcción de tablas de verdad con filtros avanzados y consistencia mínima.

        **3. QCA Engine (`ca_engine.py`)**  
        Minimización de configuraciones utilizando Quine-McCluskey.

        **4. QCA Results Explorer (`qca_results.py`)**  
        Comparación de soluciones (parsimoniosa, intermedia, compleja).

        **5. Robustness Check (`robustness_check.py`)**  
        Sensibilidad del modelo y estabilidad de caminos causales.

        **6. Configuration Visualizer (`configuration_visualizer.py`)**  
        Mapas de configuraciones, grafos causales, diagramas y visualizaciones interactivas.

        **7. Report Generator (`report_generator.py`)**  
        Exportación de un **PDF profesional** con todos los resultados del análisis.
        """
    )

    # ======= HOW TO USE =======
    st.header("Cómo Navegar la Plataforma")
    st.write(
        """
        El flujo lógico de trabajo recomendado es:

        **1. Subir dataset →**  
        Módulo de *Calibration* para definir umbrales y conjuntos.

        **2. Construir Truth Table →**  
        Filtrar por consistencia, frecuencia o condiciones relevantes.

        **3. Ejecutar el QCA Engine →**  
        Obtener términos minimizados y soluciones.

        **4. Revisar Resultados →**  
        Métricas, configuraciones y comparaciones.

        **5. Análisis de Robustez →**  
        Validar estabilidad del modelo.

        **6. Visualización Avanzada →**  
        Construir mapas causales y representaciones gráficas.

        **7. Exportar Reporte →**  
        Generar el documento con hallazgos finales.
        """
    )

    # ======= FINAL CREDITS =======
    st.header("Créditos y Reconocimientos")
    st.write(
        """
        **Autora del proyecto:**  
        *Marla Magoher*

        **Tutor académico:**  
        *[Nombre del Tutor]*

        **Universidad:**  
        *LEAD University, Costa Rica*

        **Año:** 2025
        
        ---
        Para consultas o mejoras, este sistema fue diseñado para ser extensible y modular.
        """
    )
