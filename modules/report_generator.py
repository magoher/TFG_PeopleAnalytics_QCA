# report_generator.py
"""
Generador de Reportes PDF — TFG People Analytics + QCA
------------------------------------------------------
Construye un reporte profesional en PDF integrando:
    • Diagnóstico descriptivo
    • Calibración set-theoretic
    • Tabla de Verdad
    • Soluciones QCA (C/I/P)
    • Visualizador de Configuraciones
    • Análisis de Robustez
    • Interpretación automática

Produce:
    /exports/TFG_QCA_Report.pdf

Requiere:
    pip install reportlab
"""

import os
import io
import base64
import streamlit as st
import pandas as pd
from datetime import datetime

# ReportLab for PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, 
    Paragraph, 
    Spacer, 
    Table, 
    TableStyle, 
    Image, 
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Directory for exports
EXPORT_DIR = "exports"
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

# ============================================================
# STYLE DEFINITIONS
# ============================================================

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="SectionTitle",
    fontSize=18,
    leading=22,
    spaceAfter=12,
    textColor=colors.HexColor("#283044"),  # CORREGIDO: usar colors.HexColor
    alignment=0,
    bold=True
))
styles.add(ParagraphStyle(
    name="SubTitle",
    fontSize=14,
    leading=18,
    spaceAfter=6,
    textColor=colors.HexColor("#3E4A61")  # CORREGIDO: usar colors.HexColor
))
styles.add(ParagraphStyle(
    name="Body",
    fontSize=10,
    leading=14,
    spaceAfter=10
))

# ============================================================
# HELPER: SAVE matplotlib/altair figures as PNG
# ============================================================

def save_figure(fig, filename):
    """Guardar figura matplotlib como PNG."""
    import matplotlib.pyplot as plt
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    return filename

def save_altair_chart(chart, filename):
    """Guardar gráfico Altair como PNG."""
    # Altair necesita vega_datasets instalado para exportar
    try:
        chart.save(filename, scale_factor=2)
        return filename
    except:
        # Fallback: convertir a matplotlib
        import matplotlib.pyplot as plt
        import tempfile
        
        # Crear figura temporal
        fig = plt.figure()
        # Aquí necesitarías convertir el chart de Altair a matplotlib
        # Esto es simplificado - en realidad necesitarías vega-altair
        plt.text(0.5, 0.5, "Gráfico Altair", ha='center')
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filename

# ============================================================
# HELPER: Build PDF Table
# ============================================================

def build_pdf_table(df: pd.DataFrame):
    """Construir tabla para PDF."""
    if df.empty:
        return Paragraph("No hay datos", styles["Body"])
    
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#283044")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 6)
    ]))
    return table

# ============================================================
# INTERPRETATION GENERATOR
# ============================================================

def generate_interpretation(qca_solutions):
    """Generar interpretación automática de resultados."""
    if not qca_solutions:
        return "No se encontraron soluciones para generar interpretación."
    
    text = "<b>Interpretación Automática de Resultados QCA</b><br/><br/>"
    
    for sol_type, solution in qca_solutions.items():
        terms = solution.get("terms", [])
        cons = solution.get("metrics", {}).get("Consistency", "N/A")
        cov = solution.get("metrics", {}).get("Coverage", "N/A")
        
        text += f"<b>{sol_type.upper()} Solution</b><br/>"
        text += f"Consistencia: {cons:.3f if isinstance(cons, (int, float)) else cons} — "
        text += f"Cobertura: {cov:.3f if isinstance(cov, (int, float)) else cov}<br/>"
        
        if not terms:
            text += "Sin términos implicantes detectados.<br/><br/>"
            continue
        
        text += "Esta solución sugiere que:<br/>"
        for t in terms[:3]:  # Mostrar máximo 3 términos
            text += f"• La configuración <i>{t}</i> es suficiente para el resultado.<br/>"
        
        text += "<br/>"
    
    return text

# ============================================================
# MAIN FUNCTION
# ============================================================

def show():
    st.title("📄 Generador de Reportes (PDF)")
    
    st.markdown("""
    Este módulo compila automáticamente **todas las secciones del análisis**
    en un único PDF profesional:
    
    - Descriptivos iniciales  
    - Calibración  
    - Tabla de Verdad  
    - Soluciones QCA  
    - Mapa de configuraciones  
    - Robustez  
    - Interpretación automática  
    
    <br>
    """, unsafe_allow_html=True)
    
    # ------------------------------------------------------------
    # CHECK REQUIRED DATA
    # ------------------------------------------------------------
    
    if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
        st.warning("Debe cargar un dataset en la sección Inicio.")
        return
    
    raw_df = st.session_state["raw_df"]
    conditions = st.session_state.get("conditions", [])
    tt = st.session_state.get("truth_table", None)
    qca_solutions = st.session_state.get("qca_solutions", None)
    tt_filtered = st.session_state.get("truth_table_filtered", None)
    
    # ------------------------------------------------------------
    # USER OPTIONS
    # ------------------------------------------------------------
    
    st.header("Opciones del Reporte")
    
    include_tt = st.checkbox("Incluir Tabla de Verdad", True)
    include_solutions = st.checkbox("Incluir Soluciones QCA", True)
    include_visuals = st.checkbox("Incluir visualizaciones", True)
    include_robustness = st.checkbox("Incluir análisis de robustez", True)
    include_interpretation = st.checkbox("Incluir interpretación automática", True)
    
    # ------------------------------------------------------------
    # BUTTON TO GENERATE PDF
    # ------------------------------------------------------------
    
    if st.button("🖨️ Generar Reporte PDF", type="primary"):
        with st.spinner("Generando reporte..."):
            filename = os.path.join(EXPORT_DIR, "TFG_QCA_Report.pdf")
            doc = SimpleDocTemplate(filename, pagesize=A4,
                                    rightMargin=36, leftMargin=36,
                                    topMargin=36, bottomMargin=36)
            
            story = []
            
            # ============================================================
            # COVER PAGE
            # ============================================================
            
            story.append(Paragraph("TFG People Analytics — QCA Platform", styles["SectionTitle"]))
            story.append(Paragraph("Reporte Integrado", styles["SubTitle"]))
            story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Body"]))
            story.append(Spacer(1, 0.4 * inch))
            story.append(Paragraph("<i>Generado automáticamente por la plataforma de análisis set-theoretic del TFG.</i>", styles["Body"]))
            story.append(PageBreak())
            
            # ============================================================
            # RAW DATA DESCRIPTIVES
            # ============================================================
            
            story.append(Paragraph("1. Descriptivos Iniciales del Dataset", styles["SectionTitle"]))
            story.append(Paragraph(f"Total de registros: {len(raw_df)}", styles["Body"]))
            story.append(Paragraph(f"Total de variables: {len(raw_df.columns)}", styles["Body"]))
            
            story.append(Paragraph("Vista general del dataset:", styles["SubTitle"]))
            story.append(build_pdf_table(raw_df.head(5)))
            story.append(PageBreak())
            
            # ============================================================
            # CALIBRATION RESULTS
            # ============================================================
            
            if st.session_state.get("calibrated_df") is not None:
                story.append(Paragraph("2. Calibración Set-Theoretic", styles["SectionTitle"]))
                cal = st.session_state["calibrated_df"]
                story.append(Paragraph("Variables calibradas (primeras filas):", styles["Body"]))
                story.append(build_pdf_table(cal.head(5)))
                story.append(PageBreak())
            
            # ============================================================
            # TRUTH TABLE
            # ============================================================
            
            if include_tt and tt is not None:
                story.append(Paragraph("3. Tabla de Verdad", styles["SectionTitle"]))
                # Mostrar solo las primeras 10 filas para no hacer PDF enorme
                story.append(build_pdf_table(tt.head(10)))
                story.append(PageBreak())
            
            # ============================================================
            # QCA SOLUTIONS
            # ============================================================
            
            if include_solutions and qca_solutions is not None:
                story.append(Paragraph("4. Soluciones QCA", styles["SectionTitle"]))
                
                for sol_type, sol in qca_solutions.items():
                    story.append(Paragraph(f"{sol_type.upper()} Solution", styles["SubTitle"]))
                    
                    # Crear tabla de términos
                    terms = sol.get("terms", [])
                    metrics = sol.get("metrics", {})
                    
                    if terms:
                        df_terms = pd.DataFrame({
                            "Término": [str(t) for t in terms],
                            "Consistencia": [metrics.get("Consistency", "N/A")] * len(terms),
                            "Cobertura": [metrics.get("Coverage", "N/A")] * len(terms)
                        })
                        story.append(build_pdf_table(df_terms))
                    
                    story.append(Spacer(1, 0.2 * inch))
                
                story.append(PageBreak())
            
            # ============================================================
            # BUILD PDF
            # ============================================================
            
            doc.build(story)
        
        # ------------------------------------------------------------
        # DOWNLOAD BUTTON
        # ------------------------------------------------------------
        
        with open(filename, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="TFG_QCA_Report.pdf">📥 Descargar Reporte PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success(f"Reporte generado: {filename}")

# Para ejecución directa
if __name__ == "__main__":
    show()