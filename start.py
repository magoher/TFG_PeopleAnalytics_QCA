import streamlit as st
import pandas as pd
import io

# ============================================================
#  START PAGE — CONTROL PANEL OF ANALYSIS
# ============================================================

def show():
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; margin-bottom: 30px">
        <h1 style="color: white; margin: 0">Data Input & Setup</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0">
            Upload your dataset and configure the analysis parameters for QCA
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state keys if they don't exist
    if 'analysis_ready' not in st.session_state:
        st.session_state.analysis_ready = False
    
    # ============================================================
    # 1. UPLOAD DATA SECTION
    # ============================================================
    
    st.markdown("## Upload Your Dataset")
    
    # Create two columns for delimiter and file upload
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### CSV Settings")
        delimiter_option = st.selectbox(
            "CSV Delimiter:",
            ["Comma (,)", "Semicolon (;)", "Tab", "Pipe (|)", "Space", "Custom"],
            help="Select the character that separates columns in your CSV file",
            key="delimiter_option"
        )
        
        # Map delimiter options to actual characters
        delimiter_map = {
            "Comma (,)": ",",
            "Semicolon (;)": ";",
            "Tab": "\t",
            "Pipe (|)": "|",
            "Space": " ",
            "Custom": None
        }
        
        delimiter = delimiter_map[delimiter_option]
        
        if delimiter_option == "Custom":
            delimiter = st.text_input("Enter custom delimiter:", value=",", max_chars=5, key="custom_delimiter")
        
        # Missing values handling
        st.markdown("#### Missing Values")
        missing_strategy = st.radio(
            "Handle missing values:",
            ["Remove rows", "Keep as-is", "Fill with median", "Fill with mean"],
            horizontal=True,
            help="How to handle empty or NaN values in the dataset",
            key="missing_strategy"
        )
    
    with col2:
        st.markdown("#### File Upload")
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV file here",
            type=["csv", "txt", "xlsx", "xls"],
            help="Supported formats: CSV, TXT, Excel",
            key="file_uploader"
        )
    
    # ============================================================
    # DATA PROCESSING
    # ============================================================
    
    if uploaded_file is not None:
        try:
            # Get file extension
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            # Read file content as bytes
            content_bytes = uploaded_file.getvalue()
            
            if file_ext in ['csv', 'txt']:
                # Try multiple encodings
                encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                content_str = None
                
                for encoding in encodings_to_try:
                    try:
                        content_str = content_bytes.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content_str is None:
                    # Fallback to utf-8 with errors ignored
                    content_str = content_bytes.decode('utf-8', errors='ignore')
                
                # Determine actual delimiter to use
                if delimiter_option == "Custom" and 'custom_delimiter' in st.session_state:
                    actual_delimiter = st.session_state.custom_delimiter
                else:
                    actual_delimiter = delimiter if delimiter is not None else ","
                
                # Create StringIO object from content
                string_data = io.StringIO(content_str)
                
                # Try reading with selected delimiter first
                try:
                    df = pd.read_csv(string_data, delimiter=actual_delimiter, engine='python')
                    
                    # Check if we got more than 1 column
                    if df.shape[1] <= 1:
                        st.warning(f"Only {df.shape[1]} column detected with delimiter '{actual_delimiter}'. Trying auto-detection...")
                        string_data.seek(0)
                        df = pd.read_csv(string_data, sep=None, engine='python')
                        
                except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                    # If fails, try auto-detection
                    string_data.seek(0)
                    df = pd.read_csv(string_data, sep=None, engine='python')
                    st.info(f"Used auto-detected delimiter. Detected {df.shape[1]} columns.")
                    
            elif file_ext in ['xlsx', 'xls']:
                # For Excel files, read directly
                df = pd.read_excel(uploaded_file)
                
            else:
                st.error(f"Unsupported file format: .{file_ext}")
                return
            
            # Reset file pointer for potential future use
            uploaded_file.seek(0)
            
            # Store original content in session state
            st.session_state["uploaded_file_content"] = content_bytes
            st.session_state["uploaded_file_name"] = uploaded_file.name
            
            st.success(f"File loaded: **{uploaded_file.name}**")
            
            # Display file stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                missing_count = df.isnull().sum().sum()
                st.metric("Missing Values", missing_count, delta_color="inverse")
            
            # Handle missing values
            df_original_shape = df.shape
            if missing_strategy == "Remove rows":
                df = df.dropna()
                rows_removed = df_original_shape[0] - df.shape[0]
                if rows_removed > 0:
                    st.info(f"Removed {rows_removed} rows with missing values")
            elif missing_strategy == "Fill with median":
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                st.info("Filled numeric columns with median values")
            elif missing_strategy == "Fill with mean":
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                st.info("Filled numeric columns with mean values")
            
            # Store cleaned dataframe in session state
            st.session_state["raw_df"] = df
            
            # ============================================================
            # PREVIEW SECTION
            # ============================================================
            
            tab1, tab2, tab3 = st.tabs(["Data Preview", "Column Types", "Missing Analysis"])
            
            with tab1:
                st.dataframe(
                    df.head(20),
                    use_container_width=True,
                    height=400
                )
                
            with tab2:
                # Show column data types
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Unique Values': df.nunique(),
                    'Sample Values': df.iloc[0].astype(str) if len(df) > 0 else ''
                })
                st.dataframe(dtype_df, use_container_width=True)
                
            with tab3:
                if df.isnull().sum().sum() > 0:
                    missing_df = pd.DataFrame({
                        'Column': df.columns,
                        'Missing Count': df.isnull().sum(),
                        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    st.dataframe(missing_df, use_container_width=True)
                    
                    if not missing_df.empty:
                        st.bar_chart(missing_df.set_index('Column')['Missing %'])
                else:
                    st.success("No missing values found!")
            
            # ============================================================
            # 2. ANALYSIS CONFIGURATION
            # ============================================================
            
            st.markdown("---")
            st.markdown("## Analysis Configuration")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.markdown("#### Analysis Type")
                analysis_type = st.selectbox(
                    "Select primary analysis:",
                    [
                        "QCA — Qualitative Comparative Analysis",
                        "Truth Table Analysis",
                        "Calibration & Fuzzy Sets",
                        "Necessary Condition Analysis (NCA)",
                        "Robustness & Sensitivity",
                        "Comparative Case Analysis"
                    ],
                    help="Choose the type of analysis you want to perform"
                )
                st.session_state["analysis_type"] = analysis_type
            
            with config_col2:
                st.markdown("#### Outcome Selection")
                columns = df.columns.tolist()
                
                outcome_col = st.selectbox(
                    "Select outcome variable (Y):",
                    columns,
                    index=len(columns)-1 if len(columns) > 0 else 0,
                    help="The dependent variable you want to explain"
                )
                st.session_state["outcome"] = outcome_col
            
            # ============================================================
            # 3. CONDITIONS SELECTION
            # ============================================================
            
            st.markdown("#### Condition Variables Selection")
            
            # Auto-suggest conditions (all columns except outcome)
            suggested_conditions = [col for col in columns if col != outcome_col]
            
            conditions = st.multiselect(
                "Select explanatory conditions (X):",
                suggested_conditions,
                default=suggested_conditions[:min(8, len(suggested_conditions))],
                help="Select the independent variables (conditions) for the analysis"
            )
            
            if len(conditions) == 0:
                st.warning("Please select at least one condition variable.")
                return
            
            st.session_state["conditions"] = conditions
            
            # Display selected configuration
            st.info(f"""
            **Configuration Summary:**
            - **Analysis Type:** {analysis_type}
            - **Outcome Variable:** {outcome_col}
            - **Condition Variables:** {len(conditions)} selected
            - **Total Cases:** {df.shape[0]}
            """)
            
            # ============================================================
            # 4. CONTINUE BUTTON
            # ============================================================
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("Start Analysis", type="primary", use_container_width=True):
                    st.session_state["ready_for_analysis"] = True
                    st.session_state["analysis_ready"] = True
                    
                    st.success("Analysis configuration saved successfully!")
                    
                    st.markdown("""
                    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; margin-top: 20px">
                        <h4 style="margin: 0; color: #2e7d32">Next Steps:</h4>
                        <p style="margin: 5px 0; color: #555">
                        Navigate to <strong>Calibration</strong> in the sidebar to begin the QCA process.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ============================================================
            # DATA DOWNLOAD (CLEANED VERSION)
            # ============================================================
            
            with st.expander("Download Cleaned Dataset"):
                csv = df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"cleaned_{uploaded_file.name}",
                    mime="text/csv",
                    help="Download the cleaned dataset with missing values handled"
                )
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("""
            **Troubleshooting tips:**
            1. Check if the delimiter is correct
            2. Ensure file encoding is UTF-8
            3. Verify the file is not corrupted
            4. Try opening in Excel and re-saving as CSV
            """)
    
    else:
        # Show upload instructions when no file is uploaded
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 30px; border-radius: 10px; text-align: center; border: 2px dashed #ccc">
            <h3 style="color: #666">Ready to Upload</h3>
            <p style="color: #888">
            Upload a CSV or Excel file to begin your QCA analysis.<br>
            The file should contain your cases (rows) and variables (columns).
            </p>
            <div style="margin-top: 20px">
                <div style="display: inline-block; background-color: #e3f2fd; padding: 10px 20px; border-radius: 5px; margin: 5px">
                    <strong>CSV Requirements:</strong>
                </div>
                <div style="display: inline-block; background-color: #f3e5f5; padding: 10px 20px; border-radius: 5px; margin: 5px">
                    <strong>Numeric/Text Data</strong>
                </div>
                <div style="display: inline-block; background-color: #e8f5e9; padding: 10px 20px; border-radius: 5px; margin: 5px">
                    <strong>Clear Column Names</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# For direct execution
if __name__ == "__main__":
    show()