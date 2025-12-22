import pandas as pd
import ast
import os

# List of files to process
files = [
    "ket_qua_cluster_ga_full.csv",
    "ket_qua_dfa_full.csv",
    "ket_qua_esa_full.csv",
    "ket_qua_ga_ombuki_full.csv",
    "ket_qua_ga_pd_hct_full.csv"
]

all_dataframes = []

for filename in files:
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            
            # Function to parse the details string safely
            def parse_details(details_str):
                try:
                    return ast.literal_eval(details_str)
                except (ValueError, SyntaxError):
                    return {}

            # Apply parsing
            if 'details' in df.columns:
                # Expand the details column into separate columns
                details_expanded = df['details'].apply(parse_details).apply(pd.Series)
                
                # Drop the original details column and concatenate with the new columns
                df_expanded = pd.concat([df.drop('details', axis=1), details_expanded], axis=1)
                all_dataframes.append(df_expanded)
            else:
                # If no details column, just append the original df
                all_dataframes.append(df)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if all_dataframes:
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort if columns exist
    sort_cols = [col for col in ['dataset', 'solver', 'seed'] if col in final_df.columns]
    if sort_cols:
        final_df.sort_values(by=sort_cols, inplace=True)
    
    output_filename = "ket_qua_tong_hop_final_expanded.csv"
    final_df.to_csv(output_filename, index=False)
    
    print(f"File saved to: {output_filename}")
    print(final_df.head().to_markdown(index=False, numalign="left", stralign="left"))
else:
    print("No data found.")