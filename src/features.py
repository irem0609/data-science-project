import pandas as pd
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(file_dir)
# Target directory for processed data
processed_path = os.path.join(proj_root, "data", "processed")


def process_gaming_features(df):
    # Ratio of daily gaming hours to a full 24-hour day
    df["gaming_intensity"] = df["gaming_hours"] / 24
    # Ratio of study hours to gaming hours
    df["academic_pressure_index"] = df["study_hours_per_day"] / (df["gaming_hours"] + 1)
    # Balance between sleep and gaming
    df["sleep_gaming_balance"] = df["sleep_hours_per_day"] / (df["gaming_hours"] + 1)
    # Combined hours spent on social activities and gaming
    df["total_screen_time"] = df["social_hours_day"] + df["gaming_hours"]
    return df


def process_mental_features(df):
    # Aggregate score of binary mental health indicators
    df["total_mental_risk"] = df["depression"] + df["anxiety"] + df["panic_attack"]
    # CGPA relative to age
    df["academic_progress_rate"] = df["final_cgpa"] / df["age"]
    return df


def process_student_features(df):
    # Calculate study efficiency if social hours data is available
    if "social_hours_week" in df.columns:
        df["study_efficiency"] = df["study_hours_per_day"] / (
            df["social_hours_week"] / 7 + 1
        )
    # Combined impact of attendance and previous academic performance
    df["attendance_impact"] = (df["attendance"] * df["previous_cgpa"]) / 100
    return df


def build_features():
    try:
        # Process and save gaming dataset
        g_in = os.path.join(processed_path, "cleaned_gaming.csv")
        df_g = pd.read_csv(g_in)
        df_g = process_gaming_features(df_g)
        df_g.to_csv(
            os.path.join(processed_path, "final_gaming_features.csv"), index=False
        )
        print("Gaming features successfully built.")

        # Process and save mental health dataset
        m_in = os.path.join(processed_path, "cleaned_mental.csv")
        df_m = pd.read_csv(m_in)
        df_m = process_mental_features(df_m)
        df_m.to_csv(
            os.path.join(processed_path, "final_mental_features.csv"), index=False
        )
        print("Mental features successfully built.")

        # Process and save student dataset
        s_in = os.path.join(processed_path, "cleaned_student.csv")
        df_s = pd.read_csv(s_in)
        df_s = process_student_features(df_s)
        df_s.to_csv(
            os.path.join(processed_path, "final_student_features.csv"), index=False
        )
        print("Student features successfully built.")

        print(
            "\n[SUCCESS] All features have been generated and saved to 'data/processed'."
        )

    except FileNotFoundError as e:
        print(
            f"\n[ERROR] File not found. Please check the 'data/processed' directory.\nDetails: {e}"
        )
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    build_features()
