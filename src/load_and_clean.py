import pandas as pd
import os

# No prefix is needed as they will not be joined together.
do_prefix : bool = False
file_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(file_dir)


def fit_into_cgpa_range(cgpa):
    if cgpa < 2.0:
        return pd.Interval(0, 1.99, closed='both')
    elif cgpa < 2.5:
        return pd.Interval(2.0, 2.49, closed='both')
    elif cgpa < 3.0:
        return pd.Interval(2.5, 2.99, closed='both')
    elif cgpa < 3.5:
        return pd.Interval(3.0, 3.49, closed='both')
    elif cgpa <= 4.0:
        return pd.Interval(3.5, 4.0, closed='both')
    else:
        raise ValueError(f"CGPA {cgpa} is out of range")

def clean_data():
    gender_map = {'Female': 0, 'female': 0, 'Male': 1, 'male': 1, 'Other': 2, 'other': 2}
    # load the gaming file.
    gaming_df = pd.read_csv(os.path.join(proj_root, 'data', 'raw', 'gaming.csv'))
    # Drop na rows
    gaming_df = gaming_df.dropna()
    # There is no correlation between the student_id and literally any other column, so we can drop it.
    # Make sure the column names matches other files so joining will be easier later on.
    gaming_df['student_id'] = gaming_df['student_id'].apply(lambda x: f'gaming_{x}')
    gaming_df = gaming_df.rename(columns={
        'study_hours': 'study_hours_per_day',
        'sleep_hours': 'sleep_hours_per_day',
        'social_activity': 'social_hours_day',
    })
    
    gaming_df['gender'] = gaming_df['gender'].map(gender_map).astype(int)
    # One-Hot encode the gender column
    gaming_df['is_female'] = (gaming_df['gender'] == 0).astype(int)
    gaming_df['is_male'] = (gaming_df['gender'] == 1).astype(int)
    gaming_df['is_other'] = (gaming_df['gender'] == 2).astype(int)

    # Add a social_hours_week column by multiplying the social_hours_per_day by 7
    gaming_df['social_hours_week'] = gaming_df['social_hours_day'] * 7
    gaming_df['final_cgpa'] = (gaming_df['grades'] / 25).round(2)
    # Prefix with 'gaming_' to all columns
    if do_prefix:
        gaming_df = gaming_df.add_prefix('gaming_')
    # Create the processed directory if it doesn't exist
    if not os.path.exists(os.path.join(proj_root, 'data', 'processed')):
        os.mkdir(os.path.join(proj_root, 'data', 'processed'))
    # Save the cleaned data to a new csv file
    gaming_df['source'] = 'gaming'
    gaming_df.to_csv(os.path.join(proj_root, 'data', 'processed', 'cleaned_gaming.csv'), index=False)

    # load the mental file
    mental_df = pd.read_csv(os.path.join(proj_root, 'data', 'raw', 'mental.csv'))
    # Drop na rows
    mental_df = mental_df.dropna()
    mental_df = mental_df.drop(columns=['Timestamp']) # drop the timestamp as it does not mean anything for our analysis
    # Rename the columns
    mental_df = mental_df.rename(columns={
        'Choose your gender' : 'gender',
        'Age': 'age',
        'What is your course?': 'course',
        'Your current year of Study': 'year_of_study',
        'What is your CGPA?': 'cgpa_range',
        'Marital status': 'marital_status',
        'Do you have Depression?': 'depression',
        'Do you have Anxiety?': 'anxiety',
        'Do you have Panic attack?': 'panic_attack',
        'Did you seek any specialist for a treatment?': 'seek_treatment'
        })
    # convert Yes and No to 1 and 0
    yes_no_map = {'Yes': 1, 'No': 0}
    mental_df['marital_status'] = mental_df['marital_status'].map(yes_no_map)
    mental_df['depression'] = mental_df['depression'].map(yes_no_map)
    mental_df['anxiety'] = mental_df['anxiety'].map(yes_no_map)
    mental_df['panic_attack'] = mental_df['panic_attack'].map(yes_no_map)
    mental_df['seek_treatment'] = mental_df['seek_treatment'].map(yes_no_map)

    mental_df['gender'] = mental_df['gender'].map(gender_map).astype(int)
    # One-Hot encode the gender column
    mental_df['is_female'] = (mental_df['gender'] == 0).astype(int)
    mental_df['is_male'] = (mental_df['gender'] == 1).astype(int)
    mental_df['is_other'] = (mental_df['gender'] == 2).astype(int)

    # Parse the '[yY]ear (?\d+)' pattern to extract the year of study as an integer
    mental_df['year_of_study'] = mental_df['year_of_study'].str.extract(r'[yY]ear\s*(\d+)').astype(int)
    # Parse the cgpa range to pandas.Interval
    lower, upper = mental_df['cgpa_range'].str.extract(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)')
    mental_df['cgpa_range'] = mental_df.apply(lambda row: pd.Interval(float(row['cgpa_range'].split(' - ')[0]), float(row['cgpa_range'].split(' - ')[1]), closed='both'), axis=1)
    mental_df['final_cgpa'] = mental_df['cgpa_range'].apply(lambda x: (x.left + x.right) / 2)
    mental_df['age'] = mental_df['age'].astype(int)
    # generate a student_id column by prefixing 'mental_' to the index of the dataframe
    mental_df['student_id'] = mental_df.index.to_series().apply(lambda x: f'mental_{x}')
    # Prefix with 'mental_' to all columns
    if do_prefix:
        mental_df = mental_df.add_prefix('mental_')
    # Save the cleaned data to a new csv file
    mental_df['source'] = 'mental'
    mental_df.to_csv(os.path.join(proj_root, 'data', 'processed', 'cleaned_mental.csv'), index=False)

    # load the student file
    student_df = pd.read_csv(os.path.join(proj_root, 'data', 'raw', 'student.csv'))
    # Drop na rows
    student_df = student_df.dropna()
    # convert all column names to the lower case
    student_df.columns = student_df.columns.str.lower()
    student_df = student_df.rename(columns={
        'attendance_pct': 'attendance',
        'sleep_hours': 'sleep_hours_per_day',
    })
    student_df['gender'] = student_df['gender'].map(gender_map).astype(int)
    # one-hot encode the gender column
    student_df['is_female'] = (student_df['gender'] == 0).astype(int)
    student_df['is_male'] = (student_df['gender'] == 1).astype(int)
    student_df['is_other'] = (student_df['gender'] == 2).astype(int)
    
    student_df['student_id'] = student_df['student_id'].apply(lambda x: f'student_{x}')
    # Prefix with 'student_' to all columns
    if do_prefix:
        student_df = student_df.add_prefix('student_')
    student_df['source'] = 'student'
    # Save the cleaned data to a new csv file
    student_df.to_csv(os.path.join(proj_root, 'data', 'processed', 'cleaned_student.csv'), index=False)


if __name__ == '__main__':
    clean_data()