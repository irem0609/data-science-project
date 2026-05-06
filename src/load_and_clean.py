import pandas as pd
import os

# No prefix is needed as they will not be joined together.
do_prefix : bool = False
file_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(file_dir)

def clean_data():
    # load the gaming file.
    gaming_df = pd.read_csv(os.path.join(proj_root, 'data', 'raw', 'gaming.csv'))
    # Drop na rows
    gaming_df = gaming_df.dropna()
    # There is no correlation between the student_id and literally any other column, so we can drop it.
    gaming_df = gaming_df.drop(columns=['student_id'])
    # Prefix with 'gaming_' to all columns
    if do_prefix:
        gaming_df = gaming_df.add_prefix('gaming_')
    # Create the processed directory if it doesn't exist
    if not os.path.exists(os.path.join(proj_root, 'data', 'processed')):
        os.mkdir(os.path.join(proj_root, 'data', 'processed'))
    # Save the cleaned data to a new csv file
    gaming_df.to_csv(os.path.join(proj_root, 'data', 'processed', 'cleaned_gaming.csv'), index=False)


    # load the mental file
    mental_df = pd.read_csv(os.path.join(proj_root, 'data', 'raw', 'mental.csv'))
    # Drop na rows
    mental_df = mental_df.dropna()
    # Drop the timestamp column
    mental_df = mental_df.drop(columns=['Timestamp'])
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
    # Parse the '[yY]ear (?\d+)' pattern to extract the year of study as an integer
    mental_df['year_of_study'] = mental_df['year_of_study'].str.extract(r'[yY]ear\s*(\d+)').astype(int)
    # Parse the cgpa range to pandas.Interval
    lower, upper = mental_df['cgpa_range'].str.extract(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)')
    mental_df['cgpa_range'] = mental_df.apply(lambda row: pd.Interval(float(row['cgpa_range'].split(' - ')[0]), float(row['cgpa_range'].split(' - ')[1]), closed='both'), axis=1)

    # Prefix with 'mental_' to all columns
    if do_prefix:
        mental_df = mental_df.add_prefix('mental_')
    # Save the cleaned data to a new csv file
    mental_df.to_csv(os.path.join(proj_root, 'data', 'processed', 'cleaned_mental.csv'), index=False)

    # load the student file
    student_df = pd.read_csv(os.path.join(proj_root, 'data', 'raw', 'student.csv'))
    # Drop na rows
    student_df = student_df.dropna()
    student_df = student_df.drop(columns=['Student_ID'])
    # convert all column names to the lower case
    student_df.columns = student_df.columns.str.lower()
    # Prefix with 'student_' to all columns
    if do_prefix:
        student_df = student_df.add_prefix('student_')
    # Save the cleaned data to a new csv file
    student_df.to_csv(os.path.join(proj_root, 'data', 'processed', 'cleaned_student.csv'), index=False)


if __name__ == '__main__':
    clean_data()