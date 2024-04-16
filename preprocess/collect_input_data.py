from tqdm import tqdm 
from xml.etree import ElementTree as ET
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt


def parse_date(date_str):
    try:
        output = datetime.strptime(date_str, "%B %d, %Y")
    except:
        try:
            output = datetime.strptime(date_str, "%B %Y")
        except Exception as e:
            print(e)
            raise e
    return output

def calculate_duration(start_date, completion_date):
    # Unit: days
    if start_date and completion_date:
        start_date = parse_date(start_date)
        completion_date = parse_date(completion_date)
        duration = (completion_date - start_date).days
    else:
        duration = -1

    return duration

def xmlfile2date(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    try:
        start_date = root.find('start_date').text
    except:
        start_date = ''
    try:
        completion_date = root.find('primary_completion_date').text
    except:
        try:
            completion_date = root.find('completion_date').text 
        except:
            completion_date = ''

    return start_date, completion_date


if __name__ == "__main__":
    date_list = []

    # 478504 lines
    with open("data/trials/all_xml.txt", "r") as file:
        for xml_path in tqdm(file):
            xml_path = f"data/{xml_path.strip()}"

            # NCT00000150 <- raw_data/NCT0000xxxx/NCT00000150.xml
            nct_id = re.search(r"/([^/]+)\.xml$", xml_path).group(1)
            
            start_date, completion_date = xmlfile2date(xml_path)

            if start_date and completion_date:
                duration = calculate_duration(start_date, completion_date)
            else:
                duration = -1

            date_list.append((nct_id, start_date, completion_date, duration))


    date_df = pd.DataFrame(date_list, columns=['nctid', 'start_date', 'completion_date', 'time_day'])
    date_df = date_df[date_df['time_day'] > 0]

    # filter: time_day < 10 years
    date_df = date_df[date_df['time_day'] < 3650]
    # date_df.to_csv('data/ntcid_time.csv', index=False, sep='\t')

    raw_df = pd.read_csv('data/raw_data.csv', sep=',')

    input_df = pd.merge(date_df, raw_df[['nctid', 'phase', 'diseases', 'drugs', 'criteria']], on='nctid', how='inner')

    input_df['diseases'] = input_df['diseases'].apply(lambda x: ';'.join(eval(x)))
    input_df['drugs'] = input_df['drugs'].apply(lambda x: ';'.join(eval(x)))

    # filter phase
    input_df = input_df[input_df['phase'].isin(['phase 1', 'phase 2', 'phase 3', 'phase 4'])]

    print(f"Total: {len(input_df)}")

    input_df.to_csv('data/time_prediction_input.csv', index=False, sep='\t')
