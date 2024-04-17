import pandas as pd
import numpy as np
import os

# Inplace ... with the real Nasdaq data path.
NEWS_PATH = ...
STOCK_PATH = ...

def move_open_in_place(directory_path) -> None:
    """
    Moves the "open" column to the last column in all CSV files within a directory.

    Parameters:
    - directory_path (str): The path to the directory containing the CSV files.
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                
                if 'open' in df.columns and 'open' != df.columns[-1]:
                    columns = df.columns.tolist()
                    columns.remove('open')
                    columns.append('open')
                    df = df[columns]

                df.to_csv(file_path, index=False)
    return None

def process_news_files(directory_path) -> None:
    """
    Renames the "datetime" column to "date", changes the date format, and combines the "title" and "summary" columns.
    Then, groups the data by the "date" column and combines the information for each day.

    Parameters:
    - directory_path (str): The path to the directory containing the CSV files.
    """
    out_path = r"C:\Users\50-Cyan\Desktop\Mimosa\Research\Codes\Grad\dataset\Nasdaq\test\\"
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)              

                if 'datetime' in df.columns:
                    df = df.sort_values(by='datetime')
                    df['date'] = pd.to_datetime(df['datetime']).dt.date
                    df.drop(columns=['datetime'], inplace=True)

                related = df['related'][0]
                df['information'] = df.apply(lambda row: f"Headline: {row['headline']}, source:{row['source']}, summary: {row['summary'] if 'Zacks.com' not in str(row['summary']) else 'No summary'}", axis=1)
                df = df[['date', 'information']]

                df_grouped = df.groupby('date')['information'].apply(lambda x: ', '.join(x)).reset_index()
                df_grouped["related"] = related
                # Save the modified DataFrame to a new CSV file to avoid overwriting the original

                output_file_path = os.path.splitext(file_path)[0] + '.csv'
                df_grouped.to_csv(output_file_path, index=True)

class NewsDatasetHandler(object):
    def __init__(self, symbol_list, slc, news_data_dir, **kwargs):
        self.symbol_list = symbol_list
        self.slc = slc
        self.news_data_dir = news_data_dir
        self.start = slc.start
        self.end = slc.stop
        self.check_timestamp()
    
class NewsDataProcessor:
    def __init__(self, symbol_list, slc, news_data_dirs, local_calendar_dir):
        self.news_data_dir = news_data_dirs
        self.symbol_list = symbol_list
        self.local_calendar_dir = local_calendar_dir
        self.start = slc.start
        self.end = slc.stop
        self._check_timestamp()
        self.trading_calendar = self._get_trading_day()

    def _get_trading_day(self):
        try:
            from qlib.data import D
            return D.calendar(start_time=str(self.start), end_date=str(self.end))
        except:
            assert os.path.exists(self.local_calendar_dir)
            calendar = pd.read_csv(self.local_calendar_dir, engine='python', names=["date"])
            calendar['date'] = pd.to_datetime(calendar['date'])
            calendar = calendar[(calendar['date'] >= self.start) & (calendar['date'] <= self.end)] 
            calendar['date'] = pd.to_datetime(calendar['date'].dt.date)
            return calendar.reset_index()

    def _check_timestamp(self):
        if not isinstance(self.start, pd.Timestamp) or not isinstance(self.end, pd.Timestamp):
            try: 
                self.start = pd.to_datetime(self.start)
                self.end = pd.to_datetime(self.end)
            except:
                raise ValueError("The start and end of the time range must be datetime or could be inverted to datetime.")

        return None

    def read_and_align(self) -> pd.DataFrame:
        ult_df = pd.DataFrame()
        dfs = []
        for symbol in self.symbol_list:
            file_path = os.path.join(self.news_data_dir, f"{symbol}.csv")
            assert os.path.exists(file_path), f"{symbol}.csv not exist!"
            news = pd.read_csv(file_path)
            related = news["related"]
            news['date'] = pd.to_datetime(news['date'])
            news = news[(news['date'] >= self.start) & (news['date'] <= self.end)]
            news['date'] = news['date'].map(lambda x: self.trading_calendar['date'].loc[self.trading_calendar['date'] >= x].min())
            merged_df = self.trading_calendar.join(news.set_index('date'), on='date', how='left')
            merged_df['information'] = merged_df['information'].apply(lambda x: str(x))
            merged_df = merged_df.groupby('date')['information'].agg(lambda x: ''.join(x)).reset_index(name='information')
            merged_df["related"] = related
            merged_df = merged_df.set_index(['date', 'related'])
            dfs.append(merged_df)
        ult_df = pd.concat(dfs, ignore_index=False).sort_index()
        return ult_df
    
if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    preprocess = False
    news_data_dir = r"C:\Users\50-Cyan\Desktop\Mimosa\Research\Codes\Grad\dataset\Nasdaq\news1"
    local_calendar_dir = r"C:\Users\50-Cyan\.qlib\qlib_data\my_data\calendars\day.txt"
    if preprocess:    
        process_news_files(news_data_dir)
    slc = ["2023-01-03", "2023-1-11"]
    symbol_list = ["AAPL","ABNB","ADBE"]
    n = NewsDataProcessor(symbol_list, slc, news_data_dir, local_calendar_dir)
    print(n.read_and_align())
    # calendar =D.calendar(start_time='2015-01-01',end_time='2016-01-01',freg='day')
