import pandas as pd
from pyspark.sql.streaming import StreamingQuery

from SRC.LIBRARIES.spark_utils import show_pandas, monitor_stop_event

pd.DataFrame.show = show_pandas
StreamingQuery.monitor_stop_event = monitor_stop_event

