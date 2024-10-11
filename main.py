import dataScrape as ds
import information as info

ticker = 'AAPL'
df = ds.dataScrapeMain(info.user, info.password, ticker)

