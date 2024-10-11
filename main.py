import dataScrape as ds
import information as info

ticker = 'AAPL'
df = ds.ScrapeMain(info.user, info.password, ticker)

