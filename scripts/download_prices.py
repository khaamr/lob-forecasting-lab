#download fixed date daily prices for the sector extension
import hashlib
import json
from pathlib import Path as path
import time
import pandas as pd
import yfinance as yf

root = path(__file__).resolve().parents[1]
assets = {'aapl':'consumer devices', 'goog':'internet services', 'nvda':'semiconductors',
          'msft':'enterprise software', 'amzn':'ecommerce and cloud', 'intc':'semiconductors',
          'enph':'solar energy', 'crsp':'gene editing', 'rklb':'space systems',
          'ionq':'quantum computing', 'qqq':'nasdaq 100 etf'}

def main():
    frames, records = [], []
    for ticker, field in assets.items():
        p = root / 'data/raw' / f'{ticker}_daily_2022_2025.csv'
        if not p.exists():
            for attempt in range(4):
                frame = yf.download(ticker.upper(), start='2022-01-01', end='2026-01-01', auto_adjust=True, progress=False, threads=False)
                if not frame.empty:
                    break
                time.sleep(2)
            if frame.empty:
                raise RuntimeError(f'no real daily data received for {ticker}')
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame.columns = frame.columns.str.lower()
            frame.index.name = 'date'
            frame.to_csv(p)
        frame = pd.read_csv(p)
        assert len(frame) > 900 and (frame.close > 0).all()
        frame['asset'], frame['field'] = ticker, field
        frames.append(frame)
        records.append(dict(asset=ticker, field=field, source=f'https://finance.yahoo.com/quote/{ticker.upper()}/history/',
                            query_start='2022-01-01', query_end_exclusive='2026-01-01', auto_adjust=True,
                            rows=len(frame), path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        print(f'downloaded daily prices: {ticker}, {len(frame)} rows', flush=True)
    pd.concat(frames, ignore_index=True).to_csv(root / 'data/processed/daily_prices.csv', index=False)
    (root / 'data/raw/daily_manifest.json').write_text(json.dumps(records, indent=2))

if __name__ == '__main__':
    main()
