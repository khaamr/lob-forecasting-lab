#download public research samples
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path as path
import time
import zipfile
import requests

root = path(__file__).resolve().parents[1]
raw = root / 'data' / 'raw'
raw.mkdir(parents=True, exist_ok=True)

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def fetch_file(url, destination):
    if not destination.exists():
        for attempt in range(4):
            try:
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with destination.with_suffix('.part').open('wb') as f:
                        for chunk in response.iter_content(1048576):
                            f.write(chunk)
                destination.with_suffix('.part').replace(destination)
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2)
    return destination

def lobster(ticker):
    name = f'LOBSTER_SampleFile_{ticker.upper()}_2012-06-21_10.zip'
    url = f'https://php.lobsterdata.com/info/sample/{name}'
    p = fetch_file(url, raw / f'{ticker}_2012-06-21_10.zip')
    assert zipfile.is_zipfile(p), 'expected a zip archive, not an html page'
    with zipfile.ZipFile(p) as archive:
        assert archive.testzip() is None
    print(f'downloaded {ticker}: {p.stat().st_size:,} bytes', flush=True)
    return dict(asset=ticker, kind='lobster', date='2012-06-21', source=url,
                path=str(p.relative_to(root)), sha256=digest(p), selection='complete public sample day')

def crypto(exchange, symbol, rows=60000):
    url = f'https://datasets.tardis.dev/v1/{exchange}/book_snapshot_25/2024/01/01/{symbol.upper()}.csv.gz'
    p = raw / f'{exchange}_{symbol}_2024-01-01_prefix.csv.gz'
    if not p.exists():
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as stream:
                reader = io.TextIOWrapper(stream)
                with gzip.open(p.with_suffix('.part'), 'wt') as out:
                    out.write(next(reader))
                    for i, line in enumerate(reader):
                        out.write(line)
                        if i + 1 >= rows:
                            break
        p.with_suffix('.part').replace(p)
    print(f'downloaded {exchange}: {p.stat().st_size:,} bytes', flush=True)
    return dict(asset=f'{exchange}_{symbol}', kind='tardis', date='2024-01-01', source=url,
                path=str(p.relative_to(root)), sha256=digest(p), selection=f'first {rows} captured snapshots; not a full day')

def main():
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs.extend(pool.submit(lobster, t) for t in ['aapl', 'amzn', 'goog', 'intc', 'msft'])
        jobs.append(pool.submit(crypto, 'binance', 'btcusdt'))
        jobs.append(pool.submit(crypto, 'coinbase', 'btc-usd'))
        records = [j.result() for j in jobs]
    (raw / 'manifest.json').write_text(json.dumps(records, indent=2))

if __name__ == '__main__':
    main()
