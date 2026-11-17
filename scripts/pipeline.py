#research pipeline
import copy
import hashlib
import json
import math
import os
from pathlib import Path as path
import random
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler as standard_scaler
from sklearn.linear_model import LogisticRegression as logistic_regression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, log_loss, confusion_matrix
import torch
from torch import nn
from torch.utils.data import Dataset as dataset, DataLoader as data_loader

root = path(__file__).resolve().parents[1]
features = [f'{name}_{level}' for level in range(1, 11)
            for name in ['ask_price', 'ask_size', 'bid_price', 'bid_size']]
classes = ['down', 'stable', 'up']

def seed_all(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def clean_book(frame, equity=True):
    n = len(frame)
    frame = frame.copy()
    values = frame[features].to_numpy()
    valid = np.isfinite(values).all(axis=1)
    valid &= (np.abs(values[:, 0::2]) < 1e8).all(axis=1)
    valid &= (values[:, 1::2] > 0).all(axis=1)
    valid &= (values[:, 0::4] > 0).all(axis=1) & (values[:, 2::4] > 0).all(axis=1)
    valid &= (np.diff(values[:, 0::4], axis=1) > 0).all(axis=1)
    valid &= (np.diff(values[:, 2::4], axis=1) < 0).all(axis=1)
    valid &= values[:, 0] > values[:, 2]
    frame = frame.loc[valid]
    after_quality = len(frame)
    if equity:
        frame = frame.loc[frame.seconds.between(34800, 57000, inclusive='both')]
    after_hours = len(frame)
    frame = frame.drop_duplicates('timestamp', keep='last').sort_values('timestamp', kind='stable').reset_index(drop=True)
    assert frame.timestamp.is_monotonic_increasing
    assert not frame.timestamp.duplicated().any()
    frame['mid'] = (frame.ask_price_1 + frame.bid_price_1) / 2
    return frame, dict(raw_rows=n, invalid_rows=n-after_quality,
                       outside_session=after_quality-after_hours,
                       duplicate_timestamps=after_hours-len(frame), clean_rows=len(frame))


def load_books():
    manifest = json.loads((root / 'data/raw/manifest.json').read_text())
    books, audit = {}, []
    for record in manifest:
        p = root / record['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256']
        equity = record['kind'] == 'lobster'
        if equity:
            with zipfile.ZipFile(p) as z:
                book_name = next(n for n in z.namelist() if 'orderbook' in n and n.endswith('.csv'))
                message_name = next(n for n in z.namelist() if 'message' in n and n.endswith('.csv'))
                frame = pd.read_csv(z.open(book_name), header=None, names=features)
                messages = pd.read_csv(z.open(message_name), header=None, dtype={0: str})
                assert len(messages) == len(frame)
                #nanosecond timestamps without binary floating point rounding
                seconds = messages[0].str.split('.', expand=True)
                frame['timestamp'] = seconds[0].astype('int64') * 10**9 + seconds[1].fillna('0').str.pad(9, side='right', fillchar='0').str[:9].astype('int64')
                frame['seconds'] = frame.timestamp / 1e9
                frame[features[0::2]] = frame[features[0::2]] / 10000
            tick = .01
        else:
            source = pd.read_csv(p)
            frame = pd.DataFrame({f'{side}_{kind}_{j+1}': source[f'{side}s[{j}].{field}']
                      for j in range(10) for side in ['ask', 'bid']
                      for kind, field in [('price', 'price'), ('size', 'amount')]})
            frame['timestamp'] = source.timestamp.astype('int64') * 1000
            frame['seconds'] = (frame.timestamp - frame.timestamp.iloc[0]) / 1e9
            #sample grid assumption checked against all downloaded quotes
            tick = .01
            quotes = frame[features[0::2]].to_numpy()
            assert np.max(np.abs(quotes / tick - np.rint(quotes / tick))) < 1e-5
        frame, counts = clean_book(frame, equity)
        books[record['asset']] = dict(frame=frame, tick=tick, equity=equity, source=record)
        audit.append(dict(asset=record['asset'], **counts, first_seconds=frame.seconds.iloc[0],
                          last_seconds=frame.seconds.iloc[-1], observed_span_seconds=(frame.timestamp.iloc[-1]-frame.timestamp.iloc[0])/1e9))
    return books, pd.DataFrame(audit)


def labels(mid, horizon, tick):
    #integer half ticks preserve inclusive one tick boundaries
    half_ticks = np.rint(np.asarray(mid) * 2 / tick).astype('int64')
    delta = half_ticks[horizon:] - half_ticks[:-horizon]
    return np.where(delta <= -2, 0, np.where(delta >= 2, 2, 1)).astype('int64')


def stationary_features(frame, tick):
    x = frame[features].to_numpy(dtype='float64').copy()
    x[:, 0::2] = (x[:, 0::2] - frame.mid.to_numpy()[:, None]) / tick
    x[:, 1::2] = np.log1p(x[:, 1::2])
    return x


def split_indices(n, horizon, lookback=100, validation_cap=2048, test_cap=4096):
    b1, b2 = int(n * .6), int(n * .8)
    #each window and future target stay entirely inside their split
    train = np.arange(lookback-1, b1-horizon)
    valid = np.arange(b1+lookback-1, b2-horizon)
    test = np.arange(b2+lookback-1, n-horizon)
    if validation_cap:
        valid = valid[:validation_cap]
    if test_cap:
        test = test[:test_cap]
    assert train[-1] + horizon < valid[0] - lookback + 1
    assert valid[-1] + horizon < test[0] - lookback + 1
    return train, valid, test


def prepare(book, horizon, representation='raw', validation_cap=2048, test_cap=4096):
    frame = book['frame']
    indices = split_indices(len(frame), horizon, validation_cap=validation_cap, test_cap=test_cap)
    x = frame[features].to_numpy() if representation == 'raw' else stationary_features(frame, book['tick'])
    #standard scaler
    scaler = standard_scaler().fit(x[:int(len(frame)*.6)])
    x = scaler.transform(x).astype('float32')
    assert np.isfinite(x).all()
    return dict(x=x, y=labels(frame.mid, horizon, book['tick']), indices=indices,
                scaler=scaler, representation=representation, horizon=horizon)


def balanced_indices(y, indices, cap, seed):
    rng = np.random.default_rng(seed)
    groups = [indices[y[indices] == c] for c in range(3)]
    count = min(cap, min(map(len, groups)))
    if count == 0:
        raise ValueError('a class is missing in training; cannot balance three classes')
    result = np.concatenate([rng.choice(g, count, replace=False) for g in groups])
    rng.shuffle(result)
    return result


class windows(dataset):
    def __init__(self, x, y, indices):
        self.x, self.y, self.indices = x, y, indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        t = self.indices[i]
        return torch.from_numpy(self.x[t-99:t+1]), torch.tensor(self.y[t])


class deeplob(nn.Module):
    def __init__(self):
        super().__init__()
        def conv(a, b, kernel, stride=(1, 1), padding=0):
            return [nn.Conv2d(a, b, kernel, stride=stride, padding=padding), nn.LeakyReLU(.01), nn.BatchNorm2d(b)]
        blocks = []
        for i, width in enumerate([2, 2, 10]):
            blocks += conv(1 if i == 0 else 32, 32, (1, width), (1, width) if i < 2 else (1, 1))
            blocks += conv(32, 32, (4, 1)) + conv(32, 32, (4, 1))
        self.conv = nn.Sequential(*blocks)
        self.branch1 = nn.Sequential(*(conv(32, 64, (1, 1)) + conv(64, 64, (3, 1), padding=(1, 0))))
        self.branch2 = nn.Sequential(*(conv(32, 64, (1, 1)) + conv(64, 64, (5, 1), padding=(2, 0))))
        self.branch3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), *conv(32, 64, (1, 1)))
        self.recurrent = nn.LSTM(192, 64, batch_first=True)
        self.head = nn.Linear(64, 3)
    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], 1)
        x = x.squeeze(-1).transpose(1, 2)
        x, _ = self.recurrent(x)
        return self.head(x[:, -1])


class transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(40, 32)
        self.position = nn.Parameter(torch.randn(1, 100, 32) * .01)
        layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(32, 3)
    def forward(self, x):
        #attention sees only the historical input window
        x = self.encoder(self.embed(x) + self.position)
        return self.head(x[:, -1])


class graph_lob(nn.Module):
    def __init__(self, edges=True):
        super().__init__()
        adjacency = torch.eye(20)
        if edges:
            for side in range(2):
                for level in range(9):
                    i, j = level*2+side, (level+1)*2+side
                    adjacency[i, j] = adjacency[j, i] = 1
            for level in range(10):
                adjacency[2*level, 2*level+1] = adjacency[2*level+1, 2*level] = 1
        degree = adjacency.sum(1).pow(-.5)
        self.register_buffer('adjacency', degree[:, None]*adjacency*degree[None, :])
        self.embed = nn.Linear(2, 16)
        self.message = nn.Linear(16, 16)
        self.recurrent = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, 3)
    def forward(self, x):
        #twenty nodes represent ten ask and ten bid price volume pairs
        x = x.reshape(len(x), 100, 20, 2)
        x = torch.relu(self.embed(x))
        x = torch.einsum('ij,btjf->btif', self.adjacency, x)
        x = torch.relu(self.message(x))
        x = torch.cat([x[:, :, 0::2].mean(2), x[:, :, 1::2].mean(2)], -1)
        x, _ = self.recurrent(x)
        return self.head(x[:, -1])


class diffusion_lob(nn.Module):
    def __init__(self, steps=16, draws=24):
        super().__init__()
        self.steps, self.draws = steps, draws
        self.context = nn.GRU(40, 32, batch_first=True)
        self.time_embed = nn.Embedding(steps, 8)
        self.denoise = nn.Sequential(nn.Linear(43, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 3))
        #cosine schedule for a conditional diffusion model over class embeddings
        t = torch.linspace(0, 1, steps+1)
        cumulative = torch.cos((t+.008)/1.008*math.pi/2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1-cumulative[1:]/cumulative[:-1]).clamp(.0001, .999)
        self.register_buffer('alpha_bar', (1-beta).cumprod(0))
        generator = torch.Generator().manual_seed(901)
        self.register_buffer('initial_noise', torch.randn(draws, 3, generator=generator))
    def encode(self, x):
        _, hidden = self.context(x)
        return hidden[-1]
    def epsilon(self, z, t, context):
        return self.denoise(torch.cat([z, self.time_embed(t), context], -1))
    def training_loss(self, x, y):
        context = self.encode(x)
        clean = nn.functional.one_hot(y, 3).float()*2-1
        t = torch.randint(self.steps, (len(x),))
        noise = torch.randn_like(clean)
        a = self.alpha_bar[t, None]
        z = a.sqrt()*clean + (1-a).sqrt()*noise
        return nn.functional.mse_loss(self.epsilon(z, t, context), noise)
    def forward(self, x):
        context = self.encode(x).repeat_interleave(self.draws, 0)
        z = self.initial_noise.repeat(len(x), 1)
        #deterministic reverse diffusion using fixed monte carlo initial states
        for k in reversed(range(self.steps)):
            t = torch.full((len(z),), k, dtype=torch.long)
            a = self.alpha_bar[k]
            previous = self.alpha_bar[k-1] if k else torch.tensor(1.)
            eps = self.epsilon(z, t, context)
            clean = ((z-(1-a).sqrt()*eps)/a.sqrt()).clamp(-3, 3)
            z = previous.sqrt()*clean + (1-previous).sqrt()*eps
        votes = nn.functional.one_hot(z.argmax(-1), 3).float().reshape(len(x), self.draws, 3).sum(1)
        return ((votes+1)/(self.draws+3)).log()


def predict(model, prepared, indices):
    loader = data_loader(windows(prepared['x'], prepared['y'], indices), batch_size=128, shuffle=False)
    model.eval()
    values = []
    with torch.no_grad():
        for x, _ in loader:
            values.append(torch.softmax(model(x), -1).numpy())
    return np.concatenate(values)


def train_model(name, prepared, config, seed=17):
    seed_all(seed)
    constructors = dict(deeplob=deeplob, transformer=transformer, graph=graph_lob,
                        graph_no_edges=lambda: graph_lob(False), diffusion=diffusion_lob)
    model = constructors[name]()
    lr = 6e-5 if name == 'deeplob' else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=.01)
    history, best, saved, stale = [], np.inf, None, 0
    start = time.perf_counter()
    for epoch in range(config['epochs']):
        indices = balanced_indices(prepared['y'], prepared['indices'][0], config['per_class'], seed+epoch)
        loader = data_loader(windows(prepared['x'], prepared['y'], indices), batch_size=32, shuffle=False)
        model.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad()
            loss = model.training_loss(x, y) if name == 'diffusion' else nn.functional.cross_entropy(model(x), y)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(loss.item())
        probability = predict(model, prepared, prepared['indices'][1])
        validation_loss = log_loss(prepared['y'][prepared['indices'][1]], probability, labels=[0, 1, 2])
        history.append(dict(epoch=epoch+1, training_loss=np.mean(losses), validation_loss=validation_loss))
        if validation_loss < best-1e-6:
            best, saved, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= config['patience']:
            break
    model.load_state_dict(saved)
    return model, history, time.perf_counter()-start


def transaction_set(signals):
    state, start = 0, None
    transactions = []
    for i, signal in enumerate(signals):
        if signal == 0:
            continue
        if state == 0:
            state, start = int(signal), i
        elif signal != state:
            transactions.append((start, i, state))
            state, start = int(signal), i
    return transactions


def transaction_metrics(targets, predictions, confidence=None, threshold=.3, tolerance=0):
    true = np.asarray(targets, dtype=int)-1
    pred = np.asarray(predictions, dtype=int)-1
    if confidence is not None:
        pred = np.where(np.asarray(confidence) >= threshold, pred, 0)
    potential = transaction_set(true)
    executed = transaction_set(pred)
    #figure twelve matches entry and exit inside the corresponding directional regimes
    regime = np.full(len(true), -1, dtype=int)
    state, run = 0, -1
    for i, value in enumerate(true):
        if value and value != state:
            run += 1
            state = value
        if value:
            regime[i] = run
    expected = {(regime[a], regime[b], s) for a, b, s in potential}
    matched = set()
    for a, b, sign in executed:
        if not tolerance:
            candidate = (regime[a], regime[b], sign)
            if true[a] == sign and true[b] == -sign and candidate in expected:
                matched.add(candidate)
        else:
            entries = set(regime[max(0, a-tolerance):min(len(true), a+tolerance+1)][true[max(0, a-tolerance):min(len(true), a+tolerance+1)] == sign])
            exits = set(regime[max(0, b-tolerance):min(len(true), b+tolerance+1)][true[max(0, b-tolerance):min(len(true), b+tolerance+1)] == -sign])
            #one to one matching prevents one forecast from earning multiple transactions
            candidates = sorted((i, j, sign) for i in entries for j in exits if (i, j, sign) in expected and (i, j, sign) not in matched)
            if candidates:
                matched.add(candidates[0])
    pt, tt, ct = len(potential), len(executed), len(matched)
    denominator = pt+tt-ct
    return dict(potential=pt, executed=tt, correct=ct, p_t=ct/denominator if denominator else np.nan)


def score(y, probability, threshold=.3):
    pred, conf = probability.argmax(1), probability.max(1)
    keep = conf >= threshold
    transaction = transaction_metrics(y, pred, conf, threshold)
    return dict(threshold=threshold, count=int(keep.sum()), coverage=keep.mean(),
                accuracy=accuracy_score(y[keep], pred[keep]) if keep.any() else np.nan,
                macro_f1=f1_score(y[keep], pred[keep], labels=[0, 1, 2], average='macro', zero_division=0) if keep.any() else np.nan,
                mcc=matthews_corrcoef(y[keep], pred[keep]) if keep.any() else np.nan,
                **transaction)


def fit_temperature(y, probability):
    candidates = np.geomspace(.25, 4, 41)
    losses = [log_loss(y, softmax(np.log(probability.clip(1e-9))/t, axis=1), labels=[0, 1, 2]) for t in candidates]
    return float(candidates[np.argmin(losses)])


def microstructure(books):
    rows = []
    for asset, book in books.items():
        frame, tick = book['frame'], book['tick']
        spread = (frame.ask_price_1-frame.bid_price_1)/tick
        changes = (np.diff(frame.mid) != 0).sum()
        rows.append(dict(asset=asset, mean_mid=frame.mid.mean(), mean_spread_ticks=spread.mean(),
            tick_group='large' if spread.mean() <= 1.5 else ('small' if spread.mean() >= 3 else 'medium'),
            mean_top_depth=(frame.ask_size_1+frame.bid_size_1).mean(),
            mean_book_width_ticks=((frame.ask_price_10-frame.ask_price_1 + frame.bid_price_1-frame.bid_price_10)/2/tick).mean(),
            one_tick_fraction=np.isclose(spread, 1).mean(), updates=len(frame), price_changes=changes,
            information_richness=np.log(len(frame)/changes) if changes else np.nan))
    return pd.DataFrame(rows)


def baseline_probabilities(prepared):
    tr, va, te = prepared['indices']
    indices = balanced_indices(prepared['y'], tr, 2000, 17)
    model = logistic_regression(max_iter=500, random_state=17).fit(prepared['x'][indices], prepared['y'][indices])
    result = {'logistic': (model.predict_proba(prepared['x'][va]), model.predict_proba(prepared['x'][te]))}
    frequency = np.bincount(prepared['y'][tr], minlength=3)/len(tr)
    result['prior'] = (np.tile(frequency, (len(va), 1)), np.tile(frequency, (len(te), 1)))
    #persistence uses only the preceding horizon of observed prices
    return result


def run_benchmark(books, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows, histories, runtime = [], [], []
    for asset, book in books.items():
        for horizon in [10, 50, 100]:
            prepared = prepare(book, horizon, validation_cap=config['validation_cap'], test_cap=config['test_cap'])
            tr, va, te = prepared['indices']
            target = prepared['y'][te]
            probabilities = baseline_probabilities(prepared)
            #persistence maps trailing observed increments to the same three classes
            mids = np.rint(book['frame'].mid.to_numpy()*2/book['tick']).astype('int64')
            def persistence(ii):
                past = mids[ii]-mids[np.maximum(0, ii-horizon)]
                label = np.where(past <= -2, 0, np.where(past >= 2, 2, 1))
                return np.eye(3)[label]*.98+.02/3
            probabilities['persistence'] = persistence(va), persistence(te)
            for name in config['models']:
                key = f'{asset}_h{horizon}_{name}'
                cache = directory / f'{key}.npz'
                checkpoint = directory / f'{key}.pt'
                if cache.exists() and not config.get('retrain', False):
                    result = np.load(cache)
                    pv, pt = result['validation'], result['test']
                    assert np.array_equal(result['indices'], te)
                else:
                    model, history, seconds = train_model(name, prepared, config)
                    pv, pt = predict(model, prepared, va), predict(model, prepared, te)
                    np.savez_compressed(cache, validation=pv, test=pt, indices=te, targets=target)
                    torch.save(model.state_dict(), checkpoint)
                    (directory / f'{key}_training.json').write_text(json.dumps(dict(history=history, seconds=seconds, parameters=sum(p.numel() for p in model.parameters()))))
                record = json.loads((directory / f'{key}_training.json').read_text())
                histories.extend(dict(asset=asset, horizon=horizon, model=name, **h) for h in record['history'])
                runtime.append(dict(asset=asset, horizon=horizon, model=name, seconds=record['seconds'], parameters=record['parameters']))
                probabilities[name] = pv, pt
                print(f'completed {key}', flush=True)
            for name, (pv, pt) in probabilities.items():
                assert np.isfinite(pt).all() and np.allclose(pt.sum(1), 1, atol=1e-6)
                np.savez_compressed(directory / f'{asset}_h{horizon}_{name}_predictions.npz', validation=pv, test=pt, indices=te, targets=target)
                for threshold in [.3, .4, .5, .6, .7, .8, .9]:
                    rows.append(dict(asset=asset, horizon=horizon, model=name, **score(target, pt, threshold)))
    return pd.DataFrame(rows), pd.DataFrame(histories), pd.DataFrame(runtime)


def block_bootstrap(y, probability, block=256, repeats=200, seed=73):
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = []
    #resample contiguous blocks and reset transaction state at each block boundary
    for _ in range(repeats):
        starts = rng.integers(0, n-block+1, size=math.ceil(n/block))
        chunks = [np.arange(s, s+block) for s in starts]
        index = np.concatenate(chunks)[:n]
        mcc = matthews_corrcoef(y[index], probability[index].argmax(1))
        counts = [transaction_metrics(y[c], probability[c].argmax(1)) for c in chunks]
        pt, tt, ct = [sum(c[k] for c in counts) for k in ['potential', 'executed', 'correct']]
        samples.append([mcc, ct/(pt+tt-ct) if pt+tt-ct else np.nan])
    return np.nanquantile(samples, [.025, .975], axis=0)


def execution_stress(frame, indices, probability, tick, delay=0, fee_bps=0, threshold=.3):
    signal = np.where(probability.max(1) >= threshold, probability.argmax(1)-1, 0)
    trades = transaction_set(signal)
    returns = []
    for a, b, side in trades:
        entry, exit_ = indices[a]+delay, indices[b]+delay
        if exit_ >= len(frame):
            continue
        if side == 1:
            p0, p1 = frame.ask_price_1.iloc[entry], frame.bid_price_1.iloc[exit_]
            gross = p1-p0
        else:
            p0, p1 = frame.bid_price_1.iloc[entry], frame.ask_price_1.iloc[exit_]
            gross = p0-p1
        returns.append((gross-(p0+p1)*fee_bps/10000)/tick)
    return dict(delay_updates=delay, fee_bps=fee_bps, trades=len(returns),
                mean_net_ticks=float(np.mean(returns)) if returns else np.nan,
                positive_fraction=float(np.mean(np.asarray(returns)>0)) if returns else np.nan)


def rolling_daily_scale(days):
    #five completed days only; current day never enters its own scaler
    output, lineage = {}, []
    dates = sorted(days)
    for i in range(5, len(dates)):
        previous = dates[i-5:i]
        reference = np.concatenate([days[d][features].to_numpy() for d in previous])
        scaler = standard_scaler().fit(reference)
        output[dates[i]] = scaler.transform(days[dates[i]][features]).astype('float32')
        lineage.append(dict(date=dates[i], fitted_days=previous))
    return output, lineage
