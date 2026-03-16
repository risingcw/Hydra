# 🐍 Hydra — GPU Private Key, Seed & Passphrase Recovery Tool

EDIT WIF mode speed improvement

> **CUDA-accelerated brute-force recovery for partial Bitcoin and Ethereum keys.**  
> Designed for users who have lost part of a private key, WIF, BIP39 seed phrase, or BIP39 passphrase and need to recover access to their own wallet.

---

## ⚠️ Legal Disclaimer

This tool is intended **exclusively** for recovering access to wallets you own or have legal authorization to access. The authors accept no liability for misuse.

---

## Compilation

```bash
git clone https://github.com/Julienbxl/hydra.git
cd hydra
make
```

See the `Makefile` — adjust `-arch` for your GPU:

| GPU generation | Flag |
|---|---|
| RTX 50xx (Blackwell) | `sm_120` |
| RTX 40xx (Ada Lovelace) | `sm_89` |
| RTX 30xx (Ampere) | `sm_86` |
| RTX 20xx / GTX 16xx (Turing) | `sm_75` |

---

## Configuration

Open `Hydra.cu` and fill in your credentials near the top of the file:

```cpp
static const char* ETHERSCAN_API_KEY  = "YOUR_ETHERSCAN_API_KEY";
static const char* TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN";
static const char* TELEGRAM_CHAT_ID   = "YOUR_TELEGRAM_CHAT_ID";
```

**Etherscan API key** — free tier available at [etherscan.io/apis](https://etherscan.io/apis). Required for ETH balance verification on bloom hits.

**Telegram bot** — create a bot via [@BotFather](https://t.me/BotFather), then retrieve your chat ID from `https://api.telegram.org/bot<TOKEN>/getUpdates`. If left empty, Telegram notifications are skipped silently and all hits are saved to `errors.json`.

---

## Usage

```
./Hydra <key_or_phrase> <target>
```

The `<target>` can be:
- A **BTC legacy address** — `1ABC...`
- A **BTC SegWit address** — `bc1q...`
- An **ETH address** — `0x1234...abcd`
- A **bloom keyword** — `bloom`, `bloombtc`, or `bloometh`

---

## Mode 1 — Hex Private Key

Recover a 256-bit private key where some **nibbles** (hex characters) are unknown. Use `#` as the wildcard for each unknown nibble.

```bash
# 4 unknown nibbles = 16 bits — target BTC legacy
./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 1AddressBTC

# Target BTC SegWit
./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 bc1qYourAddress

# Target ETH
./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 0x1234...abcd

# Bloom — BTC only (faster)
./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 bloombtc

# Bloom — ETH only (faster)
./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 bloometh

# Bloom — BTC + ETH (useful when you don't know which chain holds the funds)
./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 bloom
```

Each `#` represents **4 unknown bits**. 

**How it works:** the fixed part of the key is precomputed as an ECC point `P_base = k_fixed × G` on CPU (OpenSSL). An affine dictionary of `2^LOW_BITS` precomputed increments handles the low-order bits on GPU at zero ECC cost; the high-order bits are enumerated via Gray code (one point addition per step).

---

## Mode 2 — BIP39 Seed Phrase

Recover 12 or 24 words BIP39 phrase where some words are unknown. Use `#` as a placeholder for each missing word.

```bash
# 2 unknown words out of 12 — 2048² ≈ 4M candidates (÷16 after BIP39 checksum filter)
./Hydra "word1 word2 # word4 # word6 word7 word8 word9 word10 word11 word12" 1AddressBTC

# Target BTC SegWit
./Hydra "word1 word2 # word4 ..." bc1qYourAddress

# Target ETH (derives via BIP44 m/44'/60'/0'/0/0)
./Hydra "word1 word2 # word4 ..." 0x1234...abcd

# Bloom — BTC + ETH 
./Hydra "word1 word2 # word4 ..." bloom
```

**Notes:**
- If the **last word** is `#`, Hydra forces the correct BIP39 checksum automatically — no wasted candidates.
- BTC derivation path: `m/44'/0'/0'/0/0`. ETH derivation path: `m/44'/60'/0'/0/0`.
- The GPU pipeline is 3-stage: **K1** filters on BIP39 checksum (~40 registers, eliminates 15 out of 16 candidates instantly), **K2a** runs PBKDF2-HMAC-SHA512, **K2b/c** handle BIP32 hardened derivation + ECC + address comparison.

---

## Mode 3 — BIP39 Passphrase (25th word)

All 12 or 24 words are known but the BIP39 passphrase (sometimes called the "25th word") is unknown. Hydra brute-forces it from a dictionary file.

```bash
# All 12 words known — reads candidates from dictionary.txt, one per line
./Hydra "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12" 1AddressBTC

# Target ETH
./Hydra "word1 ... word12" 0x1234...abcd

# Bloom — BTC + ETH
./Hydra "word1 ... word12" bloom
```

Place your candidate passphrases in `dictionary.txt`, one per line. On a bloom hit, Hydra derives the full BIP44 private key CPU-side and verifies the live balance before confirming.

---

## Mode 4 — WIF (Wallet Import Format)

Recover a compressed WIF private key (52 characters, starting with `K` or `L`) with some characters unknown. Use `#` for each unknown Base58 character.

```bash
# 3 unknown WIF characters — 58³ = 195,112 candidates
./Hydra KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qY#rFej#um7Wt#CRUx 1AddressBTC

# Target BTC SegWit
./Hydra KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qY#rFej#um7Wt#CRUx bc1qYourAddress

# Bloom — 'bloom' and 'bloombtc' are equivalent in WIF mode (always BTC)
./Hydra KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qY#rFej#um7Wt#CRUx bloom
```

A SHA256×2 checksum pre-filter eliminates the overwhelming majority of candidates before the ECC step, making this mode fast even with several unknowns.

---

## Bloom Filter Mode

Instead of searching for a single address, Hydra can test every candidate against a **bloom filter** preloaded in VRAM — effectively scanning against millions of addresses simultaneously with no per-address overhead.

### Keywords

| Keyword | Behavior |
|---|---|
| `bloombtc` | Checks BTC legacy + SegWit only (fastest) |
| `bloometh` | Checks ETH only (fastest) |
| `bloom` | Checks BTC + ETH — use when the chain is unknown |

### Creating a bloom filter

```bash
# One or more input files (mixed BTC/ETH formats), one address per line
python3 create_bloom.py addresses.txt bloom.bin

# Multiple input files
python3 create_bloom.py btc.txt eth.txt bloom.bin
```

The script accepts BTC legacy (`1...`), BTC SegWit (`bc1q...`), and ETH (`0x...`) addresses mixed in the same file. The output `bloom.bin` is loaded directly by Hydra at startup when using a bloom keyword.

**Dependencies:**
```bash
pip install mmh3 base58 bech32 bitarray tqdm
```

### Checking an address against the filter

```bash
python3 check_bloom.py <address>
```

Checks whether a single address (BTC or ETH) is present in `bloom.bin`. Useful to verify that a filter was built correctly before starting a long run.

### Tuning for limited VRAM

The default filter is calibrated for **110 million addresses** and produces a **2 GB** output file (`TARGET_SIZE_GB = 2` in `create_bloom.py`), which corresponds to exactly `2^34` bits. If your GPU has limited VRAM, reduce this value:

```python
# Default — 2 GB, 2^34 bits, optimal for 100M addresses
TARGET_SIZE_GB = 2

# ~1 GB VRAM
TARGET_SIZE_GB = 1    # 2^33 bits

# ~512 MB VRAM
TARGET_SIZE_GB = 0.5  # 2^32 bits
```

> ⚠️ **`TARGET_SIZE_GB` must be a power of two in gigabytes** (0.5, 1, 2, 4…).  
> Internally, `FILTER_BITS = TARGET_SIZE_GB × 8 × 1024³`. For the GPU bitmask to work correctly, this value must be an exact power of 2. Any other size will silently produce incorrect results.

Reducing the filter size increases the false positive rate (more spurious API balance checks), but does not affect correctness — every true hit will still be found.

### False positive rate

With `k=16` hash functions and `n=100,000,000` addresses:

```
p = (1 - e^(-k·n/m))^k
```

| `TARGET_SIZE_GB` | Filter bits | False positives / Gkey |
|---|---|---|
| 0.5 GB | `2^32` | ~27 |
| 1 GB | `2^33` | ~0.0019 |
| **2 GB** *(default)* | **`2^34`** | **~0.0000** |

The false positive rate scales with the number of addresses loaded. The table above assumes 110 million addresses — consistent with observed results (~6 false positives over a full 2^40 run at ~1 GK/s).

---

## Performance Tuning for HEX mode — `LOW_BITS`

Hydra splits the search space into two parts. **High bits** are enumerated via Gray code — each step costs exactly one ECC point addition on GPU. **Low bits** are handled by a precomputed affine dictionary stored in GPU constant memory, with zero ECC cost per lookup. The dictionary has `2^LOW_BITS` entries.

The constant is defined in `HydraCommon.h`:

```cpp
#define LOW_BITS 9   // Default — 512-entry dictionary, ~24 KB constant memory
```

| `LOW_BITS` | Dictionary size | Constant memory | Recommended for |
|---|---|---|---|
| `6` | 64 entries | ~3 KB | Low-end or older GPUs |
| `7` | 128 entries | ~6 KB | Mid-range cards |
| `8` | 256 entries | ~12 KB | Good general balance |
| `9` *(default)* | 512 entries | ~24 KB | RTX 30xx / 40xx / 50xx |
| `10` | 1024 entries | ~48 KB | High-end cards with large L1 |

Higher `LOW_BITS` = more work per Gray-code step = fewer kernel launches = higher throughput. However, if constant memory fills up, GPU occupancy drops and performance may degrade. If you observe lower-than-expected throughput on a weaker GPU, try stepping `LOW_BITS` down by 1 or 2. Recompile after any change.

---

## Testing

Four Python test scripts verify that Hydra works correctly after compilation. Each script generates 10 **random known cases**, runs Hydra against them, and checks that the correct answer is found. All four are self-contained (Python stdlib only, no external dependencies).

```bash
# 10 tests — random private key, 8 unknown nibbles, cycles through BTC legacy / SegWit / ETH
python3 testhex.py [./Hydra]

# 10 tests — random WIF, 5 unknown characters, alternates BTC legacy / SegWit
python3 testwif.py [./Hydra]

# 10 tests — random BIP39 phrase, unknown words, BTC and ETH targets
python3 testseed.py [./Hydra]

# 10 tests — known mnemonic, brute-forces passphrase
python3 testpass.py [./Hydra]
```

Each script outputs `✅ PASS` or `❌ FAIL` per test, with a final `N/10` summary. Run the full suite after every recompile or after changing `LOW_BITS`:

```bash
python3 testhex.py && python3 testwif.py && python3 testseed.py && python3 testpass.py
```

---

## Performance Benchmarks

*Measured on RTX 5060 (30 SM, Blackwell sm_120)*

| Mode | Target | Throughput |
|---|---|---|
| Hex | BTC legacy / SegWit | ~1,200 MK/s |
| Hex | Bloom BTC | ~1,000 MK/s |
| Hex | ETH & Bloom ETH | ~700 MK/s |
| Hex | Bloom (BTC+ETH) | ~550 MK/s |
| Seed | All modes | ~1,700,000 seeds/s |
| Passphrase | All modes | ~100,000 pass/s |
| WIF | All modes | ~2500 MK/s |

---

## Output & Notifications

When a match is found, Hydra:

1. Prints a victory block to stdout with the private key (hex) and the address(es) relevant to the search mode — BTC addresses in BTC mode, ETH address in ETH mode.
2. Sends a Telegram message with the key and the matched address.
3. In bloom mode: verifies the live balance via blockchain API before confirming. Zero-balance hits (false positives) are silently discarded and the search continues automatically.
4. Any hit that could not be verified due to a network error is written to `errors.json` for manual review after the run.

---

## Support the Project

If Hydra helped you recover your funds, consider a donation.

**BTC:** `bc1qsn23hyqhwkw4775ssykdtegxqgmwpe9qns3y0m`  
**ETH / ERC-20:** `0x8f00CbC520876a62eE07b54c2266d988fE61cD86`

---
