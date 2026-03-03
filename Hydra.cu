/*
 * ======================================================================================
 * HYDRA V1.0
 * ======================================================================================
 * Tool for recovering a partial private key or partial BIP39 seed.
 *
 * USAGE :
 *   Hex key mode  (unknown nibbles = '#') :
 *     ./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 1AddressBTC
 *     ./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 0x1234...abcd
 *
 *   BIP39 seed mode (unknown words = '#') :
 *     ./Hydra "word1 word2 # word4 # word6 word7 word8 word9 word10 word11 word12" 1AddressBTC
 *
 * ======================================================================================
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <csignal>
#include <mutex>
#include <atomic>
#include <ctime>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// OpenSSL for CPU-side ECC precomputation + HTTPS
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

#include "HydraCommon.h"
#include "Bloom.h"
#include "ECC.h"
#include "Hash.cuh"
#include "Gray.h"
#include "BIP39_Dict.h"
#include "Seed.cuh"
#include "Wif.cuh"

// =================================================================================
// SIGNAL HANDLER
// =================================================================================
static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

// =================================================================================
// API CONFIGURATION & NOTIFICATIONS
// =================================================================================
static const char* ETHERSCAN_API_KEY  = "YOUR_API_KEY";
static const char* TELEGRAM_BOT_TOKEN = "YOUR_TOKEN";
static const char* TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID";

static std::mutex g_log_mutex;
static std::atomic<int> g_api_errors{0};  // bloom hits that could not be verified (network error)

// Print end-of-run summary (called in all 4 modes)
static void print_search_summary(bool found) {
    if (!found) {
        std::cout << "  No wallet found.\n";
    }
    if (g_api_errors > 0) {
        std::cout << "  /!\\ " << g_api_errors << " unverified API error(s) — check errors.json\n";
    }
}

// =================================================================================
// NETWORK & NOTIFICATION HELPERS
// =================================================================================

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

// Minimal HTTPS GET via OpenSSL (no libcurl required)
static std::string https_get(const std::string& host, const std::string& path) {
    SSL_CTX* ctx = nullptr;
    SSL* ssl     = nullptr;
    int sock     = -1;
    std::string response;

    try {
        SSL_library_init();
        OpenSSL_add_all_algorithms();
        SSL_load_error_strings();

        ctx = SSL_CTX_new(TLS_client_method());
        if (!ctx) throw std::runtime_error("SSL context failed");

        struct hostent* he = gethostbyname(host.c_str());
        if (!he) throw std::runtime_error("DNS failed: " + host);

        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(443);
        addr.sin_addr   = *((struct in_addr*)he->h_addr);

        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) throw std::runtime_error("socket() failed");
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sock); throw std::runtime_error("connect() failed: " + host);
        }

        ssl = SSL_new(ctx);
        SSL_set_tlsext_host_name(ssl, host.c_str());
        SSL_set_fd(ssl, sock);
        if (SSL_connect(ssl) <= 0) throw std::runtime_error("SSL_connect failed");

        std::string req = "GET " + path + " HTTP/1.0\r\nHost: " + host +
                          "\r\nConnection: close\r\nUser-Agent: Hydra\r\n\r\n";
        SSL_write(ssl, req.c_str(), req.length());

        char buf[4096]; int n;
        while ((n = SSL_read(ssl, buf, sizeof(buf)-1)) > 0) {
            buf[n] = 0; response += buf;
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[HTTPS] " << e.what() << "\n";
    }

    if (ssl)  { SSL_shutdown(ssl); SSL_free(ssl); }
    if (sock >= 0) close(sock);
    if (ctx)  SSL_CTX_free(ctx);

    size_t p = response.find("\r\n\r\n");
    return (p != std::string::npos) ? response.substr(p + 4) : "";
}

// Log error to errors.json (Telegram failure or balance API failure)
static void log_error(const std::string& type, const std::string& detail) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::ofstream f("errors.json", std::ios::app);
    if (!f.is_open()) { std::cerr << "[Logger] Cannot open errors.json\n"; return; }

    std::time_t now = std::time(nullptr);
    char tbuf[32];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    f << "{\"ts\":\"" << tbuf << "\","
      << "\"type\":\"" << json_escape(type) << "\","
      << "\"detail\":\"" << json_escape(detail) << "\"}\n";
    f.flush();
}

// Returns true if the message was delivered (Telegram API ack ok:true)
static bool send_telegram(const std::string& message) {
    if (!TELEGRAM_BOT_TOKEN || std::string(TELEGRAM_BOT_TOKEN).empty()) return false;

    std::ostringstream oss;
    oss << std::hex;
    for (unsigned char c : message) {
        if (isalnum(c) || c=='-' || c=='_' || c=='.' || c=='~') oss << c;
        else oss << '%' << std::uppercase << std::setw(2) << std::setfill('0') << (int)c;
    }
    std::string path = "/bot" + std::string(TELEGRAM_BOT_TOKEN) +
                       "/sendMessage?chat_id=" + std::string(TELEGRAM_CHAT_ID) +
                       "&text=" + oss.str() + "&parse_mode=Markdown";
    try {
        std::string resp = https_get("api.telegram.org", path);
        // Telegram API returns {"ok":true,...} or {"ok":false,...}
        if (resp.find("\"ok\":true") != std::string::npos) {
            std::cout << "  [Telegram] Message delivered.\n";
            return true;
        } else {
            std::string err = (resp.size() > 200) ? resp.substr(0, 200) : resp;
            std::cout << "  [Telegram] Failed : " << err << "\n";
            log_error("telegram_failed", err);
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "  [Telegram] Network error : " << e.what() << "\n";
        log_error("telegram_network_error", e.what());
        return false;
    }
}

static double parse_btc_balance(const std::string& raw) {
    if (raw.empty()) return 0.0;
    try { return std::stold(raw) / 1e8; } catch (...) { return 0.0; }
}

static double parse_eth_balance(const std::string& json) {
    std::string key = "\"result\":\"";
    size_t pos = json.find(key);
    if (pos == std::string::npos) return 0.0;
    size_t s = pos + key.length();
    size_t e = json.find("\"", s);
    if (e == std::string::npos) return 0.0;
    try { return std::stold(json.substr(s, e-s)) / 1e18; } catch (...) { return 0.0; }
}

// Log API error : only log hits we could not verify (network down).
// Sole purpose : review manually after the run.
static void log_api_error(
    const std::string& priv_hex,
    const std::string& btc_legacy,
    const std::string& btc_segwit,
    const std::string& eth_addr,
    const std::string& extra = "")
{
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_api_errors++;

    std::ofstream f("errors.json", std::ios::app);
    if (!f.is_open()) { std::cerr << "[Logger] Cannot open errors.json\n"; return; }

    std::time_t now = std::time(nullptr);
    char tbuf[32];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    f << "{\"ts\":\"" << tbuf << "\","
      << "\"type\":\"api_balance_error\","
      << "\"pk\":\"" << json_escape(priv_hex) << "\","
      << "\"btc_legacy\":\"" << json_escape(btc_legacy) << "\","
      << "\"btc_segwit\":\"" << json_escape(btc_segwit) << "\","
      << "\"eth\":\"" << json_escape(eth_addr) << "\"";
    if (!extra.empty())
        f << ",\"extra\":\"" << json_escape(extra) << "\"";
    f << "}\n";
    f.flush();
}

// Send a victory notification via Telegram for all modes.
// key_info  : private key hex, WIF, or seed phrase
// addr_info : relevant address(es) to display
static void notify_victory(const std::string& mode_title,
                           const std::string& key_info,
                           const std::string& addr_info) {
    std::ostringstream msg;
    msg << "*HYDRA - " << mode_title << "* \xF0\x9F\x8F\x86\n\n"
        << key_info << "\n\n"
        << addr_info;
    bool ok = send_telegram(msg.str());
    if (!ok) {
        log_error("telegram_victory_lost",
                  mode_title + " | " + key_info + " | " + addr_info);
        std::cout << "  /!\\ Telegram failed — details saved to errors.json\n";
    }
}

// Returns true if balance > 0 (real hit), false on false positive or network error.
// On network error : logs to errors.json and continues.
static bool check_balances_and_notify(
    const uint8_t* key32,
    const std::string& btc_legacy,
    const std::string& btc_segwit,
    const std::string& eth_addr)
{
    std::ostringstream pk_ss;
    pk_ss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) pk_ss << std::setw(2) << (int)key32[i];
    const std::string priv_hex = pk_ss.str();

    std::cout << "\n!!! BLOOM HIT !!!\n";
    std::cout << "  Private key : " << priv_hex << "\n";
    std::cout << "  BTC legacy  : " << btc_legacy << "\n";
    std::cout << "  BTC segwit  : " << btc_segwit << "\n";
    std::cout << "  ETH         : " << eth_addr << "\n";
    std::cout << "  [API] Checking balances...\n";

    bool network_error = false;
    double btc_bal = 0.0, eth_bal = 0.0;

    try {
        std::string btc_raw = https_get("blockchain.info", "/q/addressbalance/" + btc_legacy);
        if (btc_raw.empty()) network_error = true;
        else btc_bal = parse_btc_balance(btc_raw);

        std::string eth_path = "/v2/api?chainid=1&module=account&action=balance&address=" +
                               eth_addr + "&tag=latest&apikey=" + std::string(ETHERSCAN_API_KEY);
        std::string eth_raw = https_get("api.etherscan.io", eth_path);
        if (eth_raw.empty()) network_error = true;
        else eth_bal = parse_eth_balance(eth_raw);

    } catch (...) { network_error = true; }

    if (network_error) {
        std::cout << "  > Network error — logged to errors.json, continuing.\n";
        log_api_error(priv_hex, btc_legacy, btc_segwit, eth_addr);
        return false;
    }

    std::cout << "  > BTC : " << std::fixed << std::setprecision(8) << btc_bal << " BTC\n";
    std::cout << "  > ETH : " << std::fixed << std::setprecision(8) << eth_bal << " ETH\n";

    if (btc_bal > 0.0 || eth_bal > 0.0) {
        std::cout << "\n*** NON-ZERO BALANCE — WALLET FOUND ***\n";
        std::ostringstream key_info, addr_info;
        key_info << "*Private Key:*\n`" << priv_hex << "`";
        if (btc_bal > 0.0 && eth_bal > 0.0) {
            addr_info << "*BTC:* `" << btc_legacy << "`\n`"
                      << std::fixed << std::setprecision(8) << btc_bal << " BTC`\n"
                      << "*ETH:* `" << eth_addr << "`\n`"
                      << std::fixed << std::setprecision(8) << eth_bal << " ETH`";
        } else if (btc_bal > 0.0) {
            addr_info << "*BTC:* `" << btc_legacy << "`\n`"
                      << std::fixed << std::setprecision(8) << btc_bal << " BTC`";
        } else {
            addr_info << "*ETH:* `" << eth_addr << "`\n`"
                      << std::fixed << std::setprecision(8) << eth_bal << " ETH`";
        }
        notify_victory("WALLET FOUND \xF0\x9F\x92\xB8", key_info.str(), addr_info.str());
        return true;
    }

    std::cout << "  > Zero balance — false positive, continuing.\n";
    return false;
}

// =================================================================================
// 1. CPU HELPERS : SHA256 + BASE58 + KECCAK (for decoding the target address)
// =================================================================================

static const uint32_t K_CPU[64] = {
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,
    0x923F82A4,0xAB1C5ED5,0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,
    0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,0xE49B69C1,0xEFBE4786,
    0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,
    0x06CA6351,0x14292967,0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,
    0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,0xA2BFE8A1,0xA81A664B,
    0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,
    0x5B9CCA4F,0x682E6FF3,0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,
    0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
};

#define ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z)  (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define SIG0(x) (ROTR(x,2)^ROTR(x,13)^ROTR(x,22))
#define SIG1(x) (ROTR(x,6)^ROTR(x,11)^ROTR(x,25))
#define sig0(x) (ROTR(x,7)^ROTR(x,18)^((x)>>3))
#define sig1(x) (ROTR(x,17)^ROTR(x,19)^((x)>>10))

static void sha256_cpu_block(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i=0; i<16; i++)
        w[i]=(block[i*4]<<24)|(block[i*4+1]<<16)|(block[i*4+2]<<8)|block[i*4+3];
    for (int i=16; i<64; i++)
        w[i]=sig1(w[i-2])+w[i-7]+sig0(w[i-15])+w[i-16];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],
             e=state[4],f=state[5],g=state[6],h=state[7];
    for (int i=0; i<64; i++) {
        uint32_t t1=h+SIG1(e)+CH(e,f,g)+K_CPU[i]+w[i];
        uint32_t t2=SIG0(a)+MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

static void sha256_cpu(const uint8_t *data, size_t len, uint8_t out[32]) {
    uint32_t state[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                       0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t block[64]={0};
    size_t remain = len;
    const uint8_t *ptr = data;
    while (remain >= 64) { sha256_cpu_block(state, ptr); ptr+=64; remain-=64; }
    memcpy(block, ptr, remain);
    block[remain]=0x80;
    if (remain >= 56) {
        sha256_cpu_block(state, block);
        memset(block, 0, 64);
    }
    uint64_t bits = len*8;
    for (int i=0; i<8; i++) block[63-i]=(bits>>(i*8))&0xFF;
    sha256_cpu_block(state, block);
    for (int i=0; i<8; i++) {
        out[i*4]=(state[i]>>24)&0xFF; out[i*4+1]=(state[i]>>16)&0xFF;
        out[i*4+2]=(state[i]>>8)&0xFF; out[i*4+3]=state[i]&0xFF;
    }
}

// BTC Base58Check → hash160
static bool base58Decode(const std::string &addr, uint8_t out[25]) {
    static const char *alpha="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    uint8_t result[25]={0};
    for (char c : addr) {
        const char *p=strchr(alpha, c);
        if (!p) return false;
        int carry=(int)(p-alpha);
        for (int i=24; i>=0; --i) { carry+=58*result[i]; result[i]=carry%256; carry/=256; }
    }
    memcpy(out, result, 25);
    return true;
}

static bool addrToHash160(const std::string &addr, uint8_t hash160[20]) {
    if (addr.size() < 26 || addr.size() > 35) return false;
    uint8_t decoded[25];
    if (!base58Decode(addr, decoded)) return false;
    uint8_t check[32];
    sha256_cpu(decoded, 21, check);
    sha256_cpu(check, 32, check);
    if (memcmp(check, decoded+21, 4)!=0) return false;
    memcpy(hash160, decoded+1, 20);
    return true;
}

// ETH 0x... → 20 bytes

// SegWit bech32 decode → hash160 (P2WPKH bc1q...)
static const int8_t BECH32_REV[128] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    15,-1,10,17,21,20,26,30, 7, 5,-1,-1,-1,-1,-1,-1,
    -1,29,-1,24,13,25, 9, 8,23,-1,18,22,31,27,19,-1,
     1, 0, 3,16,11,28,12,14, 6, 4, 2,-1,-1,-1,-1,-1,
    -1,29,-1,24,13,25, 9, 8,23,-1,18,22,31,27,19,-1,
     1, 0, 3,16,11,28,12,14, 6, 4, 2,-1,-1,-1,-1,-1
};
static bool bech32_decode_hash160(const std::string &addr, uint8_t hash160[20]) {
    size_t sep = addr.rfind('1');
    if(sep == std::string::npos || sep < 2) return false;
    if(addr.substr(0,sep) != "bc") return false;
    std::string ds = addr.substr(sep+1);
    std::vector<uint8_t> d5;
    for(char ch : ds){
        unsigned char uc=(unsigned char)ch;
        if(uc>=128) return false;
        int v=BECH32_REV[uc]; if(v<0) return false;
        d5.push_back((uint8_t)v);
    }
    if(d5.size()!=39 || d5[0]!=0) return false;
    uint32_t acc=0; int bits=0,out_idx=0;
    for(int i=1;i<=32;i++){
        acc=(acc<<5)|d5[i]; bits+=5;
        if(bits>=8){ bits-=8; if(out_idx>=20) return false; hash160[out_idx++]=(uint8_t)(acc>>bits); }
    }
    return out_idx==20;
}
static bool addrToHash160Any(const std::string &addr, uint8_t hash160[20]) {
    if(addr.size()>=4 && addr[0]=='b' && addr[1]=='c' && addr[2]=='1')
        return bech32_decode_hash160(addr,hash160);
    return addrToHash160(addr, hash160);
}

static bool ethAddrToBytes(const std::string &addr, uint8_t out[20]) {
    std::string s = addr;
    if (s.size() >= 2 && s[0]=='0' && (s[1]=='x'||s[1]=='X')) s=s.substr(2);
    if (s.size() != 40) return false;
    for (int i=0; i<20; i++) {
        auto h=[](char c)->int{
            if(c>='0'&&c<='9') return c-'0';
            if(c>='a'&&c<='f') return c-'a'+10;
            if(c>='A'&&c<='F') return c-'A'+10;
            return -1;
        };
        int hi=h(s[i*2]), lo=h(s[i*2+1]);
        if(hi<0||lo<0) return false;
        out[i]=(uint8_t)(hi<<4|lo);
    }
    return true;
}

// =================================================================================
// 2. HEX MASK PARSING (HEX MODE)
//    "7cb5...XX...e4" → k_fixed (var bits = 0) + var_bit_positions[]
// =================================================================================

struct MaskParseResult {
    uint8_t  k_fixed[32];          // fixed key (big-endian, var bits = 0)
    std::vector<int> var_bit_positions; // variable bit positions (LSB=0)
    bool valid = false;
    uint64_t total_candidates = 0;
};

static MaskParseResult parseMask(const std::string &mask) {
    MaskParseResult r;

    // Strip optional 0x prefix
    std::string s = mask;
    if (s.size() >= 2 && s[0]=='0' && (s[1]=='x'||s[1]=='X')) s=s.substr(2);

    if (s.size() != 64) {
        std::cerr << "Error: hex mask must be exactly 64 characters (32 bytes).\n";
        return r;
    }

    memset(r.k_fixed, 0, 32);

    // Walk nibble by nibble (left to right = MSB to LSB)
    for (int n = 0; n < 64; n++) {
        char c = s[n];
        // Nibble position in k_fixed (big-endian) :
        // nibble n → byte n/2, bits [7-4] if even, [3-0] if odd
        int byte_idx = n / 2;
        int shift = (n % 2 == 0) ? 4 : 0;

        if (c == '#') {
            // 4 variable bits
            // Global bit index (LSB=0) : nibble n maps to bits
            // [255-n*4 .. 255-n*4-3] in MSB=255 notation
            for (int b = 3; b >= 0; b--) {
                int bit_index = (63 - n) * 4 + b; // bit global (LSB=0)
                r.var_bit_positions.push_back(bit_index);
            }
            // k_fixed keeps 0 for these nibbles
        } else {
            int nibble_val = -1;
            if (c>='0'&&c<='9') nibble_val=c-'0';
            else if (c>='a'&&c<='f') nibble_val=c-'a'+10;
            else if (c>='A'&&c<='F') nibble_val=c-'A'+10;
            else {
                std::cerr << "Error: invalid character '" << c << "' in mask.\n";
                return r;
            }
            r.k_fixed[byte_idx] |= (uint8_t)(nibble_val << shift);
        }
    }

    int m = (int)r.var_bit_positions.size();
    if (m == 0) {
        std::cerr << "Error: no variable nibble ('#') in mask.\n";
        return r;
    }
    if (m > MAX_VAR_BITS) {
        std::cerr << "Error: too many variable bits (" << m << " > " << MAX_VAR_BITS << ").\n";
        return r;
    }

    r.total_candidates = 1ULL << m;
    r.valid = true;

    std::cout << "Mask parsed  : " << m << " variable bits, "
              << r.total_candidates << " candidates (2^" << m << ")\n";
    return r;
}

// =================================================================================
// 3. CPU ECC PRECOMPUTATION (OpenSSL)
//    P_base = k_fixed * G
//    Q_i    = 2^(var_bit_positions[i]) * G
// =================================================================================

// Convert uint8_t[32] big-endian to uint64_t[4] little-endian (ECC.h format)
static void be32_to_le4(const uint8_t be[32], uint64_t le[4]) {
    for (int i = 0; i < 4; i++) {
        le[i] = 0;
        for (int b = 0; b < 8; b++)
            le[i] |= ((uint64_t)be[31 - i*8 - b]) << (b*8);
    }
}

// Compute scalar * G and return x,y as little-endian uint64_t[4]
// scalar est un uint8_t[32] big-endian
static bool ec_mul_G(const uint8_t scalar_be[32], uint64_t x_le[4], uint64_t y_le[4]) {
    EC_GROUP *group = EC_GROUP_new_by_curve_name(NID_secp256k1);
    EC_POINT *P     = EC_POINT_new(group);
    BIGNUM   *k     = BN_bin2bn(scalar_be, 32, nullptr);
    BN_CTX   *ctx   = BN_CTX_new();

    bool ok = (EC_POINT_mul(group, P, k, nullptr, nullptr, ctx) == 1);

    if (ok) {
        BIGNUM *bx = BN_new(), *by = BN_new();
        EC_POINT_get_affine_coordinates(group, P, bx, by, ctx);

        uint8_t xb[32]={0}, yb[32]={0};
        BN_bn2binpad(bx, xb, 32);
        BN_bn2binpad(by, yb, 32);
        be32_to_le4(xb, x_le);
        be32_to_le4(yb, y_le);

        BN_free(bx); BN_free(by);
    }

    BN_free(k); BN_CTX_free(ctx);
    EC_POINT_free(P); EC_GROUP_free(group);
    return ok;
}

// DictX, DictY, DictValid sont définis dans Gray.h (inclus ci-dessus)

static bool precompute_ecc(const MaskParseResult &mask_r, HydraData &hd) {
    std::cout << "ECC precomputation (CPU/OpenSSL)...\n";

    int total_bits = (int)mask_r.var_bit_positions.size();
    int low_bits   = std::min(LOW_BITS, total_bits);  // bits bas → dictionnaire
    int high_bits  = total_bits - low_bits;            // bits hauts → Gray Code

    hd.num_var_bits     = (uint32_t)total_bits;
    hd.num_high_bits    = (uint32_t)high_bits;
    hd.total_candidates = mask_r.total_candidates;
    hd.high_candidates  = (uint64_t)1 << high_bits;
    hd.gray_offset_start = 0;

    // P_base = k_fixed * G (high bits=0 AND low bits=0)
    if (!ec_mul_G(mask_r.k_fixed, hd.base_x, hd.base_y)) {
        std::cerr << "Error: P_base computation failed.\n"; return false;
    }
    std::cout << "  P_base computed\n";

    // HIGH deltas : Q_i = 2^(var_bit_positions[low_bits + i]) * G
    for (int i = 0; i < high_bits; i++) {
        int bit_pos = mask_r.var_bit_positions[low_bits + i];
        uint8_t scalar[32] = {0};
        int byte_idx = 31 - (bit_pos / 8);
        int bit_off  = bit_pos % 8;
        if (byte_idx >= 0 && byte_idx < 32)
            scalar[byte_idx] = (uint8_t)(1 << bit_off);
        if (!ec_mul_G(scalar, hd.delta_x[i], hd.delta_y[i])) {
            std::cerr << "Error: high delta " << i << " failed\n"; return false;
        }
    }
    std::cout << "  " << high_bits << " high deltas computed\n";

    // -----------------------------------------------------------------------
    // LOW DICTIONARY : precompute 2^low_bits affine points
    // R_k = sum of (bit_j * Q_j) for j in the active low_bits of k
    // Built by Gray Code : R_0=infinity, R_k = R_{k-1} ± Q_j
    // -----------------------------------------------------------------------
    // Base intermediate points : Q_j = 2^(var_bit_positions[j]) * G
    uint64_t Q_low_x[LOW_BITS][4], Q_low_y[LOW_BITS][4];
    for (int j = 0; j < low_bits; j++) {
        int bit_pos = mask_r.var_bit_positions[j];
        uint8_t scalar[32] = {0};
        int byte_idx = 31 - (bit_pos / 8);
        int bit_off  = bit_pos % 8;
        if (byte_idx >= 0 && byte_idx < 32)
            scalar[byte_idx] = (uint8_t)(1 << bit_off);
        if (!ec_mul_G(scalar, Q_low_x[j], Q_low_y[j])) {
            std::cerr << "Error: low delta " << j << " failed\n"; return false;
        }
    }

    // Allocate dictionary on CPU side
    // Build by direct binary decoding via OpenSSL EC_POINT_add
    static uint64_t dict_x[LOW_SIZE][4];
    static uint64_t dict_y[LOW_SIZE][4];
    static uint8_t  dict_valid[LOW_SIZE];

    // For each k, add the active Q_low[j] points via OpenSSL EC_POINT_add
    // O(2^low_bits * low_bits) additions = 64*6 = 384 ops → negligible

    // Initialize with OpenSSL
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX   *ctx = BN_CTX_new();

    auto limbs_to_bn = [](const uint64_t limbs[4]) -> BIGNUM* {
        BIGNUM *bn = BN_new();
        uint8_t buf[32];
        for (int i=0;i<4;i++){
            uint64_t w = limbs[3-i];
            buf[i*8+0]=(w>>56)&0xFF; buf[i*8+1]=(w>>48)&0xFF;
            buf[i*8+2]=(w>>40)&0xFF; buf[i*8+3]=(w>>32)&0xFF;
            buf[i*8+4]=(w>>24)&0xFF; buf[i*8+5]=(w>>16)&0xFF;
            buf[i*8+6]=(w>> 8)&0xFF; buf[i*8+7]=(w    )&0xFF;
        }
        BN_bin2bn(buf, 32, bn);
        return bn;
    };

    auto bn_to_limbs = [](const BIGNUM *bn, uint64_t limbs[4]) {
        uint8_t buf[32] = {0};
        int len = BN_num_bytes(bn);
        BN_bn2bin(bn, buf + (32 - len));
        for (int i=0;i<4;i++){
            limbs[3-i] = ((uint64_t)buf[i*8+0]<<56)|((uint64_t)buf[i*8+1]<<48)
                        |((uint64_t)buf[i*8+2]<<40)|((uint64_t)buf[i*8+3]<<32)
                        |((uint64_t)buf[i*8+4]<<24)|((uint64_t)buf[i*8+5]<<16)
                        |((uint64_t)buf[i*8+6]<< 8)|((uint64_t)buf[i*8+7]);
        }
    };

    // Create EC_POINT for each Q_low[j]
    EC_POINT *Qpts[LOW_BITS];
    for (int j=0;j<low_bits;j++){
        Qpts[j] = EC_POINT_new(grp);
        BIGNUM *qx = limbs_to_bn(Q_low_x[j]);
        BIGNUM *qy = limbs_to_bn(Q_low_y[j]);
        EC_POINT_set_affine_coordinates(grp, Qpts[j], qx, qy, ctx);
        BN_free(qx); BN_free(qy);
    }

    // Build dict[k] for k=0..LOW_SIZE-1
    // k=0 : identity (DictValid=0)
    dict_valid[0] = 0;
    memset(dict_x[0], 0, sizeof(dict_x[0]));
    memset(dict_y[0], 0, sizeof(dict_y[0]));

    EC_POINT *acc = EC_POINT_new(grp);
    for (int k=1; k<LOW_SIZE; k++){
        // Decode k in binary and add the active Q_low[j] points
        EC_POINT_set_to_infinity(grp, acc);
        for (int j=0;j<low_bits;j++){
            if ((k>>j)&1){
                EC_POINT_add(grp, acc, acc, Qpts[j], ctx);
            }
        }
        if (EC_POINT_is_at_infinity(grp, acc)){
            dict_valid[k]=0;
            memset(dict_x[k],0,sizeof(dict_x[k]));
            memset(dict_y[k],0,sizeof(dict_y[k]));
        } else {
            BIGNUM *rx=BN_new(), *ry=BN_new();
            EC_POINT_get_affine_coordinates(grp, acc, rx, ry, ctx);
            bn_to_limbs(rx, dict_x[k]);
            bn_to_limbs(ry, dict_y[k]);
            BN_free(rx); BN_free(ry);
            dict_valid[k]=1;
        }
    }

    // Upload dictionary to constant memory
    cudaMemcpyToSymbol(DictX,     dict_x,     sizeof(dict_x));
    cudaMemcpyToSymbol(DictY,     dict_y,     sizeof(dict_y));
    cudaMemcpyToSymbol(DictValid, dict_valid, sizeof(dict_valid));

    // Cleanup
    EC_POINT_free(acc);
    for (int j=0;j<low_bits;j++) EC_POINT_free(Qpts[j]);
    EC_GROUP_free(grp); BN_CTX_free(ctx);

    std::cout << "  " << LOW_SIZE << " dictionary points precomputed (low bits)\n";
    return true;
}


// =================================================================================
// CPU : PRIVATE KEY → BTC ADDRESSES (legacy, segwit) + ETH
// =================================================================================

// RIPEMD-160 (compact implementation)

static void ripemd160_cpu(const uint8_t* data, size_t len, uint8_t out[20]) {
    #define RMD_F(x,y,z) ((x)^(y)^(z))
    #define RMD_G(x,y,z) (((x)&(y))|(~(x)&(z)))
    #define RMD_H(x,y,z) (((x)|(~(y)))^(z))
    #define RMD_I(x,y,z) (((x)&(z))|((y)&~(z)))
    #define RMD_J(x,y,z) ((x)^((y)|(~(z))))
    #define ROL32(x,n) (((x)<<(n))|((x)>>(32-(n))))
    #define RMD_FF(a,b,c,d,e,x,s)  a=ROL32(a+RMD_F(b,c,d)+x,s)+e;c=ROL32(c,10)
    #define RMD_GG(a,b,c,d,e,x,s)  a=ROL32(a+RMD_G(b,c,d)+x+0x5A827999u,s)+e;c=ROL32(c,10)
    #define RMD_HH(a,b,c,d,e,x,s)  a=ROL32(a+RMD_H(b,c,d)+x+0x6ED9EBA1u,s)+e;c=ROL32(c,10)
    #define RMD_II(a,b,c,d,e,x,s)  a=ROL32(a+RMD_I(b,c,d)+x+0x8F1BBCDCu,s)+e;c=ROL32(c,10)
    #define RMD_JJ(a,b,c,d,e,x,s)  a=ROL32(a+RMD_J(b,c,d)+x+0xA953FD4Eu,s)+e;c=ROL32(c,10)
    #define RMD_FFF(a,b,c,d,e,x,s) a=ROL32(a+RMD_F(b,c,d)+x,s)+e;c=ROL32(c,10)
    #define RMD_GGG(a,b,c,d,e,x,s) a=ROL32(a+RMD_G(b,c,d)+x+0x7A6D76E9u,s)+e;c=ROL32(c,10)
    #define RMD_HHH(a,b,c,d,e,x,s) a=ROL32(a+RMD_H(b,c,d)+x+0x6D703EF3u,s)+e;c=ROL32(c,10)
    #define RMD_III(a,b,c,d,e,x,s) a=ROL32(a+RMD_I(b,c,d)+x+0x5C4DD124u,s)+e;c=ROL32(c,10)
    #define RMD_JJJ(a,b,c,d,e,x,s) a=ROL32(a+RMD_J(b,c,d)+x+0x50A28BE6u,s)+e;c=ROL32(c,10)
    uint32_t h0=0x67452301u,h1=0xEFCDAB89u,h2=0x98BADCFEu,h3=0x10325476u,h4=0xC3D2E1F0u;
    size_t total=((len+9+63)/64)*64;
    std::vector<uint8_t> msg(total,0);
    memcpy(msg.data(),data,len); msg[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for(int i=0;i<8;i++) msg[total-8+i]=(uint8_t)(bits>>(i*8));
    for(size_t blk=0;blk<total;blk+=64){
        uint32_t X[16];
        for(int i=0;i<16;i++) X[i]=(uint32_t)msg[blk+i*4]|((uint32_t)msg[blk+i*4+1]<<8)|((uint32_t)msg[blk+i*4+2]<<16)|((uint32_t)msg[blk+i*4+3]<<24);
        uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,aa=h0,bb=h1,cc=h2,dd=h3,ee=h4;
        RMD_FF(a,b,c,d,e,X[0],11);RMD_FF(e,a,b,c,d,X[1],14);RMD_FF(d,e,a,b,c,X[2],15);RMD_FF(c,d,e,a,b,X[3],12);
        RMD_FF(b,c,d,e,a,X[4],5);RMD_FF(a,b,c,d,e,X[5],8);RMD_FF(e,a,b,c,d,X[6],7);RMD_FF(d,e,a,b,c,X[7],9);
        RMD_FF(c,d,e,a,b,X[8],11);RMD_FF(b,c,d,e,a,X[9],13);RMD_FF(a,b,c,d,e,X[10],14);RMD_FF(e,a,b,c,d,X[11],15);
        RMD_FF(d,e,a,b,c,X[12],6);RMD_FF(c,d,e,a,b,X[13],7);RMD_FF(b,c,d,e,a,X[14],9);RMD_FF(a,b,c,d,e,X[15],8);
        RMD_GG(e,a,b,c,d,X[7],7);RMD_GG(d,e,a,b,c,X[4],6);RMD_GG(c,d,e,a,b,X[13],8);RMD_GG(b,c,d,e,a,X[1],13);
        RMD_GG(a,b,c,d,e,X[10],11);RMD_GG(e,a,b,c,d,X[6],9);RMD_GG(d,e,a,b,c,X[15],7);RMD_GG(c,d,e,a,b,X[3],15);
        RMD_GG(b,c,d,e,a,X[12],7);RMD_GG(a,b,c,d,e,X[0],12);RMD_GG(e,a,b,c,d,X[9],15);RMD_GG(d,e,a,b,c,X[5],9);
        RMD_GG(c,d,e,a,b,X[2],11);RMD_GG(b,c,d,e,a,X[14],7);RMD_GG(a,b,c,d,e,X[11],13);RMD_GG(e,a,b,c,d,X[8],12);
        RMD_HH(d,e,a,b,c,X[3],11);RMD_HH(c,d,e,a,b,X[10],13);RMD_HH(b,c,d,e,a,X[14],6);RMD_HH(a,b,c,d,e,X[4],7);
        RMD_HH(e,a,b,c,d,X[9],14);RMD_HH(d,e,a,b,c,X[15],9);RMD_HH(c,d,e,a,b,X[8],13);RMD_HH(b,c,d,e,a,X[1],15);
        RMD_HH(a,b,c,d,e,X[2],14);RMD_HH(e,a,b,c,d,X[7],8);RMD_HH(d,e,a,b,c,X[0],13);RMD_HH(c,d,e,a,b,X[6],6);
        RMD_HH(b,c,d,e,a,X[13],5);RMD_HH(a,b,c,d,e,X[11],12);RMD_HH(e,a,b,c,d,X[5],7);RMD_HH(d,e,a,b,c,X[12],5);
        RMD_II(c,d,e,a,b,X[1],11);RMD_II(b,c,d,e,a,X[9],12);RMD_II(a,b,c,d,e,X[11],14);RMD_II(e,a,b,c,d,X[10],15);
        RMD_II(d,e,a,b,c,X[0],14);RMD_II(c,d,e,a,b,X[8],15);RMD_II(b,c,d,e,a,X[12],9);RMD_II(a,b,c,d,e,X[4],8);
        RMD_II(e,a,b,c,d,X[13],9);RMD_II(d,e,a,b,c,X[3],14);RMD_II(c,d,e,a,b,X[7],5);RMD_II(b,c,d,e,a,X[15],6);
        RMD_II(a,b,c,d,e,X[14],8);RMD_II(e,a,b,c,d,X[5],6);RMD_II(d,e,a,b,c,X[6],5);RMD_II(c,d,e,a,b,X[2],12);
        RMD_JJ(b,c,d,e,a,X[4],9);RMD_JJ(a,b,c,d,e,X[0],15);RMD_JJ(e,a,b,c,d,X[5],5);RMD_JJ(d,e,a,b,c,X[9],11);
        RMD_JJ(c,d,e,a,b,X[7],6);RMD_JJ(b,c,d,e,a,X[12],8);RMD_JJ(a,b,c,d,e,X[2],13);RMD_JJ(e,a,b,c,d,X[10],12);
        RMD_JJ(d,e,a,b,c,X[14],5);RMD_JJ(c,d,e,a,b,X[1],12);RMD_JJ(b,c,d,e,a,X[3],13);RMD_JJ(a,b,c,d,e,X[8],14);
        RMD_JJ(e,a,b,c,d,X[11],11);RMD_JJ(d,e,a,b,c,X[6],8);RMD_JJ(c,d,e,a,b,X[15],5);RMD_JJ(b,c,d,e,a,X[13],6);
        RMD_JJJ(aa,bb,cc,dd,ee,X[5],8);RMD_JJJ(ee,aa,bb,cc,dd,X[14],9);RMD_JJJ(dd,ee,aa,bb,cc,X[7],9);RMD_JJJ(cc,dd,ee,aa,bb,X[0],11);
        RMD_JJJ(bb,cc,dd,ee,aa,X[9],13);RMD_JJJ(aa,bb,cc,dd,ee,X[2],15);RMD_JJJ(ee,aa,bb,cc,dd,X[11],15);RMD_JJJ(dd,ee,aa,bb,cc,X[4],5);
        RMD_JJJ(cc,dd,ee,aa,bb,X[13],7);RMD_JJJ(bb,cc,dd,ee,aa,X[6],7);RMD_JJJ(aa,bb,cc,dd,ee,X[15],8);RMD_JJJ(ee,aa,bb,cc,dd,X[8],11);
        RMD_JJJ(dd,ee,aa,bb,cc,X[1],14);RMD_JJJ(cc,dd,ee,aa,bb,X[10],14);RMD_JJJ(bb,cc,dd,ee,aa,X[3],12);RMD_JJJ(aa,bb,cc,dd,ee,X[12],6);
        RMD_III(ee,aa,bb,cc,dd,X[6],9);RMD_III(dd,ee,aa,bb,cc,X[11],13);RMD_III(cc,dd,ee,aa,bb,X[3],15);RMD_III(bb,cc,dd,ee,aa,X[7],7);
        RMD_III(aa,bb,cc,dd,ee,X[0],12);RMD_III(ee,aa,bb,cc,dd,X[13],8);RMD_III(dd,ee,aa,bb,cc,X[5],9);RMD_III(cc,dd,ee,aa,bb,X[10],11);
        RMD_III(bb,cc,dd,ee,aa,X[14],7);RMD_III(aa,bb,cc,dd,ee,X[15],7);RMD_III(ee,aa,bb,cc,dd,X[8],12);RMD_III(dd,ee,aa,bb,cc,X[12],7);
        RMD_III(cc,dd,ee,aa,bb,X[4],6);RMD_III(bb,cc,dd,ee,aa,X[9],15);RMD_III(aa,bb,cc,dd,ee,X[1],13);RMD_III(ee,aa,bb,cc,dd,X[2],11);
        RMD_HHH(dd,ee,aa,bb,cc,X[15],9);RMD_HHH(cc,dd,ee,aa,bb,X[5],7);RMD_HHH(bb,cc,dd,ee,aa,X[1],15);RMD_HHH(aa,bb,cc,dd,ee,X[3],11);
        RMD_HHH(ee,aa,bb,cc,dd,X[7],8);RMD_HHH(dd,ee,aa,bb,cc,X[14],6);RMD_HHH(cc,dd,ee,aa,bb,X[6],6);RMD_HHH(bb,cc,dd,ee,aa,X[9],14);
        RMD_HHH(aa,bb,cc,dd,ee,X[11],12);RMD_HHH(ee,aa,bb,cc,dd,X[8],13);RMD_HHH(dd,ee,aa,bb,cc,X[12],5);RMD_HHH(cc,dd,ee,aa,bb,X[2],14);
        RMD_HHH(bb,cc,dd,ee,aa,X[10],13);RMD_HHH(aa,bb,cc,dd,ee,X[0],13);RMD_HHH(ee,aa,bb,cc,dd,X[4],7);RMD_HHH(dd,ee,aa,bb,cc,X[13],5);
        RMD_GGG(cc,dd,ee,aa,bb,X[8],15);RMD_GGG(bb,cc,dd,ee,aa,X[6],5);RMD_GGG(aa,bb,cc,dd,ee,X[4],8);RMD_GGG(ee,aa,bb,cc,dd,X[1],11);
        RMD_GGG(dd,ee,aa,bb,cc,X[3],14);RMD_GGG(cc,dd,ee,aa,bb,X[11],14);RMD_GGG(bb,cc,dd,ee,aa,X[15],6);RMD_GGG(aa,bb,cc,dd,ee,X[0],14);
        RMD_GGG(ee,aa,bb,cc,dd,X[5],6);RMD_GGG(dd,ee,aa,bb,cc,X[12],9);RMD_GGG(cc,dd,ee,aa,bb,X[2],12);RMD_GGG(bb,cc,dd,ee,aa,X[13],9);
        RMD_GGG(aa,bb,cc,dd,ee,X[9],12);RMD_GGG(ee,aa,bb,cc,dd,X[7],5);RMD_GGG(dd,ee,aa,bb,cc,X[10],15);RMD_GGG(cc,dd,ee,aa,bb,X[14],8);
        RMD_FFF(bb,cc,dd,ee,aa,X[12],8);RMD_FFF(aa,bb,cc,dd,ee,X[15],5);RMD_FFF(ee,aa,bb,cc,dd,X[10],12);RMD_FFF(dd,ee,aa,bb,cc,X[4],9);
        RMD_FFF(cc,dd,ee,aa,bb,X[1],12);RMD_FFF(bb,cc,dd,ee,aa,X[5],5);RMD_FFF(aa,bb,cc,dd,ee,X[8],14);RMD_FFF(ee,aa,bb,cc,dd,X[7],6);
        RMD_FFF(dd,ee,aa,bb,cc,X[6],8);RMD_FFF(cc,dd,ee,aa,bb,X[2],13);RMD_FFF(bb,cc,dd,ee,aa,X[13],6);RMD_FFF(aa,bb,cc,dd,ee,X[14],5);
        RMD_FFF(ee,aa,bb,cc,dd,X[0],15);RMD_FFF(dd,ee,aa,bb,cc,X[3],13);RMD_FFF(cc,dd,ee,aa,bb,X[9],11);RMD_FFF(bb,cc,dd,ee,aa,X[11],11);
        uint32_t t=h1+c+dd;h1=h2+d+ee;h2=h3+e+aa;h3=h4+a+bb;h4=h0+b+cc;h0=t;
    }
    auto rle=[](uint32_t v,uint8_t* p){p[0]=v&0xFF;p[1]=(v>>8)&0xFF;p[2]=(v>>16)&0xFF;p[3]=(v>>24)&0xFF;};
    rle(h0,out);rle(h1,out+4);rle(h2,out+8);rle(h3,out+12);rle(h4,out+16);
    #undef RMD_F
    #undef RMD_G
    #undef RMD_H
    #undef RMD_I
    #undef RMD_J
    #undef ROL32
    #undef RMD_FF
    #undef RMD_GG
    #undef RMD_HH
    #undef RMD_II
    #undef RMD_JJ
    #undef RMD_FFF
    #undef RMD_GGG
    #undef RMD_HHH
    #undef RMD_III
    #undef RMD_JJJ
}
// hash160 = RIPEMD160(SHA256(data))
static void hash160_cpu(const uint8_t* data, size_t len, uint8_t h160[20]) {
    uint8_t sha[32];
    sha256_cpu(data, len, sha);
    ripemd160_cpu(sha, 32, h160);
}

// Keccak-256 (for ETH) — pure C implementation
static void keccak256_cpu(const uint8_t* data, size_t len, uint8_t out[32]) {
    // Keccak-256 pure C implementation
    static const uint64_t RC[24] = {
        0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808AULL,0x8000000080008000ULL,
        0x000000000000808BULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
        0x000000000000008AULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000AULL,
        0x000000008000808BULL,0x800000000000008BULL,0x8000000000008089ULL,0x8000000000008003ULL,
        0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800AULL,0x800000008000000AULL,
        0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL
    };
    static const int ROT[25] = {0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};
    static const int PI[25]  = {0,10,20,5,15,16,1,11,21,6,7,17,2,12,22,23,8,18,3,13,14,24,9,19,4};
    auto rotl = [](uint64_t x, int n){ return (x<<n)|(x>>(64-n)); };
    std::vector<uint8_t> msg(data, data+len);
    msg.push_back(0x01);
    while(msg.size()%136) msg.push_back(0x00);
    msg.back() |= 0x80;
    uint64_t st[25]={};
    for(size_t bs=0;bs<msg.size();bs+=136){
        for(int i=0;i<17;i++){
            uint64_t w=0;
            for(int b=0;b<8;b++) w|=(uint64_t)msg[bs+i*8+b]<<(b*8);
            st[i]^=w;
        }
        for(int r=0;r<24;r++){
            uint64_t C[5],D[5];
            for(int x=0;x<5;x++) C[x]=st[x]^st[x+5]^st[x+10]^st[x+15]^st[x+20];
            for(int x=0;x<5;x++) D[x]=C[(x+4)%5]^rotl(C[(x+1)%5],1);
            for(int i=0;i<25;i++) st[i]^=D[i%5];
            uint64_t B[25];
            for(int i=0;i<25;i++) B[PI[i]]=rotl(st[i],ROT[i]);
            for(int y=0;y<5;y++) for(int x=0;x<5;x++) st[x+5*y]=B[x+5*y]^((~B[(x+1)%5+5*y])&B[(x+2)%5+5*y]);
            st[0]^=RC[r];
        }
    }
    for(int i=0;i<4;i++) for(int b=0;b<8;b++) out[i*8+b]=(st[i]>>(b*8))&0xFF;
}

// Base58Check encode
static const char B58C[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static std::string base58check_enc(const uint8_t* payload, size_t plen) {
    uint8_t chk[32]; sha256_cpu(payload, plen, chk);
    uint8_t chk2[32]; sha256_cpu(chk, 32, chk2);
    std::vector<uint8_t> full(payload, payload+plen);
    full.insert(full.end(), chk2, chk2+4);
    // Convert to base58
    std::vector<uint8_t> digits;
    for (auto b : full) {
        int carry = b;
        for (auto& d : digits) { carry += 256*d; d = carry%58; carry /= 58; }
        while (carry) { digits.push_back(carry%58); carry /= 58; }
    }
    std::string res;
    for (auto b : full) { if(b==0) res+='1'; else break; }
    for (int i=(int)digits.size()-1;i>=0;i--) res+=B58C[digits[i]];
    return res;
}

// Bech32 encode (P2WPKH)
static std::string bech32_enc_addr(const uint8_t h160[20]) {
    static const char CHARSET[] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    auto polymod = [](const std::vector<int>& v) -> uint32_t {
        static const uint32_t G[]={0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3};
        uint32_t chk=1;
        for(int val:v){int b=chk>>25;chk=((chk&0x1ffffff)<<5)^val;for(int i=0;i<5;i++)if((b>>i)&1)chk^=G[i];}
        return chk;
    };
    // Convert 8-bit to 5-bit groups
    std::vector<int> d5; d5.push_back(0); // witness version
    int acc=0,bits=0;
    for(int i=0;i<20;i++){acc=(acc<<8)|h160[i];bits+=8;while(bits>=5){bits-=5;d5.push_back((acc>>bits)&31);}}
    if(bits) d5.push_back((acc<<(5-bits))&31);
    // Expand HRP
    std::string hrp="bc";
    std::vector<int> enc;
    for(char c:hrp) enc.push_back(c>>5);
    enc.push_back(0);
    for(char c:hrp) enc.push_back(c&31);
    auto data=d5;
    for(int i=0;i<6;i++) data.push_back(0);
    for(auto x:data) enc.push_back(x);
    uint32_t pm=(polymod(enc)^1);
    for(int i=0;i<6;i++) d5.push_back((pm>>5*(5-i))&31);
    std::string res="bc1";
    for(int x:d5) res+=CHARSET[x];
    return res;
}

// Private key (32 bytes BE) → BTC legacy, BTC segwit, ETH
static void key_to_addresses(const uint8_t key[32],
    std::string& btc_legacy, std::string& btc_segwit, std::string& eth) {
    // 1. ECC : key → public point
    uint64_t px[4], py[4];
    if (!ec_mul_G(key, px, py)) { btc_legacy=btc_segwit=eth="ERROR"; return; }

    // 2. Compressed pubkey (33 bytes)
    uint8_t pub33[33];
    pub33[0] = (py[0]&1) ? 0x03 : 0x02;
    for (int i=0;i<4;i++) for(int b=0;b<8;b++) pub33[1+(3-i)*8+(7-b)]=(px[i]>>(b*8))&0xFF;

    // 3. hash160(pub33)
    uint8_t h160[20];
    hash160_cpu(pub33, 33, h160);

    // 4. BTC Legacy (P2PKH)
    uint8_t pl[21]; pl[0]=0x00; memcpy(pl+1, h160, 20);
    btc_legacy = base58check_enc(pl, 21);

    // 5. BTC SegWit (P2WPKH bech32)
    btc_segwit = bech32_enc_addr(h160);

    // 6. ETH : keccak256(pub64)[12:]
    uint8_t pub64[64];
    for (int i=0;i<4;i++) for(int b=0;b<8;b++){
        pub64[(3-i)*8+(7-b)] = (px[i]>>(b*8))&0xFF;
        pub64[32+(3-i)*8+(7-b)] = (py[i]>>(b*8))&0xFF;
    }
    uint8_t kh[32];
    keccak256_cpu(pub64, 64, kh);
    char ethbuf[43]; ethbuf[0]='0'; ethbuf[1]='x';
    for(int i=0;i<20;i++) snprintf(ethbuf+2+i*2, 3, "%02x", kh[12+i]);
    eth = std::string(ethbuf, 42);
}

// =================================================================================
// 4. KEY RECONSTRUCTION FROM GRAY INDEX

static void print_key(const uint8_t key[32]) {
    for (int i=0; i<32; i++) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)key[i];
    std::cout << std::dec << "\n";
}

// =================================================================================
// 5. MAIN GPU LOOP (HEX MODE)
// =================================================================================


// =================================================================================
// BLOOM FILTER — Load into VRAM
// =================================================================================
static const char* BLOOM_FILTER_FILE = "bloom.bin";

static bool load_bloom_to_target(TargetData& target) {
    std::ifstream f(BLOOM_FILTER_FILE, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open '" << BLOOM_FILTER_FILE << "'\n";
        std::cerr << "  → Create filter with : python3 create_filter_V3.py <addresses.txt> bloom.bin\n";
        return false;
    }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(bytes);
    f.read((char*)buf.data(), bytes);
    if (!f) { std::cerr << "Error reading bloom.bin\n"; return false; }

    uint64_t* d_bloom = nullptr;
    cudaMalloc(&d_bloom, bytes);
    cudaMemcpy(d_bloom, buf.data(), bytes, cudaMemcpyHostToDevice);

    // Do not overwrite target.type if already set (BLOOM_BTC / BLOOM_ETH)
    if(target.type != TargetType::BLOOM_BTC && target.type != TargetType::BLOOM_ETH)
        target.type = TargetType::BLOOM;
    target.d_bloom_filter = d_bloom;
    target.bloom_m_bits   = (uint64_t)bytes * 8ULL;

    std::cout << "Bloom filter : " << (bytes / (1024*1024)) << " MB loaded into VRAM ("
              << target.bloom_m_bits << " bits)\n";
    return true;
}

// Returns the bloom TargetType if the arg is a bloom keyword, else BTC (sentinel)
static TargetType get_bloom_type(const std::string& s) {
    std::string l = s;
    for (auto& c : l) c = tolower(c);
    if (l == "bloombtc") return TargetType::BLOOM_BTC;
    if (l == "bloometh") return TargetType::BLOOM_ETH;
    if (l == "bloom")    return TargetType::BLOOM;
    return TargetType::BTC; // not a bloom arg
}

static bool is_bloom_arg(const std::string& s) {
    std::string l = s;
    for (auto& c : l) c = tolower(c);
    return l == "bloom" || l == "bloombtc" || l == "bloometh";
}

static int run_hex_mode(const std::string &mask_str, const std::string &addr_str) {

    // --- Parse mask ---
    MaskParseResult mask_r = parseMask(mask_str);
    if (!mask_r.valid) return 1;

    // --- Decode target address ---
    TargetData target = {};
    if (is_bloom_arg(addr_str)) {
        target.type = get_bloom_type(addr_str);
        if (!load_bloom_to_target(target)) return 1;
        std::string mode_name = (target.type==TargetType::BLOOM_BTC) ? "HEX + Bloom BTC"
                              : (target.type==TargetType::BLOOM_ETH) ? "HEX + Bloom ETH"
                              : "HEX + Bloom";
        std::cout << "Mode : " << mode_name << "\n";
    } else if (addr_str.size() >= 2 && addr_str[0]=='0' && (addr_str[1]=='x'||addr_str[1]=='X')) {
        target.type = TargetType::ETH;
        if (!ethAddrToBytes(addr_str, target.hash20)) {
            std::cerr << "Error: invalid ETH address.\n"; return 1;
        }
        std::cout << "Mode : ETH\n";
    } else {
        target.type = TargetType::BTC;
        if (!addrToHash160Any(addr_str, target.hash20)) {
            std::cerr << "Error: invalid BTC address.\n"; return 1;
        }
        std::cout << "Mode : BTC\n";
    }

    // --- CPU ECC precomputation ---
    HydraData h_fd;
    if (!precompute_ecc(mask_r, h_fd)) return 1;

    // --- GPU init ---
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // --- Allocate GPU buffers ---
    HydraData   *d_fd     = nullptr;
    TargetData  *d_target = nullptr;
    HydraResult *d_result = nullptr;

    cudaMalloc(&d_fd,     sizeof(HydraData));
    cudaMalloc(&d_target, sizeof(TargetData));
    cudaMalloc(&d_result, sizeof(HydraResult));

    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);

    HydraResult h_result = {0, 0, 0};
    cudaMemcpy(d_result, &h_result, sizeof(HydraResult), cudaMemcpyHostToDevice);

    // In V4, each thread handles ONE P_base → LOW_SIZE candidates
    // wave_size = number of P_base per kernel launch
    const int threads   = 256;
    const int blocks    = prop.multiProcessorCount * 128;
    const int wave_size = (int)std::min(
        (uint64_t)(blocks * threads),
        h_fd.high_candidates);

    std::cout << "======== HYDRA V4.0 (AFFINE DICT) =================\n";
    std::cout << "GPU         : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "Blocks      : " << blocks << " x " << threads << " threads\n";
    std::cout << "Dict size   : " << LOW_SIZE << " (2^" << LOW_BITS << " bits bas)\n";
    std::cout << "Candidates  : " << h_fd.total_candidates << " (2^" << (int)h_fd.num_var_bits << ")\n";
    std::cout << "P_base pool : " << h_fd.high_candidates << " (2^" << (int)h_fd.num_high_bits << " bits hauts)\n";
    signal(SIGINT, handle_sigint);

    auto t0     = std::chrono::high_resolution_clock::now();
    auto t_last = t0;
    uint64_t offset = 0;       // en unités de P_base (index dans bits hauts)
    int found = 0;
    double keys_since_last = 0;

    while (!g_sigint && found == 0 && offset < h_fd.high_candidates) {
        uint64_t remaining = h_fd.high_candidates - offset;
        int cur_wave = (int)std::min((uint64_t)wave_size, remaining);

        h_fd.gray_offset_start = offset;
        cudaMemcpy(d_fd, &h_fd, sizeof(HydraData), cudaMemcpyHostToDevice);

        hydra_mega_kernel<<<blocks, threads>>>(d_fd, d_target, d_result, cur_wave);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";
            break;
        }

        cudaMemcpy(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost);
        offset += cur_wave;
        // Count actual candidates tested (P_base × LOW_SIZE)
        keys_since_last += (double)cur_wave * LOW_SIZE;

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - t_last).count();
        if (dt >= 1.0) {
            double speed   = keys_since_last / dt / 1e6;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            // progress in terms of actual candidates tested
            double prog    = 100.0 * (double)(offset * LOW_SIZE) / (double)h_fd.total_candidates;
            double total_keys_done = (double)offset * LOW_SIZE;
            double eta = (elapsed > 0 && total_keys_done > 0)
                ? ((double)h_fd.total_candidates - total_keys_done) / (total_keys_done / elapsed) : 0;
            int eh=(int)(eta/3600), em=(int)((eta-eh*3600)/60), es=(int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(2) << speed << " MK/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2) << eh << ":" << std::setw(2) << em << ":" << std::setw(2) << es
                      << std::flush;
            t_last = now;
            keys_since_last = 0;
        }

        // Bloom hit detected inside loop → handle here so we can continue
        if (found && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) {
            cudaMemcpy(&h_result, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

            uint64_t high_idx = h_result.index / (uint64_t)LOW_SIZE;
            uint64_t low_k    = h_result.index % (uint64_t)LOW_SIZE;
            uint8_t key[32];
            memcpy(key, mask_r.k_fixed, 32);
            uint64_t gray_high = high_idx ^ (high_idx >> 1);
            int high_bits = (int)h_fd.num_high_bits;
            int low_bits  = (int)h_fd.num_var_bits - high_bits;
            for (int i = 0; i < high_bits; i++) {
                if ((gray_high >> i) & 1ULL) {
                    int bit_pos  = mask_r.var_bit_positions[low_bits + i];
                    int byte_idx = 31 - (bit_pos / 8);
                    int bit_off  = bit_pos % 8;
                    key[byte_idx] |= (uint8_t)(1 << bit_off);
                }
            }
            for (int j = 0; j < low_bits; j++) {
                if ((low_k >> j) & 1ULL) {
                    int bit_pos  = mask_r.var_bit_positions[j];
                    int byte_idx = 31 - (bit_pos / 8);
                    int bit_off  = bit_pos % 8;
                    key[byte_idx] |= (uint8_t)(1 << bit_off);
                }
            }
            std::string addr_legacy, addr_segwit, addr_eth;
            key_to_addresses(key, addr_legacy, addr_segwit, addr_eth);

            bool victory = check_balances_and_notify(key, addr_legacy, addr_segwit, addr_eth);
            if (victory) {
                // Non-zero balance → keep found=1, exit loop → VICTORY displayed below
            } else {
                // False positive → reset and continue
                int zero = 0;
                cudaMemcpy(&d_result->found, &zero, sizeof(int), cudaMemcpyHostToDevice);
                found = 0;
            }
        }
    }
    std::cout << "\n";

    if (found) {
        cudaMemcpy(&h_result, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

        // In V4, result->index = high_idx * LOW_SIZE + low_k
        // Reconstruct : high bits = high_idx (Gray), low bits = low_k (direct binary)
        uint64_t high_idx = h_result.index / (uint64_t)LOW_SIZE;
        uint64_t low_k    = h_result.index % (uint64_t)LOW_SIZE;

        uint8_t key[32];
        memcpy(key, mask_r.k_fixed, 32);

        // High bits : decode Gray → set in key
        uint64_t gray_high = high_idx ^ (high_idx >> 1);
        int high_bits = (int)h_fd.num_high_bits;
        int low_bits  = (int)h_fd.num_var_bits - high_bits;
        for (int i = 0; i < high_bits; i++) {
            if ((gray_high >> i) & 1ULL) {
                int bit_pos  = mask_r.var_bit_positions[low_bits + i];
                int byte_idx = 31 - (bit_pos / 8);
                int bit_off  = bit_pos % 8;
                key[byte_idx] |= (uint8_t)(1 << bit_off);
            }
        }
        // Low bits : low_k is direct binary (not Gray)
        for (int j = 0; j < low_bits; j++) {
            if ((low_k >> j) & 1ULL) {
                int bit_pos  = mask_r.var_bit_positions[j];
                int byte_idx = 31 - (bit_pos / 8);
                int bit_off  = bit_pos % 8;
                key[byte_idx] |= (uint8_t)(1 << bit_off);
            }
        }

        // Compute and display addresses
        std::string addr_legacy, addr_segwit, addr_eth;
        key_to_addresses(key, addr_legacy, addr_segwit, addr_eth);

        // Bloom mode : already handled in loop (API check + reset on false positive)
        // We only reach here if victory=true (balance > 0) or direct address mode
        std::cout << "\n======== VICTORY ! KEY FOUND ==========================\n";
		std::cout << "Private key : "; print_key(key);
		if (target.type == TargetType::ETH)
			std::cout << "  ETH         : " << addr_eth << "\n";
		else if (target.type == TargetType::BLOOM_ETH)
			std::cout << "  ETH         : " << addr_eth << "\n";
		else if (!addr_str.empty() && addr_str.substr(0,3) == "bc1")
			std::cout << "  BTC segwit  : " << addr_segwit << "\n";
		else {
			std::cout << "  BTC legacy  : " << addr_legacy << "\n";
			std::cout << "  BTC segwit  : " << addr_segwit << "\n";
		}
        std::cout << "=======================================================\n";
        {
            std::ostringstream pk_ss;
            pk_ss << std::hex << std::setfill('0');
            for (int i = 0; i < 32; ++i) pk_ss << std::setw(2) << (int)key[i];
            std::string key_info = "*Private Key:*\n`" + pk_ss.str() + "`";
            // Relevant address = the one we were searching for
            std::string addr_info;
            if (target.type == TargetType::ETH)
                addr_info = "*ETH:* `" + addr_eth + "`";
            else if (!addr_str.empty() && addr_str.substr(0,3) == "bc1")
                addr_info = "*BTC segwit:* `" + addr_segwit + "`";
            else
                addr_info = "*BTC legacy:* `" + addr_legacy + "`";
            notify_victory("KEY FOUND", key_info, addr_info);
        }
    } else if (!g_sigint) {
        std::cout << "Not found in " << h_fd.total_candidates << " candidates.\n";
    }
    print_search_summary(found != 0);
    if(target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_fd); cudaFree(d_target); cudaFree(d_result);
    return found ? 0 : 2;
}

// =================================================================================
// 6. MODE DETECTION AND MAIN
// =================================================================================

// Detect if the argument looks like a hex mask (64 hex+# chars)
static bool looks_like_hex_mask(const std::string &s) {
    std::string t = s;
    if (t.size() >= 2 && t[0]=='0' && (t[1]=='x'||t[1]=='X')) t=t.substr(2);
    if (t.size() != 64) return false;
    for (char c : t) {
        if (!isxdigit(c) && c!='#') return false;
    }
    return true;
}

// Detect if this is a BIP39 phrase (contains spaces)
static bool looks_like_seed(const std::string &s) {
    return s.find(' ') != std::string::npos;
}

// BIP39 phrase with no '#' → all words known → passphrase mode
static bool is_passphrase_mode(const std::string &s) {
    return looks_like_seed(s) && s.find('#') == std::string::npos;
}

static bool looks_like_wif_mask(const std::string &s) {
    // Compressed WIF : 52 chars, Base58 alphabet + '#' for unknowns
    // must start with K, L or #
    if(s.size() != 52) return false;
    static const char* valid = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz#";
    for(char c : s) if(!strchr(valid, c)) return false;
    // doit commencer par K, L ou $ (WIF compressé)
    return (s[0] == 'K' || s[0] == 'L' || s[0] == '#');
}


// =================================================================================
// 7. BIP39 PHRASE PARSER WITH UNKNOWN POSITIONS
// =================================================================================

// Lookup table : word → BIP39 index  (built at runtime from BIP39_Dict.h)
static std::unordered_map<std::string, uint16_t> build_word_map() {
    std::unordered_map<std::string, uint16_t> m;
    for(int i=0;i<2048;i++){
        uint16_t off = h_BIP39_OFFS[i];
        uint8_t  len = h_BIP39_LENS[i];
        m[std::string((char*)h_BIP39_BLOB+off, len)] = (uint16_t)i;
    }
    return m;
}

static bool parse_seed_mask(const std::string &phrase,
                            SeedMask &mask,
                            const std::unordered_map<std::string,uint16_t> &wmap)
{
    // Tokenize on whitespace
    std::vector<std::string> tokens;
    std::istringstream ss(phrase);
    std::string tok;
    while(ss >> tok) tokens.push_back(tok);

    int n = (int)tokens.size();
    if(n!=12 && n!=15 && n!=18 && n!=21 && n!=24){
        std::cerr << "Error: phrase must have 12/15/18/21/24 words (got " << n << ")\n";
        return false;
    }
    mask.num_words   = (uint8_t)n;
    mask.num_unknown = 0;
    // checksum_bits : ENT/32 = (n*11/33)*8/32 = n*11/132 bits → 4 for 12 words, 8 for 24, etc.
    mask.checksum_bits = (uint8_t)((n * 11) / 33);  // 12→4, 15→5, 18→6, 21→7, 24→8

    for(int i=0;i<n;i++){
        if(tokens[i]=="#"){
            if(mask.num_unknown >= SEED_MAX_X){
                std::cerr << "Error: too many unknown positions (max " << SEED_MAX_X << ")\n";
                return false;
            }
            mask.unknown_pos[mask.num_unknown++] = (uint8_t)i;
            mask.word_indices[i] = 0xFFFF;
        } else {
            auto it = wmap.find(tokens[i]);
            if(it==wmap.end()){
                std::cerr << "Error: word not found in BIP39 wordlist: \"" << tokens[i] << "\"\n";
                return false;
            }
            mask.word_indices[i] = it->second;
        }
    }

    // Compute total_candidates and optimized checksum fields
    const uint8_t cs_bits = mask.checksum_bits;
    const uint8_t cs_mask_val = (uint8_t)((1u << cs_bits) - 1u);

    mask.last_word_unknown = (mask.num_unknown > 0 &&
        mask.unknown_pos[mask.num_unknown-1] == (uint8_t)(n-1));

    if(mask.num_unknown == 0){
        mask.total_candidates  = 1;
        mask.required_checksum = 0xFF;
    } else if(mask.last_word_unknown){
        // Last word unknown : iterate over 2^(11-cs_bits) entropy values,
        // checksum will be forced by K1 → always valid
        mask.total_candidates = 1;
        for(int i=0; i<mask.num_unknown-1; i++) mask.total_candidates *= 2048ULL;
        mask.total_candidates *= (1ULL << (11 - cs_bits));
        mask.required_checksum = 0xFF;
    } else {
        // Last word known → fixed checksum, K1 filters ~1/2^cs_bits
        mask.total_candidates = 1;
        for(int i=0; i<mask.num_unknown; i++) mask.total_candidates *= 2048ULL;
        mask.required_checksum = (uint8_t)(mask.word_indices[n-1] & cs_mask_val);
    }
    return true;
}

// =================================================================================
// 8. RUN SEED MODE
// =================================================================================
static int run_seed_mode(const std::string &phrase, const std::string &addr_str) {

    auto wmap = build_word_map();

    // Parse target address
    TargetData target = {};
    if(is_bloom_arg(addr_str)){
        target.type = get_bloom_type(addr_str);
        if(!load_bloom_to_target(target)) return 1;
        std::string _bmode=(target.type==TargetType::BLOOM_BTC)?"SEED + Bloom BTC":(target.type==TargetType::BLOOM_ETH)?"SEED + Bloom ETH":"SEED + Bloom";
        std::cout << "Mode : " << _bmode << "\n";
    } else if(addr_str.size()>=2 && addr_str[0]=='0' && (addr_str[1]=='x'||addr_str[1]=='X')){
        target.type = TargetType::ETH;
        if(!ethAddrToBytes(addr_str, target.hash20)){
            std::cerr << "Error: invalid ETH address.\n"; return 1;
        }
        std::cout << "Mode : ETH / BIP44 m/44'/60'/0'/0/0\n";
    } else {
        target.type = TargetType::BTC;
        if(!addrToHash160Any(addr_str, target.hash20)){
            std::cerr << "Error: invalid BTC address.\n"; return 1;
        }
        std::cout << "Mode : BTC / BIP44 m/44'/0'/0'/0/0\n";
    }

    // Parse mask
    SeedMask mask = {};
    if(!parse_seed_mask(phrase, mask, wmap)) return 1;

    // Display
    {
        const uint32_t cs_div = (1u << mask.checksum_bits);
        uint64_t effective = mask.last_word_unknown
            ? mask.total_candidates
            : mask.total_candidates / cs_div;
        std::cout << "Words       : " << (int)mask.num_words << "\n";
        std::cout << "Unknown pos : " << (int)mask.num_unknown << "\n";
        std::cout << "Candidates  : " << mask.total_candidates
                  << " raw → ~" << effective
                  << " valid (BIP39 checksum ÷" << cs_div << ")\n";
    }

    // Upload BIP39 dictionary to constant memory
    cudaMemcpyToSymbol(d_BIP39_BLOB, h_BIP39_BLOB, sizeof(h_BIP39_BLOB));
    cudaMemcpyToSymbol(d_BIP39_OFFS, h_BIP39_OFFS, sizeof(h_BIP39_OFFS));
    cudaMemcpyToSymbol(d_BIP39_LENS, h_BIP39_LENS, sizeof(h_BIP39_LENS));

    int device=0; cudaSetDevice(device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);

    SeedMask    *d_mask   = nullptr;
    TargetData  *d_target = nullptr;
    HydraResult *d_result = nullptr;
    cudaMalloc(&d_mask,   sizeof(SeedMask));
    cudaMalloc(&d_target, sizeof(TargetData));
    cudaMalloc(&d_result, sizeof(HydraResult));
    cudaMemcpy(d_mask,   &mask,   sizeof(SeedMask),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);
    HydraResult h_res={0,0,0};
    cudaMemcpy(d_result, &h_res, sizeof(HydraResult), cudaMemcpyHostToDevice);

    const int THREADS   = 256;
    // Large wave size : K1 filters 15/16, we want ~250K valid for K2
    // 250K * 16 = 4M raw → GPU stays busy between waves
    const int wave_k1   = 4 * 1024 * 1024;  // 4M raw candidates per wave
    const int blocks_k1 = (wave_k1 + THREADS - 1) / THREADS;
    const int blocks_k2 = prop.multiProcessorCount * 64;

    uint64_t *d_valid_indices = nullptr;
    int      *d_valid_count   = nullptr;
    uint8_t  *d_seeds     = nullptr;  // K2a → K2b : seeds [wave_k2 × 64]
    uint8_t  *d_intermed  = nullptr;  // K2b → K2c : priv||chain after m/44'/coin'/0' [×64]
    const int wave_k2 = wave_k1;
    cudaMalloc(&d_valid_indices, (size_t)wave_k1 * sizeof(uint64_t));
    cudaMalloc(&d_valid_count,   sizeof(int));
    cudaMalloc(&d_seeds,         (size_t)wave_k2 * 64);
    cudaMalloc(&d_intermed,      (size_t)wave_k2 * 64);

    std::cout << "======== HYDRA V4.0 (SEED MODE) ====================\n";
    std::cout << "GPU         : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "K1 checksum : " << blocks_k1 << " blocks x " << THREADS << " (40 reg, wave=" << wave_k1 << ")\n";
    std::cout << "K2a PBKDF2  : " << blocks_k2 << " blocks x " << THREADS << " (~106 reg)\n";
    std::cout << "K2b hardened: " << blocks_k2 << " blocks x " << THREADS << " (0 ECC, 0 spill)\n";
    std::cout << "K2c ECC     : " << blocks_k2 << " blocks x " << THREADS << " (~166 reg)\n";
    signal(SIGINT, handle_sigint);

    auto t0 = std::chrono::high_resolution_clock::now(), t_last = t0;
    uint64_t offset        = 0;
    uint64_t pbkdf2_done   = 0;   // for final summary only
    uint64_t scan_since_last = 0; // raw candidates since last display
    int      found         = 0;

    while(!g_sigint && found==0 && offset < mask.total_candidates){
        uint64_t remaining = mask.total_candidates - offset;
        int cur_wave = (int)std::min((uint64_t)wave_k1, remaining);

        // K1 : checksum filter (~40 reg, eliminates 15/16)
        {
            int zero = 0;
            cudaMemcpy(d_valid_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            int blk = (cur_wave + THREADS - 1) / THREADS;
            hydra_checksum_kernel<<<blk, THREADS>>>(d_mask, offset, cur_wave,
                                                     d_valid_indices, d_valid_count);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                std::cerr << "CUDA Error (checksum): " << cudaGetErrorString(cudaGetLastError()) << "\n"; break;
            }
        }

        int h_valid = 0;
        cudaMemcpy(&h_valid, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);

        // K2a/K2b/K2c : pipeline 3 kernels
        if(h_valid > 0 && !found){
            int blk = (h_valid + THREADS - 1) / THREADS;

            // K2a : word_indices → seed[64]
            hydra_k2a_pbkdf2<<<blk, THREADS>>>(
                d_mask, d_valid_indices, h_valid, d_seeds);

            // K2b : seed[64] → priv||chain after m/44'/coin'/0' (0 ECC, 0 spill)
            hydra_k2b_hardened<<<blk, THREADS>>>(
                d_target, d_seeds, d_intermed, h_valid);

            // K2c : 2 niveaux normaux (ECC×2) + ECC final + Hash + compare/Bloom
            hydra_k2c_ecc<<<blk, THREADS>>>(
                d_target, d_result, d_intermed, d_valid_indices, h_valid);

            cudaError_t err = cudaDeviceSynchronize();
            if(err != cudaSuccess){
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; break;
            }
            cudaMemcpy(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost);

            // Bloom mode : bloom hit in loop → verify balance
            if(found && (target.type==TargetType::BLOOM||target.type==TargetType::BLOOM_BTC||target.type==TargetType::BLOOM_ETH)) {
                cudaMemcpy(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

                // Reconstruire la phrase du hit
                uint16_t hit_words[SEED_MAX_WORDS];
                for(int w=0;w<mask.num_words;w++) hit_words[w]=mask.word_indices[w];
                uint64_t hidx=h_res.index;
                for(int x=(int)mask.num_unknown-1;x>=0;x--){
                    hit_words[mask.unknown_pos[x]]=(uint16_t)(hidx%2048); hidx/=2048;
                }
                std::string hit_phrase;
                for(int w=0;w<mask.num_words;w++){
                    uint16_t off=h_BIP39_OFFS[hit_words[w]]; uint8_t len=h_BIP39_LENS[hit_words[w]];
                    if(w>0) hit_phrase+=" ";
                    hit_phrase+=std::string((char*)h_BIP39_BLOB+off,len);
                }

                std::cout << "\n!!! BLOOM HIT !!!\n  Phrase : " << hit_phrase << "\n";
                // found stays 1 → exits loop → notify_victory + VICTORY displayed below
            }
        }

        offset           += cur_wave;
        pbkdf2_done      += h_valid;
        scan_since_last  += cur_wave;  // espace de recherche parcouru (brut)

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - t_last).count();
        if(dt >= 1.0){
            // Speed = search space covered (K1 included in denominator)
            double speed   = scan_since_last / dt / 1e6;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog    = 100.0 * (double)offset / (double)mask.total_candidates;
            double eta     = (elapsed>0 && offset>0)
                ? (double)(mask.total_candidates - offset) / ((double)offset / elapsed) : 0;
            int eh=(int)(eta/3600), em=(int)((eta-eh*3600)/60), es=(int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(2) << speed << " MKey/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2)<<eh<<":"<<std::setw(2)<<em<<":"<<std::setw(2)<<es
                      << std::flush;
            t_last = now; scan_since_last = 0;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t0).count();
    double avg_speed = (total_elapsed > 0) ? (double)offset / total_elapsed / 1e6 : 0;
    std::cout << "\nTime    : " << std::fixed << std::setprecision(2) << total_elapsed << " s"
              << " | " << std::setprecision(2) << avg_speed << " MKey/s avg"
              << " | PBKDF2 done : " << pbkdf2_done << "\n";

    cudaFree(d_valid_indices); cudaFree(d_valid_count);
    cudaFree(d_seeds); cudaFree(d_intermed);

    if(found){
        cudaMemcpy(&h_res,d_result,sizeof(HydraResult),cudaMemcpyDeviceToHost);
        // Reconstruit la combinaison gagnante
        uint16_t win_words[SEED_MAX_WORDS];
        for(int w=0;w<mask.num_words;w++) win_words[w]=mask.word_indices[w];
        uint64_t idx=h_res.index;
        for(int x=(int)mask.num_unknown-1;x>=0;x--){
            win_words[mask.unknown_pos[x]]=(uint16_t)(idx%2048);
            idx/=2048;
        }

        // Reconstruire la phrase
        std::string phrase;
        for(int w=0;w<mask.num_words;w++){
            uint16_t off=h_BIP39_OFFS[win_words[w]];
            uint8_t  len=h_BIP39_LENS[win_words[w]];
            if(w>0) phrase += " ";
            phrase += std::string((char*)h_BIP39_BLOB+off,len);
        }

        std::cout<<"\n======== VICTORY ! SEED FOUND =========================\n";
        std::cout<<"Phrase : " << phrase << "\n";
        std::cout<<"=======================================================\n";
        {
            std::string key_info = "*Phrase:*\n`" + phrase + "`";
            std::string addr_info = addr_str.empty() ? "" : ("*Adresse:* `" + addr_str + "`");
            notify_victory("SEED FOUND \xF0\x9F\x8C\xB1", key_info, addr_info);
        }
    } else if(!g_sigint){
        std::cout<<"Not found in "<<mask.total_candidates<<" candidates.\n";
    }
    print_search_summary(found != 0);

    if(target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_mask); cudaFree(d_target); cudaFree(d_result);
    return found ? 0 : 2;
}

// =================================================================================
// WIF MODE — parse_wif_mask
// =================================================================================
static bool parse_wif_mask(const std::string &wif_str, WifMask &mask)
{
    if(wif_str.size() != WIF_LEN){
        std::cerr << "Error: WIF mask must be exactly " << WIF_LEN
                  << " characters (got " << wif_str.size() << ")\n";
        return false;
    }

    static const char* alpha = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    mask.num_chars   = WIF_LEN;
    mask.num_unknown = 0;
    mask.total_candidates = 1;

    for(int i = 0; i < WIF_LEN; i++){
        char c = wif_str[i];
        if(c == '#'){
            if(mask.num_unknown >= WIF_MAX_UNKN){
                std::cerr << "Error: too many unknown positions (max " << WIF_MAX_UNKN << ")\n";
                return false;
            }
            mask.unknown_pos[mask.num_unknown++] = (uint8_t)i;
            mask.known_b58[i] = 0xFF;
            mask.total_candidates *= 58ULL;
        } else {
            const char *p = strchr(alpha, c);
            if(!p){
                std::cerr << "Error: invalid character '" << c << "' (not Base58 or #)\n";
                return false;
            }
            mask.known_b58[i] = (uint8_t)(p - alpha);
        }
    }

    if(mask.num_unknown == 0){
        std::cerr << "Error: no unknown position (#) in WIF mask.\n";
        return false;
    }

    std::cout << "Positions $ : " << (int)mask.num_unknown << "\n";
    std::cout << "Candidats   : " << mask.total_candidates
              << " (58^" << (int)mask.num_unknown << ")"
              << " → ~1 valide après checksum SHA256×2\n";
    return true;
}

// =================================================================================
// 8b. RUN WIF MODE
// =================================================================================
static int run_wif_mode(const std::string &wif_str, const std::string &addr_str)
{
    // Parse adresse cible (WIF = BTC uniquement, pas d'ETH)
    TargetData target = {};
    if(is_bloom_arg(addr_str)){
        // WIF is a Bitcoin-only format — bloom and bloombtc are equivalent here
        std::string l = addr_str;
        for (auto& ch : l) ch = tolower(ch);
        if (l == "bloometh") {
            std::cerr << "Error: WIF mode is Bitcoin-only. Use 'bloom' or 'bloombtc'.\n";
            return 1;
        }
        // bloom and bloombtc both map to BLOOM_BTC in WIF mode
        target.type = TargetType::BLOOM_BTC;
        if(!load_bloom_to_target(target)) return 1;
        std::cout << "Mode : WIF + Bloom BTC\n";
    } else {
        target.type = TargetType::BTC;
        if(!addrToHash160Any(addr_str, target.hash20)){
            std::cerr << "Error: invalid BTC/SegWit address.\n"; return 1;
        }
        std::cout << "Mode : WIF / BTC\n";
    }

    WifMask mask = {};
    if(!parse_wif_mask(wif_str, mask)) return 1;

    // Alloue GPU
    int device = 0; cudaSetDevice(device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);

    WifMask    *d_mask   = nullptr;
    TargetData *d_target = nullptr;
    HydraResult*d_result = nullptr;
    cudaMalloc(&d_mask,   sizeof(WifMask));
    cudaMalloc(&d_target, sizeof(TargetData));
    cudaMalloc(&d_result, sizeof(HydraResult));
    cudaMemcpy(d_mask,   &mask,   sizeof(WifMask),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);
    HydraResult h_res = {0, 0, 0};
    cudaMemcpy(d_result, &h_res, sizeof(HydraResult), cudaMemcpyHostToDevice);

    const int THREADS   = 256;
    const int blocks_k1 = prop.multiProcessorCount * 64;
    const int wave_k1   = blocks_k1 * THREADS;
    const int blocks_k2 = prop.multiProcessorCount * 32;

    uint64_t *d_valid_indices = nullptr;
    int      *d_valid_count   = nullptr;
    cudaMalloc(&d_valid_indices, (size_t)wave_k1 * sizeof(uint64_t));
    cudaMalloc(&d_valid_count,   sizeof(int));

    std::cout << "======== HYDRA V4.0 (WIF MODE) =====================\n";
    std::cout << "GPU         : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "K1 checksum : " << blocks_k1 << " blocs × " << THREADS << " (~40 reg)\n";
    std::cout << "K2 ecc      : " << blocks_k2 << " blocs × " << THREADS << "\n";
    signal(SIGINT, handle_sigint);

    auto t0 = std::chrono::high_resolution_clock::now(), t_last = t0;
    uint64_t offset        = 0;
    uint64_t checked_total = 0;
    double   checked_since = 0;
    int      found         = 0;

    while(!g_sigint && !found && offset < mask.total_candidates){
        uint64_t remaining = mask.total_candidates - offset;
        int cur_wave = (int)std::min((uint64_t)wave_k1, remaining);

        // K1 : SHA256x2 checksum filter
        {
            int zero = 0;
            cudaMemcpy(d_valid_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            int blk = (cur_wave + THREADS - 1) / THREADS;
            hydra_wif_checksum_kernel<<<blk, THREADS>>>(
                d_mask, offset, cur_wave, d_valid_indices, d_valid_count);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                std::cerr << "CUDA Error (wif checksum): " << cudaGetErrorString(cudaGetLastError()) << "\n"; break;
            }
        }

        int h_valid = 0;
        cudaMemcpy(&h_valid, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);

        // K2 : ECC + compare (rarely reached thanks to ÷2^32 filter)
        if(h_valid > 0 && !found){
            int blk = (h_valid + THREADS - 1) / THREADS;
            hydra_wif_ecc_kernel<<<blk, THREADS>>>(
                d_mask, d_target, d_result, d_valid_indices, h_valid);
            cudaError_t err = cudaDeviceSynchronize();
            if(err != cudaSuccess){
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; break;
            }
            cudaMemcpy(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost);
        }

        offset          += cur_wave;
        checked_total   += cur_wave;
        checked_since   += cur_wave;

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - t_last).count();
        if(dt >= 1.0){
            double speed   = checked_since / dt / 1e6;  // M/s (checksum SHA256 est rapide)
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog    = 100.0 * (double)offset / (double)mask.total_candidates;
            double eta     = (elapsed > 0 && offset > 0)
                ? (double)(mask.total_candidates - offset) / ((double)offset / elapsed) : 0;
            int eh = (int)(eta/3600), em = (int)((eta - eh*3600)/60), es = (int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(1) << speed << " MKey/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2) << eh << ":" << std::setw(2) << em << ":" << std::setw(2) << es
                      << std::flush;
            t_last = now; checked_since = 0;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t0).count();
    double avg_speed = (total_elapsed > 0) ? checked_total / total_elapsed / 1e6 : 0;
    std::cout << "\nTime  : " << std::fixed << std::setprecision(2) << total_elapsed << " s"
              << " | Avg speed : " << std::setprecision(1) << avg_speed << " M/s"
              << " | Tested : " << checked_total << "\n";

    if(found){
        cudaMemcpy(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

        // Reconstruct the found WIF
        uint8_t b58vals[WIF_LEN];
        for(int i = 0; i < WIF_LEN; i++) b58vals[i] = mask.known_b58[i];
        uint64_t idx = h_res.index;
        for(int x = (int)mask.num_unknown - 1; x >= 0; x--){
            b58vals[mask.unknown_pos[x]] = (uint8_t)(idx % 58);
            idx /= 58;
        }
        static const char* alpha = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        std::string wif_found(WIF_LEN, ' ');
        for(int i = 0; i < WIF_LEN; i++) wif_found[i] = alpha[b58vals[i]];

        // Decoder WIF -> cle privee -> adresses
        std::string addr_legacy, addr_segwit, addr_eth;
        {
            static const char* WIF_ALPHA = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
            BIGNUM* bn = BN_new(); BN_zero(bn);
            BN_CTX* bctx = BN_CTX_new();
            for(char c : wif_found){
                const char* p = strchr(WIF_ALPHA, c);
                int val = p ? (int)(p - WIF_ALPHA) : 0;
                BN_mul_word(bn, 58);
                BN_add_word(bn, (unsigned long)val);
            }
            uint8_t raw[38]={};
            BN_bn2binpad(bn, raw, 38);
            BN_free(bn); BN_CTX_free(bctx);
            uint8_t privkey[32];
            memcpy(privkey, raw+1, 32);
            key_to_addresses(privkey, addr_legacy, addr_segwit, addr_eth);
        }
        std::cout << "\n======== VICTORY ! WIF FOUND ==========================\n";
        std::cout << "WIF         : " << wif_found << "\n";
        std::cout << "BTC legacy  : " << addr_legacy << "\n";
        std::cout << "BTC segwit  : " << addr_segwit << "\n";
        std::cout << "=======================================================\n";
        {
            std::string key_info = "*WIF:*\n`" + wif_found + "`";
            std::string addr_info;
            if (!addr_str.empty() && addr_str.substr(0,3) == "bc1")
                addr_info = "*BTC segwit:* `" + addr_segwit + "`";
            else
                addr_info = "*BTC legacy:* `" + addr_legacy + "`";
            notify_victory("WIF FOUND", key_info, addr_info);
        }

        // En mode bloom : vérifier le solde avant de déclarer victoire
        if (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH) {
            // Récupérer la clé privée pour check_balances_and_notify
            static const char* WIF_ALPHA2 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
            BIGNUM* bn2 = BN_new(); BN_zero(bn2);
            BN_CTX* bctx2 = BN_CTX_new();
            for(char c : wif_found){
                const char* p2 = strchr(WIF_ALPHA2, c);
                int val2 = p2 ? (int)(p2 - WIF_ALPHA2) : 0;
                BN_mul_word(bn2, 58); BN_add_word(bn2, (unsigned long)val2);
            }
            uint8_t raw2[38]={}; BN_bn2binpad(bn2, raw2, 38);
            BN_free(bn2); BN_CTX_free(bctx2);
            uint8_t privkey_wif[32]; memcpy(privkey_wif, raw2+1, 32);

            bool victory = check_balances_and_notify(privkey_wif, addr_legacy, addr_segwit, addr_eth);
            if (!victory) {
                // Faux positif → continuer
                int zero = 0;
                cudaMemcpy(&d_result->found, &zero, sizeof(int), cudaMemcpyHostToDevice);
                found = 0;
            }
        }
    } else if(!g_sigint){
        std::cout << "Not found in " << checked_total << " candidates.\n";
    }
    print_search_summary(found != 0);

    if(target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_mask); cudaFree(d_target); cudaFree(d_result);
    cudaFree(d_valid_indices); cudaFree(d_valid_count);
    return found ? 0 : 2;
}

// =================================================================================
// 9. RUN PASSPHRASE MODE
// All words are known, brute-force the BIP39 passphrase from dictionary.txt
// =================================================================================
#define DICT_FILE      "dictionary.txt"
#define PASS_BATCH     491520    // 480K passphrases par batch (MAX_PASS_LEN=96 → ~45 MB)

// =================================================================================
// CPU BIP32/44 DERIVATION
// seed + passphrase → privkey (m/44'/coin'/0'/0/0)
// Used to verify bloom hits in passphrase mode (GPU does not output privkey)
// =================================================================================
#include <openssl/hmac.h>
#include <openssl/evp.h>

// HMAC-SHA512 wrapper (OpenSSL)
static void cpu_hmac_sha512(const uint8_t* key, size_t key_len,
                             const uint8_t* msg, size_t msg_len,
                             uint8_t out[64])
{
    unsigned int out_len = 64;
    HMAC(EVP_sha512(), key, (int)key_len, msg, (int)msg_len, out, &out_len);
}

// (tweak + parent) mod secp256k1_n, big-endian byte[32]
static void cpu_add_mod_n(const uint8_t tweak[32], const uint8_t parent[32], uint8_t out[32])
{
    static const uint8_t N[32] = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };
    uint32_t carry = 0;
    uint8_t tmp[32];
    for (int i = 31; i >= 0; i--) {
        uint32_t s = (uint32_t)tweak[i] + (uint32_t)parent[i] + carry;
        tmp[i] = (uint8_t)(s & 0xFF);
        carry  = s >> 8;
    }
    // if tmp >= N, subtract N
    bool ge = (carry > 0);
    if (!ge) for (int i = 0; i < 32; i++) {
        if (tmp[i] > N[i]) { ge = true; break; }
        if (tmp[i] < N[i]) break;
    }
    if (ge) {
        uint32_t borrow = 0;
        for (int i = 31; i >= 0; i--) {
            int32_t s = (int32_t)tmp[i] - (int32_t)N[i] - (int32_t)borrow;
            out[i] = (uint8_t)((s + 256) & 0xFF);
            borrow = (s < 0) ? 1 : 0;
        }
    } else {
        memcpy(out, tmp, 32);
    }
}

// BIP32 hardened child derivation (CPU)
static void cpu_bip32_hardened(const uint8_t par_priv[32], const uint8_t par_chain[32],
                                uint32_t index, uint8_t ch_priv[32], uint8_t ch_chain[32])
{
    uint8_t data[37];
    data[0] = 0x00;
    memcpy(data + 1, par_priv, 32);
    data[33] = (uint8_t)(index >> 24); data[34] = (uint8_t)(index >> 16);
    data[35] = (uint8_t)(index >>  8); data[36] = (uint8_t)(index);
    uint8_t out[64];
    cpu_hmac_sha512(par_chain, 32, data, 37, out);
    cpu_add_mod_n(out, par_priv, ch_priv);
    memcpy(ch_chain, out + 32, 32);
}

// BIP32 normal child derivation — needs compressed pubkey (CPU via OpenSSL 3.0 API)
static void cpu_bip32_normal(const uint8_t par_priv[32], const uint8_t par_chain[32],
                              uint32_t index, uint8_t ch_priv[32], uint8_t ch_chain[32])
{
    // Use EC_GROUP/EC_POINT directly (EC_KEY API deprecated since OpenSSL 3.0)
    EC_GROUP* grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BIGNUM*   bn  = BN_bin2bn(par_priv, 32, nullptr);
    EC_POINT* pub = EC_POINT_new(grp);
    EC_POINT_mul(grp, pub, bn, nullptr, nullptr, nullptr);

    uint8_t pubkey[33];
    EC_POINT_point2oct(grp, pub, POINT_CONVERSION_COMPRESSED, pubkey, 33, nullptr);
    EC_POINT_free(pub); BN_free(bn); EC_GROUP_free(grp);

    uint8_t data[37];
    memcpy(data, pubkey, 33);
    data[33] = (uint8_t)(index >> 24); data[34] = (uint8_t)(index >> 16);
    data[35] = (uint8_t)(index >>  8); data[36] = (uint8_t)(index);
    uint8_t out[64];
    cpu_hmac_sha512(par_chain, 32, data, 37, out);
    cpu_add_mod_n(out, par_priv, ch_priv);
    memcpy(ch_chain, out + 32, 32);
}

// Full pipeline: mnemonic phrase + passphrase → privkey via BIP44 m/44'/coin'/0'/0/0
// coin_type: 0=BTC, 60=ETH
// Returns true on success
static bool cpu_derive_key(const std::string& phrase, const std::string& passphrase,
                            uint32_t coin_type, uint8_t privkey[32])
{
    // 1. PBKDF2-HMAC-SHA512 : mnemonic → seed[64]
    const std::string salt = "mnemonic" + passphrase;
    uint8_t seed[64];
    PKCS5_PBKDF2_HMAC(phrase.c_str(), (int)phrase.size(),
                      (const uint8_t*)salt.c_str(), (int)salt.size(),
                      2048, EVP_sha512(), 64, seed);

    // 2. BIP32 master key : HMAC-SHA512("Bitcoin seed", seed)
    static const uint8_t BITCOIN_SEED[] = "Bitcoin seed";
    uint8_t master[64];
    cpu_hmac_sha512(BITCOIN_SEED, 12, seed, 64, master);
    uint8_t mpriv[32], mchain[32];
    memcpy(mpriv,  master,      32);
    memcpy(mchain, master + 32, 32);

    // 3. BIP44 : m/44'/coin'/0'/0/0
    uint8_t k0[32],c0[32], k1[32],c1[32], k2[32],c2[32], k3[32],c3[32];
    cpu_bip32_hardened(mpriv,  mchain,  0x8000002C,           k0, c0);  // 44'
    cpu_bip32_hardened(k0, c0, 0x80000000 | coin_type,        k1, c1);  // coin'
    cpu_bip32_hardened(k1, c1, 0x80000000,                    k2, c2);  // 0'
    cpu_bip32_normal  (k2, c2, 0,                             k3, c3);  // 0
    cpu_bip32_normal  (k3, c3, 0,                        privkey, c0);  // 0

    return true;
}

static int run_passphrase_mode(const std::string &phrase, const std::string &addr_str) {


    // ── Parse target address ─────────────────────────────────────────────────
    TargetData target = {};
    if(is_bloom_arg(addr_str)){
        target.type = get_bloom_type(addr_str);
        if(!load_bloom_to_target(target)) return 1;
        std::string _bmode=(target.type==TargetType::BLOOM_BTC)?"PASSPHRASE + Bloom BTC":(target.type==TargetType::BLOOM_ETH)?"PASSPHRASE + Bloom ETH":"PASSPHRASE + Bloom";
        std::cout << "Mode : " << _bmode << "\n";
    } else if(addr_str.size()>=2 && addr_str[0]=='0' && (addr_str[1]=='x'||addr_str[1]=='X')){
        target.type = TargetType::ETH;
        if(!ethAddrToBytes(addr_str, target.hash20)){
            std::cerr << "Error: invalid ETH address.\n"; return 1;
        }
        std::cout << "Mode : PASSPHRASE / ETH BIP44 m/44'/60'/0'/0/0\n";
    } else {
        target.type = TargetType::BTC;
        if(!addrToHash160Any(addr_str, target.hash20)){
            std::cerr << "Error: invalid BTC address.\n"; return 1;
        }
        std::cout << "Mode : PASSPHRASE / BTC BIP44 m/44'/0'/0'/0/0\n";
    }

    // ── Build mnemonic string from phrase ────────────────────────────────────
    // phrase = "word1 word2 ... word12" (all known, no #)
    auto wmap = build_word_map();
    SeedMask mask = {};
    if(!parse_seed_mask(phrase, mask, wmap)) return 1;

    // Rebuild mnemonic string (PBKDF2 password)
    // Using CPU-side build_mnemonic (simple CPU version)
    std::string mnemonic_str;
    {
        std::istringstream ss(phrase);
        std::string tok;
        bool first = true;
        while(ss >> tok) {
            if(!first) mnemonic_str += ' ';
            mnemonic_str += tok;
            first = false;
        }
    }
    const uint32_t mnemonic_len = (uint32_t)mnemonic_str.size();

    // ── Open dictionary.txt ──────────────────────────────────────────────────
    FILE* dict = fopen(DICT_FILE, "r");
    if(!dict){
        std::cerr << "Error: cannot open " << DICT_FILE << "\n";
        std::cerr << "Place your dictionary at " << DICT_FILE << "\n";
        return 1;
    }

    // Count lines for ETA
    uint64_t total_lines = 0;
    {
        char buf[256];
        while(fgets(buf, sizeof(buf), dict)) total_lines++;
        rewind(dict);
    }
    std::cout << "Dictionary : " << DICT_FILE << " (" << total_lines << " entries)\n";

    // ── GPU setup ────────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const int THREADS = 256;
    const int blocks_pass = (prop.multiProcessorCount * 2048) / THREADS; // ~occupancy max
    const int wave        = PASS_BATCH;

    // Allocate device buffers
    uint8_t  *d_mnemonic    = nullptr;
    uint8_t  *d_passphrases = nullptr;
    uint8_t  *d_pass_lens   = nullptr;
    uint8_t  *d_seeds       = nullptr;
    uint8_t  *d_intermed    = nullptr;
    TargetData *d_target    = nullptr;
    HydraResult *d_result   = nullptr;

    cudaMalloc(&d_mnemonic,    mnemonic_len);
    cudaMalloc(&d_passphrases, (size_t)wave * MAX_PASS_LEN);
    cudaMalloc(&d_pass_lens,   (size_t)wave);
    cudaMalloc(&d_seeds,       (size_t)wave * 64);
    cudaMalloc(&d_intermed,    (size_t)wave * 64);
    cudaMalloc(&d_target,      sizeof(TargetData));
    cudaMalloc(&d_result,      sizeof(HydraResult));

    // Upload mnemonic + target
    cudaMemcpy(d_mnemonic, mnemonic_str.c_str(), mnemonic_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);

    // Initialize BIP39 dictionary in constant memory
    cudaMemcpyToSymbol(d_BIP39_BLOB, h_BIP39_BLOB, sizeof(h_BIP39_BLOB));
    cudaMemcpyToSymbol(d_BIP39_OFFS, h_BIP39_OFFS, sizeof(h_BIP39_OFFS));
    cudaMemcpyToSymbol(d_BIP39_LENS, h_BIP39_LENS, sizeof(h_BIP39_LENS));

    // Precompute H_ipad/H_opad from mnemonic (1 thread, once)
    hydra_passphrase_setup<<<1, 1>>>(d_mnemonic, mnemonic_len);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "CUDA Error (passphrase setup): " << cudaGetErrorString(cudaGetLastError()) << "\n";
        return 1;
    }

    HydraResult h_res = {};
    cudaMemcpy(d_result, &h_res, sizeof(HydraResult), cudaMemcpyHostToDevice);

    std::cout << "======== HYDRA V4.0 (PASSPHRASE MODE) ====================\n";
    std::cout << "GPU          : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "Mnemonic     : " << mnemonic_str << "\n";
    std::cout << "K2a passphrase: " << blocks_pass << " blocks x " << THREADS << "\n";
    std::cout << "K2b hardened : " << blocks_pass << " blocks x " << THREADS << "\n";
    std::cout << "K2c ECC      : " << blocks_pass << " blocks x " << THREADS << "\n";

    // ── Host buffers for dictionary reading ──────────────────────────────────
    std::vector<uint8_t> h_passphrases((size_t)wave * MAX_PASS_LEN, 0);
    std::vector<uint8_t> h_pass_lens(wave, 0);

    // ── Main loop ────────────────────────────────────────────────────────────
    auto t0     = std::chrono::high_resolution_clock::now();
    auto t_last = t0;
    uint64_t total_tested  = 0;
    uint64_t since_last    = 0;
    int      found         = 0;
    char     line_buf[256];
    bool     eof           = false;

    while(!g_sigint && !found && !eof){

        // Fill batch from dictionary
        int batch_count = 0;
        while(batch_count < wave && !eof){
            if(!fgets(line_buf, sizeof(line_buf), dict)){
                eof = true; break;
            }
            // Strip \n and \r
            int len = (int)strlen(line_buf);
            while(len > 0 && (line_buf[len-1]=='\n' || line_buf[len-1]=='\r')) len--;
            line_buf[len] = '\0';

            // Skip lines that are too long
            if(len > MAX_PASS_LEN) continue;

            uint8_t* dst = h_passphrases.data() + (size_t)batch_count * MAX_PASS_LEN;
            memset(dst, 0, MAX_PASS_LEN);
            memcpy(dst, line_buf, len);
            h_pass_lens[batch_count] = (uint8_t)len;
            batch_count++;
        }

        if(batch_count == 0) break;

        // Upload batch
        cudaMemcpy(d_passphrases, h_passphrases.data(),
                   (size_t)batch_count * MAX_PASS_LEN, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pass_lens, h_pass_lens.data(),
                   (size_t)batch_count, cudaMemcpyHostToDevice);

        int blk = (batch_count + THREADS - 1) / THREADS;

        // K2a : passphrase → seed[64]
        hydra_k2a_passphrase<<<blk, THREADS>>>(
            d_passphrases, d_pass_lens, batch_count, d_seeds);

        // K2b : seed → priv||chain after m/44'/coin'/0'
        hydra_k2b_hardened<<<blk, THREADS>>>(
            d_target, d_seeds, d_intermed, batch_count);

        // K2c : normal derivations + ECC + Hash + compare
        // Note : nullptr for valid_indices (unused in passphrase mode)
        // d_pass_lens reused as index proxy (unused when no match)
        hydra_k2c_ecc_pass<<<blk, THREADS>>>(
            d_target, d_result, d_intermed, batch_count);

        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << "CUDA Error (k2c_ecc_pass): " << cudaGetErrorString(cudaGetLastError()) << "\n"; break;
        }
        cudaMemcpy(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost);

        // Bloom mode : passphrase hit → derive privkey CPU → check balance
        if(found && (target.type==TargetType::BLOOM||target.type==TargetType::BLOOM_BTC||target.type==TargetType::BLOOM_ETH)) {
            cudaMemcpy(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);
            uint64_t local_idx = h_res.index;
            std::string hit_pass;
            if(local_idx < (uint64_t)h_passphrases.size() / MAX_PASS_LEN){
                const uint8_t* p = h_passphrases.data() + local_idx * MAX_PASS_LEN;
                hit_pass = std::string((const char*)p, h_pass_lens[local_idx]);
            }
            std::cout << "\n!!! BLOOM HIT !!!\n  Mnemonic   : " << mnemonic_str
                      << "\n  Passphrase : \"" << hit_pass << "\"\n";
            std::cout << "  [CPU] BIP44 derivation...\n";

            // Derive privkey CPU (BTC coin=0, ETH coin=60)
            // For BLOOM_BTC → BTC derivation only
            // For BLOOM_ETH → ETH derivation only
            // For BLOOM    → try BTC first, then ETH if no balance
            bool victory = false;

            if (target.type != TargetType::BLOOM_ETH) {
                uint8_t privkey_btc[32];
                cpu_derive_key(mnemonic_str, hit_pass, 0, privkey_btc);
                std::string addr_legacy, addr_segwit, addr_eth;
                key_to_addresses(privkey_btc, addr_legacy, addr_segwit, addr_eth);
                victory = check_balances_and_notify(privkey_btc, addr_legacy, addr_segwit, addr_eth);
            }
            if (!victory && target.type != TargetType::BLOOM_BTC) {
                uint8_t privkey_eth[32];
                cpu_derive_key(mnemonic_str, hit_pass, 60, privkey_eth);
                std::string addr_legacy, addr_segwit, addr_eth;
                key_to_addresses(privkey_eth, addr_legacy, addr_segwit, addr_eth);
                victory = check_balances_and_notify(privkey_eth, addr_legacy, addr_segwit, addr_eth);
            }
            if (!victory) {
                // False positive → reset and continue
                int zero = 0;
                cudaMemcpy(&d_result->found, &zero, sizeof(int), cudaMemcpyHostToDevice);
                found = 0;
            }
            // If victory=true → found stays 1 → exits loop → VICTORY block below
        }

        total_tested += batch_count;
        since_last   += batch_count;

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - t_last).count();
        if(dt >= 1.0){
            double speed   = since_last / dt / 1e6;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog    = (total_lines > 0) ? 100.0 * total_tested / total_lines : 0.0;
            double eta     = (elapsed > 0 && total_tested > 0 && total_lines > 0)
                ? (double)(total_lines - total_tested) / ((double)total_tested / elapsed) : 0;
            int eh=(int)(eta/3600), em=(int)((eta-eh*3600)/60), es=(int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(2) << speed << " MKey/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2)<<eh<<":"<<std::setw(2)<<em<<":"<<std::setw(2)<<es
                      << std::flush;
            t_last = now; since_last = 0;
        }
    }
    fclose(dict);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t0).count();
    double avg_speed = (total_elapsed > 0) ? (double)total_tested / total_elapsed / 1e6 : 0;
    std::cout << "\nTime    : " << std::fixed << std::setprecision(2) << total_elapsed << " s"
              << " | " << std::setprecision(2) << avg_speed << " MKey/s avg"
              << " | Tested  : " << total_tested << "\n";

    if(found){
        cudaMemcpy(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);
        // result->index = position within current batch
        // Retrieve passphrase from host buffer (still in memory)
        uint64_t local_idx = h_res.index;
        std::string found_pass;
        if(local_idx < (uint64_t)h_passphrases.size() / MAX_PASS_LEN){
            const uint8_t* p = h_passphrases.data() + local_idx * MAX_PASS_LEN;
            found_pass = std::string((const char*)p, h_pass_lens[local_idx]);
        }
        std::cout << "\n======== VICTORY ! PASSPHRASE FOUND ==================\n";
        std::cout << "Mnemonic   : " << mnemonic_str << "\n";
        std::cout << "Passphrase : \"" << found_pass << "\"\n";
        std::cout << "=======================================================\n";
        {
            std::string key_info = "*Mnemonic:*\n`" + mnemonic_str + "`\n\n"
                                 + "*Passphrase:*\n`" + found_pass + "`";
            // Bloom mode : addr_str = "bloom/bloombtc/bloometh" → not useful to display
            // Derive the real address CPU-side for the Telegram message
            std::string addr_info;
            if (is_bloom_arg(addr_str)) {
                // Derive the real address from the recovered key
                uint32_t coin = (target.type == TargetType::BLOOM_ETH) ? 60u : 0u;
                uint8_t privkey[32];
                cpu_derive_key(mnemonic_str, found_pass, coin, privkey);
                std::string al, as_, ae;
                key_to_addresses(privkey, al, as_, ae);
                if (target.type == TargetType::BLOOM_ETH)
                    addr_info = "*ETH:* `" + ae + "`";
                else
                    addr_info = "*BTC legacy:* `" + al + "`\n*BTC segwit:* `" + as_ + "`";
            } else {
                addr_info = addr_str.empty() ? "" : ("*Address:* `" + addr_str + "`");
            }
            notify_victory("PASSPHRASE FOUND \xF0\x9F\x94\x91", key_info, addr_info);
        }
    } else {
        std::cout << "Passphrase not found in " << DICT_FILE << "\n";
    }
    print_search_summary(found != 0);

    cudaFree(d_mnemonic); cudaFree(d_passphrases); cudaFree(d_pass_lens);
    cudaFree(d_seeds); cudaFree(d_intermed);
    if(target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_target); cudaFree(d_result);
    return found ? 0 : 1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "  Hex mode        : ./Hydra <64hex_mask_#> <address|bloom>\n";
        std::cerr << "  Seed mode       : ./Hydra \"<phrase with # for unknowns>\" <address|bloom>\n";
        std::cerr << "  Passphrase mode : ./Hydra \"<12 full words>\" <address|bloom>\n";
        std::cerr << "                    (reads dictionary.txt)\n";
        std::cerr << "  WIF mode        : ./Hydra <wif_mask_#> <address|bloom>\n";
        std::cerr << "  Bloom filter    : replace <address> with bloom/bloombtc/bloometh\n";
        std::cerr << "                    bloom    = search BTC + ETH in bloom.bin\n";
        std::cerr << "                    bloombtc = BTC only (faster)\n";
        std::cerr << "                    bloometh = ETH only (faster)\n";
        return 1;
    }

    std::string arg1 = argv[1];

    if (argc >= 3 && looks_like_hex_mask(arg1)) {
        return run_hex_mode(arg1, argv[2]);
    } else if (argc >= 3 && is_passphrase_mode(arg1)) {
        return run_passphrase_mode(arg1, argv[2]);
    } else if (argc >= 3 && looks_like_seed(arg1)) {
        return run_seed_mode(arg1, argv[2]);
    } else if (argc >= 3 && looks_like_wif_mask(arg1)) {
        return run_wif_mode(arg1, argv[2]);
    } else {
        std::cerr << "Error: unrecognized argument or missing address.\n";
        return 1;
    }
}
