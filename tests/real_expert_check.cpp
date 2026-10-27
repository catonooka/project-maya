// real_expert_check.cpp - read REAL experts from the 2.7 pack, compare the CPU lane kernel vs the reference
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <map>
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/cpu/exl3_ref.hpp"
#include "strata/kernels/f16_bits.hpp"

namespace kc = strata::kernels::cpu;
namespace ref = strata::kernels::cpu;

struct TInfo { std::string dtype; uint64_t off, end; std::vector<uint64_t> shape; };
struct Pack {
    std::ifstream f; std::map<std::string, TInfo> t; uint64_t data0 = 0;
    Pack(const std::string& path) {
        f.open(path, std::ios::binary);
        if (!f) { std::printf("cannot open %s\n", path.c_str()); return; }
        uint64_t n = 0; f.read((char*) &n, 8);
        std::string h(n, 0); f.read(h.data(), n);
        data0 = 8 + n;
        const size_t H = h.size();
        auto skip_ws = [&](size_t& p) { while (p < H && (h[p] == ' ' || h[p] == '\n' || h[p] == '\r' || h[p] == '\t')) ++p; };
        auto read_str = [&](size_t& p) -> std::string {
            if (p >= H || h[p] != '"') return {};
            ++p; size_t s = p;
            while (p < H && h[p] != '"') ++p;
            std::string out = h.substr(s, p - s);
            if (p < H) ++p;
            return out;
        };
        auto read_num = [&](size_t& p) -> uint64_t {
            uint64_t v = 0; bool any = false;
            while (p < H && h[p] >= '0' && h[p] <= '9') { v = v * 10 + (h[p] - '0'); ++p; any = true; }
            return any ? v : 0;
        };
        size_t p = 0;
        skip_ws(p);
        if (p >= H || h[p] != '{') return;
        ++p;
        while (p < H) {
            skip_ws(p);
            if (p < H && h[p] == '}') break;
            if (p >= H || h[p] != '"') break;
            std::string name = read_str(p);
            skip_ws(p);
            if (p >= H || h[p] != ':') break;
            ++p; skip_ws(p);
            if (p >= H || h[p] != '{') break;
            ++p;
            TInfo ti;
            while (p < H) {
                skip_ws(p);
                if (p < H && h[p] == '}') { ++p; break; }
                std::string key = read_str(p);
                skip_ws(p);
                if (p >= H || h[p] != ':') break;
                ++p; skip_ws(p);
                if (p < H && h[p] == '"') {
                    std::string v = read_str(p);
                    if (key == "dtype") ti.dtype = v;
                } else if (p < H && h[p] == '[') {
                    ++p;
                    std::vector<uint64_t> arr;
                    while (p < H && h[p] != ']') {
                        skip_ws(p);
                        if (p < H && h[p] == ']') break;
                        if (h[p] == ',') { ++p; continue; }
                        arr.push_back(read_num(p));
                    }
                    if (p < H) ++p;
                    if (key == "data_offsets") { ti.off = arr.size() > 0 ? arr[0] : 0; ti.end = arr.size() > 1 ? arr[1] : ti.off; }
                    else if (key == "shape") ti.shape = arr;
                } else {
                    uint64_t v = read_num(p);
                    (void) v;
                }
                skip_ws(p);
                if (p < H && h[p] == ',') { ++p; continue; }
                if (p < H && h[p] == '}') { ++p; break; }
            }
            t[name] = ti;
            skip_ws(p);
            if (p < H && h[p] == ',') { ++p; continue; }
            break;
        }
    }
    bool read(const std::string& name, std::vector<uint8_t>& out) {
        auto it = t.find(name);
        if (it == t.end()) return false;
        out.resize(it->second.end - it->second.off);
        f.seekg((std::streamoff) data0 + (std::streamoff) it->second.off);
        f.read((char*) out.data(), out.size());
        return true;
    }
};

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/home/aisever/models/DSV41-EXL3-2.7-0xSero";
    const int layer = argc > 2 ? std::atoi(argv[2]) : 2;
    char fname[256];
    std::snprintf(fname, sizeof fname, "%s/model-exl3-L%02d.safetensors", dir.c_str(), layer);
    Pack pack(fname);
    if (pack.t.empty()) { std::printf("pack empty\n"); return 1; }
    const char* parts[3] = {"w1", "w3", "w2"};

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> ux(-0.5f, 0.5f);

    {
        const char* head_env = std::getenv("HEAD_FILE");
        if (head_env != nullptr) {
            Pack hp(head_env);
            const char* hnames[] = {"head"};
            char tn2[64];
            std::snprintf(tn2, sizeof tn2, "head.trellis");
            auto hit = hp.t.find(tn2);
            if (hit != hp.t.end()) {
                const int k2 = (int) (hit->second.shape[2] / 8);
                const int k = (int) (hit->second.shape[0] * 16), nn = (int) (hit->second.shape[1] * 16);
                std::vector<uint8_t> tr2, su2, sv2;
                hp.read("head.suh", su2); hp.read("head.svh", sv2); hp.read("head.trellis", tr2);
                kc::Exl3Part m;
                m.trellis = (const uint32_t*) tr2.data();
                m.suh = (const uint16_t*) su2.data();
                m.svh = (const uint16_t*) sv2.data();
                m.k = k; m.n = nn; m.k2 = k2;
                std::vector<float> x(k);
                for (auto& v : x) v = ux(rng);
                // ref on a SLICE (full n=129280 w = 2.6GB - too big): check first 1024 cols only
                const int nsub = 1024;
                std::vector<float> w((size_t) k * nsub), y_ref(nsub, 0.0f);
                kc::exl3_inner_k2(m.trellis, nn / 16, k, nsub, k2, 2, w.data());
                kc::exl3_mv(w.data(), m.suh, m.svh, k, nsub, x.data(), y_ref.data());
                std::vector<float> xh(k), y_got(nsub, 0.0f);
                const float xs = kc::exl3_cpu_prepare(m, x.data(), xh.data());
                for (int c0 = 0; c0 < nsub; c0 += 128) kc::exl3_cpu_band(m, xh.data(), xs, c0, y_got.data() + c0);
                double worst = 0, den = 0;
                for (int i = 0; i < nsub; ++i) {
                    worst = std::max(worst, (double) std::fabs(y_ref[i] - y_got[i]));
                    den = std::max(den, (double) std::fabs(y_ref[i]));
                }
                std::printf("head K%-3g: worst %9.4g rel %7.3g  %s\n", k2 / 2.0, worst, den > 0 ? worst / den : worst,
                            den > 0 && worst / den < 2e-3 ? "ok" : "FAIL");
            }
        }
        const char* names[] = {"layers.2.attn.wq_a", "layers.2.ffn.shared_experts.w1", "layers.2.ffn.shared_experts.w2"};
        for (const char* nm : names) {
            char tn[160];
            std::snprintf(tn, sizeof tn, "%s.trellis", nm);
            auto it = pack.t.find(tn);
            if (it == pack.t.end()) { std::printf("%s: no trellis\n", nm); continue; }
            const int k2 = (int) (it->second.shape[2] / 8);
            const int k = (int) (it->second.shape[0] * 16), nn = (int) (it->second.shape[1] * 16);
            std::vector<uint8_t> tr2, su2, sv2;
            std::snprintf(tn, sizeof tn, "%s.suh", nm); pack.read(tn, su2);
            std::snprintf(tn, sizeof tn, "%s.svh", nm); pack.read(tn, sv2);
            std::snprintf(tn, sizeof tn, "%s.trellis", nm); pack.read(tn, tr2);
            if (tr2.empty() || su2.empty() || sv2.empty()) { std::printf("%s: read failed\n", nm); continue; }
            kc::Exl3Part m;
            m.trellis = (const uint32_t*) tr2.data();
            m.suh = (const uint16_t*) su2.data();
            m.svh = (const uint16_t*) sv2.data();
            m.k = k; m.n = nn; m.k2 = k2;
            std::vector<float> x(k);
            for (auto& v : x) v = ux(rng);
            std::vector<float> w((size_t) k * nn), y_ref(nn, 0.0f);
            kc::exl3_inner_k2(m.trellis, nn / 16, k, nn, k2, 2, w.data());
            kc::exl3_mv(w.data(), m.suh, m.svh, k, nn, x.data(), y_ref.data());
            std::vector<float> xh(k), y_got(nn, 0.0f);
            const float xs = kc::exl3_cpu_prepare(m, x.data(), xh.data());
            for (int c0 = 0; c0 < nn; c0 += 128) kc::exl3_cpu_band(m, xh.data(), xs, c0, y_got.data() + c0);
            double worst = 0, den = 0;
            for (int i = 0; i < nn; ++i) {
                worst = std::max(worst, (double) std::fabs(y_ref[i] - y_got[i]));
                den = std::max(den, (double) std::fabs(y_ref[i]));
            }
            double rel = den > 0 ? worst / den : worst;
            std::printf("%-36s K%-3g k%-5d n%-5d: worst %9.4g rel %7.3g  %s\n", nm, k2 / 2.0, k, nn, worst, rel,
                        rel < 2e-3 ? "ok" : "FAIL");
        }
    }
    for (int e : {0}) {
        for (int r = 0; r < 3; ++r) {
            if (std::getenv("ONLY_R") != nullptr && r != std::atoi(std::getenv("ONLY_R"))) continue;
            char tn[160];
            std::snprintf(tn, sizeof tn, "layers.%d.ffn.experts.%d.%s.trellis", layer, e, parts[r]);
            auto it = pack.t.find(tn);
            if (it == pack.t.end()) continue;
            const int k2 = (int) (it->second.shape[2] / 8);
            const int k = (int) (it->second.shape[0] * 16), nn = (int) (it->second.shape[1] * 16);
            std::vector<uint8_t> tr, su, sv;
            std::snprintf(tn, sizeof tn, "layers.%d.ffn.experts.%d.%s.suh", layer, e, parts[r]); pack.read(tn, su);
            std::snprintf(tn, sizeof tn, "layers.%d.ffn.experts.%d.%s.svh", layer, e, parts[r]); pack.read(tn, sv);
            std::snprintf(tn, sizeof tn, "layers.%d.ffn.experts.%d.%s.trellis", layer, e, parts[r]); pack.read(tn, tr);
            if (tr.empty() || su.empty() || sv.empty()) { std::printf("read failed e%d r%d\n", e, r); return 1; }
            if (std::getenv("TR_FILE") != nullptr && r == 0 && e == 0) {
                std::FILE* tf = std::fopen(std::getenv("TR_FILE"), "rb");
                if (tf != nullptr) {
                    std::fread(tr.data(), 1, tr.size(), tf);
                    std::fclose(tf);
                    std::printf("    trellis overridden from %s\n", std::getenv("TR_FILE"));
                }
            }
            std::printf("  e%d %s: k2=%d k=%d n=%d tr=%zu su=%zu sv=%zu\n", e, parts[r], k2, k, nn, tr.size(), su.size(), sv.size()); std::fflush(stdout);

            std::vector<float> x(k);
            for (auto& v : x) v = ux(rng);

            kc::Exl3Part m;
            m.trellis = (const uint32_t*) tr.data();
            m.suh = (const uint16_t*) su.data();
            m.svh = (const uint16_t*) sv.data();
            m.k = k; m.n = nn; m.k2 = k2;
            std::vector<uint8_t> su_big;
            if (std::getenv("ONES") != nullptr) {   // (all-ones scales: isolates scale-value dependence)
                su_big.assign((size_t) k * 2, 0);
                uint16_t* s = (uint16_t*) su_big.data();
                uint16_t one = strata::kernels::f16_from_f32(1.0f);
                for (int i = 0; i < k; ++i) s[i] = one;
                m.suh = (const uint16_t*) su_big.data();
                sv.assign((size_t) nn * 2, 0);
                uint16_t* v = (uint16_t*) sv.data();
                for (int i = 0; i < nn; ++i) v[i] = one;
                m.svh = (const uint16_t*) v;
            }
            if (std::getenv("SCALE_UP") != nullptr) {   // (suh x16, x /16: the same product, a safe range)
                su_big = su;
                uint16_t* s = (uint16_t*) su_big.data();
                for (int i = 0; i < k; ++i) {
                    float v = strata::kernels::f32_from_f16(s[i]) * 16.0f;
                    s[i] = strata::kernels::f16_from_f32(v);
                }
                m.suh = (const uint16_t*) su_big.data();
                for (auto& v : x) v /= 16.0f;
            }

            // reference: w = exl3_inner_k2(trellis), y = exl3_mv
            std::vector<float> w((size_t) k * nn), y_ref(nn, 0.0f);
            kc::exl3_inner_k2(m.trellis, nn / 16, k, nn, k2, 2, w.data());
            kc::exl3_mv(w.data(), m.suh, m.svh, k, nn, x.data(), y_ref.data());

            // engine CPU kernel - the band path (as the engine itself uses it: prepare's sum feeds band)
            std::vector<float> xh(k), y_got(nn, 0.0f);
            const float xs = kc::exl3_cpu_prepare(m, x.data(), xh.data());
            for (int c0 = 0; c0 < nn; c0 += 128) kc::exl3_cpu_band(m, xh.data(), xs, c0, y_got.data() + c0);
            if (std::getenv("DUMP_XH") != nullptr) {
                std::printf("    x[:4]: %.5g %.5g  xh[:4]: %.5g %.5g %.5g %.5g\n", x[0], x[1],
                            xh[0], xh[1], xh[2], xh[3]);
                std::printf("    ref w row0 [0..3]: %.5g %.5g %.5g %.5g\n", w[0], w[1], w[2], w[3]);
                // ref xh for comparison
                std::vector<float> rxh(k);
                kc::Exl3Part pm = m;
                pm.trellis = nullptr;  // (prepare only)
                // emulate the ref's xh: x*suh, had128, round f16 - via the public ref helpers on a copy
                std::vector<float> xs(k);
                for (int i = 0; i < k; ++i) xs[i] = x[i] * strata::kernels::f32_from_f16(m.suh[i]);
                kc::exl3_had128(xs.data(), (size_t) k);
                for (int i = 0; i < k; ++i) rxh[i] = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(xs[i]));
                std::printf("    ref xh[:4]: %.5g %.5g %.5g %.5g\n", rxh[0], rxh[1], rxh[2], rxh[3]);
                // single-column kernel output vs ref for column 0: y = had( w col ) * svh
                double c0 = 0;
                for (int i = 0; i < k; ++i) c0 += (double) rxh[i] * w[(size_t) i * nn];
                std::vector<float> y0(128, 0.0f);
                kc::exl3_had128(y0.data(), 128);
                // (skip had emulation; just print the raw dot)
                std::printf("    dot(xh_ref, w_col0) = %.6g\n", c0);
            }

            double worst = 0, den = 0;
            for (int i = 0; i < nn; ++i) {
                worst = std::max(worst, (double) std::fabs(y_ref[i] - y_got[i]));
                den = std::max(den, (double) std::fabs(y_ref[i]));
            }
            double rel = den > 0 ? worst / den : worst;
            std::printf("L%d e%-3d %s K%-3g: worst %9.4g rel %7.3g  %s\n", layer, e, parts[r], k2 / 2.0, worst, rel,
                        rel < 2e-3 ? "ok" : "FAIL");
            if (rel >= 2e-3) {
                std::printf("    ref[0..3]: %.5g %.5g %.5g   got[0..3]: %.5g %.5g %.5g\n",
                            y_ref[0], y_ref[1], y_ref[2], y_got[0], y_got[1], y_got[2]);
                double rs = 0, gs = 0;
                for (int i = 0; i < nn; ++i) rs += (double) y_ref[i] * y_ref[i], gs += (double) y_got[i] * y_got[i];
                std::printf("    |ref| %.4g  |got| %.4g  ratio %.4g\n", std::sqrt(rs), std::sqrt(gs),
                            rs > 0 ? std::sqrt(gs / rs) : 0.0);
            }
        }
    }
    return 0;
}
