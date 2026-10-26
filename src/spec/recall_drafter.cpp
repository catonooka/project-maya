// src/spec/recall_drafter.cpp - see include/strata/spec/recall_drafter.hpp.
#include "strata/spec/recall_drafter.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace strata::spec {
namespace {
// The record on disk: an identity header, then appended chunks of [int64 n][n int32 tokens].  A record
// made by another build (or a truncated tail) never loads as this one's: the header check rejects the
// file, a short chunk stops the load at the last whole chunk.
constexpr uint64_t kMagic = 0x52454341'4C4C3031ull;   // "RECALL01"

// Does the record's held tail already end with this exact run of `n` tokens?
bool run_held(const std::vector<int32_t>& tail, const int32_t* tokens, int n) {
    if ((int) tail.size() < n) return false;
    const size_t off = tail.size() - (size_t) n;
    for (int i = 0; i < n; ++i)
        if (tail[(size_t) i + off] != tokens[i]) return false;
    return true;
}
}  // namespace

RecallDrafter::RecallDrafter(size_t capacity_tokens, std::string persist_path)
    : sfx_(3, 256, capacity_tokens), path_(std::move(persist_path)), cap_tokens_(capacity_tokens) {
    // (min_match 3: the drafter finds the longest match itself; the OFFER floor is this class's rule)
    tail_.reserve((size_t) dedup_run + 1);
    if (path_.empty()) return;

    bool fresh = true;
    if (std::FILE* in = std::fopen(path_.c_str(), "rb")) {
        uint64_t magic = 0;
        if (std::fread(&magic, sizeof magic, 1, in) == 1 && magic == kMagic) {
            fresh = false;
            int64_t total = 0;
            std::vector<int32_t> buf;
            for (;;) {
                int64_t n = 0;
                // a damaged or misaligned tail must not size an allocation: a run is capped by the
                // capacity (a chunk that claims more than fits, or a token pair read as a length
                // like 0x43800001000, ends the load - the record's good prefix stays)
                if (std::fread(&n, sizeof n, 1, in) != 1 || n <= 0 || n > (int64_t) cap_tokens_) break;
                buf.assign((size_t) n, 0);
                if (std::fread(buf.data(), 4, (size_t) n, in) != (size_t) n) break;
                sfx_.append(buf.data(), (size_t) n);
                for (int32_t t : buf) {
                    tail_.push_back(t);
                    if ((int) tail_.size() > dedup_run) tail_.erase(tail_.begin());
                }
                total += n;
            }
            if (total > 0)
                std::fprintf(stderr, "recall: loaded %lld tokens from %s\n", (long long) total, path_.c_str());
        }
        std::fclose(in);
    }

    f_ = std::fopen(path_.c_str(), fresh ? "w+b" : "ab");
    if (f_ == nullptr) {
        std::fprintf(stderr, "recall: %s: %s (the record stays in memory)\n", path_.c_str(),
                     std::strerror(errno));
        return;
    }
    if (fresh) {
        const uint64_t magic = kMagic;
        if (std::fwrite(&magic, sizeof magic, 1, f_) != 1) {
            std::fclose(f_);
            f_ = nullptr;
        }
    }
}

RecallDrafter::~RecallDrafter() {
    if (f_ != nullptr) {
        flush();
        std::fclose(f_);
    }
}

void RecallDrafter::flush() {
    if (f_ == nullptr || pending_.empty()) return;
    const int64_t cnt = (int64_t) pending_.size();
    if (std::fwrite(&cnt, sizeof cnt, 1, f_) != 1 ||
        std::fwrite(pending_.data(), 4, pending_.size(), f_) != pending_.size()) {
        std::fprintf(stderr, "recall: %s: write failed (%s)\n", path_.c_str(), std::strerror(errno));
        pending_.clear();
        return;
    }
    pending_.clear();
}

void RecallDrafter::append(const int32_t* tokens, size_t n) {
    if (n == 0) return;
    // Recall's dedup: a run the record already holds at its end is not added again (a request sent
    // twice adds nothing the second time).  The tail holds the record's last dedup_run tokens for this.
    const int run = (int) std::min<size_t>(n, (size_t) dedup_run);
    if (run >= 8 && run_held(tail_, tokens, run)) return;

    sfx_.append(tokens, n);
    for (size_t i = 0; i < n; ++i) {
        tail_.push_back(tokens[i]);
        if ((int) tail_.size() > dedup_run) tail_.erase(tail_.begin());
    }
    if (f_ == nullptr) return;
    pending_.insert(pending_.end(), tokens, tokens + n);
    if (pending_.size() >= 256) flush();   // a prompt's new part, a reply's end, a burst of tokens
}

bool RecallDrafter::offer(int dspark_kept) {
    if (cooldown_ > 0) {
        --cooldown_;
        return false;
    }
    if (since_probe_ < probe_every_) {   // while probes miss they come rarer (Recall's decay)
        ++since_probe_;
        return false;
    }
    // The expected-tokens check needs the match length, so it happens in propose(): offer() only says
    // whether the record may be asked at all this round.
    (void) dspark_kept;
    return true;
}

int RecallDrafter::propose(int max_k, int32_t* out, int dspark_kept) {
    // A cheap probe first: the drafter's own history gives the match length without copying.
    int32_t scratch[1];
    const int m = sfx_.propose(1, scratch);
    if (m <= 0) return 0;
    const int match = sfx_.last_match();
    const bool tier_b = match < kMinOfferMatch;
    if (tier_b && match < kMinBMatch) return 0;
    // Recall's rule: a copy is offered when its expected kept tokens beat the drafter's measured
    // kept-tokens-a-round.  The expectation is the bucket's land rate times a typical landing (the
    // matched run's length is the ceiling; the rate is how often copies at this length land at all).
    const double rate = tier_b ? land_rate_b() : land_rate(match);
    const double expect = rate * (double) std::min(match, tier_b ? 8 : 16);
    if (dspark_kept > 0) {
        // tier A must merely beat the drafter; tier B must clear it by kBarB (its misses cost the
        // round more, and its rate estimate is noisier at these lengths)
        const double bar = tier_b ? (double) dspark_kept * kBarB : (double) dspark_kept;
        if (expect <= bar) return 0;
    }
    const int k = sfx_.propose(max_k, out);
    if (k <= 0) return 0;
    if (tier_b) ++offers_b_; else ++offers_;
    return k;
}

void RecallDrafter::land(int offered, int kept) {
    if (offered <= 0) return;
    const bool tier_b = last_match() < kMinOfferMatch;
    if (tier_b) {
        if (kept > 0) {
            ++landed_b_;
            ++lands_;
            kept_tokens_ += kept;
            kept_b_ += kept;
            cooldown_ = 0;
            probe_every_ = 1;
        } else {
            ++missed_b_;
            cooldown_ = kCooldownRoundsB;
            probe_every_ = std::min(probe_every_ * 2, 64);
        }
        since_probe_ = 1;
        return;
    }
    const int b = bucket(last_match());
    if (kept > 0) {
        ++landed_[b];
        ++lands_;
        kept_tokens_ += kept;
        cooldown_ = 0;
        probe_every_ = 1;
    } else {
        ++missed_[b];
        cooldown_ = kCooldownRounds;
        probe_every_ = std::min(probe_every_ * 2, 64);
    }
    since_probe_ = 1;
}

}  // namespace strata::spec
