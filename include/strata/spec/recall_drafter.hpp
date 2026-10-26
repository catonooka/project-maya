// include/strata/spec/recall_drafter.hpp - a persistent record drafter for maya-ds4's serve loop.
//
// TensorFold's `--recall` (docs/recall.md, MIT) cross-request idea, on this engine's own suffix-lookup
// drafter (suffix_drafter.hpp): the serve loop keeps one token-id record across requests and restarts,
// and when the stream's own history has a long enough match into the record, the earlier continuation
// is proposed instead of (before) the DSpark drafter.  The target model verifies every copied token, so
// a reply's tokens are the same with Recall on, off or read-only - the only thing that changes is how
// many tokens a round commits.
//
// The offer policy is Recall's: land rates kept apart for matches of 16-31, 32-63 and 64+ tokens, a
// copy is offered only when its expected kept tokens beat the drafter's recently kept-tokens-a-round,
// a miss costs the round its drafts (cooldown, and probes come rarer while they miss), and runs the
// record already holds are not re-added.
#pragma once

#include "strata/spec/suffix_drafter.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::spec {

class RecallDrafter {
public:
    /// `capacity_tokens` bounds the in-memory record (the suffix drafter's table); `persist_path`
    /// empty: this process only.
    RecallDrafter(size_t capacity_tokens, std::string persist_path);
    ~RecallDrafter();

    size_t capacity() const { return cap_tokens_; }   // the loader's chunk-length sanity bound

    /// Tokens into the record: a prompt's newly-read part, then every committed token.  A run the
    /// record already holds at its end is not added again.
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }

    /// May a copy be offered this round?  `dspark_kept` is the drafter's recently kept-tokens-a-round
    /// (0: nothing measured yet - then a long enough match is offered once).
    bool offer(int dspark_kept);

    /// Up to `max_k` proposal tokens into `out` (0: nothing worth verifying).  `dspark_kept` is the
    /// same as offer()'s: a copy whose expected kept tokens do not beat it proposes nothing.
    int propose(int max_k, int32_t* out, int dspark_kept);

    /// A copy landed `kept` of its `offered` tokens.  `kept == 0` is a miss: cooldown, slower probes.
    void land(int offered, int kept);

    /// The record's last match length (0: none) - for STAT and tests.
    int last_match() const { return sfx_.last_match(); }
    size_t size() const { return sfx_.size(); }

    /// Counters for the serve loop's STAT line.
    int64_t offers() const { return offers_; }
    int64_t lands() const { return lands_; }
    int64_t kept_tokens() const { return kept_tokens_; }

    static constexpr int dedup_run = 128;   // Recall's rule: a held run of this length is not re-added

private:
    static constexpr int kBuckets = 3;          // 16-31, 32-63, 64+
    static constexpr int kBucketsB = 1;         // tier B: 8-15 (semi-repeats - a paraphrase, an edited
                                                // file, structured output whose header repeats).  The
                                                // suffix drafter finds these matches already (its key
                                                // is a 3-gram); tier A's 16 floor was the only block.
    static constexpr int kMinOfferMatch = 16;   // tier A's floor (Recall's own)
    static constexpr int kMinBMatch = 8;        // tier B: shorter matches, lower prior, tighter rule
    static constexpr int kCooldownRounds = 4;   // a missed copy leaves the record alone this long
    static constexpr int kCooldownRoundsB = 8;  // (tier B misses cost the round more: cool down longer)
    static constexpr double kPriorQ[kBuckets] = {0.75, 0.88, 0.93};
    static constexpr double kPriorQB = 0.55;    // tier B lands far less often (beaten only when clear)
    static constexpr double kPriorN = 4.0;
    static constexpr double kBarB = 1.35;       // tier B must beat the drafter by 35% (misses cost more)

    int bucket(int match) const { return match < 32 ? 0 : match < 64 ? 1 : 2; }
    double land_rate(int match) const {
        const int b = bucket(match);
        return (landed_[b] + kPriorN * kPriorQ[b]) / (landed_[b] + missed_[b] + kPriorN);
    }
    double land_rate_b() const {   // tier B's own rate, its own prior
        return (landed_b_ + kPriorN * kPriorQB) / (landed_b_ + missed_b_ + kPriorN);
    }
    void flush();

    SuffixDrafter sfx_;            // the record (its history IS the record)
    std::string path_;
    std::FILE* f_ = nullptr;
    size_t cap_tokens_ = 0;        // the capacity the loader's chunk lengths are bounded by
    std::vector<int32_t> tail_;    // the record's last dedup_run tokens (the dedup check)
    std::vector<int32_t> pending_; // tokens not yet on disk (flushed in batches and at the end)
    int64_t landed_[kBuckets] = {}, missed_[kBuckets] = {};
    int64_t landed_b_ = 0, missed_b_ = 0;   // tier B's semi-repeat stats
    int cooldown_ = 0, probe_every_ = 1, since_probe_ = 1;
    int64_t offers_ = 0, lands_ = 0, kept_tokens_ = 0;
    int64_t offers_b_ = 0, kept_b_ = 0;     // tier B's, for the STAT line
};

}  // namespace strata::spec
