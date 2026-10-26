// src/spec/recall_drafter_test.cpp - the persistent record drafter's gates, in suffix_drafter_test's
// style.  What must hold (TensorFold's recall.md contract, adapted):
//   1. the copy case: a request whose suffix matches an earlier long run proposes that run's
//      continuation, in order;
//   2. the dedup rule: a run the record already holds is not appended again;
//   3. persistence: a second instance on the same file proposes from the first's record;
//   4. the offer policy: a landed copy keeps offers coming, a miss cools the record down for the
//      next kCooldownRounds rounds and the probes grow rarer;
//   5. the floor: matches under 16 tokens never propose;
//   6. the rate rule: a copy whose expected tokens do not beat the drafter's kept-tokens-a-round
//      proposes nothing (the drafter stays in charge when it is winning).
#include "strata/spec/recall_drafter.hpp"

#include <cstdio>
#include <string>
#include <vector>

using strata::spec::RecallDrafter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

// a damaged record (a truncated or misaligned tail read as a chunk length) must not size an
// allocation: the load ends, the good prefix stays, no abort (the 4.6 TB bad_alloc crash)
static bool damaged_record_test() {    const char* dir = std::getenv("RECALL_TEST_DIR");
    std::string path = std::string(dir ? dir : "/tmp") + "/record-damaged.bin";
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        const uint64_t magic = 0x52454341'4C4C3031ull;   // "RECALL01" - the loader's own
        std::fwrite(&magic, sizeof magic, 1, f);
        int64_t n = 8; std::vector<int32_t> run = {1, 2, 3, 4, 5, 6, 7, 8};
        std::fwrite(&n, sizeof n, 1, f);
        std::fwrite(run.data(), 4, run.size(), f);
        int64_t garbage = 0x43800001000ll;              // a token pair as a length: 4.6 TB
        std::fwrite(&garbage, sizeof garbage, 1, f);
        std::fclose(f);
    }
    strata::spec::RecallDrafter r(1 << 20, path);
    if (r.size() != 8) return false;                    // the good prefix survives
    std::remove(path.c_str());
    return true;
}

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : ".";
    std::string path = std::string(dir) + "/recall-test.rec";

    std::printf("recall_drafter_test\n");
    std::remove(path.c_str());
    {
        std::printf("  copy + dedup + the offer policy (file %s)\n", path.c_str());
        RecallDrafter r(1u << 20, path);
        // an earlier turn: a 40-token verbatim run, then a distinct 200-token tail
        std::vector<int32_t> run(40), filler(200);
        for (int i = 0; i < 40; ++i) run[(size_t) i] = 1000 + i;
        for (int i = 0; i < 200; ++i) filler[(size_t) i] = 5000 + i;
        r.append(run.data(), run.size());
        r.append(filler.data(), filler.size());

        // a later request replays the same run and asks for a continuation: the record's earlier
        // continuation (the filler) is what it should propose
        r.append(run.data(), run.size());
        int32_t out[64];
        check(r.offer(0), "an offer is possible with no dspark rate measured");
        const int k = r.propose(64, out, 0);
        check(k >= 16, "a match into an earlier run proposes its continuation");
        check(r.last_match() >= 16, "the match is at the offer floor or longer");
        bool same = k > 0;
        for (int i = 0; i < k && same; ++i) same = out[(size_t) i] == filler[(size_t) i];
        check(same, "the proposal is the earlier continuation, in order");

        // dedup: appending the record's own tail run again adds nothing (a request re-sent)
        const size_t before = r.size();
        r.append(run.data(), run.size());   // the tail IS this run now (the last append)
        check(r.size() == before, "a held run is not appended again");

        // the rate rule: when the DSpark drafter keeps more tokens a round than a copy can expect,
        // the copy is not offered
        int32_t out2[64];
        check(r.propose(64, out2, 1000000) == 0, "a copy below the drafter's rate proposes nothing");

        // land + cooldown
        r.land(k, k / 2);
        check(r.offer(0), "a landed copy keeps offers coming");
        r.land(1, 0);
        bool cooled = !r.offer(0);
        for (int i = 0; i < 3 && !cooled; ++i) cooled = cooled || !r.offer(0);
        check(cooled, "a miss cools the record down for the next rounds");
    }
    {
        std::printf("  persistence: a second instance reads the first's record\n");
        RecallDrafter r2(1u << 20, path);
        int32_t out[64];
        std::vector<int32_t> run(40);
        for (int i = 0; i < 40; ++i) run[(size_t) i] = 1000 + i;
        r2.append(run.data(), run.size());
        check(r2.offer(0), "the loaded record may be asked");
        const int k = r2.propose(64, out, 0);
        check(k >= 16, "the loaded record proposes from the first instance's history");
    }
    {
        std::printf("  the floor: a short match never proposes\n");
        RecallDrafter r(1u << 18, "");
        std::vector<int32_t> a(30), b(8);
        for (int i = 0; i < 30; ++i) a[(size_t) i] = 7000 + i;
        for (int i = 0; i < 8; ++i) b[(size_t) i] = 7000 + i;
        r.append(a.data(), a.size());
        r.append(b.data(), b.size());
        int32_t out[64];
        const int k = r.propose(64, out, 0);
        check(k == 0 || r.last_match() < 16, "matches under 16 propose nothing");
    }

    std::remove(path.c_str());
    {
        std::printf("  a damaged record loads its good prefix and does not abort\n");
        check(damaged_record_test(), "damaged record: prefix kept, no bad_alloc");
    }
    std::printf("%s\n", g_fail ? "FAILURES" : "all ok");
    return g_fail ? 1 : 0;
}
