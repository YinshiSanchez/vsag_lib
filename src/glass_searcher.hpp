#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <ostream>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "common.h"
#include "glass_neighbor.hpp"
#include "index/glass_graph.hpp"
#include "quantization/glass_quant.hpp"

namespace vsag {

namespace glass {
using LockGuard = std::lock_guard<std::mutex>;

inline void
GenRandom(std::mt19937& rng, int* addr, const int size, const int N) {
    for (int i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (int i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    int off = rng() % N;
    for (int i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

struct RandomGenerator {
    std::mt19937 mt;

    explicit RandomGenerator(int64_t seed = 1234) : mt((unsigned int)seed) {
    }

    /// random positive integer
    int
    rand_int() {
        return mt() & 0x7fffffff;
    }

    /// random int64_t
    int64_t
    rand_int64() {
        return int64_t(rand_int()) | int64_t(rand_int()) << 31;
    }

    /// generate random integer between 0 and max-1
    int
    rand_int(int max) {
        return mt() % max;
    }

    /// between 0 and 1
    float
    rand_float() {
        return mt() / float(mt.max());
    }

    double
    rand_double() {
        return mt() / double(mt.max());
    }
};
struct SearcherBase {
    virtual void
    SetData(const float* data, int n, int dim) = 0;
    virtual void
    Optimize(int num_threads = 0) = 0;
    virtual void
    Search(const float* q, int k, int* dst) const = 0;
    virtual void
    SetEf(int ef) = 0;
    virtual ~SearcherBase() = default;
    virtual void
    serialize(std::ostream& writer) const = 0;
    virtual void
    deserialize(std::istream& reader) = 0;
};

template <typename Quantizer>
struct Searcher : public SearcherBase {
    int d;
    int nb;
    Graph<int> graph;
    Quantizer quant;

    // Search parameters
    int ef = 32;

    // Memory prefetch parameters
    int po = 1;
    int pl = 1;

    // Optimization parameters
    constexpr static int kOptimizePoints = 1000;
    constexpr static int kTryPos = 10;
    constexpr static int kTryPls = 5;
    constexpr static int kTryK = 10;
    int sample_points_num;
    std::vector<float> optimize_queries;
    int graph_po;

    Searcher() = default;

    Searcher(Graph<int>&& graph) : graph(std::move(graph)), graph_po(graph.K / 16) {
    }

    void
    SetData(const float* data, int n, int dim) override {
        this->nb = n;
        this->d = dim;
        quant = Quantizer(d);
        quant.train(data, n);

        sample_points_num = std::min(kOptimizePoints, nb - 1);
        std::vector<int> sample_points(sample_points_num);
        std::mt19937 rng;
        GenRandom(rng, sample_points.data(), sample_points_num, nb);
        optimize_queries.resize(sample_points_num * d);
        for (int i = 0; i < sample_points_num; ++i) {
            memcpy(optimize_queries.data() + i * d, data + sample_points[i] * d, d * sizeof(float));
        }
    }

    void
    SetEf(int ef) override {
        this->ef = ef;
    }

    void
    Optimize(int num_threads = 0) override {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        std::vector<int> try_pos(std::min(kTryPos, graph.K));
        std::vector<int> try_pls(std::min(kTryPls, (int)upper_div(quant.code_size, 64)));
        std::iota(try_pos.begin(), try_pos.end(), 1);
        std::iota(try_pls.begin(), try_pls.end(), 1);
        std::vector<int> dummy_dst(kTryK);
        printf("=============Start optimization=============\n");
        {  // warmup
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
            for (int i = 0; i < sample_points_num; ++i) {
                Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
            }
        }

        float min_ela = std::numeric_limits<float>::max();
        int best_po = 0, best_pl = 0;
        for (auto try_po : try_pos) {
            for (auto try_pl : try_pls) {
                this->po = try_po;
                this->pl = try_pl;
                auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
                for (int i = 0; i < sample_points_num; ++i) {
                    Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
                }

                auto ed = std::chrono::high_resolution_clock::now();
                auto ela = std::chrono::duration<double>(ed - st).count();
                if (ela < min_ela) {
                    min_ela = ela;
                    best_po = try_po;
                    best_pl = try_pl;
                }
            }
        }
        this->po = 1;
        this->pl = 1;
        auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (int i = 0; i < sample_points_num; ++i) {
            Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
        }
        auto ed = std::chrono::high_resolution_clock::now();
        float baseline_ela = std::chrono::duration<double>(ed - st).count();
        printf(
            "settint best po = %d, best pl = %d\n"
            "gaining %.2f%% performance improvement\n============="
            "Done optimization=============\n",
            best_po,
            best_pl,
            100.0 * (baseline_ela / min_ela - 1));
        this->po = best_po;
        this->pl = best_pl;
    }

    void
    Search(const float* q, int k, int* dst) const override {
        auto computer = quant.get_computer(q);
        searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type> pool(
            nb, std::max(k, ef), k);
        graph.initialize_search(pool, computer);
        SearchImpl(pool, computer);
        quant.reorder(pool, q, dst, k);
    }

    template <typename Pool, typename Computer>
    void
    SearchImpl(Pool& pool, const Computer& computer) const {
        while (pool.has_next()) {
            auto u = pool.pop();
            graph.prefetch(u, graph_po);
            for (int i = 0; i < po; ++i) {
                int to = graph.at(u, i);
                computer.prefetch(to, pl);
            }
            for (int i = 0; i < graph.K; ++i) {
                int v = graph.at(u, i);
                if (v == -1) {
                    break;
                }
                if (i + po < graph.K && graph.at(u, i + po) != -1) {
                    int to = graph.at(u, i + po);
                    computer.prefetch(to, pl);
                }
                if (pool.vis.get(v)) {
                    continue;
                }
                pool.vis.set(v);
                auto cur_dist = computer(v);
                pool.insert(v, cur_dist);
            }
        }
    }

    void
    serialize(std::ostream& writer) const override {
        writer.write((char*)&d, sizeof(d));
        writer.write((char*)&nb, sizeof(nb));
        writer.write((char*)&ef, sizeof(ef));
        writer.write((char*)&po, sizeof(po));
        writer.write((char*)&pl, sizeof(pl));
        writer.write((char*)&graph_po, sizeof(graph_po));

        graph.save(writer);

        quant.serialize(writer);
    }

    void
    deserialize(std::istream& reader) override {
        reader.read((char*)&d, sizeof(d));
        reader.read((char*)&nb, sizeof(nb));
        reader.read((char*)&ef, sizeof(ef));
        reader.read((char*)&po, sizeof(po));
        reader.read((char*)&pl, sizeof(pl));
        reader.read((char*)&graph_po, sizeof(graph_po));

        graph.load(reader);

        quant.deserialize(reader);
    }
};

inline std::unique_ptr<SearcherBase>
create_searcher(Graph<int>&& graph, const std::string& metric, int level = 1) {
    auto m = metric_map[metric];
    if (level == 0) {
        if (m == Metric::L2) {
            return std::make_unique<Searcher<glass::FP32Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            return std::make_unique<Searcher<FP32Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (level == 1) {
        if (m == Metric::L2) {
            return std::make_unique<Searcher<SQ8Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            return std::make_unique<Searcher<SQ8Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (level == 2) {
        if (m == Metric::L2) {
            return std::make_unique<Searcher<SQ4Quantizer<Metric::L2>>>(std::move(graph));
        } else if (m == Metric::IP) {
            return std::make_unique<Searcher<SQ4Quantizer<Metric::IP>>>(std::move(graph));
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else {
        printf("Quantizer type not supported\n");
        return nullptr;
    }
}

inline std::unique_ptr<SearcherBase>
create_searcher(const std::string& metric, int level = 1) {
    auto m = metric_map[metric];
    if (level == 0) {
        if (m == Metric::L2) {
            return std::make_unique<Searcher<glass::FP32Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            return std::make_unique<Searcher<FP32Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (level == 1) {
        if (m == Metric::L2) {
            return std::make_unique<Searcher<SQ8Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            return std::make_unique<Searcher<SQ8Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else if (level == 2) {
        if (m == Metric::L2) {
            return std::make_unique<Searcher<SQ4Quantizer<Metric::L2>>>();
        } else if (m == Metric::IP) {
            return std::make_unique<Searcher<SQ4Quantizer<Metric::IP>>>();
        } else {
            printf("Metric not suppported\n");
            return nullptr;
        }
    } else {
        printf("Quantizer type not supported\n");
        return nullptr;
    }
}

}  // namespace glass

}  // namespace vsag