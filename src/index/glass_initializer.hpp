#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "glass_memory.hpp"

namespace vsag {
namespace glass {

struct GraphInitializer {
    int N, K;
    int ep;
    std::vector<int> levels;
    std::vector<std::vector<int, align_alloc<int>>> lists;
    GraphInitializer() = default;
    ~GraphInitializer() = default;

    explicit GraphInitializer(int n, int K = 0) : N(n), K(K), levels(n), lists(n) {
    }

    GraphInitializer(const GraphInitializer& rhs) = default;

    int
    at(int level, int u, int i) const {
        return lists[u][(level - 1) * K + i];
    }

    int&
    at(int level, int u, int i) {
        return lists[u][(level - 1) * K + i];
    }

    const int*
    edges(int level, int u) const {
        return lists[u].data() + (level - 1) * K;
    }

    int*
    edges(int level, int u) {
        return lists[u].data() + (level - 1) * K;
    }

    template <typename Pool, typename Computer>
    void
    initialize(Pool& pool, const Computer& computer) const {
        int u = ep;
        auto cur_dist = computer(u);
        for (int level = levels[u]; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                const int* list = edges(level, u);
                for (int i = 0; i < K && list[i] != -1; ++i) {
                    int v = list[i];
                    auto dist = computer(v);
                    if (dist < cur_dist) {
                        cur_dist = dist;
                        u = v;
                        changed = true;
                    }
                }
            }
        }
        pool.insert(u, cur_dist);
        pool.vis.set(u);
    }

    void
    load(std::ifstream& reader) {
        reader.read((char*)&N, 4);
        reader.read((char*)&K, 4);
        reader.read((char*)&ep, 4);
        for (int i = 0; i < N; ++i) {
            int cur;
            reader.read((char*)&cur, 4);
            levels[i] = cur / K;
            if (cur > 0) {
                lists[i].assign(cur, -1);
                reader.read((char*)lists[i].data(), cur * 4);
            }
        }
    }

    void
    load(std::istream& reader) {
        auto start = std::chrono::high_resolution_clock::now();
        reader.read((char*)&N, 4);
        reader.read((char*)&K, 4);
        reader.read((char*)&ep, 4);
        for (int i = 0; i < N; ++i) {
            int cur;
            reader.read((char*)&cur, 4);
            levels[i] = cur / K;
            if (cur > 0) {
                lists[i].assign(cur, -1);
                reader.read((char*)lists[i].data(), cur * 4);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "glass initializer deserialize: " << duration.count() << " ms" << std::endl;
    }

    void
    save(std::ofstream& writer) const {
        writer.write((char*)&N, 4);
        writer.write((char*)&K, 4);
        writer.write((char*)&ep, 4);
        for (int i = 0; i < N; ++i) {
            int cur = levels[i] * K;
            writer.write((char*)&cur, 4);
            if (cur > 0) {
                writer.write((char*)lists[i].data(), cur * 4);
            }
        }
    }

    void
    save(std::ostream& writer) const {
        auto start = std::chrono::high_resolution_clock::now();
        writer.write((char*)&N, 4);
        writer.write((char*)&K, 4);
        writer.write((char*)&ep, 4);
        for (int i = 0; i < N; ++i) {
            int cur = levels[i] * K;
            writer.write((char*)&cur, 4);
            if (cur > 0) {
                writer.write((char*)lists[i].data(), cur * 4);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "glass initializer serialize: " << duration.count() << " ms" << std::endl;
    }
};

}  // namespace glass
}  // namespace vsag