#pragma once

#include <cmath>
#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <regex>
#include <string>
#include <variant>
#include <vector>

#include "zlib.h"

template<typename T>
constexpr auto sqr(const T& x) {
    return x * x;
}

template<typename T>
constexpr auto mod_std(T x, T y) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    if constexpr (std::is_integral_v<T>) {
        const auto ix = int64_t(x), iy = int64_t(y);
        return (ix % iy + iy) % iy;
    } else {
        const auto mod = std::fmod(x, y);
        return mod < 0 ? mod + y : mod;
    }
}

struct slab_shuffle {
    float shift_x, shift_y; // [0.0, 1.0), always float
    bool flip_x, flip_y;    // {false, true}
};

template<size_t MAP_SIZE, typename T, typename U>
auto shuffle_angle(const T x, const T y, const U angle_factor, const slab_shuffle& shuffle) {
    const auto shuffle_x = shuffle.shift_x;
    const auto shuffle_y = shuffle.shift_y;
    const auto flip_x = shuffle.flip_x ? T(-1) : T(1);
    const auto flip_y = shuffle.flip_y ? T(-1) : T(1);
    const auto x_pix = mod_std(flip_x * x * T(angle_factor) + shuffle_x, T(1)) * MAP_SIZE;
    const auto y_pix = mod_std(flip_y * y * T(angle_factor) + shuffle_y, T(1)) * MAP_SIZE;
    if (x_pix < 0 || x_pix > 1.0001 * MAP_SIZE ||
        y_pix < 0 || y_pix > 1.0001 * MAP_SIZE
    ) {
        printf("out of range pix = (%g, %g)\n", x_pix, y_pix);
        std::exit(1);
    }
    const auto x_idx = static_cast<int64_t>(x_pix);
    const auto y_idx = static_cast<int64_t>(y_pix);
    const auto x_frac = x_pix - x_idx;
    const auto y_frac = y_pix - y_idx;
    return std::make_tuple(x_idx, y_idx, x_frac, y_frac);
}

// x, y in [0, 1]
template<typename T, typename U>
auto bilinear_interpolation(const T v00, const T v01, const T v10, const T v11, const U x, const U y) {
    return (1 - x) * ((1 - y) * v00 + y * v01) + x * ((1 - y) * v10 + y * v11);
}

// pick value from array with periodical boundary condition
template<size_t MAP_SIZE, typename Array>
auto pbc_pick(const Array& array, int64_t i, int64_t j) {
    const auto ir = mod_std(i, int64_t(MAP_SIZE));
    const auto jr = mod_std(j, int64_t(MAP_SIZE));
    // if (ir != (i % MAP_SIZE) || jr != (j % MAP_SIZE)) {
    //     printf("i = %d, j = %d, size = %d, ir = %d, jr = %d, imod = %d, jmod = %d\n",
    //         int(i), int(j), int(MAP_SIZE), int(ir), int(jr), int(i % MAP_SIZE), int(j % MAP_SIZE));
    //     std::exit(1);
    // }
    return array[ir * MAP_SIZE + jr];
}

void vectorized_gaussian_inner_loop(
    float* __restrict__ dest, float* __restrict__ dy2, int N,
    const float half_r2_inv, const float dx2
) {
    for (int i = 0; i < N; ++i) {
        dest[i] = expf(half_r2_inv * (dx2 + dy2[i]));
    }
}

// map from [-0.5, 0.5] to [0, UINT32_MAX]
template<typename T>
inline constexpr auto to_uint32_position(T x) {
    static_assert(std::is_floating_point_v<T>);
    constexpr auto factor = int64_t(UINT32_MAX) + 1;
    constexpr auto offset = factor / 2;
    const auto intpos = std::llroundf(x * T(factor));
    return static_cast<uint32_t>(intpos + offset);
}

template<typename T>
void assert_size(const std::vector<T>& x, size_t size, std::string name) {
    if (x.size() != size) {
        printf("%s has a size of %zu; it is expected to be %zu\n",
            name.c_str(), x.size(), size);
        std::exit(1);
    }
}

inline auto list_files(std::string dir, std::string file_pattern) {
    auto files = std::vector<std::string>{};
    auto regex = std::regex(file_pattern);
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const auto filename = entry.path().filename().string();
        if (std::regex_match(filename, regex)) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(begin(files), end(files));
    return files;
}

template<typename T>
void binary_read(std::string file, T* buffer, size_t size) {
    std::ifstream str(file, std::ios::binary);
    if (!str.is_open()) {
        printf("failed to open binary file %s\n", file.c_str());
        std::exit(1);
    }
    str.read(reinterpret_cast<char*>(buffer), size * sizeof(T));
    if (str.fail()) {
        printf("failed in reading binary file %s\n", file.c_str());
        std::exit(1);
    }
}

template<typename T>
auto binary_read(std::string file) -> std::vector<T> {
    static_assert(sizeof(char) == 1u);
    static_assert(std::is_trivially_copyable_v<T>);
    std::ifstream str(file, std::ios::binary);
    if (!str.is_open()) {
        printf("failed to open binary file %s\n", file.c_str());
        std::exit(1);
    }
    const auto file_size = static_cast<size_t>(std::filesystem::file_size(file));

    if (file_size % sizeof(T) != 0u) {
        printf("The size %zu of file %s is not divisible by sizeof(%s).\n", file_size, file.c_str(), typeid(T).name());
        std::exit(1);
    }
    auto data = std::vector<T>(file_size / sizeof(T));
    str.read(reinterpret_cast<char*>(data.data()), file_size);
    return data;
}

template<typename T>
void binary_write(std::string file, const T* buffer, size_t size) {
    std::ofstream str(file, std::ios::binary);
    if (!str.is_open()) {
        printf("failed to open file %s\n", file.c_str());
        std::exit(1);
    }
    str.write(reinterpret_cast<const char*>(buffer), size * sizeof(T));
    str.close();
    if (str.fail()) {
        printf("failed in writing file %s\n", file.c_str());
        std::exit(1);
    }
}

template<typename T> struct is_variant : std::false_type {};

template<typename ...Args>
struct is_variant<std::variant<Args...>> : std::true_type {};

template<typename T>
inline constexpr bool is_variant_v=is_variant<T>::value;

template<typename T>
void tsv_write(std::string file, const std::vector<std::vector<T>>& data) {
    std::ofstream str(file);
    if (!str.is_open()) {
        printf("failed to open file %s\n", file.c_str());
        std::exit(1);
    }
    for (const auto& row : data) {
        const auto row_size = row.size();
        for (size_t i = 0; i < row_size; ++i) {
            str << (i > 0 ? "\t" : "");
            if constexpr (is_variant_v<T>) {
                std::visit([&](const auto& val) { str << val; }, row.at(i));
            } else {
                str << row.at(i);
            }
        }
        str << '\n';
    }
    str.close();
    if (str.fail()) {
        printf("failed in writing file %s\n", file.c_str());
        std::exit(1);
    }
}

inline void _swap_bytes_2(uint8_t* data) {
    std::swap(data[0], data[1]);
}

inline void _swap_bytes_4(uint8_t* data) {
    std::swap(data[0], data[3]);
    std::swap(data[1], data[2]);
}

inline void _swap_bytes_8(uint8_t* data) {
    std::swap(data[0], data[7]);
    std::swap(data[1], data[6]);
    std::swap(data[2], data[5]);
    std::swap(data[3], data[4]);
}

template<typename T>
void swap_bytes(T* data) {
    auto buffer = std::array<uint8_t, sizeof(T)>{};
    std::memcpy(buffer.data(), data, sizeof(T));
    if constexpr (sizeof(T) == 2) {
        _swap_bytes_2(buffer.data());
    } else if constexpr (sizeof(T) == 4) {
        _swap_bytes_4(buffer.data());
    } else if constexpr (sizeof(T) == 8) {
        _swap_bytes_8(buffer.data());
    } else {
        static_assert(sizeof(T) == 2, "cannot do byte swap");
    }
    std::memcpy(data, buffer.data(), sizeof(T));
}

inline auto estimate_map_size(size_t size) {
    const auto N = static_cast<size_t>(std::llround(std::sqrt(size)));
    if (size != N * N) {
        printf("estimate_map_size(size = %zu) failed, N = %zu\n", size, N);
        std::exit(1);
    }
    return N;
}

using position_t = std::array<uint32_t, 3u>;

// the distance between two coordinates, in the units of simulation box size
inline double distance(const position_t& p1, const position_t& p2) {
    constexpr auto factor = double(1.0) / ((long long)(std::numeric_limits<uint32_t>::max()) + 1);
    auto x1 = factor * p1[0],  y1 = factor * p1[1],  z1 = factor * p1[2];
    auto x2 = factor * p2[0],  y2 = factor * p2[1],  z2 = factor * p2[2];
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    if (z1 > z2) std::swap(z1, z2);
    return std::sqrt(
        sqr(std::min(x2 - x1, 1.0 + (x1 - x2))) +
        sqr(std::min(y2 - y1, 1.0 + (y1 - y2))) +
        sqr(std::min(z2 - z1, 1.0 + (z1 - z2))));
}

struct tipsy_particle_t {
    float mass{};
    std::array<float, 3> pos{};
    std::array<float, 3> vel{};
    float eps{};
    float phi{};
};

inline auto read_tipsy(std::string bin_file) {
    auto num_particles = uint32_t{};
    auto time = double{};
    auto positions = std::vector<position_t>{};

    constexpr auto tipsy_buffer_size = size_t(256*1024*1024);
    auto tipsy_buffer = std::vector<tipsy_particle_t>(tipsy_buffer_size);

    auto str = std::fopen(bin_file.c_str(), "rb");
    if (!str) {
        printf("Failed to open file %s\n", bin_file.c_str());
        std::exit(1);
    }
    printf("reading TIPSY file...\n");
    [[maybe_unused]] auto _0 = std::fread(&time, 8u, 1u, str);
    [[maybe_unused]] auto _1 = std::fread(&num_particles, 4u, 1u, str);
    std::fseek(str, 20, SEEK_CUR);
    swap_bytes(&time);
    swap_bytes(&num_particles);
    positions.resize(num_particles);

    printf("  time = %g, N = %d\n", time, int(num_particles));
    
    for (size_t i = 0; i < num_particles; i += tipsy_buffer_size) {
        const auto num_read = std::min(tipsy_buffer_size, num_particles - i);
        printf("Thread %d starts to read %zu particles from index %zu.\n", omp_get_thread_num(), num_read, i);
        const auto num_actual = std::fread(
            tipsy_buffer.data(), sizeof(tipsy_particle_t), num_read, str);
        printf("Thread %d finishes reading, %zu particles read.\n", omp_get_thread_num(), num_actual);
        if (num_read != num_actual) {
            printf("Expecting %zu particles, only %zu are read.\n", num_read, num_actual);
            std::exit(1);
        }
        const int num_threads = std::min(omp_get_max_threads(), 12); // 12 threads max
        printf("Extracting particle positions with %d threads...\n", num_threads);
        #pragma omp parallel for num_threads(num_threads)
        for (size_t j = 0; j < num_read; ++j) {
            auto& pos = tipsy_buffer[j].pos;
            auto& outpos = positions.at(i + j);
            swap_bytes(&pos[0]);
            swap_bytes(&pos[1]);
            swap_bytes(&pos[2]);
            outpos[0] = to_uint32_position(pos[0]);
            outpos[1] = to_uint32_position(pos[1]);
            outpos[2] = to_uint32_position(pos[2]);
        }
    }

    std::fclose(str);

    printf("\n");
    printf("position[0] = (%u,%u,%u)\n", positions[0][0], positions[0][1], positions[0][2]);
    printf("position[n-1] = (%u,%u,%u)\n",
        positions.back()[0], positions.back()[1], positions.back()[2]);
    return positions;
}

inline auto get_halo_files(std::string snapshot_file, std::string halos_dir) {
    constexpr auto SNAPSHOT_ID_LENGTH = size_t(5);
    if (snapshot_file.size() < SNAPSHOT_ID_LENGTH) {
        printf("Cannot extract the snapshot ID from file %s\n", snapshot_file.c_str());
    }
    std::string id = snapshot_file.substr(snapshot_file.size() - SNAPSHOT_ID_LENGTH, SNAPSHOT_ID_LENGTH);
    if (!std::all_of(begin(id), end(id), [](char c) { return '0' <= c && c <= '9'; })) {
        printf("Cannot extract the snapshot ID from file %s\n", snapshot_file.c_str());
    }
    return list_files(halos_dir, ".+" + id + "\\..+\\.bin");
}

// The following declaration is taken from
// https://bitbucket.org/gfcstanford/rockstar/src/main/io/io_internal.h
#define BINARY_HEADER_SIZE 256
#define VERSION_MAX_SIZE 12
struct rockstar_binary_output_header_t {
  uint64_t magic;
  int64_t snap, chunk;
  float scale, Om, Ol, h0;
  float bounds[6];
  int64_t num_halos, num_particles;
  float box_size, particle_mass;
  int64_t particle_type;
  int32_t format_revision;
  char rockstar_version[VERSION_MAX_SIZE];
  char unused[BINARY_HEADER_SIZE - (sizeof(char)*VERSION_MAX_SIZE) - (sizeof(float)*12) - sizeof(int32_t) - (sizeof(int64_t)*6)];
};
#undef VERSION_MAX_SIZE
#undef BINARY_HEADER_SIZE

// The following declaration is taken from
// https://bitbucket.org/gfcstanford/rockstar/src/main/halo.h
struct rockstar_halo_t {
  int64_t id;
  float pos[6], corevel[3], bulkvel[3];
  float m, r, child_r, vmax_r, mgrav, vmax, rvmax, rs, klypin_rs, vrms,
    J[3], energy, spin, alt_m[4], Xoff, Voff, b_to_a, c_to_a, A[3],
    b_to_a2, c_to_a2, A2[3],
    bullock_spin, kin_to_pot, m_pe_b, m_pe_d, halfmass_radius;
  int64_t num_p, num_child_particles, p_start, desc, flags, n_core;
  float min_pos_err, min_vel_err, min_bulkvel_err;
};

struct halo_t {
    position_t pos{};  // center of the halo
    float r_vir{};     // virial radius
    float c{};         // concentration parameter
    uint32_t weight{}; // number of halo particles within the virial radius
    float mass{};      // weight * particle mass [M_sun/h]

    halo_t() = default;
    halo_t(position_t pos, float r_vir, float c) : pos{pos}, r_vir{r_vir}, c{c} {}
};

template<typename T>
struct summary_statistics_t {
    std::string name;
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::min();
    double sum{}, sum2{};
    size_t count{};

    summary_statistics_t(std::string name) : name{name} {};

    void operator+=(T other) {
        min = std::min(min, other);
        max = std::max(max, other);
        sum += double(other);
        sum2 += double(other) * double(other);
        ++count;
    }

    void print() {
        std::cout << "    Summary of " << name << ": "
            << "count = " << count << ", "
            << "range = (" << min << ", " << max << "), "
            << "mean = " << sum / double(count) << ", "
            << "stddev = " << std::sqrt(sum2 / double(count) - sqr(sum / double(count))) << "\n";
    }
};


constexpr auto ZLIB_CHUNK_SIZE = size_t(1) << 20;
constexpr auto ZLIB_COMPRESSION_LEVEL = 5;
using ZLIB_BUFFER_CHAR = unsigned char;

template<typename T>
inline auto zlib_compress(T* data, size_t size) {
    const auto start_time = omp_get_wtime();
    static_assert(sizeof(ZLIB_BUFFER_CHAR) == 1u);
    const auto num_bytes = sizeof(T) * size;
    auto out_buffer = std::vector<ZLIB_BUFFER_CHAR>(num_bytes);

    auto src_iter = reinterpret_cast<ZLIB_BUFFER_CHAR*>(data);
    auto src_end = src_iter + num_bytes;
    auto dst_iter = out_buffer.data();
    auto dst_end = dst_iter + num_bytes;

    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    const auto init_ret = deflateInit(&strm, ZLIB_COMPRESSION_LEVEL);
    if (init_ret != Z_OK) {
        printf("zlib initialization failed.\n");
        std::exit(1);
    }

    /* compress until end of file */
    auto flush = Z_NULL;
    do {
        const auto size_available = std::min(size_t(src_end - src_iter), ZLIB_CHUNK_SIZE);
        strm.avail_in = size_available;
        flush = (size_available < ZLIB_CHUNK_SIZE) ? Z_FINISH : Z_NO_FLUSH;
        strm.next_in = src_iter;
        src_iter += size_available;

        do {
            if (dst_iter >= dst_end) {
                printf("zlib out buffer is too small in compression.\n");
                std::exit(1);
            }
            strm.avail_out = std::min(size_t(dst_end - dst_iter), ZLIB_CHUNK_SIZE);
            strm.next_out = dst_iter;
            deflate(&strm, flush);
            dst_iter += ZLIB_CHUNK_SIZE - strm.avail_out;
        } while (strm.avail_out == 0);
        fflush(stdout);
    } while (flush != Z_FINISH);

    deflateEnd(&strm);
    const auto out_size = size_t(dst_iter - (dst_end - num_bytes));
    out_buffer.resize(out_size);
    printf("Compression took %g seconds, compression rate = %g%%.\n",
        omp_get_wtime() - start_time, 100.0 * (out_size / double(num_bytes)));
    return out_buffer;
}

template<typename T>
auto zlib_decompress(std::vector<ZLIB_BUFFER_CHAR> in_buffer) {
    // const auto start_time = omp_get_wtime();
    auto src_iter = in_buffer.data();
    auto src_end = src_iter + in_buffer.size();
    auto out_buffer = std::vector<T>{};
    auto dst_begin = (ZLIB_BUFFER_CHAR*)nullptr;
    auto dst_iter = (ZLIB_BUFFER_CHAR*)nullptr;
    auto dst_end = (ZLIB_BUFFER_CHAR*)nullptr;

    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    const auto init_ret = inflateInit(&strm);
    if (init_ret != Z_OK) {
        printf("zlib initialization failed.\n");
        std::exit(1);
    }

    auto ret = Z_NULL;
    do {
        const auto size_available = std::min(size_t(src_end - src_iter), ZLIB_CHUNK_SIZE);
        if (size_available == 0)
            break;
        strm.avail_in = size_available;
        strm.next_in = src_iter;
        src_iter += size_available;

        do {
            if (dst_end - dst_iter < ptrdiff_t(ZLIB_CHUNK_SIZE)) {
                // not enough buffer
                out_buffer.resize(out_buffer.size() + ((ZLIB_CHUNK_SIZE - 1u) / sizeof(T) + 1u));
                const auto diff = reinterpret_cast<ZLIB_BUFFER_CHAR*>(&(*begin(out_buffer))) - dst_begin;
                dst_begin += diff;
                dst_iter += diff;
                dst_end = reinterpret_cast<ZLIB_BUFFER_CHAR*>(&(*end(out_buffer)));
            }
            strm.avail_out = ZLIB_CHUNK_SIZE;
            strm.next_out = dst_iter;
            ret = inflate(&strm, Z_NO_FLUSH);
            switch (ret) {
            case Z_NEED_DICT:
                ret = Z_DATA_ERROR;
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                inflateEnd(&strm);
                printf("zlib out buffer is too small in compression.\n");
                std::exit(1);
            }
            dst_iter += ZLIB_CHUNK_SIZE - strm.avail_out;
        } while (strm.avail_out == 0);
    } while (ret != Z_STREAM_END);
    inflateEnd(&strm);

    if ((dst_iter - dst_begin) % sizeof(T)) {
        printf("number of bytes after decompression is not divisible by sizeof(T).\n");
        std::exit(1);
    }
    out_buffer.resize((dst_iter - dst_begin) / sizeof(T));
    // printf("Decompression took %g seconds.\n", omp_get_wtime() - start_time);
    return out_buffer;
}

template<typename T>
void print_vector(const std::vector<T>& vec, std::string name = "") {
    if (name != "") {
        std::cout << name << ": ";
    }
    std::cout << "{";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << (i > 0 ? ", " : "") << vec[i];
    }
    std::cout << "}\n";
}
