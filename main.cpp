#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <array>
#include <complex>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>
#include <set>
#include <string>

#include <fftw3.h>
#include <omp.h>

#include "utils.h"

// calculations use single-precision numbers
using FLOAT = float;
using COMPLEX = std::complex<FLOAT>;

constexpr auto DEGREE = M_PI / 180.0;

// set parameters below
constexpr size_t MAP_SIZE = size_t(208);
constexpr size_t TOMO_MAP_SIZE = size_t(104);
constexpr FLOAT  FIELD_SIZE_DEG = FLOAT(3.0);
constexpr FLOAT  FIELD_SIZE_RADIAN = FLOAT(FIELD_SIZE_DEG * DEGREE);
constexpr size_t POTENTIAL_SIZE_LEVEL = size_t(13);
constexpr size_t POTENTIAL_SIZE = size_t(1) << POTENTIAL_SIZE_LEVEL;
constexpr FLOAT  SIMULATION_SIZE_MPCH = FLOAT(500);
constexpr size_t SIMULATION_SIDE_LENGTH = size_t(1024);

// S16A parameters
constexpr auto S16A_MEAN_RMS_E = FLOAT(0.4);
constexpr auto S16A_MEAN_M = FLOAT(-0.1132);
constexpr auto S16A_NUM_Z_BIN = size_t(4);
constexpr auto S16A_NUM_Z_BIN_PAIRS = size_t(10);
constexpr size_t S16A_Z_BIN_PAIRS[S16A_NUM_Z_BIN_PAIRS][2] = {
    {0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}, {1, 2}, {1, 3}, {2, 2}, {2, 3}, {3, 3}
};
constexpr FLOAT S16A_Z_BIN_ERROR[S16A_NUM_Z_BIN] = { // See Hikage et al. (2020)
    FLOAT(0.0285), FLOAT(0.0135), FLOAT(0.0383), FLOAT(0.0376)
};

// cosmological parameters
constexpr auto OMEGA_B = FLOAT(0.0493);
constexpr auto PARTICLE_MASS_MSUNH = [] (double omega_m) {
    constexpr auto res = double(SIMULATION_SIZE_MPCH) / SIMULATION_SIDE_LENGTH;
    return omega_m * 2.7753312e11 * res * res * res;
};

// halo parameters
constexpr auto HALO_CONCENTRATION_MIN = FLOAT(1);
constexpr auto HALO_CONCENTRATION_MAX = FLOAT(32);

// BCM parameters
constexpr auto SUPER_SAMPLING_FACTOR = size_t(16);
constexpr auto LINE_OF_SIGHT_RESOLUTION = size_t(100);
constexpr auto RADIAL_RESOLUTION = size_t(4000);
constexpr auto BCM_DERIVATIVE_DR = double(0.01);
constexpr auto FN_HALO_RVIR_PIXEL = [](size_t i) { return std::pow(10.0, 0.02 * i + 0.5); };
constexpr auto HALO_RVIR_COUNT = size_t(71);
constexpr auto FN_HALO_C = [](size_t i) { return std::pow(10.0, 0.05 * i + 0.0); };
constexpr auto HALO_C_COUNT = size_t(31);

struct real_galaxy_measure {
    FLOAT x{}, y{};
    FLOAT e1{}, e2{};
    FLOAT w{}, m{}, m_bias{}, c1{}, c2{};
    FLOAT sigma_e{}, rms_e{};
    FLOAT z_best{}, z_rt{};
    int z_bin{};
};

struct sim_galaxy_measure {
    FLOAT kappa, gamma1, gamma2;
};

struct bcm_parameters {
    double Mc;   // Halo mass scale for retaining half of the total gas                  10^[12,16] h^-1 M_sun
    double M1_0; // Characteristic halo mass for a galaxy mass fraction epsilon = 0.023  10^[10,13] h^-1 M_sun
    double eta;  // Maximum distance of gas ejection in terms of the halo escape radius  10^[-0.7,0.5]
    double beta; // Slope of the gas fraction as a function of halo mass                 10^[-1.0,0.5]
};

struct slab_entry_t {
    std::string filename;
    double omega_m;
    double hubble_0;
    double redshift;
    double thickness_mpch;
    double distance_mpch;
};

void get_slab_randomizations(
    uint32_t seed,
    std::vector<uint32_t>& slab_choices,     // which slab to pick
    std::vector<slab_shuffle>& slab_shuffles // how to shift and flip each slab
) {
    auto engine = std::mt19937(seed);
    if (engine.min() != 0 || engine.max() != UINT32_MAX) {
        printf("ERROR: The random engine does not generate random integers in [0, UINT32_MAX].\n");
        std::exit(1);
    }
    for (auto& choice : slab_choices) {
        choice = engine();
    }
    for (auto& shuffle : slab_shuffles) {
        shuffle.shift_x = engine() / FLOAT(1ull << 32);
        shuffle.shift_y = engine() / FLOAT(1ull << 32);
        shuffle.flip_x = bool(engine() & 1u);
        shuffle.flip_y = bool(engine() & 1u);
    }
}

auto read_real_galaxy_measures(std::string text_file) {
    auto catalog = std::vector<real_galaxy_measure>{};
    std::ifstream str(text_file);
    if (!str.is_open()) {
        printf("ERROR: Failed to open file %s\n", text_file.c_str());
        std::exit(1);
    }
    std::string buffer;
    std::getline(str, buffer, '\n');
    if (buffer != "id,w,m,c1,c2,sigma_e,rms_e,e1,e2,psf_m11,psf_m22,psf_m12,ra,dec,z_best,z_bin,z_rt,field,x,y,m_bias") {
        printf("ERROR: Unexpected csv header %s\n", buffer.c_str());
        std::exit(1);
    }
    constexpr auto NUM_COLUMNS = size_t(21);
    while (std::getline(str, buffer, '\n')) {
        std::array<FLOAT, NUM_COLUMNS> vals{};
        auto begin = buffer.c_str();
        for (size_t i = 0; i < NUM_COLUMNS; ++i) {
            char* end = nullptr;
            vals[i] = std::strtof(begin, &end);
            begin = end + 1;
        }

        auto g = real_galaxy_measure{};
        g.w       = vals[1];
        g.m       = vals[2];
        g.c1      = vals[3];
        g.c2      = vals[4];
        g.sigma_e = vals[5];
        g.rms_e   = vals[6];
        g.e1      = vals[7];
        g.e2      = vals[8];
        g.z_best  = vals[14];
        g.z_bin   = static_cast<int>(vals[15]);
        g.z_rt    = vals[16];
        g.x       = vals[18];
        g.y       = vals[19];
        g.m_bias  = vals[20];

        if (g.x == 0.0 || g.y == 0.0 || std::abs(g.x) > 1.5 || std::abs(g.y) > 1.5 || g.m_bias == 0.0) {
            printf("ERROR: Error in reading %s\n", text_file.c_str());
            printf("ERROR: Line number = %zu, (g.x, g.y) = (%g, %g)\n", catalog.size() + 1, g.x, g.y);
            exit(1);
        }
        catalog.push_back(g);
    }
    return catalog;
}

auto read_slab_config(std::string text_file) {
    auto config = std::vector<slab_entry_t>{};
    std::ifstream str(text_file);
    if (!str.is_open()) {
        printf("ERROR: Failed to open file %s\n", text_file.c_str());
        std::exit(1);
    }
    std::string buffer;
    std::getline(str, buffer, '\n');
    if (buffer != "filename\tomega_m\thubble_0\tredshift\tthickness\tdistance") {
        printf("ERROR: Unexpected csv header %s\n", buffer.c_str());
        std::exit(1);
    }
    while (std::getline(str, buffer, '\n')) {
        if (std::count(begin(buffer), end(buffer), '\t') == 5) {
            auto entry = slab_entry_t{};
            double* const pointer_to_fields[] = {
                nullptr, &entry.omega_m, &entry.hubble_0,
                &entry.redshift, &entry.thickness_mpch, &entry.distance_mpch
            };
            auto field_begin = begin(buffer);
            for (size_t i = 0; i < 6; ++i) {
                auto field_end = std::find(field_begin, end(buffer), '\t');
                if (i == 0) {
                    entry.filename = std::string(field_begin, field_end);
                } else {
                    *pointer_to_fields[i] = std::stod(std::string(field_begin, field_end));
                }
                field_begin = field_end + 1;
            }
            if (entry.omega_m <= OMEGA_B) {
                printf("ERROR: omega_m < omega_b, omega_m = %g, omega_b = %g\n", entry.omega_m, OMEGA_B);
                std::exit(1);
            }
            config.push_back(entry);
        }
    }
    return config;
}

auto read_bcm_sobol_sequence(std::string text_file) {
    auto bcm_list = std::vector<bcm_parameters>{};
    std::ifstream str(text_file);
    if (!str.is_open()) {
        printf("ERROR: Failed to open file %s\n", text_file.c_str());
        std::exit(1);
    }
    std::string buffer;
    std::getline(str, buffer, '\n');
    if (buffer != "Mc,M1_0,eta,beta") {
        printf("ERROR: Unexpected csv header %s\n", buffer.c_str());
        std::exit(1);
    }
    constexpr auto NUM_COLUMNS = size_t(4);
    while (std::getline(str, buffer, '\n')) {
        if (std::count(begin(buffer), end(buffer), ',') == NUM_COLUMNS - 1) {
            auto values = std::array<double, NUM_COLUMNS>{};
            auto field_begin = begin(buffer);
            for (size_t i = 0; i < NUM_COLUMNS; ++i) {
                auto field_end = std::find(field_begin, end(buffer), ',');
                values[i] = std::stod(std::string(field_begin, field_end));
                field_begin = field_end + 1;
            }
            bcm_list.push_back(bcm_parameters{values[0], values[1], values[2], values[3]});
        }
    }
    auto bcm_members = std::vector{
        &bcm_parameters::Mc, &bcm_parameters::M1_0, &bcm_parameters::eta, &bcm_parameters::beta
    };
    auto bcm_names = std::vector{"Mc", "M1_0", "eta", "beta"};

    printf("BCM parameter size: %d, ranges: ", int(bcm_list.size()));
    for (size_t i = 0; i < 4; ++i) {
        const auto [min, max] = std::minmax_element(begin(bcm_list), end(bcm_list),
            [&](auto const& a, auto const& b) { return a.*bcm_members[i] < b.*bcm_members[i]; }
        );
        printf("%s = [%g, %g], ", bcm_names[i], (*min).*bcm_members[i], (*max).*bcm_members[i]);
    }
    printf("\n");
    return bcm_list;
}

// linear interpolation
auto get_redshift_to_distance(const std::vector<slab_entry_t>& config) {
    auto redshifts = std::vector{FLOAT(0)};
    auto distances = std::vector{FLOAT(0)};
    for (const auto& entry : config) {
        redshifts.push_back(entry.redshift);
        distances.push_back(entry.distance_mpch);
    }
    return [=] (FLOAT z) {
        if (z < redshifts.front() || z > redshifts.back()) {

            printf("ERROR: Redshift %g is outside the range [%g,%g] specified by the slab config\n",
                z, redshifts.front(), redshifts.back());
            std::exit(1);
        }
        auto i = std::lower_bound(begin(redshifts), end(redshifts), z) - begin(redshifts);
        return z == redshifts[i] ? distances[i] :
            distances[i - 1] + ((z - redshifts[i - 1])
                / (redshifts[i] - redshifts[i - 1]))
                * (distances[i] - distances[i - 1]);
    };
}

auto read_mask(std::string bin_file) {
    auto mask = binary_read<FLOAT>(bin_file);
    assert_size(mask, MAP_SIZE * MAP_SIZE, "galaxy mask");
    return mask;
}

inline auto read_halos(const std::vector<std::string>& bin_files,
    const std::vector<position_t>& particle_pos,
    const double min_halo_mass_msunh)
{
    auto all_halos = std::vector<halo_t>{};
    auto all_halo_particles = std::vector<std::vector<uint32_t>>{};

    if (bin_files.size() == 0) {
        printf("There is no halo files to be read.\n");
        return std::make_pair(std::move(all_halos), std::move(all_halo_particles));
    }
    printf("Reading halos (>= %g Msun/h) from %zu files: %s, ...\n",
        min_halo_mass_msunh, bin_files.size(), bin_files[0].c_str());

    auto stat_m = summary_statistics_t<double>("m_vir");
    auto stat_r = summary_statistics_t<double>("r_vir");
    auto stat_rs = summary_statistics_t<double>("r_s");
    auto stat_num_p = summary_statistics_t<double>("num_p");
    auto stat_num_vir = summary_statistics_t<double>("num_vir");
    auto stat_id = summary_statistics_t<int64_t>("particle_id");
    auto stat_x = summary_statistics_t<double>("pos_x");
    auto stat_c = summary_statistics_t<double>("v_vir/r_s");
    
    for (const auto& file : bin_files) {
        auto str = std::fopen(file.c_str(), "rb");
        auto header = rockstar_binary_output_header_t{};
        if (std::fread(&header, sizeof(header), 1, str) != 1u) {
            printf("ERROR: Failed to read the header from %s\n", file.c_str());
            std::exit(1);
        }

        auto halos = std::vector<rockstar_halo_t>(header.num_halos);
        auto particles = std::vector<int64_t>(header.num_particles);
        if (std::fread(halos.data(), sizeof(halos[0]), halos.size(), str) != halos.size()) {
            printf("ERROR: Failed to read halos from %s\n", file.c_str());
            std::exit(1);
        }
        if (std::fread(particles.data(), sizeof(particles[0]), particles.size(), str) != particles.size()) {
            printf("ERROR: Failed to read halo particle IDs from %s\n", file.c_str());
            std::exit(1);
        }

        for (auto& p : particles) {
            stat_id += p;
        }

        auto particles_begin = begin(particles);
        for (const auto& halo : halos) {
            if (double(halo.num_p) * header.particle_mass >= min_halo_mass_msunh) {
                // only record halos with mass greater than the threshold
                const auto halo_center_pos = position_t{
                    to_uint32_position(halo.pos[0] / header.box_size - 0.5),
                    to_uint32_position(halo.pos[1] / header.box_size - 0.5),
                    to_uint32_position(halo.pos[2] / header.box_size - 0.5)
                };
                auto halo_entry = halo_t(halo_center_pos, halo.r,
                    halo.r >= halo.rs ? std::min(halo.r / halo.rs, HALO_CONCENTRATION_MAX) : 0);
                auto particles_end = particles_begin + halo.num_p;
                if (particles_end > particles.end()) {
                    printf("ERROR: There are not enough particles to be read in halo file %s\n", file.c_str());
                    std::exit(1);
                }

                auto rescaled_halo_r = halo.r / (1000.0 * header.box_size);
                auto particles_entry = std::vector<uint32_t>{};
                for (; particles_begin != particles_end; ++particles_begin) {
                    if (distance(halo_center_pos, particle_pos[*particles_begin]) < rescaled_halo_r) {
                        particles_entry.push_back(*particles_begin);
                    }
                }
                halo_entry.weight = particles_entry.size();
                halo_entry.mass = halo_entry.weight * header.particle_mass;

                if ((halo_entry.weight * 2 < halo.num_p) && (halo_entry.c > 0)) {
                    // the virial radius of the halo should include at least half of its particles
                    // the concentration parameter of the halo should be valid
                    continue;
                }

                stat_x += halo.pos[0] / header.box_size;
                stat_m += halo.m;
                stat_r += halo_entry.r_vir;
                stat_num_p += halo.num_p;
                stat_c += halo_entry.c;
                stat_num_vir += halo_entry.weight;

                all_halos.push_back(std::move(halo_entry));
                all_halo_particles.push_back(std::move(particles_entry));
            } else {
                particles_begin += halo.num_p;
            }
        }
        if (particles_begin != particles.end()) {
            printf("ERROR: There are more particles than enough in halo file %s\n", file.c_str());
            std::exit(1);
        }

        printf("box_size = %g, num_halos = %ld, num_particles = %ld, particle_mass = %g\n",
            header.box_size, header.num_halos, header.num_particles, header.particle_mass);
    }

    // remove duplicate halos
    auto halo_set = std::set<std::pair<position_t, uint32_t>>{};
    for (size_t i = 0; i < all_halos.size(); ++i) {
        const auto key = std::make_pair(all_halos[i].pos, all_halos[i].weight);
        auto iter = halo_set.find(key);
        if (iter == halo_set.end()) {
            halo_set.insert(key);
        } else {
            // this halo is duplicate with another halo; remove the current one
            printf("A halo [pos=(%zu,%zu,%zu), weight=%zu] is removed due to duplication.\n",
                size_t(key.first[0]), size_t(key.first[1]), size_t(key.first[2]), size_t(key.second));
            all_halos.erase(begin(all_halos) + i);
            all_halo_particles.erase(begin(all_halo_particles) + i);
            --i;
        }
    }
        
    stat_x.print();
    stat_m.print();
    stat_r.print();
    stat_rs.print();
    stat_num_p.print();
    stat_num_vir.print();
    stat_id.print();
    stat_c.print();
    return std::make_pair(std::move(all_halos), std::move(all_halo_particles));
}

void dft_2d_inplace(std::complex<FLOAT>* data, size_t N0, size_t N1, int sign) {
    // fftw_plan / fftwf_plan is not thread-safe
    static std::mutex mut;

    if constexpr (sizeof(FLOAT) == 8u) {
        auto plan = fftw_plan{};
        {
            auto lock = std::lock_guard{mut};
            plan = fftw_plan_dft_2d(int(N0), int(N1),
                reinterpret_cast<fftw_complex*>(data),
                reinterpret_cast<fftw_complex*>(data),
                sign, FFTW_ESTIMATE);
        }
        if (!plan) {
            printf("ERROR: dft_2d_inplace failed.\n");
            std::exit(1);
        }
        fftw_execute(plan);
        {
            auto lock = std::lock_guard{mut};
            fftw_destroy_plan(plan);
        }
    } else {
        auto plan = fftwf_plan{};
        {
            auto lock = std::lock_guard{mut};
            plan = fftwf_plan_dft_2d(int(N0), int(N1),
                reinterpret_cast<fftwf_complex*>(data),
                reinterpret_cast<fftwf_complex*>(data),
                sign, FFTW_ESTIMATE);
        }
        if (!plan) {
            printf("ERROR: dft_2d_inplace failed.\n");
            std::exit(1);
        }
        fftwf_execute(plan);
        {
            auto lock = std::lock_guard{mut};
            fftwf_destroy_plan(plan);
        }
    }
    const auto scale = FLOAT(1.0 / std::sqrt(N0 * N1));
    std::transform(data, data + (N0 * N1), data, [=](auto x) { return scale * x; });
}

void gaussian_blur(std::vector<COMPLEX>& map, const FLOAT radius) {
    const auto N = estimate_map_size(map.size());
    // printf("gaussian_blur(...) map N = %zu, radius = %lf\n", N, radius);

    thread_local auto kernel_fft = std::vector<COMPLEX>{};
    thread_local auto cached_N = size_t{};
    thread_local auto cached_radius = FLOAT{};

    if (cached_N != N || cached_radius != radius) {
        cached_N = N;
        cached_radius = radius;
        kernel_fft.clear();
        kernel_fft.resize(N * N);
        const auto norm = FLOAT(1.0 / (2 * M_PI * radius * radius));
        const auto r4 = std::llround(std::min(std::ceil(4 * radius), FLOAT(0.4) * N));
        for (auto di = -r4; di <= r4; ++di) {
            for (auto dj = -r4; dj <= r4; ++dj) {
                const auto i = di + size_t(di < 0) * N;
                const auto j = dj + size_t(dj < 0) * N;
                kernel_fft[i * N + j] =
                    norm * std::exp(-(di * di + dj * dj) / (2 * radius * radius));
            }
        }
        dft_2d_inplace(kernel_fft.data(), N, N, FFTW_FORWARD);
        for (auto& val : kernel_fft) {
            val = FLOAT(N) * val.real();
        }
    }

    dft_2d_inplace(map.data(), N, N, FFTW_FORWARD);
    std::transform(begin(map), end(map), begin(kernel_fft),
        begin(map), std::multiplies<>{});
    dft_2d_inplace(map.data(), N, N, FFTW_BACKWARD);
}

void gaussian_blur(std::vector<FLOAT>& map, const FLOAT radius) {
    std::vector<COMPLEX> copy(map.size());
    std::copy(begin(map), end(map), begin(copy));
    gaussian_blur(copy, radius);
    std::transform(begin(copy), end(copy), begin(map),
        [](const auto& x) { return x.real(); });
}

template<typename T, typename U, typename V>
void put_patch(
    T* back, size_t BM, size_t BN,
    U* front, size_t FM, size_t FN,
    int64_t BM_signed_offset, int64_t BN_signed_offset, V alpha
) {
    static_assert(std::is_convertible_v<decltype(U{} *  V{}), T>);
    if (FM > BM || FN > BN) {
        printf("foreground image should be smaller than background image.\n");
        std::exit(0);
    }

    const auto BM_offset = mod_std(BM_signed_offset, int64_t(BM));
    const auto BN_offset = mod_std(BN_signed_offset, int64_t(BN));
    const auto FM_split = std::min(BM - BM_offset, FM);
    const auto FN_split = std::min(BN - BN_offset, FN);

    for (size_t fm = 0, bm = BM_offset; fm < FM_split; ++fm, ++bm) {
        const auto f_base = front + fm * FN;
        const auto b_base = back + bm * BN;
        for (size_t fn = 0, bn = BN_offset; fn < FN_split; ++fn, ++bn) {
            b_base[bn] += alpha * f_base[fn];
        }
        for (size_t fn = FN_split, bn = 0; fn < FN; ++fn, ++bn) {
            b_base[bn] += alpha * f_base[fn];
        }
    }
    for (size_t fm = FM_split, bm = 0; fm < FM; ++fm, ++bm) {
        const auto f_base = front + fm * FN;
        const auto b_base = back + bm * BN;
        for (size_t fn = 0, bn = BN_offset; fn < FN_split; ++fn, ++bn) {
            b_base[bn] += alpha * f_base[fn];
        }
        for (size_t fn = FN_split, bn = 0; fn < FN; ++fn, ++bn) {
            b_base[bn] += alpha * f_base[fn];
        }
    }
}

auto gaussian_matrix(const FLOAT sigma, const size_t size) {
    auto matrix = std::vector<FLOAT>(size * size);
    auto sum = double{}; // always double
    const auto offset = (FLOAT(size) - 1) / 2;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            const auto di = i - offset, dj = j - offset;
            const auto r2 = di * di + dj * dj;
            const auto val = std::exp(-r2 / (2 * sigma * sigma));
            matrix[i * size + j] = val;
            sum += val;
        }
    }
    for (auto& val : matrix) {
        val = FLOAT(val / sum);
    }
    return matrix;
}

template<typename T, typename U>
void list_convolve(
    const std::vector<U>& kernel, size_t KM, size_t KN,
    const std::vector<T>& map, size_t PM, size_t PN,
    std::vector<decltype(T{} * U{})>& ret
) {
    if ((KM % 2u == 0u) || (KN % 2u == 0u)) {
        printf("ERROR: Kernel size should be an odd number.\n");
        std::exit(1);
    } else if (KM >= PM || KN >= PN) {
        printf("ERROR: Kernel size should be less than map size.\n");
        std::exit(1);
    }
    assert_size(ret, PM * PN, "result of list_convolve");
    std::fill(begin(ret), end(ret), T{} * U{});

    for (size_t km = 0; km < KM; ++km) {
        for (size_t kn = 0; kn < KN; ++kn) {
            const auto di = int64_t(km) - int64_t((KM - 1u) / 2u);
            const auto dj = int64_t(kn) - int64_t((KN - 1u) / 2u);
            const auto alpha = kernel[km * KN + kn];
            if (alpha == U(0)) {
                continue;
            }
            put_patch(ret.data(), PM, PN, map.data(), PM, PN, di, dj, alpha);
        }
    }
}

template<typename T, typename U>
auto list_convolve(
    const std::vector<U>& kernel, size_t KM, size_t KN,
    const std::vector<T>& map, size_t PM, size_t PN
) {

    auto ret = std::vector<decltype(T{} * U{})>(PM * PN);
    list_convolve(kernel, KM, KN, map, PM, PN, ret);
    return ret;
}

auto split_catalog_to_tomographic_bins(const std::vector<real_galaxy_measure>& catalog) {
    auto tomographic_catalogs = std::vector<std::vector<real_galaxy_measure>>(S16A_NUM_Z_BIN);
    for (const auto& entry : catalog) {
        if (0 <= entry.z_bin && entry.z_bin < int(S16A_NUM_Z_BIN)) {
            tomographic_catalogs[entry.z_bin].push_back(entry);
        } else {
            printf("ERROR: An entry (z_bin = %d, z_best = %g) in the catalog have z_bin outside [0, S16A_NUM_Z_BIN)\n",
                entry.z_bin, entry.z_best);
            std::exit(1);
        }
    }
    printf("Number of galaxies for each tomographic bin: %zu", tomographic_catalogs[0].size());
    for (size_t i = 1; i < S16A_NUM_Z_BIN; ++i) {
        printf(", %zu", tomographic_catalogs[i].size());
    }
    printf("\n");
    return tomographic_catalogs;
}

// auto calc_shear_field(
//     const std::vector<real_galaxy_measure>& catalog,
//     const std::vector<uint8_t>& mask,
//     const double smoothing_radius_arcmin = 1.0
// ) {
//     constexpr auto N = MAP_SIZE + 2 * MAP_PADDING;
//     auto map_w   = std::vector<FLOAT>(N * N);
//     auto map_wr2 = std::vector<FLOAT>(N * N);
//     auto map_wm  = std::vector<FLOAT>(N * N);
//     auto map_wc  = std::vector<COMPLEX>(N * N);
//     auto map_we  = std::vector<COMPLEX>(N * N);
//     auto map_g   = std::vector<COMPLEX>(N * N);
//     assert_size(mask, N * N, "galaxy mask");
//
//     constexpr auto scale = MAP_SIZE / FIELD_SIZE_DEG; // convert from degrees (x, y) to pixels (i, j)
//     for (const auto& g : catalog) {
//         const auto i =
//             static_cast<size_t>(std::floor((N / 2u) + scale * g.x)) * N +
//             static_cast<size_t>(std::floor((N / 2u) + scale * g.y));
//         map_w  [i] += g.w;
//         map_wr2[i] += g.w * g.rms_e * g.rms_e;
//         map_wm [i] += g.w * (g.m + g.m_bias);
//         map_wc [i] += g.w * COMPLEX(g.c1, g.c2);
//         map_we [i] += g.w * COMPLEX(g.e1, g.e2);
//     }
//     auto total_weight = double{}; // always double precision
//     auto total_pixel = size_t{};
//     for (size_t i = 0; i < N * N; ++i) {
//         total_weight += mask[i] * map_w[i];
//         total_pixel += mask[i];
//     }
//     const auto mean_weight = FLOAT(total_weight / total_pixel);
//     for (size_t i = 0; i < N * N; ++i) {
//         const auto null_weight = (mean_weight - map_w[i]) * (FLOAT(1) - mask[i]);
//         map_w[i]   += null_weight;
//         map_wr2[i] += null_weight * (S16A_MEAN_RMS_E * S16A_MEAN_RMS_E);
//         map_wm[i]  += null_weight * S16A_MEAN_M;
//     }
//
//     const auto radius = scale * FLOAT(smoothing_radius_arcmin / 60.0); // in pixels
//     gaussian_blur(map_w,   radius);
//     gaussian_blur(map_wr2, radius);
//     gaussian_blur(map_wm,  radius);
//     gaussian_blur(map_wc,  radius);
//     gaussian_blur(map_we,  radius);
//
//     for (size_t i = 0; i < N; ++i) {
//         for (size_t j = 0; j < N; ++j) {
//             const auto index = i * N + j;
//             const auto w_inv = 1 / map_w[index];
//             const auto r2 = w_inv * map_wr2[index];
//             const auto m  = w_inv * map_wm[index];
//             const auto c  = w_inv * map_wc[index];
//             const auto e  = w_inv * map_we[index];
//             map_g[i * N + j] = (e / (2 * (1 - r2)) - c) / (1 + m);
//         }
//     }
//     return map_g;
// }

// auto calc_shear_field(
//     const std::vector<std::vector<real_galaxy_measure>>& catalogs,
//     const std::vector<uint8_t>& mask,
//     const double smoothing_radius_arcmin = 1.0
// ) {
//     auto shear_fields = std::vector<std::vector<COMPLEX>>{};
//     for (size_t i = 0; i < S16A_NUM_Z_BIN; ++i) {
//         shear_fields.push_back(calc_shear_field(catalogs[i], mask, smoothing_radius_arcmin));
//     }
//     return shear_fields;
// }

template<size_t N = MAP_SIZE>
auto calc_weighted_shear_field(const std::vector<real_galaxy_measure>& catalog) {
    constexpr auto scale = double(N) / double(FIELD_SIZE_DEG);
    auto map_we  = std::vector<COMPLEX>(N * N);

    for (const auto& g : catalog) {
        const auto i = static_cast<size_t>(std::floor((N / 2u) + scale * g.x));
        const auto j = static_cast<size_t>(std::floor((N / 2u) + scale * g.y));
        if (i >= N || j >= N) {
            printf("ERROR: Position of a galaxy (%g, %g) is out of bounds.", g.x, g.y);
            std::exit(1);
        }
        map_we[i * N + j] += g.w * COMPLEX(g.e1, g.e2);
    }
    return map_we;
}

template<size_t N = TOMO_MAP_SIZE>
auto calc_weighted_shear_tomo_field(const std::vector<std::vector<real_galaxy_measure>>& catalogs) {
    assert_size(catalogs, S16A_NUM_Z_BIN, "tomographic catalogs");
    // constexpr auto scale = double(N) / double(FIELD_SIZE_DEG);
    auto maps_we  = std::vector<COMPLEX>{};

    for (size_t i = 0; i < S16A_NUM_Z_BIN; ++i) {
        const auto map_we = calc_weighted_shear_field<N>(catalogs.at(i));
        maps_we.insert(end(maps_we), begin(map_we), end(map_we));
    }
    return maps_we;
}

auto combine_real_and_sim_catalog(
    const std::vector<real_galaxy_measure>& real_catalog,
    const std::vector<sim_galaxy_measure>& sim_catalog,
    const FLOAT randomness_factor = 1.0
) {
    thread_local auto random_device = std::random_device{};
    thread_local auto random_engine = std::mt19937_64(random_device());
    thread_local auto phase_dist = std::uniform_real_distribution<FLOAT>(-M_PI, M_PI);

    const auto n_gal = real_catalog.size();
    assert_size(sim_catalog, n_gal, "sim_catalog");
    printf("genrating mock shear catalog... n_gal = %zu\n", n_gal);

    // summary_statistics_t<double> g1_stat("g1");
    // summary_statistics_t<double> g2_stat("g2");
    // summary_statistics_t<double> e1_stat("e1");
    // summary_statistics_t<double> e2_stat("e2");
    // summary_statistics_t<double> factor_stat("factor");

    auto copy_catalog = real_catalog;
    for (size_t i = 0; i < real_catalog.size(); ++i) {
        auto& copy = copy_catalog[i];
        const auto& real = real_catalog[i];
        const auto& sim = sim_catalog[i];

        constexpr auto GAMMA_NORM_MAX = FLOAT(1.0);
        constexpr auto KAPPA_NORM_MAX = FLOAT(0.5);

        const auto kappa = std::max(std::min(sim.kappa, KAPPA_NORM_MAX), -KAPPA_NORM_MAX);
        const auto gamma_norm = std::sqrt(sqr(sim.gamma1) + sqr(sim.gamma2));
        if (gamma_norm >= FLOAT(1)) {
            printf("[WARNING] Large gamma detected, gamma = (%g, %g).\n", sim.gamma1, sim.gamma2);
        }
        const auto gamma_factor = GAMMA_NORM_MAX / std::max(gamma_norm, GAMMA_NORM_MAX);
        const auto gamma1 = gamma_factor * sim.gamma1;
        const auto gamma2 = gamma_factor * sim.gamma2;

        const auto e_obs = COMPLEX(real.e1, real.e2);
        const auto e_ran = e_obs * std::exp(COMPLEX(0, 1) * phase_dist(random_engine));
        const auto b_ratio = 1 + real.m + real.m_bias;

        const auto dis1 = 2 * gamma1 * b_ratio / (1 - kappa);
        const auto dis2 = 2 * gamma2 * b_ratio / (1 - kappa);
        const auto e_norm2 = sqr(std::abs(e_obs)) / (1 + sqr(real.sigma_e / real.rms_e));
        copy.e1 = randomness_factor * e_ran.real() + dis1 * (1 - e_norm2 / 2);
        copy.e2 = randomness_factor * e_ran.imag() + dis2 * (1 - e_norm2 / 2);

        // g1_stat += gamma_factor * sim.gamma1;
        // g2_stat += gamma_factor * sim.gamma2;
        // e1_stat += copy.e1;
        // e2_stat += copy.e2;
        // factor_stat += factor;

        copy.c1 = FLOAT(0);
        copy.c2 = FLOAT(0);
    }
    // g1_stat.print();
    // g2_stat.print();
    // e1_stat.print();
    // e2_stat.print();
    // factor_stat.print();
    return copy_catalog;
}

auto get_fourier_n(size_t N) {
    if (N % 2u != 0u) {
        printf("ERROR: Size N in fourier_n must be an even number.\n");
        std::exit(1);
    }
    auto fourier_n = std::vector<FLOAT>(N);
    std::iota(begin(fourier_n), begin(fourier_n) + (N / 2), FLOAT(0));
    std::iota(begin(fourier_n) + (N / 2), end(fourier_n), -FLOAT(N / 2));
    return fourier_n;
}

/// cut a slab from a simulation box
/// 0 <= lower/upper_bound <= 1
/// returns bin counts of size (POTENTIAL_SIZE, POTENTIAL_SIZE)
template<typename ReturnType = uint32_t>
auto get_slab(
    const std::vector<position_t>& positions,
    size_t axis, double lower_bound, double upper_bound
) {
    constexpr auto N = POTENTIAL_SIZE;
    const auto z_base = &positions[0][axis];
    const auto x_base = &positions[0][(axis + 1) % 3];
    const auto y_base = &positions[0][(axis + 2) % 3];
    const auto z_lower = std::llround(lower_bound * std::pow(2.0, 32.0));
    const auto z_upper = std::llround(upper_bound * std::pow(2.0, 32.0));
    const auto num_particles = positions.size();
    if (size_t(uint32_t(num_particles)) != num_particles) {
        printf("ERROR: There are more than 2^32 particles.\n");
        std::exit(1);
    }

    auto slab = std::vector<ReturnType>(N * N);
    for (size_t i = 0; i < num_particles; ++i) {
        const auto z = static_cast<decltype(z_lower)>(z_base[3 * i]);
        if (z_lower <= z && z < z_upper) {
            const auto x_bin = x_base[3 * i] >> (32 - POTENTIAL_SIZE_LEVEL);
            const auto y_bin = y_base[3 * i] >> (32 - POTENTIAL_SIZE_LEVEL);
            slab.at(x_bin * N + y_bin) += ReturnType(1);
        }
    }
    return slab;
}


template<typename SlabElement>
auto slab_to_delta(const std::vector<SlabElement>& slab, FLOAT kernel_radius, size_t kernel_size) {
    constexpr auto N = POTENTIAL_SIZE;
    assert_size(slab, N * N, "density slab");
    const auto mean_density = std::accumulate(begin(slab), end(slab), double{}) / double(N * N);
    auto delta = std::vector<FLOAT>(N * N);
    std::transform(begin(slab), end(slab), begin(delta), [&](auto x){
        return FLOAT(x / mean_density) - 1;
    });

    if (kernel_radius > 0) {
        delta = list_convolve(
        gaussian_matrix(kernel_radius, kernel_size), kernel_size, kernel_size,
        delta, POTENTIAL_SIZE, POTENTIAL_SIZE);
    } 
    return delta;
}

constexpr auto contrast_to_density_factor(
    FLOAT omega_m, FLOAT size_mpch, FLOAT thickness_mpch,
    FLOAT comoving_distance_mpch, FLOAT redshift
) {
    constexpr auto N = POTENTIAL_SIZE;
    constexpr auto h0_unit_m_s = 100000.;
    constexpr auto c_m_s = 299792458.;
    constexpr auto prefactor = 1.5 * sqr(h0_unit_m_s / c_m_s);
    const auto factor = FLOAT(prefactor * omega_m * sqr(size_mpch) * thickness_mpch
        * (1 + redshift) / (comoving_distance_mpch * N * N));
    return factor;
}

template<typename SlabElement>
auto slab_to_potential(const std::vector<SlabElement>& slab,
    const double omega_m, const double hubble_0, const double redshift, 
    const double size_mpch, const double thickness_mpch,
    const double comoving_distance_mpch
) {
    constexpr auto N = POTENTIAL_SIZE;
    assert_size(slab, N * N, "density slab");
    constexpr auto kernel_radius = FLOAT(0.0);
    constexpr auto kernel_size = size_t(7);
    auto delta = slab_to_delta(slab, kernel_radius, kernel_size);

    printf("slab_to_potential(Om=%g, h0=%g, z=%g, size=%gMpc/h, thickness=%gMpc/h, distance=%gMpc/h)\n",
        omega_m, hubble_0, redshift, size_mpch, thickness_mpch, comoving_distance_mpch);

    const auto ARCMIN_TO_PX = (1.0 / 60) * DEGREE * comoving_distance_mpch / (SIMULATION_SIZE_MPCH / POTENTIAL_SIZE);
    const auto SLAB_SMOOTH_SCALE_PX = FLOAT(0.25 * ARCMIN_TO_PX);
    thread_local auto k2_inv = std::vector<FLOAT>{};
    if (k2_inv.size() != N * N) {
        auto fourier_n = get_fourier_n(N);
        k2_inv.resize(N * N);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                const auto index = i * N + j;
                const auto ni = fourier_n[i];
                const auto nj = fourier_n[j];
                const auto l2 = std::max(FLOAT(1), ni * ni + nj * nj) / (N * N);
                k2_inv[index] = FLOAT(
                    std::exp(-0.5 * sqr(2 * M_PI * SLAB_SMOOTH_SCALE_PX) * l2) / (l2 * sqr(2 * M_PI)));
            }
        }
        k2_inv[0] = FLOAT(0);
    }

    const auto time_start = omp_get_wtime();

    auto fourier = std::vector<COMPLEX>(N * N);
    std::copy(begin(delta), end(delta), begin(fourier));
    dft_2d_inplace(fourier.data(), N, N, FFTW_FORWARD);
    std::transform(begin(fourier), end(fourier), begin(k2_inv), begin(fourier), [](auto x, auto y) {
       return FLOAT(-2) * x * y;
    });
    dft_2d_inplace(fourier.data(), N, N, FFTW_BACKWARD);
    
    const auto factor = contrast_to_density_factor(omega_m, size_mpch, thickness_mpch, comoving_distance_mpch, redshift);
    auto& potential = delta;
    std::transform(begin(fourier), end(fourier), begin(potential), [=](auto x) {
        return factor * x.real();
    });
    printf("slab_to_potential(...) took %g seconds.\n", omp_get_wtime() - time_start);

    return potential;
}

inline auto get_ray_trace_derivatives(
    const std::vector<FLOAT>& phi, const FLOAT phi_distance_mpch,
    const int64_t i, const int64_t j, const FLOAT flip_x, const FLOAT flip_y
) {
    constexpr auto N = POTENTIAL_SIZE;
    const auto a = phi_distance_mpch / (SIMULATION_SIZE_MPCH / FLOAT(N));
    const auto t = sqr(a);

    const auto v_00 = pbc_pick<N>(phi, i, j);
    const auto v_h0 = pbc_pick<N>(phi, i + 1, j);
    const auto v_l0 = pbc_pick<N>(phi, i - 1, j);
    const auto v_0h = pbc_pick<N>(phi, i, j + 1);
    const auto v_0l = pbc_pick<N>(phi, i, j - 1);
    const auto v_hh = pbc_pick<N>(phi, i + 1, j + 1);
    const auto v_hl = pbc_pick<N>(phi, i + 1, j - 1);
    const auto v_lh = pbc_pick<N>(phi, i - 1, j + 1);
    const auto v_ll = pbc_pick<N>(phi, i - 1, j - 1);

    auto alpha = std::array<FLOAT, 2u>{
        flip_x * (a / 2) * (v_h0 - v_l0),
        flip_y * (a / 2) * (v_0h - v_0l)
    };
    auto T = std::array<FLOAT, 3u>{
        t * (v_h0 + v_l0 - 2 * v_00),
        flip_x * flip_y * (t / 4) * (v_hh + v_ll - v_hl - v_lh),
        t * (v_0h + v_0l - 2 * v_00)
    };
    return std::make_pair(alpha, T);
}

auto intrinsic_alignment_Fz(const FLOAT redshift, const FLOAT omega_m) {
    auto D = [](FLOAT z, FLOAT Omega_m0) {
        const auto Omega_Lambda0 = 1 - Omega_m0;
        const auto Omega_m = 1 / (1 + (Omega_Lambda0 / Omega_m0) / std::pow(1 + z, 3));
        const auto Omega_Lambda = 1 - Omega_m;
        const auto D_num = FLOAT(2.5) * Omega_m / (1 + z);
        const auto D_den = std::pow(Omega_m, FLOAT(4) / 7) - Omega_Lambda + (1 + Omega_m / 2) * (1 + Omega_Lambda0 / 70);
        return D_num / D_den;
    };
    constexpr auto factor = FLOAT(-0.0138769);
    return factor * omega_m * (D(0, omega_m) / D(redshift, omega_m));
}

// should not be called in parallel
auto ray_trace(
    const std::vector<std::vector<FLOAT>>& phis,
    const std::vector<slab_entry_t>& config,
    std::vector<std::array<FLOAT, 2u>> gal_angles,
    std::vector<FLOAT> gal_distances_mpch,
    std::vector<int> gal_z_bins,
    std::vector<size_t> gal_ranges_all_fields,
    const std::vector<slab_shuffle>& slab_shuffles
) {
    const auto N_phi = phis.size();
    const auto N_gal = gal_angles.size();
    assert_size(config, N_phi, "potential configurations");
    for (const auto& phi: phis) {
        assert_size(phi, POTENTIAL_SIZE * POTENTIAL_SIZE, "potential map");
    }

    auto phi_distances_mpch = std::vector<FLOAT>{};
    auto phi_redshifts = std::vector<FLOAT>{};
    for (const auto& entry : config) {
        phi_distances_mpch.push_back(entry.distance_mpch);
        phi_redshifts.push_back(entry.redshift);
    }
    if (!std::is_sorted(begin(phi_distances_mpch), end(phi_distances_mpch))) {
        printf("ERROR: phi_distances_mpch should be sorted from low to high.\n");
        std::exit(1);
    }

    print_vector(gal_ranges_all_fields, "gal_ranges_all_fields");
    if (!(gal_ranges_all_fields.size() >= 2 &&
        gal_ranges_all_fields.front() == 0 && gal_ranges_all_fields.back() == N_gal &&
        std::is_sorted(begin(gal_ranges_all_fields), end(gal_ranges_all_fields)))
    ) {
        printf("ERROR: gal_ranges_all_fields should be a sorted list in the form of {0, ..., N_gal}, N_gal = %zu.\n", N_gal);
        std::exit(1);
    }
    const auto N_field = gal_ranges_all_fields.size() - 1;

    auto [x_min, x_max] = std::minmax_element(begin(gal_angles), end(gal_angles), [](auto a, auto b) {return a[0] < b[0];});
    auto [y_min, y_max] = std::minmax_element(begin(gal_angles), end(gal_angles), [](auto a, auto b) {return a[1] < b[1];});
    printf("ray_trace(...) N_phi=%zu, chi_phi=[%g,%g]Mpc/h, gal_angle=[(%g,%g),(%g,%g)]radians, <gal_distance>=%gMpc/h\n",
    N_phi, phi_distances_mpch.front(), phi_distances_mpch.back(), (*x_min)[0], (*y_min)[1], (*x_max)[0], (*y_max)[1],
        std::accumulate(begin(gal_distances_mpch), end(gal_distances_mpch), FLOAT(0)) / FLOAT(N_gal)
    );
    if (std::max((*x_max)[0] - (*x_min)[0], (*y_max)[1] - (*y_min)[1]) > FLOAT(0.262)) {
        printf("ERROR: The angular size of the ray-tracing field is greater than 15 degrees.\n");
        std::exit(1);
    }

    assert_size(slab_shuffles, N_phi * N_field, "slab shuffle data");
    
    phi_distances_mpch.insert(begin(phi_distances_mpch), FLOAT(0));
    phi_distances_mpch.push_back(FLOAT(1.e9));
    phi_redshifts.insert(begin(phi_redshifts), FLOAT(0));
    phi_redshifts.push_back(FLOAT(1.e9));

    const auto& chi = gal_distances_mpch;
    auto& beta = gal_angles;
    auto d_beta = std::vector<std::array<FLOAT, 2u>>(N_gal, { FLOAT(0), FLOAT(0) });
    auto A = std::vector<std::array<FLOAT, 4u>>(N_gal, { FLOAT(1), FLOAT(0), FLOAT(0), FLOAT(1) });
    auto d_A = std::vector<std::array<FLOAT, 4u>>(N_gal, { FLOAT(0), FLOAT(0), FLOAT(0), FLOAT(0) });
    auto dA_IA = std::vector<std::array<FLOAT, 4u>>(N_gal, { FLOAT(0), FLOAT(0), FLOAT(0), FLOAT(0) });

    static auto random_device = std::random_device{};
    static auto random_engine = std::mt19937_64(random_device());
    static auto shuffle_dist = std::uniform_real_distribution<FLOAT>(0, 1);

    auto shear_dot = [](const std::array<FLOAT, 3u>& t, const std::array<FLOAT, 4u>& a) {
        return std::array<FLOAT, 4u>{
            a[0] * t[0] + a[2] * t[1], a[1] * t[0] + a[3] * t[1],
            a[0] * t[1] + a[2] * t[2], a[1] * t[1] + a[3] * t[2]
        };
    };
    
    for (size_t k = 0; k < N_phi; ++k) {
        // auto time_start = omp_get_wtime();
        const auto& phi = phis[k];
        const auto& entry = config[k];
        const auto d0 = phi_distances_mpch[k];
        const auto d1 = phi_distances_mpch[k + 1];
        const auto d2 = phi_distances_mpch[k + 2];
        const auto z0 = phi_redshifts[k];
        const auto z1 = phi_redshifts[k + 1];
        const auto z2 = phi_redshifts[k + 2];
        const auto Ak = (d1 / d2) * (1 + (d2 - d1) / (d1 - d0));
        const auto Ck = (d1 - d2) / d2;

        const auto angle_factor = (d1 / SIMULATION_SIZE_MPCH);
        // W_IA / W_Born = (-E(z)F(z)a(z)/D(z)) * (D(z')/D(z,z'))
        const auto Fz = intrinsic_alignment_Fz(z1, entry.omega_m);
        const auto EzDz = (z1 - z0) / (d1 - d0) / d1 / sqr(3.33564e-4);
        const auto factor_IA = FLOAT(2. / 3. / entry.omega_m * Fz * EzDz / (1 + z1));

        printf("k = %zu, z = %g, chi = %g, Fz = %g, factor_IA = %g, z012 = (%g,%g,%g), d012 = (%g,%g,%g)\n",
            k, z1, d1, Fz, factor_IA, z0, z1, z2, d0, d1, d2);

        auto trace_field = [&](size_t i_field, size_t i_begin, size_t i_end) {
            const auto& shuffle = slab_shuffles.at(k * N_field + i_field);
            const auto flip_x = shuffle.flip_x ? FLOAT(-1) : FLOAT(1);
            const auto flip_y = shuffle.flip_y ? FLOAT(-1) : FLOAT(1);

            constexpr auto MAX_NUM_Z_BIN = size_t(10);
            auto num_galaxies_in_bin  = std::array<size_t, MAX_NUM_Z_BIN>{};
            auto num_galaxies_in_slab = std::array<size_t, MAX_NUM_Z_BIN>{};

            for (size_t i = i_begin; i < i_end; ++i) {
                const auto z_bin = gal_z_bins[i];
                if (z_bin >= int(MAX_NUM_Z_BIN)) {
                    printf("ERROR: ray_trace(...) i = %zu, z_bin >= MAX_NUM_Z_BIN\n", i);
                    exit(1);
                }
                ++num_galaxies_in_bin[z_bin];
                if ((d0 + d1) / 2 <= chi[i] && chi[i] < (d1 + d2) / 2) {
                    ++num_galaxies_in_slab[z_bin];
                }
            }
            if (i_field == 0) {
                printf("    gal in bin = (%zu, %zu, %zu, %zu), gal in slab = (%zu, %zu, %zu, %zu)\n", 
                    num_galaxies_in_bin[0], num_galaxies_in_bin[1], num_galaxies_in_bin[2], num_galaxies_in_bin[3],
                    num_galaxies_in_slab[0], num_galaxies_in_slab[1], num_galaxies_in_slab[2], num_galaxies_in_slab[3]);
            }

            for (size_t i = i_begin; i < i_end; ++i) {
                const auto [x_idx, y_idx, x_frac, y_frac] = shuffle_angle<POTENTIAL_SIZE>(
                    beta[i][0], beta[i][1], angle_factor, shuffle);

                auto [alpha, T] = get_ray_trace_derivatives(phi, d1, x_idx, y_idx, flip_x, flip_y);
                const auto R = (std::min(std::max(chi[i], d1), d2) - d1) / (d2 - d1);

                d_beta[i][0] = (Ak - 1) * d_beta[i][0] + Ck * alpha[0];
                d_beta[i][1] = (Ak - 1) * d_beta[i][1] + Ck * alpha[1];
                beta[i][0] += R * d_beta[i][0];
                beta[i][1] += R * d_beta[i][1];
                const auto TA = shear_dot(T, A[i]);
                d_A[i][0] = (Ak - 1) * d_A[i][0] + Ck * TA[0];
                d_A[i][1] = (Ak - 1) * d_A[i][1] + Ck * TA[1];
                d_A[i][2] = (Ak - 1) * d_A[i][2] + Ck * TA[2];
                d_A[i][3] = (Ak - 1) * d_A[i][3] + Ck * TA[3];
                A[i][0] += R * d_A[i][0];
                A[i][1] += R * d_A[i][1];
                A[i][2] += R * d_A[i][2];
                A[i][3] += R * d_A[i][3];

                const auto factor_nz = FLOAT(num_galaxies_in_slab[gal_z_bins[i]])
                    / (z1 - z0) / num_galaxies_in_bin[gal_z_bins[i]];
                dA_IA[i][0] += -factor_IA * factor_nz * T[0];
                dA_IA[i][1] += -factor_IA * factor_nz * T[1];
                dA_IA[i][2] += -factor_IA * factor_nz * T[1];
                dA_IA[i][3] += -factor_IA * factor_nz * T[2];
            }
        };

        const auto num_threads = std::min(omp_get_max_threads(), 12); // 12 threads max
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        for (size_t i_field = 0; i_field < N_field; ++i_field) {
            trace_field(i_field, gal_ranges_all_fields[i_field], gal_ranges_all_fields[i_field + 1]);
        }
        // printf("total time = %g\n", omp_get_wtime() - time_start);
        // t0.print();
        // t1.print();
        // t2.print();
        // a00.print();
        // a01.print();
        // a10.print();
        // a11.print();
    }

    return std::make_pair(A, dA_IA);
}

template<typename RhoR>
auto get_component_patch(RhoR rho_r, const size_t radius_px) {
    if (!(radius_px >= 1u)) {
        printf("ERROR: Component patches should have radius_px >= 1\n");
        std::exit(1);
    }
    const auto N = radius_px * 2 + 1;
    const auto N_ss = SUPER_SAMPLING_FACTOR * N;

    thread_local auto z2 = std::vector<double>{};
    thread_local auto rho = std::vector<double>{};
    if (z2.size() != LINE_OF_SIGHT_RESOLUTION) {
        z2.resize(LINE_OF_SIGHT_RESOLUTION);
        rho.resize(LINE_OF_SIGHT_RESOLUTION);
        for (size_t i = 0; i < LINE_OF_SIGHT_RESOLUTION; ++i) {
            z2[i] = sqr((i + 0.5) / LINE_OF_SIGHT_RESOLUTION);
        }
    }
    
    auto patch = std::vector<double>(N * N, FLOAT{});
    auto element = [&](size_t i, size_t j) -> auto& {
        return patch[(i / SUPER_SAMPLING_FACTOR) * N + (j / SUPER_SAMPLING_FACTOR)];
    };
    
    const auto z2_factor = sqr(double(radius_px));
    const auto half = N_ss / 2;
    for (size_t i = 0; i < half; ++i) {
        for (size_t j = 0; j < half; ++j) {
            const auto r2_xy = (sqr(i + 0.5) + sqr(j + 0.5)) / sqr(double(SUPER_SAMPLING_FACTOR));
            for (size_t k = 0; k < LINE_OF_SIGHT_RESOLUTION; ++k) {
                const auto r = std::sqrt(z2[k] * z2_factor + r2_xy);
                rho[k] = rho_r(r);
            }
            const auto total = std::accumulate(begin(rho), end(rho), double{});
            element(half + 0 + i, half + 0 + j) += total;
            element(half + 0 + i, half - 1 - j) += total;
            element(half - 1 - i, half + 0 + j) += total;
            element(half - 1 - i, half - 1 - j) += total;
        }
    }
    const auto normalize = 1 / std::accumulate(begin(patch), end(patch), double{});
    std::transform(begin(patch), end(patch), begin(patch), [=](auto x) { return normalize * x; });
    return patch;
}

template<typename Frac>
void get_baryonic_fractions(
    const slab_entry_t& slab_entry, const bcm_parameters& bcm, const size_t halo_num_particles,
    Frac& fCG, Frac& fBG, Frac& fEGO, Frac& fEGI
) {
    const auto omega_m = slab_entry.omega_m;
    const auto z = slab_entry.redshift;
    const auto f_b = OMEGA_B / omega_m;
    const auto m200 = halo_num_particles * PARTICLE_MASS_MSUNH(omega_m);
    const auto a = 1 / (z + 1);
    const auto nu = std::exp(-4 * sqr(a));
    const auto m1 = std::pow(10, std::log10(bcm.M1_0) + (-1.793 * (a - 1) - 0.251 * z) * nu);
    const auto epsilon = std::pow(10, std::log10(0.023) + (-0.006 * (a - 1)) * nu - 0.119 * (a - 1));
    const auto alpha = -1.779 + (0.731 * (a - 1)) * nu;
    const auto delta = 4.394 + (2.608 * (a - 1) - 0.043 * z) * nu;
    const auto gamma = 0.547 + (1.319 * (a - 1) + 0.279 * z) * nu;
    const auto g = [&](double x) -> double {
        return -std::log10(std::pow(10, alpha * x) + 1) +
            delta * std::pow(std::log10(1 + std::exp(x)), gamma) / (1 + std::exp(std::pow(10, -x)));
    };
    const auto r_ej = 0.75 * bcm.eta * (0.5 * std::sqrt(200));
    const auto egi_eg = -(std::sqrt(2 / M_PI) * std::exp(-0.5 / (r_ej * r_ej))) / r_ej
        + std::erf(1 / (std::sqrt(2) * r_ej));
    fCG = epsilon * (m1 / m200) * std::pow(10, g(std::log10(m200 / m1)) - g(0));
    fBG = (f_b - fCG) / (1 + std::pow(bcm.Mc / m200, bcm.beta));
    fEGI = egi_eg * (f_b - (fCG + fBG));
    fEGO = (1 - egi_eg) * (f_b - (fCG + fBG));
}

// paint BCM patches onto a DM-only slab
void apply_bcm(
    const slab_entry_t& slab_entry,
    std::vector<FLOAT>& dm_slab,
    const std::vector<halo_t>& halos,
    const size_t axis,
    bcm_parameters bcm,
    std::string bcm_patches_dir = ""
) {
    thread_local auto offsets     = std::vector<size_t>{};
    thread_local auto patch_sizes = std::vector<size_t>{};
    thread_local auto nfw_data    = std::vector<FLOAT>{};
    thread_local auto cg_data     = std::vector<FLOAT>{};
    thread_local auto bg_data     = std::vector<FLOAT>{};
    thread_local auto rdmcg_data  = std::vector<FLOAT>{};
    thread_local auto rdmbg_data  = std::vector<FLOAT>{};
    thread_local auto rdmego_data = std::vector<FLOAT>{};
    thread_local auto rdmegi_data = std::vector<FLOAT>{};
    thread_local auto log_rvir_px = std::vector<FLOAT>{};
    thread_local auto log_c       = std::vector<FLOAT>{};

    if (offsets.empty()) {
        if (bcm_patches_dir == "") {
            printf("ERROR: BCM patches are to be loaded, but bcm_patches_dir == \"\".\n");
            std::exit(1);
        } else {
            printf("Loading BCM patches from %s.\n", bcm_patches_dir.c_str());
        }
        offsets = binary_read<size_t>(bcm_patches_dir + "/offsets.u64");
        assert_size(offsets, HALO_RVIR_COUNT * HALO_C_COUNT + 1, "BCM patch offsets");
        for (size_t i = 0; i < HALO_RVIR_COUNT * HALO_C_COUNT; ++i) {
            patch_sizes.push_back(estimate_map_size(offsets[i + 1] - offsets[i]));
        }
        auto load_patches = [](auto& data, const auto& path) {
            data = binary_read<FLOAT>(path);
            assert_size(data, offsets.back(), path);
        };
        load_patches(nfw_data,    bcm_patches_dir + "/nfw_data.f32");
        load_patches(cg_data,     bcm_patches_dir + "/cg_data.f32");
        load_patches(bg_data,     bcm_patches_dir + "/bg_data.f32");
        load_patches(rdmcg_data,  bcm_patches_dir + "/rdmcg_data.f32");
        load_patches(rdmbg_data,  bcm_patches_dir + "/rdmbg_data.f32");
        load_patches(rdmego_data, bcm_patches_dir + "/rdmego_data.f32");
        load_patches(rdmegi_data, bcm_patches_dir + "/rdmegi_data.f32");
        load_patches(nfw_data,    bcm_patches_dir + "/nfw_data.f32");
        for (size_t i = 0; i < HALO_RVIR_COUNT; ++i) {
            log_rvir_px.push_back(std::log(FN_HALO_RVIR_PIXEL(i)));
        }
        for (size_t i = 0; i < HALO_C_COUNT; ++i) {
            log_c.push_back(std::log(FN_HALO_C(i)));
        }
    }
    const auto time_start = omp_get_wtime();

    assert_size(dm_slab, POTENTIAL_SIZE * POTENTIAL_SIZE, "Dark matter-only density slab");

    auto find_closest_index = [](const auto& vec, auto val) {
        auto i = std::lower_bound(begin(vec), end(vec), val) - begin(vec);
        return (i == vec.size() || (i > 0 && (val - vec[i - 1] < vec[i] - val))) ? (i - 1) : i;
    };

    auto rvir_bin_counts = std::vector<size_t>(HALO_RVIR_COUNT);
    auto c_bin_counts = std::vector<size_t>(HALO_C_COUNT);

    auto cg_stat = summary_statistics_t<FLOAT>("cg");
    auto bg_stat = summary_statistics_t<FLOAT>("bg");
    auto ego_stat = summary_statistics_t<FLOAT>("ego");
    auto egi_stat = summary_statistics_t<FLOAT>("egi");
    auto r_ej_stat = summary_statistics_t<FLOAT>("r_ej_px");

    // std::fill(begin(dm_slab), end(dm_slab), FLOAT{});

    auto patch_sum = std::vector<FLOAT>{};
    auto eg_patch = std::vector<FLOAT>{};

    for (const auto& halo : halos) {
        const auto rvir_px = halo.r_vir / (1000 * SIMULATION_SIZE_MPCH) * POTENTIAL_SIZE;
        const auto r_vir_index = find_closest_index(log_rvir_px, std::log(rvir_px));
        const auto c_index = find_closest_index(log_c, std::log(halo.c));
        rvir_bin_counts[r_vir_index]++;
        c_bin_counts[c_index]++;
        const auto model_index = r_vir_index * HALO_C_COUNT + c_index;

        FLOAT fCG, fBG, fEGO, fEGI;
        get_baryonic_fractions(slab_entry, bcm, halo.weight, fCG, fBG, fEGO, fEGI);
        cg_stat += fCG; bg_stat += fBG; ego_stat += fEGO; egi_stat += fEGI;

        const auto patch_size = patch_sizes[model_index];
        const auto offset = offsets[model_index];

        const auto center_x = halo.pos[(axis + 1) % 3] >> (32 - POTENTIAL_SIZE_LEVEL);
        const auto center_y = halo.pos[(axis + 2) % 3] >> (32 - POTENTIAL_SIZE_LEVEL);
        const auto patch_half_size = int64_t(patch_size / 2);

        if (patch_sum.size() < patch_size * patch_size) {
            patch_sum.resize(patch_size * patch_size);
        }
        auto add_component = [&](const std::vector<FLOAT>& component_data, const FLOAT fraction) {
            std::transform(
                patch_sum.data(), patch_sum.data() + patch_size * patch_size,
                component_data.data() + offset, patch_sum.data(),
                [=](auto sum, auto val) { return sum + fraction * val; }
            );
        };
        std::fill_n(patch_sum.data(), patch_size * patch_size, FLOAT{}); // reset the sum patch
        add_component(cg_data, fCG);
        add_component(bg_data, fBG);
        add_component(rdmcg_data, fCG);
        add_component(rdmbg_data, fBG);
        add_component(rdmego_data, fEGO);
        add_component(rdmegi_data, fEGI);

        put_patch(dm_slab.data(), POTENTIAL_SIZE, POTENTIAL_SIZE,
            patch_sum.data(), patch_size, patch_size,
            int64_t(center_x) - patch_half_size, int64_t(center_y) - patch_half_size,
            halo.weight
        );

        const auto r_ej_px = rvir_px * (0.75 * bcm.eta * (0.5 * std::sqrt(200)));
        r_ej_stat += r_ej_px;
        const auto eg_half_size = static_cast<size_t>(std::ceil(r_ej_px * 2.5));
        const auto eg_size = eg_half_size * 2 + 1;
        if (eg_patch.size() < eg_size * eg_size) {
            eg_patch.resize(eg_size * eg_size);
        }
        if (eg_size > POTENTIAL_SIZE) {
            printf("ERROR: eg_size = %zu is greater than POTENTIAL_SIZE = %zu.", eg_size, POTENTIAL_SIZE);
            std::exit(1);
        }

        const auto eg_r_factor = FLOAT(-0.5 / sqr(r_ej_px));
        auto dy2 = std::vector<FLOAT>(eg_size);
        for (size_t i = 0; i < eg_size; ++i) {
            dy2[i] = sqr(FLOAT(i) - FLOAT(eg_half_size));
        }
        for (size_t i = 0; i < eg_size; ++i) {
            const auto dx2 = sqr(FLOAT(i) - FLOAT(eg_half_size));
            for (size_t j = 0; j < eg_size; ++j) {
                eg_patch[i * eg_size + j] = expf(eg_r_factor * (dx2 + dy2[j]));
            }
        }
        const auto eg_weight = halo.weight * (fEGI + fEGO)
            / std::accumulate(begin(eg_patch), begin(eg_patch) + eg_size * eg_size, FLOAT{});
        put_patch(dm_slab.data(), POTENTIAL_SIZE, POTENTIAL_SIZE,
            eg_patch.data(), eg_size, eg_size,
            int64_t(center_x) - eg_half_size, int64_t(center_y) - eg_half_size,
            eg_weight
        );

    }
    //print_vector(rvir_bin_counts, "rvir_bin_counts");
    //print_vector(c_bin_counts, "c_bin_counts");
    //cg_stat.print(); bg_stat.print(); ego_stat.print(); egi_stat.print(); r_ej_stat.print();

    printf("Adding baryons to the slab took %g seconds.\n", omp_get_wtime() - time_start);
}

template<typename RandomEngine>
void load_potentials(
    std::string slab_config, std::string potential_dir,
    RandomEngine& random_engine,
    std::vector<std::vector<FLOAT>>& phis,
    std::vector<FLOAT>& phi_distances_mpch
) {
    const auto config = read_slab_config(slab_config);
    for (size_t i = 0; i < config.size(); ++i) {
        auto files = list_files(potential_dir, "potential_order" + std::to_string(i) + ".+");
        if (files.size() == 0) {
            printf("ERROR: There is not potential file matching %s.",
                (potential_dir + "potential_order" + std::to_string(i) + "_*").c_str());
            std::exit(1);
        }
        std::sort(begin(files), end(files));
        auto file_rand = std::uniform_int_distribution(0, int(files.size()) - 1)(random_engine);
        auto shift_x = std::uniform_int_distribution(0, int(POTENTIAL_SIZE - 1))(random_engine);
        auto shift_y = std::uniform_int_distribution(0, int(POTENTIAL_SIZE - 1))(random_engine);
        printf("file = %s, shift = (%d, %d)\n", files[file_rand].c_str(), shift_x, shift_y);

        auto potential = std::vector<FLOAT>(POTENTIAL_SIZE * POTENTIAL_SIZE);
        auto potential_shift = std::vector<FLOAT>(POTENTIAL_SIZE * POTENTIAL_SIZE);
        binary_read(files[file_rand], potential.data(), potential.size());
        put_patch(
            potential_shift.data(), POTENTIAL_SIZE, POTENTIAL_SIZE,
            potential.data(), POTENTIAL_SIZE, POTENTIAL_SIZE, shift_x, shift_y, FLOAT(1));
        // binary_write("./potentialshift_" + std::to_string(i), potential_shift.data(), potential_shift.size());
        phis.push_back(std::move(potential_shift));
        phi_distances_mpch.push_back(config[i].distance_mpch);
    }
}

// load the total density slab (background + halo + bcm)
auto load_single_bcm_density_slab(
    std::string density_dir, size_t order, const slab_entry_t& entry,
    std::string bcm_patches_dir, bcm_parameters bcm, FLOAT bcm_intensity,
    uint64_t rand
) -> std::vector<FLOAT> {
    const auto time_start = omp_get_wtime();

    auto background_files = list_files(density_dir, "density_background_order" + std::to_string(order) + "_.+\\.u32\\.z");
    auto halo_density_files = list_files(density_dir, "density_halo_order" + std::to_string(order) + "_.+\\.u32\\.z");
    auto halo_info_files = list_files(density_dir, "halo_info_order" + std::to_string(order) + "_.+\\.halo_t");
    const size_t n_files = background_files.size();
    if ((n_files == 0) || (n_files % 3 != 0)) {
        printf("ERROR: The number of density files (N = %zu) is 0 or is not divisible by the number of axes (3).\n", n_files);
        std::exit(1);
    }
    assert_size(halo_density_files, n_files, "halo density files");
    assert_size(halo_info_files, n_files, "halo info files");

    std::sort(begin(background_files), end(background_files));
    std::sort(begin(halo_density_files), end(halo_density_files));
    std::sort(begin(halo_info_files), end(halo_info_files));
    const auto file_rand = rand % n_files;

    auto get_axis = [](const std::string& file) {
        const auto pos = file.find("_axis", std::max(0, int(file.size() - 25)));
        const auto axis = (pos == std::string::npos) ? -1 : (file.at(pos + 5) - '0');
        if (!(0 <= axis && axis < 3)) {
            printf("ERROR: The axis value is not valid in file %s\n", file.c_str());
            std::exit(1);
        }
        return size_t(axis);
    };
    const auto axis = get_axis(background_files[file_rand]);

    printf("Reading from %s\n", background_files[file_rand].c_str());
    
    const auto slab_bg = zlib_decompress<uint32_t>(binary_read<ZLIB_BUFFER_CHAR>(background_files[file_rand]));
    const auto slab_halo = zlib_decompress<uint32_t>(binary_read<ZLIB_BUFFER_CHAR>(halo_density_files[file_rand]));
    const auto halos = binary_read<halo_t>(halo_info_files[file_rand]);
    assert_size(slab_bg, POTENTIAL_SIZE * POTENTIAL_SIZE, "background density slab");
    assert_size(slab_halo, POTENTIAL_SIZE * POTENTIAL_SIZE, "halo density slab");

    auto slab_bcm = std::vector<FLOAT>(POTENTIAL_SIZE * POTENTIAL_SIZE);

    const auto halo_factor = 1 - bcm_intensity * (OMEGA_B / entry.omega_m);
    printf("Number of particles (bg = %g, halo = %g), weight of adding halos = %g\n",
        std::accumulate(begin(slab_bg), end(slab_bg), double{}),
        std::accumulate(begin(slab_halo), end(slab_halo), double{}), halo_factor);
    
    std::transform(begin(slab_bg), end(slab_bg), begin(slab_halo), begin(slab_bcm),
        [=](auto bg, auto halo) { return bg + halo_factor * halo; });

    printf("Loading density planes and halo information took %g seconds.\n", omp_get_wtime() - time_start);

    if (bcm_intensity == FLOAT(0)) {
        return slab_bcm;
    } else if (bcm_intensity == FLOAT(1)) {
        apply_bcm(entry, slab_bcm, halos, axis, bcm, bcm_patches_dir);
        return slab_bcm;
    } else {
        printf("ERROR: bcm_intensity should be 0 or 1, bcm_intensity = %g.\n", bcm_intensity);
        std::exit(1);
    }
}

namespace _main {

// void calc_potential_maps(std::string slab_config, std::string density_dir, std::string potential_dir) {
//     const auto config = read_slab_config(slab_config);
//     const auto config_size = config.size();
//     const auto num_threads = std::min(omp_get_max_threads(), 12); // 12 threads max
//     printf("Loading density and potential planes with %d threads...\n", num_threads);
//     #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
//     for (size_t i = 0; i < config_size; ++i) {
//         const auto n_files = list_files(density_dir, "density_background_order" + std::to_string(i) + "_.+\\.u32\\.z").size();
//         printf("Background density order %d has %d files.\n", int(i), int(n_files));
//         for (size_t axis = 0; axis < 3; ++axis) {
//             for (size_t plane = 0; plane < (n_files / 3); ++plane) {          
//                 auto slab = load_single_bcm_density_slab(
//                     density_dir, i, config[i], "", {}, 0, axis * (n_files / 3) + plane);
//                 auto phi = slab_to_potential(
//                     slab, config[i].omega_m, config[i].hubble_0, config[i].redshift,
//                     SIMULATION_SIZE_MPCH, config[i].thickness_mpch, config[i].distance_mpch);
//                 binary_write(
//                     potential_dir + "/potential_order" + std::to_string(i)
//                         + "_axis" + std::to_string(axis)
//                         + "_plane" + std::to_string(plane) + ".f32",
//                     phi.data(), phi.size()
//                 );
//             }
//         }
//     }
// }

// void save_power_spectra_as_tsv(
//     std::string tsv_file,
//     const std::vector<std::pair<std::vector<double>, std::vector<double>>>& powerspecs
// ) {
//     if (powerspecs.size() == 0) {
//         printf("There is no power spectra to be saved.\n");
//         return;
//     }
//     const auto num_bins = powerspecs[0].first.size();
//     auto tsv = std::vector<std::vector<double>>(num_bins);
//     for (size_t i = 0; i < num_bins; ++i) {
//         tsv[i].push_back(powerspecs[0].first[i]);
//         for (const auto& p : powerspecs) {
//             tsv[i].push_back(p.second[i]);
//         }
//     }
//     tsv_write(tsv_file, tsv);
// }

void calc_hsc_ray_tracing(
    std::string slab_config,
    std::string density_dir,
    std::string raytracing_dir,
    std::string catalog_dir,
    std::string bcm_patches_dir,
    std::optional<bcm_parameters> bcm,
    FLOAT bcm_intensity,
    FLOAT delta_z_intensity,
    const int num_realizations,
    const int initial_random_seed
) {
    constexpr bool OUTPUT_SHEAR_MAPS = true;
    constexpr bool OUTPUT_CATALOGS = false;
    constexpr bool KIDS450_IA_TEST = false;
    constexpr bool SIGMA8_TEST = false;

    {
        const auto print_bcm = bcm.value_or(bcm_parameters{0.0, 0.0, 0.0, 0.0});
        printf("calc_hsc_ray_tracing(...) bcm = (%g, %g, %g, %g), intensity = %g, random_seed = [%d, %d), \
OUTPUT_CATALOGS = %s, OUTPUT_SHEAR_MAPS = %s\n",
            print_bcm.Mc, print_bcm.M1_0, print_bcm.eta, print_bcm.beta, bcm_intensity,
            initial_random_seed, initial_random_seed + num_realizations,
            OUTPUT_CATALOGS ? "true" : "false", OUTPUT_SHEAR_MAPS ? "true" : "false");
    }

    const int num_threads = std::min(omp_get_max_threads(), 12); // 12 threads max

    const auto config = read_slab_config(slab_config);
    const auto config_size = config.size();
    auto z_to_chi = get_redshift_to_distance(config);
    printf("redshift -- comoving distance relation:\n");
    for (auto z = FLOAT(0); z < FLOAT(2.5); z += FLOAT(0.2)) {
        printf("   z = %g, chi = %g\n", z, z_to_chi(z));
    }

    auto catalog_files = list_files(catalog_dir, "s16a-catalog-field.+\\.csv");
    const auto N_field = catalog_files.size();
    std::sort(begin(catalog_files), end(catalog_files));
    for (auto f : catalog_files) {
        std::cout << f << "\n";
    }

    auto catalogs_all_z = std::vector<decltype(read_real_galaxy_measures(""))>(N_field);

    printf("Loading catalogs with %d threads...\n", num_threads);
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (size_t i_field = 0; i_field < N_field; ++i_field) {
        printf("catalog = %s\n", catalog_files[i_field].c_str());
        catalogs_all_z[i_field] = read_real_galaxy_measures(catalog_files[i_field]);
    }

    auto bcm_list = std::vector<bcm_parameters>(initial_random_seed + num_realizations);
    if (bcm) {
        for (auto& b : bcm_list) {
            b = bcm.value();
        }
    } else {
        // read bcm list from file
        const auto bcm_list_file = catalog_dir + "/bcm-sobol.csv";
        bcm_list = read_bcm_sobol_sequence(bcm_list_file);
        if (bcm_list.size() < size_t(initial_random_seed + num_realizations)) {
            printf("ERROR: BCM sobol sequence file %s has only %d entries, but %d is required.\n",
                bcm_list_file.c_str(), int(bcm_list.size()), initial_random_seed + num_realizations);
            exit(1);
        }
    }

    thread_local auto random_device = std::random_device{};
    thread_local auto random_engine = std::mt19937_64(random_device());
    thread_local auto z_rt_dist = std::uniform_real_distribution<FLOAT>(-0.15f, 0.15f);
    thread_local auto normal_dist = std::normal_distribution<FLOAT>(0, 1);
    thread_local auto uint32_dist = std::uniform_int_distribution<uint32_t>(0, UINT32_MAX);

    if (KIDS450_IA_TEST) {
        // Replace each of the fields by a regular grid
        constexpr auto TEST_SIZE = size_t(256);
        for (size_t i_field = 0; i_field < N_field; ++i_field) {
            auto& catalog = catalogs_all_z[i_field];
            catalog.clear();
            auto measure = real_galaxy_measure{};
            measure.w = 1;
            for (auto z_bin = 0; z_bin < 4; ++z_bin) {
                measure.z_bin = z_bin;
                const auto z_rt_center = FLOAT(0.3 * z_bin + 0.45);
                for (auto i = 0u; i < TEST_SIZE; ++i) {
                    for (auto j = 0u; j < TEST_SIZE; ++j) {
                        measure.x = (FLOAT(i) / TEST_SIZE - FLOAT(0.5)) * FIELD_SIZE_DEG;
                        measure.y = (FLOAT(j) / TEST_SIZE - FLOAT(0.5)) * FIELD_SIZE_DEG;
                        measure.z_rt = z_rt_center + z_rt_dist(random_engine);
                        measure.z_best = measure.z_rt;
                        catalog.push_back(measure);
                    }
                }
            }
        }
    }
    if (SIGMA8_TEST) {
        // Replace each of the fields by a regular grid, z at 0.4, 0.7, 1.0, 1.3
        constexpr auto TEST_SIZE = size_t(256);
        for (size_t i_field = 0; i_field < N_field; ++i_field) {
            auto& catalog = catalogs_all_z[i_field];
            catalog.clear();
            auto measure = real_galaxy_measure{};
            measure.w = 1;
            for (auto z_bin = 0; z_bin < 4; ++z_bin) {
                measure.z_bin = z_bin;
                const auto z_rt_center = FLOAT(0.3 * z_bin + 0.4);
                for (auto i = 0u; i < TEST_SIZE; ++i) {
                    for (auto j = 0u; j < TEST_SIZE; ++j) {
                        measure.x = (FLOAT(i) / TEST_SIZE - FLOAT(0.5)) * FIELD_SIZE_DEG;
                        measure.y = (FLOAT(j) / TEST_SIZE - FLOAT(0.5)) * FIELD_SIZE_DEG;
                        measure.z_rt = z_rt_center;
                        measure.z_best = measure.z_rt;
                        catalog.push_back(measure);
                    }
                }
            }
        }
    }

    auto weighted_shear_binary_data = std::vector<COMPLEX>{};
    auto weighted_tomo_shear_binary_data = std::vector<COMPLEX>{};
    auto catalogs_binary_data = std::vector<sim_galaxy_measure>{};
    auto IA_catalogs_binary_data = std::vector<sim_galaxy_measure>{};

    for (auto random_seed = initial_random_seed; random_seed < initial_random_seed + num_realizations; ++random_seed) {
        // Generate random redshift shifts for each bin
        auto z_shifts = std::array<FLOAT, S16A_NUM_Z_BIN>{};
        printf("random_seed = %d, z_shifts = (", random_seed);
        for (auto z_bin = 0; z_bin < int(S16A_NUM_Z_BIN); ++z_bin) {
            auto random = std::max(std::min(normal_dist(random_engine), FLOAT(3)), FLOAT(-3));
            z_shifts[z_bin] = delta_z_intensity * random * S16A_Z_BIN_ERROR[z_bin];
            printf("%g, ", z_shifts[z_bin]);
        }
        printf(")\n");
        auto gal_angles = std::vector<std::array<FLOAT, 2>>{};
        auto gal_distances_mpch = std::vector<FLOAT>{};
        auto gal_z_bins = std::vector<int>{};
        auto gal_ranges_all_fields = std::vector<size_t>{size_t(0)};
        for (size_t i_field = 0; i_field < N_field; ++i_field) {
            const auto& catalog = catalogs_all_z[i_field];
            for (const auto& measure : catalog) {
                const auto z_bin = int(measure.z_bin);
                if (z_bin < 0 || z_bin >= int(S16A_NUM_Z_BIN)) {
                    printf("ERROR: z_bin = %d, which is not between 0 and (S16A_NUM_Z_BIN - 1).\n", z_bin);
                    exit(1);
                }
                const auto z_rt = std::max(measure.z_rt + z_shifts[z_bin], FLOAT(0.001));
                gal_angles.push_back({FLOAT(measure.x * DEGREE), FLOAT(measure.y * DEGREE)});
                gal_distances_mpch.push_back(z_to_chi(z_rt));
                gal_z_bins.push_back(z_bin);
            }
            gal_ranges_all_fields.push_back(gal_ranges_all_fields.back() + catalog.size());
        }
        
        auto slab_choices = std::vector<uint32_t>(config_size);
        auto slab_shuffles = std::vector<slab_shuffle>(config_size * N_field);
        get_slab_randomizations(uint32_dist(random_engine), slab_choices, slab_shuffles);

        auto phis = std::vector<std::vector<FLOAT>>(config_size);

        printf("Loading density and potential planes with %d threads...\n", num_threads);
        printf("BCM = {%g, %g, %g, %g}\n", bcm_list.at(random_seed).Mc, bcm_list.at(random_seed).M1_0, bcm_list.at(random_seed).eta, bcm_list.at(random_seed).beta);
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        for (size_t i = 0; i < config.size(); ++i) {
            const auto slab = load_single_bcm_density_slab(
                density_dir, i, config[i], bcm_patches_dir, bcm_list.at(random_seed), bcm_intensity, slab_choices[i]);
            phis[i] = slab_to_potential(
                slab, config[i].omega_m, config[i].hubble_0, config[i].redshift,
                SIMULATION_SIZE_MPCH, config[i].thickness_mpch, config[i].distance_mpch);
        }

        const auto [A, dA_IA] = ray_trace(
            phis, config, 
            gal_angles, gal_distances_mpch, gal_z_bins, gal_ranges_all_fields,
            slab_shuffles);

        // phi's are no longer needed after raytracing, so free the memory
        (std::vector<std::vector<FLOAT>>{}).swap(phis);

        constexpr auto N = MAP_SIZE;
        constexpr auto TN = TOMO_MAP_SIZE;
        constexpr auto SHEAR_STEP = N * N;
        constexpr auto TOMO_SHEAR_STEP = S16A_NUM_Z_BIN * TN * TN;
        auto weighted_shear_maps = std::vector<COMPLEX>(N_field * SHEAR_STEP);
        auto weighted_IA_maps = std::vector<COMPLEX>(N_field * SHEAR_STEP);
        auto weighted_tomo_shear_maps = std::vector<COMPLEX>(N_field * TOMO_SHEAR_STEP);
        auto weighted_tomo_IA_maps = std::vector<COMPLEX>(N_field * TOMO_SHEAR_STEP);

        auto sim_catalogs = std::vector<std::vector<sim_galaxy_measure>>(N_field);
        auto sim_IA_catalogs = std::vector<std::vector<sim_galaxy_measure>>(N_field);

        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        for (size_t i_field = 0; i_field < N_field; ++i_field) {
            const auto& catalog = catalogs_all_z[i_field];
            auto& sim_catalog = sim_catalogs[i_field];
            auto& sim_IA_catalog = sim_IA_catalogs[i_field];
            const auto i_begin = gal_ranges_all_fields[i_field];
            const auto i_end = gal_ranges_all_fields[i_field + 1];
            for (size_t i = i_begin; i < i_end; ++i) {
                const auto& A_i = A[i];
                const auto& dA_IA_i = dA_IA[i];
                sim_catalog.push_back(sim_galaxy_measure{
                    1 - (A_i[0] + A_i[3]) / 2, (A_i[3] - A_i[0]) / 2, -(A_i[1] + A_i[2]) / 2
                });
                sim_IA_catalog.push_back(sim_galaxy_measure{
                    -(dA_IA_i[0] + dA_IA_i[3]) / 2, (dA_IA_i[3] - dA_IA_i[0]) / 2, -(dA_IA_i[1] + dA_IA_i[2]) / 2
                });
            }

            if (OUTPUT_SHEAR_MAPS) {
                const auto combined_shear = combine_real_and_sim_catalog(catalog, sim_catalog, 0);
                const auto combined_IA = combine_real_and_sim_catalog(catalog, sim_IA_catalog, 0);
                const auto combined_tomo_shear = split_catalog_to_tomographic_bins(combined_shear);
                const auto combined_tomo_IA = split_catalog_to_tomographic_bins(combined_IA);

                const auto weighted_shear_map = calc_weighted_shear_field(combined_shear);
                const auto weighted_IA_map = calc_weighted_shear_field(combined_IA);
                const auto weighted_tomo_shear_map = calc_weighted_shear_tomo_field(combined_tomo_shear);
                const auto weighted_tomo_IA_map = calc_weighted_shear_tomo_field(combined_tomo_IA);
                std::copy_n(begin(weighted_shear_map), SHEAR_STEP, begin(weighted_shear_maps) + i_field * SHEAR_STEP);
                std::copy_n(begin(weighted_IA_map), SHEAR_STEP, begin(weighted_IA_maps) + i_field * SHEAR_STEP);
                std::copy_n(begin(weighted_tomo_shear_map), TOMO_SHEAR_STEP, begin(weighted_tomo_shear_maps) + i_field * TOMO_SHEAR_STEP);
                std::copy_n(begin(weighted_tomo_IA_map), TOMO_SHEAR_STEP, begin(weighted_tomo_IA_maps) + i_field * TOMO_SHEAR_STEP);
            }
        }

        if (OUTPUT_CATALOGS) {
            for (const auto& sim_catalog : sim_catalogs) {
                catalogs_binary_data.insert(end(catalogs_binary_data), begin(sim_catalog), end(sim_catalog));
            }
            for (const auto& sim_IA_catalog : sim_IA_catalogs) {
                IA_catalogs_binary_data.insert(end(IA_catalogs_binary_data), begin(sim_IA_catalog), end(sim_IA_catalog));
            }
        }
        if (OUTPUT_SHEAR_MAPS) {
            weighted_shear_binary_data.insert(end(weighted_shear_binary_data),
                begin(weighted_shear_maps), end(weighted_shear_maps));
            weighted_shear_binary_data.insert(end(weighted_shear_binary_data),
                begin(weighted_IA_maps), end(weighted_IA_maps));
            weighted_tomo_shear_binary_data.insert(end(weighted_tomo_shear_binary_data),
                begin(weighted_tomo_shear_maps), end(weighted_tomo_shear_maps));
            weighted_tomo_shear_binary_data.insert(end(weighted_tomo_shear_binary_data),
                begin(weighted_tomo_IA_maps), end(weighted_tomo_IA_maps));
        }
    }

    constexpr auto seed_str_size = size_t(10);
    auto seed_str = std::to_string(initial_random_seed);
    if (seed_str.size() < seed_str_size) {
        seed_str.insert(begin(seed_str), (seed_str_size - seed_str.size()), '0');
    }
    seed_str += "." + std::to_string(num_realizations);
    const auto bcm_str = std::string(bcm_intensity > 0 ? (bcm ? ".fiducial" : ".free-test") : "");
    const auto dz_str = std::string(delta_z_intensity == FLOAT(0) ? ".no-dz" : "");
    const auto tags_str = bcm_str + dz_str;
    if (tags_str != std::string(".no-dz")) {
        printf("ERROR: tags_str must be \".no-dz\" for now\n");
        std::exit(1);
    }

    if (OUTPUT_CATALOGS) {
        binary_write(raytracing_dir + "/sim-catalog-allfields." + seed_str + tags_str + ".f32",
            catalogs_binary_data.data(), catalogs_binary_data.size());
        binary_write(raytracing_dir + "/sim-ia-catalog-allfields." + seed_str + tags_str + ".f32",
            IA_catalogs_binary_data.data(), IA_catalogs_binary_data.size());
    }
    if (OUTPUT_SHEAR_MAPS) {
        binary_write(raytracing_dir + "/sim-wg-allfields." + seed_str + tags_str + ".c64",
            weighted_shear_binary_data.data(), weighted_shear_binary_data.size());
        binary_write(raytracing_dir + "/sim-wg-tomo-allfields." + seed_str + tags_str + ".c64",
            weighted_tomo_shear_binary_data.data(), weighted_tomo_shear_binary_data.size());
    }
}

void calc_density_maps(
    std::string slab_config,
    std::string halos_dir,
    std::string density_dir,
    const double min_halo_mass_msunh
) {
    const auto config = read_slab_config(slab_config);

    omp_set_max_active_levels(2);
    const int num_tasks = std::min(omp_get_max_threads(), 2); // 2 threads max
    printf("Iterating over slab config with %d tasks...\n", num_tasks);
    #pragma omp parallel for num_threads(num_tasks) schedule(dynamic, 1)
    for (size_t order = 0; order < config.size(); ++order) {
        const auto& entry = config[order];
        const auto dz = entry.thickness_mpch / SIMULATION_SIZE_MPCH;
        if (dz > 1.0) {
            printf("ERROR: dz > 1.0, no plane available.\n");
            std::exit(1);
        }
        constexpr auto MAX_NUM_SLABS = size_t(4);
        const auto num_slabs = std::min(MAX_NUM_SLABS, static_cast<size_t>(std::floor(1.0 / dz)));
        printf("box size = %g Mpc/h, thickness = %g Mpc/h, N = %zu, bounds =",
            SIMULATION_SIZE_MPCH, entry.thickness_mpch, num_slabs);
        auto bounds = std::vector<std::pair<double, double>>{};
        for (size_t i = 0; i < num_slabs; ++i) {
            bounds.push_back({double(i) / num_slabs, double(i) / num_slabs + dz});
            printf(" (%g,%g)", bounds.back().first, bounds.back().second);
        }
        printf("\n");

        const auto positions = read_tipsy(entry.filename);
        const auto halo_info = read_halos(
            get_halo_files(entry.filename, halos_dir), positions, min_halo_mass_msunh);
        const auto& halos = halo_info.first;
        const auto& halo_particles = halo_info.second;

        std::vector<std::pair<size_t, size_t>> args{};
        for (size_t axis = 0; axis < 3u; ++axis) {
            for (size_t i = 0; i < num_slabs; ++i) {
                args.push_back({axis, i});
            }
        }

        // density planes are generated in parallel
        auto generate_density_plane = [&](size_t axis, size_t i) {
            const auto lower = bounds[i].first, upper = bounds[i].second;
            auto slab = get_slab(positions, axis, lower, upper);
            auto halos_in_slab = std::vector<halo_t>{};
            auto positions_in_slab = std::vector<position_t>{};
            const auto num_halos = halos.size();
            for (size_t i = 0; i < num_halos; ++i) {
                const auto halo_center_z = halos[i].pos[axis] / double(int64_t(UINT32_MAX) + 1);
                const auto halo_r = halos[i].r_vir / (1000.0 * SIMULATION_SIZE_MPCH);
                if (lower < halo_center_z - halo_r && halo_center_z + halo_r < upper) {
                    halos_in_slab.push_back(halos[i]);
                    for (auto& id : halo_particles[i]) {
                        positions_in_slab.push_back(positions[id]);
                    }
                }
            }
            auto halo_slab = get_slab(positions_in_slab, axis, lower, upper);
            auto halo_slab_particle_count = std::accumulate(begin(halo_slab), end(halo_slab), size_t{});
            printf("Within slab (axis = %zu, plane = %zu), n_halos = %zu, n_halo_particles = %zu, n_in_slab = %zu\n",
                axis, i, halos_in_slab.size(), positions_in_slab.size(), halo_slab_particle_count);
            std::transform(begin(slab), end(slab), begin(halo_slab), begin(slab), std::minus<>{});

            // if particle count is negative in a cell, give a warning and set it to zero
            static_assert(sizeof(slab[0]) == 4u);
            for (auto& x : slab) {
                if (static_cast<int32_t>(x) < 0) {
                    printf("WARNING: Particle count (N=%d) in background (axis=%zu, plane=%zu) is negative.\n",
                        static_cast<int32_t>(x), axis, i);
                    x = 0;
                }
            }
            
            const auto slab_z = zlib_compress(slab.data(), slab.size());
            binary_write(
                density_dir + "/density_background_order" + std::to_string(order)
                    + "_axis" + std::to_string(axis)
                    + "_plane" + std::to_string(i) + ".u32.z",
                slab_z.data(), slab_z.size()
            );
            const auto halo_slab_z = zlib_compress(halo_slab.data(), halo_slab.size());
            binary_write(
                density_dir + "/density_halo_order" + std::to_string(order)
                    + "_axis" + std::to_string(axis)
                    + "_plane" + std::to_string(i) + ".u32.z",
                halo_slab_z.data(), halo_slab_z.size()
            );
            binary_write(
                density_dir + "/halo_info_order" + std::to_string(order)
                    + "_axis" + std::to_string(axis)
                    + "_plane" + std::to_string(i) + ".halo_t",
                halos_in_slab.data(), halos_in_slab.size()
            );
        };

        const int num_threads = std::min(omp_get_max_threads(), 12); // 12 threads max
        printf("Generating density planes with %d threads...\n", num_threads);
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        for (auto task = 0u; task < args.size(); ++task) {
            auto [axis, i] = args.at(task);
            generate_density_plane(axis, i);
        }
    }
}

void calc_bcm_patches(std::string bcm_patches_dir) {
    printf("calc_bcm_patches(), bcm_patches_dir = %s\n", bcm_patches_dir.c_str());
    printf("    Range of r_vir: [%g, %g] kpc/h, c: [%g, %g]\n",
        FN_HALO_RVIR_PIXEL(0) * (SIMULATION_SIZE_MPCH * 1000.0 / POTENTIAL_SIZE),
        FN_HALO_RVIR_PIXEL(HALO_RVIR_COUNT - 1) * (SIMULATION_SIZE_MPCH * 1000.0 / POTENTIAL_SIZE),
        FN_HALO_C(0), FN_HALO_C(HALO_C_COUNT - 1));

    auto offsets = std::vector<size_t>{0};
    auto nfw_data = std::vector<FLOAT>{};
    auto bg_data = std::vector<FLOAT>{};
    auto cg_data = std::vector<FLOAT>{};
    auto rdmbg_data = std::vector<FLOAT>{};
    auto rdmcg_data = std::vector<FLOAT>{};
    auto rdmego_data = std::vector<FLOAT>{}; // ejected gas beyond the virial radius
    auto rdmegi_data = std::vector<FLOAT>{}; // ejected gas within the virial radius

    auto nfw_dMdr = std::vector<double>(RADIAL_RESOLUTION);
    auto nfw_Mr = std::vector<double>(RADIAL_RESOLUTION);
    auto baryon_dMdr = std::vector<double>(RADIAL_RESOLUTION);
    auto baryon_Mr = std::vector<double>(RADIAL_RESOLUTION);
    auto rdm_rho_r = std::vector<double>(RADIAL_RESOLUTION);
    auto r_marks = std::vector<double>(RADIAL_RESOLUTION);
    auto rf_marks = std::vector<double>(RADIAL_RESOLUTION);
    for (size_t i = 0; i < RADIAL_RESOLUTION; ++i) {
        r_marks[i] = (i + 0.5) / RADIAL_RESOLUTION;
    }

    for (size_t i = 0; i < HALO_RVIR_COUNT; ++i) {
        for (size_t j = 0; j < HALO_C_COUNT; ++j) {
            // halo constants
            constexpr auto SQRT_5 = 2.2360679774997896964;
            const auto a          = 0.68; // r_i/r_f -- M_f/M_i exponent
            const auto rvir_px    = FN_HALO_RVIR_PIXEL(i);
            const auto c          = FN_HALO_C(j);
            const auto rs_px      = rvir_px / c;
            const auto radius_px  = static_cast<size_t>(std::ceil(rvir_px));
            const auto rvir5_px   = rvir_px / SQRT_5;
            const auto c5         = c / SQRT_5;
            const auto gamma_eff  = ((1 + 3 * c5) * std::log(1 + c5)) / ((1 + c5) * std::log(1 + c5) - c5);
            const auto edge_px    = std::max(0.0, rvir_px - 1.5); // the starting radius of the edge
            const auto edge_slope = 1 / (rvir_px - edge_px);

            // profile of the components, NFW, CG, BG, EGO, EGI
            auto nfw = [=](auto r_px) {
                const auto x = r_px / rs_px;
                return r_px >= rvir_px ? 0.0 : 1 / (x * sqr(1 + x));
            };
            auto cg = [=](auto r_px) {
                return r_px >= rvir_px ? 0.0 : std::exp(-sqr(r_px) / (sqr(2 * 0.015 * rvir_px)));
            };
            auto bg = [=](auto r_px) {
                const auto x = r_px / rs_px;
                return r_px >= rvir_px ? 0.0 :
                    r_px >= rvir5_px ? (c5 * sqr(1 + c5)) / (x * sqr(1 + x)) :
                    std::pow((std::log(1 + x) / x) / (std::log(1 + c5) / c5), gamma_eff);
            };
            auto ego = [=](auto r_px) {
                return r_px <= edge_px ? 0.0 : edge_slope * (r_px - edge_px);
            };
            auto egi = [=](auto r_px) {
                return 1.0;
            };

            auto nfw_patch = get_component_patch(nfw, radius_px);
            nfw_data.insert(end(nfw_data), begin(nfw_patch), end(nfw_patch));
            auto cg_patch = get_component_patch(cg, radius_px);
            cg_data.insert(end(cg_data), begin(cg_patch), end(cg_patch));
            auto bg_patch = get_component_patch(bg, radius_px);
            bg_data.insert(end(bg_data), begin(bg_patch), end(bg_patch));
            offsets.push_back(nfw_data.size());

            auto calc_Mr = [&](auto& dMdr, auto& Mr, const auto& rho_r) {
                std::transform(begin(r_marks), end(r_marks), begin(dMdr),
                    [&](auto f) { return rho_r(f * rvir_px) * (f * f); });
                std::partial_sum(begin(dMdr), end(dMdr), begin(Mr));
                auto sum = Mr.back();
                std::transform(begin(dMdr), end(dMdr), begin(dMdr),
                    [factor = RADIAL_RESOLUTION / sum](auto x) { return factor * x; });
                std::transform(begin(Mr), end(Mr), begin(Mr),
                    [factor = 1 / sum](auto x) { return factor * x; });
            };
            // calculate M(r) and dM/dr of NFW
            calc_Mr(nfw_dMdr, nfw_Mr, nfw);

            auto interpolate = [](const auto& arr, double index) {
                const auto integer = static_cast<ssize_t>(std::floor(index));
                const auto fraction = index - integer;
                return integer < 0 ? arr.front() :
                    integer + 1 >= ssize_t(arr.size()) ? arr.back() :
                    (1 - fraction) * arr[integer] + fraction * arr[integer + 1];
            };

            auto calc_rdm_data = [&](auto& rdm_data, const auto& baryon_rho_r) {
                // calculate M(r) and dM/dr of the baryonic component
                calc_Mr(baryon_dMdr, baryon_Mr, baryon_rho_r);
                // r_f(r_i)
                for (size_t i = 0; i < RADIAL_RESOLUTION; ++i) {
                    const auto ri_mark = double(i + 1) / RADIAL_RESOLUTION;
                    rf_marks[i] = ri_mark - BCM_DERIVATIVE_DR * (
                        (baryon_Mr[i] - nfw_Mr[i]) / (nfw_dMdr[i] + nfw_Mr[i] / ri_mark / a));
                }
                // rho(r) with BCM_DERIVATIVE_DR of the baryonic component
                for (size_t i = 0; i < RADIAL_RESOLUTION; ++i) {
                    const auto ri_mark = double(i + 1) / RADIAL_RESOLUTION;
                    const auto rf_idx = std::min(ssize_t(RADIAL_RESOLUTION) - 1, 
                        std::upper_bound(begin(rf_marks), end(rf_marks), ri_mark) - begin(rf_marks));
                    const auto residual = rf_idx == 0 ? (ri_mark - rf_marks[0]) / (rf_marks[0] - 0) :
                        (ri_mark - rf_marks[rf_idx]) / (rf_marks[rf_idx] - rf_marks[rf_idx - 1]);                    
                    const auto ri_diff = FLOAT(1) / RADIAL_RESOLUTION;
                    const auto rf_diff = rf_idx == 0 ? rf_marks[0] : rf_marks[rf_idx] - rf_marks[rf_idx - 1];
                    rdm_rho_r[i] = interpolate(nfw_dMdr, rf_idx + 0.5 + residual) * (ri_diff / rf_diff) / sqr(r_marks[i]);
                    // rdm_rho_r[i] = nfw_dMdr[rf_idx] * (ri_diff / rf_diff) / sqr(r_marks[i]);
                    // const auto rf_mark = double(rf_idx + 0.5) / RADIAL_RESOLUTION;
                    // rdm_rho_r[i] = nfw(rvir_px * rf_mark) * sqr(rf_mark) * (ri_diff / rf_diff) / sqr(r_marks[i]);
                }
                auto rdm_rho = [&](auto r_px) {
                    const auto idx = static_cast<ssize_t>(std::floor(RADIAL_RESOLUTION * (r_px / rvir_px)));
                    return idx >= ssize_t(RADIAL_RESOLUTION) ? FLOAT(0) : rdm_rho_r[idx];
                };
                // if (offsets.size() == 1114) {
                //     printf("rvir = %g, c = %g\n", rvir_px, c);
                //     binary_write("rdm_rho_r.f32", rdm_rho_r.data(), rdm_rho_r.size());
                //     binary_write("nfw_dMdr.f32", nfw_dMdr.data(), nfw_dMdr.size());
                //     binary_write("nfw_Mr.f32", nfw_Mr.data(), nfw_Mr.size());
                //     binary_write("baryon_dMdr.f32", baryon_dMdr.data(), baryon_dMdr.size());
                //     binary_write("baryon_Mr.f32", baryon_Mr.data(), baryon_Mr.size());
                //     binary_write("rf_marks.f32", rf_marks.data(), rf_marks.size());
                // }
                auto rdm_patch = get_component_patch(rdm_rho, radius_px);
                // if (offsets.size() == 1114) {
                //     binary_write("nfw_patch.f32", nfw_patch.data(), nfw_patch.size());
                //     binary_write("rdm_patch.f32", rdm_patch.data(), rdm_patch.size());
                // }
                std::transform(begin(rdm_patch), end(rdm_patch), begin(nfw_patch), begin(rdm_patch),
                    [](auto rdm, auto nfw) { return (rdm - nfw) / BCM_DERIVATIVE_DR; });
                // if (offsets.size() == 1114) {
                //     binary_write("rdm_diff_patch.f32", rdm_patch.data(), rdm_patch.size());
                // }
                rdm_data.insert(end(rdm_data), begin(rdm_patch), end(rdm_patch));
            };

            calc_rdm_data(rdmcg_data, cg);
            calc_rdm_data(rdmbg_data, bg);
            calc_rdm_data(rdmego_data, ego);
            calc_rdm_data(rdmegi_data, egi);
        }
    }

    binary_write(bcm_patches_dir + "/offsets.u64", offsets.data(), offsets.size());
    binary_write(bcm_patches_dir + "/nfw_data.f32", nfw_data.data(), nfw_data.size());
    binary_write(bcm_patches_dir + "/cg_data.f32", cg_data.data(), cg_data.size());
    binary_write(bcm_patches_dir + "/bg_data.f32", bg_data.data(), bg_data.size());
    binary_write(bcm_patches_dir + "/rdmcg_data.f32", rdmcg_data.data(), rdmcg_data.size());
    binary_write(bcm_patches_dir + "/rdmbg_data.f32", rdmbg_data.data(), rdmbg_data.size());
    binary_write(bcm_patches_dir + "/rdmego_data.f32", rdmego_data.data(), rdmego_data.size());
    binary_write(bcm_patches_dir + "/rdmegi_data.f32", rdmegi_data.data(), rdmegi_data.size());
}

}

int main(int argc, char* argv[]) {
    fftw_init_threads();
    fftwf_init_threads();
    fftw_plan_with_nthreads(1);
    fftwf_plan_with_nthreads(1);

    printf("Simulation box size:     %g Mpc/h\n", SIMULATION_SIZE_MPCH);
    printf("Potential plane size:    %d pixels\n", int(POTENTIAL_SIZE));
    printf("Convergence map FOV:     %g degrees\n", FIELD_SIZE_DEG);
    printf("Convergence map size:    %d pixels\n", int(MAP_SIZE));
    // printf("Convergence map padding: %d pixels\n", int(MAP_PADDING));

    if (argc >= 10 && argv[1] == std::string("calc_hsc_ray_tracing")) {
        const auto baryon_model = std::string(argv[7]);
        auto bcm_intensity = FLOAT{};
        auto bcm = std::optional<bcm_parameters>{};
        if (baryon_model == "fiducial") {
            bcm_intensity = FLOAT(1);
            bcm = bcm_parameters{3.3e13, 8.63e11, 0.54, 0.12};
        } else if (baryon_model == "free") {
            bcm_intensity = FLOAT(1);
            bcm = std::nullopt;
        } else if (baryon_model == "none") {
            bcm_intensity = FLOAT(0);
            bcm = bcm_parameters{3.3e13, 8.63e11, 0.54, 0.12};
        } else {
            printf("ERROR: baryon_model must be one of \"fiducial\", \"free\", or \"none\".\n");
            exit(1);
        }
        const auto delta_z_intensity = FLOAT(0);
        _main::calc_hsc_ray_tracing(argv[2], argv[3], argv[4], argv[5], argv[6], bcm, bcm_intensity, delta_z_intensity, std::atoi(argv[8]), std::atoi(argv[9]));
    } else if (argc >= 4 && argv[1] == std::string("calc_hsc_real_maps")) {
        // _main::calc_hsc_real_maps(argv[2], argv[3]);
    } else if (argc >= 6 && argv[1] == std::string("calc_density_maps")) {
        _main::calc_density_maps(argv[2], argv[3], argv[4], std::atof(argv[5]));
    } else if (argc >= 3 && argv[1] == std::string("calc_bcm_patches")) {
        _main::calc_bcm_patches(argv[2]);
    } else {
        printf(R"(usage:
hsclens calc_hsc_ray_tracing  <slab_config> <density_dir> <raytracing_dir> <catalog_dir> <bcm_patches_dir> <baryon_model> <num_realizations> <first_seed>
hsclens calc_hsc_real_maps    <raytracing_dir> <catalog_dir>
hsclens calc_density_maps     <slab_config> <halos_dir> <density_dir> <min_halo_mass[M_sun/h] (3e12)>
hsclens calc_bcm_patches      <bcm_patches_dir>
)");
        std::exit(1);
    }

    return 0;
}