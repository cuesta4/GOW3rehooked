// SPDX-License-Identifier: GPL-2.0-or-later
#include "main/config.h"
#include "main/host.h"
#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <set>
#include <span>
#include <stdexcept>

namespace Gow3 {
namespace {
std::string Trim(std::string_view text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start == text.npos) return {};
    return std::string(text.substr(start, text.find_last_not_of(" \t\r\n") - start + 1));
}

bool Boolean(std::string_view value, bool& out) {
    if (value != "true" && value != "false") return false;
    out = value == "true";
    return true;
}

constexpr std::array<std::string_view, 3> DisplayModes{"windowed", "borderless", "fullscreen"};
constexpr std::array<std::string_view, 3> VSyncModes{"fifo", "mailbox", "off"};
constexpr std::array<std::string_view, 3> AntiAliasingModes{"none", "cmaa2", "faaa"};
constexpr std::array<std::string_view, 2> ScalingFilters{"bilinear", "fsr1"};
constexpr std::array<std::string_view, 2> ShaderModes{"sync", "async"};
constexpr std::array<std::string_view, 2> ShaderMechanisms{"jit", "aot"};

template <typename Enum, size_t N>
bool Choice(std::string_view value, Enum& out, const std::array<std::string_view, N>& choices) {
    const auto it = std::ranges::find(choices, value);
    if (it == choices.end()) return false;
    out = static_cast<Enum>(it - choices.begin());
    return true;
}
}

Config LoadConfig(const std::filesystem::path& path) {
    Config config;
    if (!std::filesystem::exists(path)) return config;
    std::ifstream file(path);
    if (!file) { Log("Cannot read config.cfg; using defaults"); return config; }
    std::set<std::string> seen;
    std::string line;
    unsigned number{};
    while (std::getline(file, line)) {
        ++number;
        if (number == 1 && line.starts_with("\xEF\xBB\xBF")) line.erase(0, 3);
        line = Trim(line.substr(0, line.find_first_of("#;")));
        if (line.empty()) continue;
        auto equals = line.find('=');
        if (equals == line.npos) { Log("Invalid config line " + std::to_string(number)); continue; }
        const auto key = Trim(line.substr(0, equals));
        const auto value = Trim(line.substr(equals + 1));
        bool valid{};
        if (!seen.insert(key).second) Log("Duplicate config option: " + key);
        if (key == "resolution") {
            const auto it = std::ranges::find(Resolutions, value, &Resolution::name);
            if ((valid = it != Resolutions.end())) config.resolution = unsigned(it - Resolutions.begin());
        } else if (key == "fps") {
            const auto it = std::ranges::find_if(FrameRates, [&](const auto& rate) {
                return value == rate.name || value == std::to_string(rate.nominal);
            });
            if ((valid = it != FrameRates.end())) config.fps = it->nominal;
        } else if (key == "anisotropy") {
            unsigned parsed{};
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if ((valid = error == std::errc{} && end == value.data() + value.size() &&
                         std::ranges::find(AnisotropyLevels, parsed) != AnisotropyLevels.end())) config.anisotropy = parsed;
        } else if (key == "msaa") {
            unsigned parsed{};
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if ((valid = error == std::errc{} && end == value.data() + value.size() &&
                         std::ranges::find(MsaaLevels, parsed) != MsaaLevels.end())) config.msaa = parsed;
        } else if (key == "skip_intro") {
            valid = Boolean(value, config.skip_intro);
        } else if (key == "skip_videos") {
            valid = Boolean(value, config.skip_videos);
        } else if (key == "cinematics_4k") {
            valid = Boolean(value, config.cinematics_4k);
        } else if (key == "display_mode") {
            valid = Choice(value, config.display_mode, DisplayModes);
        } else if (key == "vsync") {
            valid = Choice(value, config.vsync, VSyncModes);
        } else if (key == "vrr") {
            valid = Boolean(value, config.vrr);
        } else if (key == "vrr_pacing") {
            bool ignored{};
            valid = Boolean(value, ignored); // compatibility with older config files
        } else if (key == "nvidia_reflex") {
            valid = Boolean(value, config.nvidia_reflex);
        } else if (key == "anti_aliasing") {
            valid = Choice(value, config.anti_aliasing, AntiAliasingModes);
        } else if (key == "scaling_filter") {
            if (value == "none" || value == "nearest") config.scaling_filter = ScalingFilter::Bilinear, valid = true;
            else valid = Choice(value, config.scaling_filter, ScalingFilters);
        } else if (key == "sharpening" || key == "rcas_attenuation" || key == "language") {
            valid = true; // options of the removed launcher, still present in older config files
        } else if (key == "shader_recompilation") {
            valid = Choice(value, config.shader_recompilation, ShaderModes);
        } else if (key == "shader_mechanism") {
            valid = Choice(value, config.shader_mechanism, ShaderMechanisms);
        } else if (key == "lazy_aot") {
            valid = Boolean(value, config.lazy_aot);
        } else if (key == "vibration") {
            unsigned parsed{};
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if ((valid = error == std::errc{} && end == value.data() + value.size() && parsed <= 100)) config.vibration = parsed;
        } else if (key == "cheat_infinite_health") {
            valid = Boolean(value, config.cheat_infinite_health);
        } else if (key == "cheat_infinite_magic") {
            valid = Boolean(value, config.cheat_infinite_magic);
        } else if (key == "cheat_infinite_item") {
            valid = Boolean(value, config.cheat_infinite_item);
        } else if (key == "cheat_infinite_rage") {
            valid = Boolean(value, config.cheat_infinite_rage);
        } else if (key == "cheat_max_red_orbs") {
            valid = Boolean(value, config.cheat_max_red_orbs);
        } else if (key == "cheat_infinite_double_jumps") {
            valid = Boolean(value, config.cheat_infinite_double_jumps);
        } else if (key == "cheat_unlock_chaos") {
            valid = Boolean(value, config.cheat_unlock_chaos);
        } else if (key == "cheat_auto_qte") {
            valid = Boolean(value, config.cheat_auto_qte);
        } else if (key == "remove_motion_blur") {
            valid = Boolean(value, config.remove_motion_blur);
        } else if (key == "remove_dof_spartan") {
            valid = Boolean(value, config.remove_dof_spartan);
        }
        if (!valid) Log(std::format("Invalid or unsupported config option: {}={}; using default", key, value));
    }
    if (file.bad()) Log("Error reading config.cfg; unspecified options use defaults");
    return config;
}

namespace {
void WriteConfigValues(const std::filesystem::path& path,
                       std::span<const std::pair<std::string_view, std::string>> values) {
    std::string content;
    if (std::filesystem::exists(path)) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot read config.cfg");
        content.assign(std::istreambuf_iterator<char>(file), {});
        if (file.bad()) throw std::runtime_error("Cannot read config.cfg");
    }
    std::string output;
    std::vector<bool> found(values.size());
    for (size_t start = 0; start < content.size();) {
        const size_t end = content.find('\n', start), next = end == content.npos ? content.size() : end + 1;
        std::string line = content.substr(start, next - start);
        const size_t prefix = start == 0 && line.starts_with("\xEF\xBB\xBF") ? 3 : 0;
        const auto equals = line.find('=', prefix), comment = line.find_first_of("#;", prefix);
        const auto key = equals != line.npos && (comment == line.npos || equals < comment) ? Trim(line.substr(prefix, equals - prefix)) : "";
        const auto it = std::ranges::find(values, key, &std::pair<std::string_view, std::string>::first);
        if (it != values.end()) {
            if (prefix) output.append(line, 0, prefix);
            output += std::string(it->first) + "=" + it->second;
            if (line.ends_with("\r\n")) output += "\r\n";
            else if (line.ends_with('\n')) output += '\n';
            found[size_t(it - values.begin())] = true;
        } else output += line;
        start = next;
    }
    for (size_t i = 0; i < values.size(); ++i) if (!found[i]) {
        if (!output.empty() && output.back() != '\n') output += '\n';
        output += std::string(values[i].first) + "=" + values[i].second + "\n";
    }
    auto temporary = path;
#ifdef _WIN32
    temporary += L"." + std::to_wstring(GetCurrentProcessId()) + L".tmp";
#else
    temporary += "." + std::to_string(getpid()) + ".tmp";
#endif
    try {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(output.data(), std::streamsize(output.size()));
        file.flush();
        if (!file) throw std::runtime_error("Cannot save config.cfg");
        file.close();
        if (!ReplaceFile(temporary, path))
            throw std::runtime_error(HostError("Save config.cfg"));
    } catch (...) { std::filesystem::remove(temporary); throw; }
}

std::array<std::pair<std::string_view, std::string>, 11> ExtrasValues(const Config& config) {
    const auto flag = [](bool value) { return value ? "true" : "false"; };
    return {{
        {"vibration", std::to_string(config.vibration)},
        {"cheat_infinite_health", flag(config.cheat_infinite_health)},
        {"cheat_infinite_magic", flag(config.cheat_infinite_magic)},
        {"cheat_infinite_item", flag(config.cheat_infinite_item)},
        {"cheat_infinite_rage", flag(config.cheat_infinite_rage)},
        {"cheat_max_red_orbs", flag(config.cheat_max_red_orbs)},
        {"cheat_infinite_double_jumps", flag(config.cheat_infinite_double_jumps)},
        {"cheat_unlock_chaos", flag(config.cheat_unlock_chaos)},
        {"cheat_auto_qte", flag(config.cheat_auto_qte)},
        {"remove_motion_blur", flag(config.remove_motion_blur)},
        {"remove_dof_spartan", flag(config.remove_dof_spartan)},
    }};
}
} // namespace

void SaveMenuConfig(const std::filesystem::path& path, const Config& config) {
    const auto rate = std::ranges::find(FrameRates, config.fps, &FrameRate::nominal);
    if (config.resolution >= Resolutions.size() || rate == FrameRates.end() ||
        std::ranges::find(AnisotropyLevels, config.anisotropy) == AnisotropyLevels.end() ||
        std::ranges::find(MsaaLevels, config.msaa) == MsaaLevels.end() ||
        unsigned(config.display_mode) >= DisplayModes.size() || unsigned(config.vsync) >= VSyncModes.size() ||
        unsigned(config.scaling_filter) >= ScalingFilters.size() || unsigned(config.anti_aliasing) >= AntiAliasingModes.size() ||
        unsigned(config.shader_recompilation) >= ShaderModes.size() ||
        unsigned(config.shader_mechanism) >= ShaderMechanisms.size() ||
        config.vibration > 100)
        throw std::runtime_error("Invalid menu configuration");
    const auto flag = [](bool value) { return value ? "true" : "false"; };
    std::vector<std::pair<std::string_view, std::string>> values{
        {"resolution", std::string(Resolutions[config.resolution].name)}, {"fps", std::string(rate->name)},
        {"display_mode", std::string(DisplayModes[size_t(config.display_mode)])}, {"vrr", flag(config.vrr)},
        {"vsync", std::string(VSyncModes[size_t(config.vsync)])},
        {"nvidia_reflex", flag(config.nvidia_reflex)}, {"msaa", std::to_string(config.msaa)}, {"anisotropy", std::to_string(config.anisotropy)},
        {"anti_aliasing", std::string(AntiAliasingModes[size_t(config.anti_aliasing)])}, {"scaling_filter", std::string(ScalingFilters[size_t(config.scaling_filter)])},
        {"skip_intro", flag(config.skip_intro)},
        {"skip_videos", flag(config.skip_videos)}, {"cinematics_4k", flag(config.cinematics_4k)},
        {"shader_recompilation", std::string(ShaderModes[size_t(config.shader_recompilation)])},
        {"shader_mechanism", std::string(ShaderMechanisms[size_t(config.shader_mechanism)])}, {"lazy_aot", flag(config.lazy_aot)},
    };
    const auto extras = ExtrasValues(config);
    values.insert(values.end(), extras.begin(), extras.end());
    WriteConfigValues(path, values);
}

void SaveExtrasConfig(const std::filesystem::path& path, const Config& config) {
    if (config.vibration > 100) throw std::runtime_error("Invalid extras configuration");
    const auto values = ExtrasValues(config);
    WriteConfigValues(path, values);
}
}
