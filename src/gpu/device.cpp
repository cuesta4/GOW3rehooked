// SPDX-License-Identifier: GPL-2.0-or-later
// The Vulkan instance, window surface and device shared by the renderer and the presenter, and the renderer's own device.
#include "gpu/device.h"
#include "gpu/renderer_impl.h"
#include <cctype>
#include <cstdlib>
#ifndef _WIN32
#include <dlfcn.h>
#include <execinfo.h>
#endif
#ifndef GOW3_OFFLINE  // the window's instance needs SDL and the emulator's runtime (framecap builds without them)
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "main/host.h"

// ---- Shared live Vulkan instance, window surface and device ----

namespace Gow3::Vulkan {
namespace {
Device context;
std::mutex queue_mutex;
} // namespace

void Check(VkResult result, const char* what) {
    if (result < 0) throw std::runtime_error(std::format("{} failed: {}", what, int(result)));
}

const Device& Get() { return context; }
std::mutex& QueueMutex() { return queue_mutex; }

void Initialize(SDL_Window* window) {
    u32 extension_count{};
    const auto* extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
    if (!extensions) throw std::runtime_error(std::string("SDL_Vulkan_GetInstanceExtensions: ") + SDL_GetError());
    const VkApplicationInfo app{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "God of War III",
                                .apiVersion = VK_API_VERSION_1_3};
    // GOW3_VK_VALIDATE=1 or GOW3_DEBUG=2: Khronos validation, each message ID logged a few times (skipped if layer is absent).
    bool validate = DebugDeep("GOW3_VK_VALIDATE");
    std::vector<const char*> instance_extensions(extensions, extensions + extension_count);
    const bool temp_fault = DebugAuto("GOW3_TEMP_FAULT");  // fault diagnostics: GPU address bindings
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (validate) {
        u32 layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> layers(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
        if (std::none_of(layers.begin(), layers.end(), [layer](const VkLayerProperties& p) { return std::strcmp(p.layerName, layer) == 0; })) {
            Log("Vulkan validation requested, but VK_LAYER_KHRONOS_validation is not available");
            validate = false;
        }
    }
    if (validate || temp_fault || DebugAuto("GOW3_TEMP_LABELS")) instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);  // labels
    u32 count = 0;
    std::string invalid_shader_path;
    const auto requested_shader_path = Gpu::RequestedShaderPath(&invalid_shader_path);
    if (!invalid_shader_path.empty())
        Log(std::format("Ignoring unrecognized GOW3_SHADER_PATH='{}' (expected auto, objects, gpl, or monolithic)", invalid_shader_path));
#ifdef _WIN32
    auto has = [](std::span<const VkExtensionProperties> list, const char* name) {
        return std::ranges::any_of(list, [&](const auto& e) { return !std::strcmp(e.extensionName, name); });
    };
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> instance_available(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, instance_available.data());
    const bool caps2 = has(instance_available, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);  // needed by full_screen_exclusive
    if (caps2) instance_extensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
#endif
    const VkInstanceCreateInfo instance_info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
        .enabledLayerCount = validate ? 1u : 0u, .ppEnabledLayerNames = &layer,
        .enabledExtensionCount = u32(instance_extensions.size()), .ppEnabledExtensionNames = instance_extensions.data()};
    Check(vkCreateInstance(&instance_info, nullptr, &context.instance), "Create Vulkan instance");

    auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(context.instance, "vkCreateDebugUtilsMessengerEXT"));
    auto add_messenger = [&](const VkDebugUtilsMessengerCreateInfoEXT& info) {
        VkDebugUtilsMessengerEXT handle{};
        if (create_messenger) create_messenger(context.instance, &info, nullptr, &handle);
    };
    if (validate) {
        add_messenger({.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
            .pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                  const VkDebugUtilsMessengerCallbackDataEXT* data, void*) -> VkBool32 {
                static std::mutex lock;
                static std::unordered_map<s32, u32> seen;
                std::scoped_lock guard(lock);
                if (const u32 n = ++seen[data->messageIdNumber]; n <= 4)
                    Log(std::format("VK {} #{} {}", severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "ERROR" : "WARN", n,
                                    data->pMessage ? data->pMessage : ""));
                return VK_FALSE;
            }});
    }
    if (temp_fault) {
        add_messenger({.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT,
            .pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                  const VkDebugUtilsMessengerCallbackDataEXT* data, void*) -> VkBool32 { return Gpu::TempAddressBinding(data), VK_FALSE; }});
    }
    if (!SDL_Vulkan_CreateSurface(window, context.instance, nullptr, &context.surface))
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface: ") + SDL_GetError());

    // Prefer a discrete GPU with a queue the renderer can use that presents to the window.
    vkEnumeratePhysicalDevices(context.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(context.instance, &count, devices.data());
    int best = -1;
    std::string lacking;  // presenting devices skipped for a required extension
#ifdef __linux__
    constexpr u32 amd_vendor_id = 0x1002;
    bool best_radeon = false, best_radv = false, missing_radeon_maintenance5 = false;
#endif
    for (auto candidate : devices) {
        VkPhysicalDeviceProperties info{};
        vkGetPhysicalDeviceProperties(candidate, &info);
        const int score = info.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1 : 0;
        if (info.apiVersion < VK_API_VERSION_1_3 || score < best) continue;

        const auto properties = Gpu::QueueFamilies(candidate);
        for (u32 i = 0; i < properties.size(); ++i) {
            if (!Gpu::QueueFamilySuitable(properties[i].queueFlags)) continue;
            VkBool32 presents{};
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, context.surface, &presents);
            if (!presents) continue;
            if (const auto missing = Gpu::MissingExtensions(candidate); !missing.empty()) {
                lacking += std::format("{}{} lacks{}", lacking.empty() ? "" : "; ", info.deviceName, missing);
                break;
            }

            bool preferred = score > best;
#ifdef __linux__
            const bool radeon = info.vendorID == amd_vendor_id;
            if (radeon && !Gpu::SupportsMaintenance5(candidate)) {
                missing_radeon_maintenance5 = true;
                break;
            }
            bool radv = false;
            if (radeon) {
                VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
                VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
                vkGetPhysicalDeviceProperties2(candidate, &properties2);
                radv = driver.driverID == VK_DRIVER_ID_MESA_RADV;
                preferred = preferred || (radv && best_radeon && !best_radv);
            }
            if (preferred) best_radeon = radeon, best_radv = radv;
#endif
            if (preferred) best = score, context.physical = candidate, context.family = i;
            break;
        }
    }
    if (best < 0 && !lacking.empty()) {
        if (requested_shader_path.has_value())
            throw std::runtime_error(std::format("{} (required because GOW3_SHADER_PATH={} is set). Update the GPU's Vulkan driver or unset GOW3_SHADER_PATH.",
                                                 lacking, Gpu::ShaderPathName(*requested_shader_path)));
        throw std::runtime_error(lacking + ". Update the GPU's Vulkan driver (on Linux, Mesa 24.1 or newer).");
    }
#ifdef __linux__
    if (best < 0 && missing_radeon_maintenance5)
        throw std::runtime_error("Radeon requires VK_KHR_maintenance5. Install or update the Mesa Vulkan driver (RADV).");
#endif
    if (best < 0) throw std::runtime_error("No Vulkan 1.3 device presents to the window");

    bool unified_layouts = false;
    std::vector<const char*> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
#ifdef _WIN32
    vkEnumerateDeviceExtensionProperties(context.physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> device_available(count);
    vkEnumerateDeviceExtensionProperties(context.physical, nullptr, &count, device_available.data());
    context.full_screen_exclusive = caps2 && has(device_available, "VK_EXT_full_screen_exclusive");
    if (context.full_screen_exclusive) device_extensions.push_back("VK_EXT_full_screen_exclusive");
#endif
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
    vkGetPhysicalDeviceProperties2(context.physical, &properties);
    std::string requested;
    Gpu::DynamicCaps caps{};
    const VkResult created = Gpu::CreateDevice(context.physical, context.family, device_extensions, &context.device, &context.present_wait,
                                               &context.reflex, &unified_layouts, &requested, &caps);
    Log(std::format("Vulkan device: {} ({} {}); extensions:{}", properties.properties.deviceName, driver.driverName, driver.driverInfo, requested));
    {  // VRAM: the largest device-local heap (AMD's 256 MB BAR window is another one over the same VRAM)
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(context.physical, &mp);
        VkDeviceSize vram = 0;
        for (u32 i = 0; i < mp.memoryHeapCount; ++i)
            if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram = std::max(vram, mp.memoryHeaps[i].size);
        std::string available;
#ifdef _WIN32
        if (MEMORYSTATUSEX ms{sizeof(ms)}; GlobalMemoryStatusEx(&ms))
            available = std::format(" ({} MB available, commit limit {} MB)", ms.ullAvailPhys >> 20, ms.ullTotalPageFile >> 20);
#endif
        Log(std::format("Memory: RAM {} MB{}, VRAM {} MB{}", SDL_GetSystemRAM(), available, vram >> 20,
                        properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? " (integrated GPU: part of the RAM)" : ""));
    }
    Check(created, "Create Vulkan device");
    Log(std::format(
        "Vulkan shader path: {}{} (cache_control {} m5 {} fast_link {} unrestricted_topo {} depth_bounds {} storage_read_nofmt {} | eds3: neg_one {} clamp {} samples {} a2c {} blend_en {} blend_eq {} mask {})",
        Gpu::ShaderPathName(caps.path), caps.forced ? " [GOW3_SHADER_PATH]" : " [auto]", caps.cache_control, caps.maintenance5,
        caps.gpl_fast_link, caps.unrestricted_topology, caps.depth_bounds, caps.storage_image_read_without_format,
        caps.depth_clip_negative_one_to_one, caps.depth_clamp_enable,
        caps.rasterization_samples, caps.alpha_to_coverage_enable, caps.color_blend_enable, caps.color_blend_equation, caps.color_write_mask));
    Log(std::format("Vulkan Unified Image Layouts: {}", unified_layouts ? "supported and enabled" : "not supported (disabled)"));
    vkGetDeviceQueue(context.device, context.family, 0, &context.queue);
}
} // namespace Gow3::Vulkan
#endif

namespace Gpu {
namespace {
std::vector<VkExtensionProperties> DeviceExtensions(VkPhysicalDevice pd) {
    u32 n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> available(n);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, available.data());
    return available;
}

bool HasExtension(std::span<const VkExtensionProperties> available, const char* name) {
    return std::any_of(available.begin(), available.end(), [&](const auto& e) { return !std::strcmp(e.extensionName, name); });
}

bool SupportsBlendInterlock(VkPhysicalDevice pd, std::span<const VkExtensionProperties> available) {
    if (!HasExtension(available, VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME) ||
        !HasExtension(available, VK_EXT_POST_DEPTH_COVERAGE_EXTENSION_NAME))
        return false;
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT interlock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &interlock};
    vkGetPhysicalDeviceFeatures2(pd, &query);
    return interlock.fragmentShaderPixelInterlock && query.features.shaderStorageImageMultisample && query.features.shaderStorageImageArrayDynamicIndexing;
}

bool SupportsMaintenance5(VkPhysicalDevice pd, std::span<const VkExtensionProperties> available) {
    if (!HasExtension(available, VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) return false;
    VkPhysicalDeviceMaintenance5FeaturesKHR m5{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &m5};
    vkGetPhysicalDeviceFeatures2(pd, &query);
    return m5.maintenance5;
}

constexpr const char* kRequiredExtensions[]{VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
                                            VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME};

bool SupportsShaderObject(VkPhysicalDevice pd, std::span<const VkExtensionProperties> available) {
    if (!HasExtension(available, VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) return false;
    VkPhysicalDeviceShaderObjectFeaturesEXT so{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &so};
    vkGetPhysicalDeviceFeatures2(pd, &query);
    return so.shaderObject != VK_FALSE;
}

bool SupportsGraphicsPipelineLibrary(VkPhysicalDevice pd, std::span<const VkExtensionProperties> available, bool* fast_link = nullptr) {
    if (fast_link) *fast_link = false;
    if (!HasExtension(available, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) ||
        !HasExtension(available, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME))
        return false;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &gpl};
    vkGetPhysicalDeviceFeatures2(pd, &query);
    if (!gpl.graphicsPipelineLibrary) return false;
    if (fast_link) {
        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &props};
        vkGetPhysicalDeviceProperties2(pd, &p2);
        *fast_link = props.graphicsPipelineLibraryFastLinking != VK_FALSE;
    }
    return true;
}

DynamicCaps QueryDynamicCaps(VkPhysicalDevice pd, std::span<const VkExtensionProperties> available) {
    DynamicCaps c{};
    VkPhysicalDeviceVulkan13Features f13_q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 f2_q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13_q};
    vkGetPhysicalDeviceFeatures2(pd, &f2_q);
    c.cache_control = f13_q.pipelineCreationCacheControl != VK_FALSE;
    c.depth_bounds = f2_q.features.depthBounds != VK_FALSE;
    c.storage_image_read_without_format = f2_q.features.shaderStorageImageReadWithoutFormat != VK_FALSE;
    c.maintenance5 = SupportsMaintenance5(pd, available);

    const bool has_eds3 = HasExtension(available, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
    VkPhysicalDeviceDriverProperties driver_p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceFloatControlsProperties fc_p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES, &driver_p};
    VkPhysicalDeviceExtendedDynamicState3PropertiesEXT eds3_p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES_EXT, &fc_p};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, has_eds3 ? static_cast<void*>(&eds3_p) : &fc_p};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    c.signed_zero_inf_nan_preserve = fc_p.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE;

    // Pre-Turing NVIDIA (Maxwell/Pascal/Volta) on the proprietary driver: VK_EXT_shader_object triggers intermittent SM
    // WRITE_INVALID device-lost faults (e.g. GTX 1080 Ti on R580); Turing+ exposes mesh shaders. Fall back to GPL/monolithic.
    const bool avoid_objects = driver_p.driverID == VK_DRIVER_ID_NVIDIA_PROPRIETARY &&
                               !HasExtension(available, VK_NV_MESH_SHADER_EXTENSION_NAME) &&
                               !HasExtension(available, VK_EXT_MESH_SHADER_EXTENSION_NAME);
    const bool has_objects = !avoid_objects && SupportsShaderObject(pd, available);
    bool fast_link = false;
    const bool has_gpl = SupportsGraphicsPipelineLibrary(pd, available, &fast_link);

    if (const auto req = RequestedShaderPath(); req.has_value()) {
        c.path = *req;
        c.forced = true;
    } else {
        c.forced = false;
        c.path = has_objects ? ShaderPath::Objects : has_gpl ? ShaderPath::Gpl : ShaderPath::Monolithic;
    }

    if (c.path == ShaderPath::Objects) {
        c.gpl_fast_link = false;
        c.unrestricted_topology = true;
        c.depth_clip_negative_one_to_one = true;
        c.depth_clamp_enable = true;
        c.rasterization_samples = true;
        c.alpha_to_coverage_enable = true;
        c.color_blend_enable = true;
        c.color_blend_equation = true;
        c.color_write_mask = true;
    } else {
        c.gpl_fast_link = (c.path == ShaderPath::Gpl) && fast_link;
        if (has_eds3) {
            VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3_q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
            VkPhysicalDeviceFeatures2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &eds3_q};
            vkGetPhysicalDeviceFeatures2(pd, &q2);

            // AMD proprietary/AMDVLK (XGL/PAL): dynamic rasterizationSamples corrupts 2x MSAA in 8x8 tiles, and PAL gfx6 (Polaris)
            // clobbers DB_SHADER_CONTROL.ALPHA_TO_MASK_DISABLE on pipeline switches with dynamic alphaToCoverageEnable.
            // Intel Windows proprietary: dynamic depthClipNegativeOneToOne is ignored (clips Z to [0, 1] instead of [-1, 1],
            // dropping UI and close-up geometry), and dynamic rasterizationSamples hangs on 2x -> 1x switches across a reused GPL pipeline.
            const bool intel_win = driver_p.driverID == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS;
            const bool bake_msaa = intel_win ||
                                   driver_p.driverID == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                   driver_p.driverID == VK_DRIVER_ID_AMD_OPEN_SOURCE;
            c.unrestricted_topology = eds3_p.dynamicPrimitiveTopologyUnrestricted != VK_FALSE;
            c.depth_clip_negative_one_to_one = !intel_win && eds3_q.extendedDynamicState3DepthClipNegativeOneToOne != VK_FALSE;
            c.depth_clamp_enable = eds3_q.extendedDynamicState3DepthClampEnable != VK_FALSE;
            c.rasterization_samples = !bake_msaa && eds3_q.extendedDynamicState3RasterizationSamples != VK_FALSE;
            c.alpha_to_coverage_enable = !bake_msaa && eds3_q.extendedDynamicState3AlphaToCoverageEnable != VK_FALSE;
            c.color_blend_enable = eds3_q.extendedDynamicState3ColorBlendEnable != VK_FALSE;
            c.color_blend_equation = eds3_q.extendedDynamicState3ColorBlendEquation != VK_FALSE;
            c.color_write_mask = eds3_q.extendedDynamicState3ColorWriteMask != VK_FALSE;
        }
    }
    return c;
}

const auto temp_bind_t0 = std::chrono::steady_clock::now();
double TempBindMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - temp_bind_t0).count(); }
} // namespace

// ---- Physical device queries and device creation ----

const char* ShaderPathName(ShaderPath path) {
    switch (path) {
    case ShaderPath::Objects: return "objects";
    case ShaderPath::Gpl: return "gpl";
    case ShaderPath::Monolithic: return "monolithic";
    }
    return "unknown";
}

std::optional<ShaderPath> RequestedShaderPath(std::string* invalid) {
    const char* env = std::getenv("GOW3_SHADER_PATH");
    if (!env || !*env) return std::nullopt;
    std::string v;
    for (const char* p = env; *p; ++p)
        if (!std::isspace(static_cast<unsigned char>(*p))) v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
    if (v.empty() || v == "auto") return std::nullopt;
    if (v == "objects" || v == "object" || v == "shader_object") return ShaderPath::Objects;
    if (v == "gpl" || v == "library") return ShaderPath::Gpl;
    if (v == "monolithic" || v == "pipeline" || v == "mono") return ShaderPath::Monolithic;
    if (invalid) *invalid = env;
    return std::nullopt;
}

DynamicCaps QueryDynamicCaps(VkPhysicalDevice pd) { return QueryDynamicCaps(pd, DeviceExtensions(pd)); }

bool QueueFamilySuitable(VkQueueFlags flags) {
    const VkQueueFlags want = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
    return (flags & want) == want;
}

std::vector<VkQueueFamilyProperties> QueueFamilies(VkPhysicalDevice pd) {
    u32 n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fp(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fp.data());
    return fp;
}

// The copy engine: a family that only transfers (~0u: none).
u32 CopyFamily(std::span<const VkQueueFamilyProperties> fp) {
    for (u32 i = 0; i < fp.size(); ++i)
        if (fp[i].queueFlags & VK_QUEUE_TRANSFER_BIT && !(fp[i].queueFlags & ~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT | VK_QUEUE_PROTECTED_BIT)))
            return i;
    return ~0u;
}

bool SupportsBlendInterlock(VkPhysicalDevice pd) { return SupportsBlendInterlock(pd, DeviceExtensions(pd)); }
bool SupportsMaintenance5(VkPhysicalDevice pd) { return SupportsMaintenance5(pd, DeviceExtensions(pd)); }

std::string MissingExtensions(VkPhysicalDevice pd) {
    const auto available = DeviceExtensions(pd);
    std::string missing;
    for (const char* name : kRequiredExtensions)
        if (!HasExtension(available, name)) missing += std::format(" {}", name);
    if (const auto req = RequestedShaderPath(); req.has_value()) {
        if (*req == ShaderPath::Objects) {
            if (!SupportsShaderObject(pd, available)) missing += std::format(" {}", VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
        } else if (*req == ShaderPath::Gpl) {
            if (!HasExtension(available, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME))
                missing += std::format(" {}", VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
            if (!HasExtension(available, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) {
                missing += std::format(" {}", VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
            } else if (HasExtension(available, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) && !SupportsGraphicsPipelineLibrary(pd, available)) {
                missing += std::format(" {}", VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
            }
        }
    }
    return missing;
}

VkResult CreateDevice(VkPhysicalDevice pd, u32 family, std::span<const char* const> extensions, VkDevice* device, bool* present_wait, bool* reflex, bool* unified_layouts,
                      std::string* requested, DynamicCaps* caps) {
    const auto available = DeviceExtensions(pd);
    const DynamicCaps dc = QueryDynamicCaps(pd, available);
    if (caps) *caps = dc;

    VkPhysicalDeviceDepthClipControlFeaturesEXT dcc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, nullptr, VK_TRUE};
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT, &dcc, VK_TRUE};  // predication
    const bool temp_stats = DebugDeep("GOW3_TEMP_STATS");  // driver statistics of each program (Compile)
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR temp_pep{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, &cr, VK_TRUE};
    void* f13_next = temp_stats ? (void*)&temp_pep : &cr;

    std::vector<const char*> ext(std::begin(kRequiredExtensions), std::end(kRequiredExtensions));
    VkPhysicalDeviceShaderObjectFeaturesEXT so{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT, nullptr, VK_TRUE};
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, nullptr, VK_TRUE};
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};

    VkPhysicalDeviceVulkan13Features sup13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features sup12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &sup13};
    VkPhysicalDeviceVulkan11Features sup11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &sup12};
    VkPhysicalDeviceFeatures2 sup2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &sup11};
    vkGetPhysicalDeviceFeatures2(pd, &sup2);
    const auto& sup = sup2.features;

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = f13.synchronization2 = VK_TRUE;
    f13.maintenance4 = sup13.maintenance4;
    f13.shaderDemoteToHelperInvocation = sup13.shaderDemoteToHelperInvocation;
    // A slot that falls back to the 1x1 dummy: its out-of-bounds writes are dropped, not stray.
    f13.robustImageAccess = sup13.robustImageAccess;

    if (dc.path == ShaderPath::Objects) {
        ext.push_back(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
        so.pNext = f13_next;
        f13_next = &so;
    } else {
        if (dc.cache_control) f13.pipelineCreationCacheControl = VK_TRUE;
        if (dc.path == ShaderPath::Gpl) {
            ext.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
            ext.push_back(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
            gpl.pNext = f13_next;
            f13_next = &gpl;
        }
        const bool any_eds3 = dc.depth_clip_negative_one_to_one || dc.depth_clamp_enable || dc.rasterization_samples ||
                                dc.alpha_to_coverage_enable || dc.color_blend_enable || dc.color_blend_equation || dc.color_write_mask;
        if (HasExtension(available, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME) && (any_eds3 || dc.unrestricted_topology)) {
            ext.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
            if (any_eds3) {
                if (dc.depth_clip_negative_one_to_one) eds3.extendedDynamicState3DepthClipNegativeOneToOne = VK_TRUE;
                if (dc.depth_clamp_enable) eds3.extendedDynamicState3DepthClampEnable = VK_TRUE;
                if (dc.rasterization_samples) eds3.extendedDynamicState3RasterizationSamples = VK_TRUE;
                if (dc.alpha_to_coverage_enable) eds3.extendedDynamicState3AlphaToCoverageEnable = VK_TRUE;
                if (dc.color_blend_enable) eds3.extendedDynamicState3ColorBlendEnable = VK_TRUE;
                if (dc.color_blend_equation) eds3.extendedDynamicState3ColorBlendEquation = VK_TRUE;
                if (dc.color_write_mask) eds3.extendedDynamicState3ColorWriteMask = VK_TRUE;
                eds3.pNext = f13_next;
                f13_next = &eds3;
            }
        }
    }
    f13.pNext = f13_next;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.bufferDeviceAddress = f12.descriptorIndexing = f12.runtimeDescriptorArray = f12.descriptorBindingPartiallyBound = VK_TRUE;
    f12.descriptorBindingSampledImageUpdateAfterBind = f12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = f12.shaderStorageImageArrayNonUniformIndexing = f12.scalarBlockLayout = VK_TRUE;
    f12.shaderInt8 = sup12.shaderInt8;
    f12.storageBuffer8BitAccess = sup12.storageBuffer8BitAccess;
    f12.samplerFilterMinmax = sup12.samplerFilterMinmax;
    f12.samplerMirrorClampToEdge = sup12.samplerMirrorClampToEdge;  // tiling.comp, S#
    f12.hostQueryReset = sup12.hostQueryReset;                      // trace probes
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = sup11.storageBuffer16BitAccess;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    auto& f = f2.features;
    f.shaderInt16 = sup.shaderInt16;
    f.geometryShader = sup.geometryShader;
    f.sparseBinding = f.sparseResidencyBuffer = VK_TRUE;
    f.shaderInt64 = sup.shaderInt64;
    f.independentBlend = sup.independentBlend;
    f.dualSrcBlend = sup.dualSrcBlend;
    f.samplerAnisotropy = sup.samplerAnisotropy;
    f.imageCubeArray = sup.imageCubeArray;
    f.fragmentStoresAndAtomics = sup.fragmentStoresAndAtomics;
    f.vertexPipelineStoresAndAtomics = sup.vertexPipelineStoresAndAtomics;
    f.depthClamp = sup.depthClamp;
    f.depthBounds = dc.depth_bounds ? VK_TRUE : VK_FALSE;
    f.shaderStorageImageReadWithoutFormat = dc.storage_image_read_without_format ? VK_TRUE : VK_FALSE;
    f.shaderStorageImageWriteWithoutFormat = sup.shaderStorageImageWriteWithoutFormat;
    f.shaderStorageImageExtendedFormats = sup.shaderStorageImageExtendedFormats;
    f.depthBiasClamp = sup.depthBiasClamp;
    f.logicOp = sup.logicOp;
    f.pipelineStatisticsQuery = sup.pipelineStatisticsQuery;
    f.occlusionQueryPrecise = sup.occlusionQueryPrecise;
    const float prio[2]{1.0f, 1.0f};
    const auto fp = QueueFamilies(pd);
    // The second queue (when the family has one) uploads the next frame while this one renders; two of the copy engine's
    // (Setup prefers them).
    VkDeviceQueueCreateInfo qci[2]{{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family, std::min(2u, fp[family].queueCount), prio}};
    u32 queue_infos = 1;
    if (const u32 cf = CopyFamily(fp); cf != ~0u) qci[queue_infos++] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, cf, std::min(2u, fp[cf].queueCount), prio};
    VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR};
    if (HasExtension(available, VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &unified};
        vkGetPhysicalDeviceFeatures2(pd, &query);
        unified.unifiedImageLayoutsVideo = VK_FALSE;
        if (unified.unifiedImageLayouts) {
            ext.push_back(VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
            unified.pNext = dcc.pNext, dcc.pNext = &unified;
        }
    }
    if (unified_layouts) *unified_layouts = unified.unifiedImageLayouts;
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT interlock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
    if (SupportsBlendInterlock(pd, available)) {
        ext.push_back(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME), ext.push_back(VK_EXT_POST_DEPTH_COVERAGE_EXTENSION_NAME);
        interlock.fragmentShaderPixelInterlock = VK_TRUE, interlock.pNext = dcc.pNext, dcc.pNext = &interlock;
        f.shaderStorageImageMultisample = f.shaderStorageImageArrayDynamicIndexing = VK_TRUE;
    }
    if (temp_stats) ext.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    VkPhysicalDeviceMaintenance5FeaturesKHR m5{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR, nullptr, VK_TRUE};  // sized index binding
    if (SupportsMaintenance5(pd, available)) ext.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME), m5.pNext = dcc.pNext, dcc.pNext = &m5;
    ext.insert(ext.end(), extensions.begin(), extensions.end());
    VkPhysicalDevicePresentIdFeaturesKHR present_id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
    VkPhysicalDevicePresentWaitFeaturesKHR wait{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
    if (reflex) *reflex = false;
    if (present_wait) {
        *present_wait = false;
        if (HasExtension(available, VK_KHR_PRESENT_ID_EXTENSION_NAME) && HasExtension(available, VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
            present_id.pNext = &wait;
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &present_id};
            vkGetPhysicalDeviceFeatures2(pd, &query);
            if (present_id.presentId && wait.presentWait) {
                ext.push_back(VK_KHR_PRESENT_ID_EXTENSION_NAME);
                ext.push_back(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
                wait.pNext = dcc.pNext;
                dcc.pNext = &present_id;
                *present_wait = true;
                if (reflex && HasExtension(available, VK_NV_LOW_LATENCY_2_EXTENSION_NAME)) {
                    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
                    VkPhysicalDeviceFeatures2 timeline_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &timeline};
                    vkGetPhysicalDeviceFeatures2(pd, &timeline_query);
                    if (timeline.timelineSemaphore) {
                        ext.push_back(VK_NV_LOW_LATENCY_2_EXTENSION_NAME);
                        f12.timelineSemaphore = VK_TRUE;
                        *reflex = true;
                    }
                }
            }
        }
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, queue_infos, qci, 0, nullptr, (u32)ext.size(), ext.data()};
    // TEMP fault (debug runs, GOW3_TEMP_FAULT=0 off): VK_EXT_device_fault and NVIDIA diagnostics (checkpoints), read when the device is lost.
    VkPhysicalDeviceFaultFeaturesEXT temp_fault{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT, nullptr, VK_TRUE, VK_FALSE};
    VkPhysicalDeviceDiagnosticsConfigFeaturesNV temp_diag{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV, nullptr, VK_TRUE};
    VkPhysicalDeviceAddressBindingReportFeaturesEXT temp_abr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT, nullptr, VK_TRUE};
    VkDeviceDiagnosticsConfigCreateInfoNV temp_diag_ci{VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV, &f2,
                                                       VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
                                                           VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
                                                           VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV};
    if (DebugAuto("GOW3_TEMP_FAULT")) {
        const bool has_fault = HasExtension(available, VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        const bool has_ckpt = HasExtension(available, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
        const bool has_abr = HasExtension(available, VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
        const bool has_diag = HasExtension(available, VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
        if (has_fault) ext.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME), temp_fault.pNext = dcc.pNext, dcc.pNext = &temp_fault;
        if (has_ckpt) ext.push_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
        if (has_abr) ext.push_back(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME), temp_abr.pNext = dcc.pNext, dcc.pNext = &temp_abr;
        // NVIDIA's automatic checkpoints, resource tracking and shader debug info cost on every draw: GOW3_TEMP_FAULT=1 only.
        if (has_diag && DebugDeep("GOW3_TEMP_FAULT")) ext.push_back(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME), temp_diag.pNext = dcc.pNext, dcc.pNext = &temp_diag, dci.pNext = &temp_diag_ci;
        dci.enabledExtensionCount = (u32)ext.size(), dci.ppEnabledExtensionNames = ext.data();
        std::fprintf(stderr, "TEMP fault: device_fault %d checkpoints %d diagnostics %d binding report %d\n",
                     has_fault, has_ckpt, has_diag && DebugDeep("GOW3_TEMP_FAULT"), has_abr);
    }
    if (requested)
        for (const char* name : ext) *requested += std::format(" {}{}", name, HasExtension(available, name) ? "" : " (MISSING)");
    return vkCreateDevice(pd, &dci, nullptr, device);
}

// ---- Renderer device setup and fault diagnostics ----

void Renderer::Impl::LoadDeviceFunctions() {
    const auto available = DeviceExtensions(pd);
    caps = QueryDynamicCaps(pd, available);
    blend_interlock = SupportsBlendInterlock(pd, available);
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p13};
    vkGetPhysicalDeviceProperties2(pd, &props);
    const bool integrated = props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    const bool has_budget = HasExtension(available, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    uma_budget = integrated && has_budget;
    uma_single_heap = integrated && mem.memoryHeapCount == 1 &&
                      Type(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != ~0u;
    direct_tables = uma_single_heap;
    use_stage_dev = !integrated;  // a VRAM twin of the staging chunks would be a second copy in the same RAM
    u64 total_heap = 0;
    for (u32 i = 0; i < mem.memoryHeapCount; ++i) total_heap += mem.memoryHeaps[i].size;
    const u32 local_type = Type(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const u32 vram_heap = local_type != ~0u ? mem.memoryTypes[local_type].heapIndex : 0;
    if (has_budget) budget_heaps = integrated ? (1u << mem.memoryHeapCount) - 1 : 1u << vram_heap;
    // An 8 GB APU reports ~6.8 GiB of heaps (carve-out + shared); 16 GB ones ~10 GiB.
    compact_memory = integrated ? total_heap <= (8ull << 30) : mem.memoryHeaps[vram_heap].size <= (4ull << 30);
    if (compact_memory) {
        chunk_size = chunk_used = 64ull << 20;
        image_block_size = 64ull << 20;
        image_reserve_blocks = 1;
        scratch_reserve = 32ull << 20;
        table_bytes = 32ull << 20;
        stage_chunk_size = 32ull << 20;
        temp_chunk_size = 16ull << 20;
    }
    image_copy_via_temp = p13.maxBufferSize < kMirrorMax;  // before SetupMirror: against the largest mirror
#define LOAD(var, name) var = (PFN_##name)vkGetDeviceProcAddr(dev, #name)
    if (caps.path == ShaderPath::Objects) {
        LOAD(CreateShaders, vkCreateShadersEXT);
        LOAD(GetShaderBinary, vkGetShaderBinaryDataEXT);
        LOAD(BindShaders, vkCmdBindShadersEXT);
        LOAD(DestroyShader, vkDestroyShaderEXT);
        LOAD(SetVertexInput, vkCmdSetVertexInputEXT);
        LOAD(SetPolygonMode, vkCmdSetPolygonModeEXT);
        LOAD(SetSampleMask, vkCmdSetSampleMaskEXT);
        LOAD(SetLogicOpEnable, vkCmdSetLogicOpEnableEXT);
    }
    if (caps.rasterization_samples) LOAD(SetSamples, vkCmdSetRasterizationSamplesEXT);
    if (caps.alpha_to_coverage_enable) LOAD(SetAlphaToCoverage, vkCmdSetAlphaToCoverageEnableEXT);
    if (caps.color_blend_enable) LOAD(SetBlendEnable, vkCmdSetColorBlendEnableEXT);
    if (caps.color_blend_equation) LOAD(SetBlendEquation, vkCmdSetColorBlendEquationEXT);
    if (caps.color_write_mask) LOAD(SetWriteMask, vkCmdSetColorWriteMaskEXT);
    if (caps.depth_clamp_enable) LOAD(SetDepthClamp, vkCmdSetDepthClampEnableEXT);
    if (caps.depth_clip_negative_one_to_one) LOAD(SetNegativeOneToOne, vkCmdSetDepthClipNegativeOneToOneEXT);
    LOAD(PushDescriptors, vkCmdPushDescriptorSetKHR);
    LOAD(BeginConditional, vkCmdBeginConditionalRenderingEXT), LOAD(EndConditional, vkCmdEndConditionalRenderingEXT);
    if (caps.maintenance5) LOAD(BindIndex2, vkCmdBindIndexBuffer2KHR);
    if (DebugAuto("GOW3_TEMP_LABELS")) LOAD(temp_label_begin, vkCmdBeginDebugUtilsLabelEXT), LOAD(temp_label_end, vkCmdEndDebugUtilsLabelEXT);
    if (DebugAuto("GOW3_TEMP_FAULT"))
        LOAD(temp_get_fault, vkGetDeviceFaultInfoEXT), LOAD(temp_set_check, vkCmdSetCheckpointNV), LOAD(temp_get_checks, vkGetQueueCheckpointDataNV);
#undef LOAD
}

// TEMP fault: GPU address ranges bound and unbound, in order (VK_EXT_device_address_binding_report).
std::mutex temp_bind_lock;
std::vector<TempBinding> temp_binds;
std::atomic<u32> temp_bind_frame{0};
void TempAddressBinding(const VkDebugUtilsMessengerCallbackDataEXT* d) {
    for (auto* p = (const VkBaseInStructure*)d->pNext; p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT) {
            const auto* b = (const VkDeviceAddressBindingCallbackDataEXT*)p;
            const VkDebugUtilsObjectNameInfoEXT* o = d->objectCount ? d->pObjects : nullptr;
            std::string who;
            if (b->size >= (8u << 20)) {
#ifdef _WIN32
                PWSTR n = nullptr;
                if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &n)) && n)
                    for (const wchar_t* c = n; *c; ++c) who += (char)*c;
                if (n) LocalFree(n);
                void* f[40];
                for (u16 i = 0, k = RtlCaptureStackBackTrace(1, 40, f, nullptr); i < k; ++i) {
                    HMODULE m = nullptr;
                    char path[MAX_PATH] = "?";
                    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)f[i], &m)) GetModuleFileNameA(m, path, MAX_PATH);
                    who += std::format(" {}+{:#x}", std::filesystem::path(path).filename().string(), (u64)f[i] - (u64)m);
                }
#else
                who = Common::GetCurrentThreadNameView();
                void* frames[40];
                for (int i = 1, count = backtrace(frames, std::size(frames)); i < count; ++i) {
                    Dl_info info{};
                    if (dladdr(frames[i], &info) && info.dli_fname)
                        who += std::format(" {}+{:#x}", std::filesystem::path(info.dli_fname).filename().string(),
                                           reinterpret_cast<u64>(frames[i]) - reinterpret_cast<u64>(info.dli_fbase));
                    else
                        who += std::format(" ?+{:#x}", reinterpret_cast<u64>(frames[i]));
                }
#endif
            }
            std::scoped_lock lock(temp_bind_lock);
            temp_binds.push_back({b->baseAddress, b->size, o ? o->objectHandle : 0, o ? (u32)o->objectType : 0, (u32)b->flags,
                                  temp_bind_frame.load(std::memory_order_relaxed), b->bindingType == VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT,
                                  temp_call ? temp_call : "-", std::hash<std::thread::id>{}(std::this_thread::get_id()) % 100000,
                                  (u32)Gow3::ThreadId(), TempBindMs(), std::move(who)});
        }
}

// TEMP device report (debug runs): what this GPU and driver answer where the renderer assumes: limits, queues, sparse memory,
// formats and extensions.
void Renderer::Impl::TempDeviceReport(std::span<const VkQueueFamilyProperties> fp) {
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, &p13};
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &p12};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p11};
    vkGetPhysicalDeviceProperties2(pd, &p2);
    const VkPhysicalDeviceProperties& p = p2.properties;
    const VkPhysicalDeviceLimits& l = p.limits;
    VkPhysicalDeviceFeatures f;
    vkGetPhysicalDeviceFeatures(pd, &f);
    Report(std::format("Device: {} vendor {:04x} device {:04x} type {} api {}.{}.{} driver {:08x} \"{}\" \"{}\" id {} conformance {}.{}.{}.{}", p.deviceName, p.vendorID,
                       p.deviceID, (int)p.deviceType, VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion),
                       p.driverVersion, p12.driverName, p12.driverInfo, (int)p12.driverID, p12.conformanceVersion.major, p12.conformanceVersion.minor,
                       p12.conformanceVersion.subminor, p12.conformanceVersion.patch));
    Report(std::format("Limits: maxPushConstantsSize {} (used {}{}) maxBufferSize {:x} (mirror {:x}{}) maxMemoryAllocationSize {:x} sparseAddressSpaceSize {:x} "
                       "maxMemoryAllocationCount {} maxSamplerAllocationCount {} maxDrawIndexedIndexValue {:x} maxImageDimension2D {} maxColorAttachments {} "
                       "maxBoundDescriptorSets {} maxUniformBufferRange {:x} maxStorageBufferRange {:x} minUniformBufferOffsetAlignment {} "
                       "minStorageBufferOffsetAlignment {} bufferImageGranularity {} nonCoherentAtomSize {} timestampPeriod {} maxSamplerAnisotropy {}",
                       l.maxPushConstantsSize, sizeof(Push), l.maxPushConstantsSize < sizeof(Push) ? ": TOO SMALL" : "", p13.maxBufferSize, mirror_size,
                       p13.maxBufferSize < mirror_size ? ": TOO SMALL" : "", p11.maxMemoryAllocationSize, l.sparseAddressSpaceSize, l.maxMemoryAllocationCount,
                       l.maxSamplerAllocationCount, l.maxDrawIndexedIndexValue, l.maxImageDimension2D, l.maxColorAttachments, l.maxBoundDescriptorSets,
                       l.maxUniformBufferRange, l.maxStorageBufferRange, l.minUniformBufferOffsetAlignment, l.minStorageBufferOffsetAlignment, l.bufferImageGranularity,
                       l.nonCoherentAtomSize, l.timestampPeriod, l.maxSamplerAnisotropy));
    Report(std::format("Bindless: uses sampled {} samplers {} storage {}; update-after-bind limits: set sampled {} samplers {} storage {}, stage sampled {} samplers {} "
                       "storage {} resources {}, all pools {}; dynamic uniform buffers per set {}",
                       18 * kTextures, kSamplers, 9 * kStorage, p12.maxDescriptorSetUpdateAfterBindSampledImages, p12.maxDescriptorSetUpdateAfterBindSamplers,
                       p12.maxDescriptorSetUpdateAfterBindStorageImages, p12.maxPerStageDescriptorUpdateAfterBindSampledImages,
                       p12.maxPerStageDescriptorUpdateAfterBindSamplers, p12.maxPerStageDescriptorUpdateAfterBindStorageImages, p12.maxPerStageUpdateAfterBindResources,
                       p12.maxUpdateAfterBindDescriptorsInAllPools, l.maxDescriptorSetUniformBuffersDynamic));
    Report(std::format("Subgroup: size {} (from {} to {}) stages {:x} sized stages {:x} operations {:x} quad in all stages {}", p11.subgroupSize, p13.minSubgroupSize,
                       p13.maxSubgroupSize, (u32)p11.subgroupSupportedStages, (u32)p13.requiredSubgroupSizeStages, (u32)p11.subgroupSupportedOperations,
                       p11.subgroupQuadOperationsInAllStages));
    Report(std::format("Sparse: binding {} residency buffer {} aliased {} non-resident strict {}; mirror page {:x} memory types {:x}, integrated with budget {}, budget heaps {:x}, single_heap {}, stage_dev {}, direct_tables {}, compact {} [chunk {} MB, img {} MB x{}, table {} MB, scratch {} MB, temp {} MB, stage {} MB], image copies through a temp buffer {}",
                       f.sparseBinding, f.sparseResidencyBuffer, f.sparseResidencyAliased, p.sparseProperties.residencyNonResidentStrict, page, mirror_bits, uma_budget,
                       budget_heaps, uma_single_heap, use_stage_dev, direct_tables, compact_memory, chunk_size >> 20, image_block_size >> 20, image_reserve_blocks,
                       table_bytes >> 20, scratch_reserve >> 20, temp_chunk_size >> 20, stage_chunk_size >> 20, image_copy_via_temp));
    std::string s;
    for (u32 i = 0; i < fp.size(); ++i)
        s += std::format(" [{}: flags {:x} x{} timestamp bits {} granularity {}x{}x{}]", i, (u32)fp[i].queueFlags, fp[i].queueCount, fp[i].timestampValidBits,
                         fp[i].minImageTransferGranularity.width, fp[i].minImageTransferGranularity.height, fp[i].minImageTransferGranularity.depth);
    Report(std::format("Queues:{}; render family {}, uploads on {}, readback {}", s, family,
                       copy_family != ~0u ? std::format("the copy engine (family {})", copy_family) : xfer != queue ? "a second render queue" : "the render queue",
                       rbq ? "on its own queue" : "with the frame (no second copy queue)"));
    s.clear();
    for (u32 i = 0; i < mem.memoryHeapCount; ++i) s += std::format(" heap {} {} MB flags {:x}", i, mem.memoryHeaps[i].size >> 20, (u32)mem.memoryHeaps[i].flags);
    for (u32 i = 0; i < mem.memoryTypeCount; ++i) s += std::format(" type {} heap {} flags {:x}", i, mem.memoryTypes[i].heapIndex, (u32)mem.memoryTypes[i].propertyFlags);
    Report("Memory:" + s);
    static constexpr VkFormat formats[]{
        VK_FORMAT_R4G4B4A4_UNORM_PACK16, VK_FORMAT_B4G4R4A4_UNORM_PACK16, VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_B5G6R5_UNORM_PACK16, VK_FORMAT_R5G5B5A1_UNORM_PACK16,
        VK_FORMAT_B5G5R5A1_UNORM_PACK16, VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UINT,
        VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UNORM_PACK32,
        VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R32_UINT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
        VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, VK_FORMAT_D16_UNORM, VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D32_SFLOAT, VK_FORMAT_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC2_UNORM_BLOCK,
        VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC6H_UFLOAT_BLOCK, VK_FORMAT_BC7_UNORM_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK};
    s.clear();
    for (const VkFormat x : formats) {
        VkFormatProperties q;
        vkGetPhysicalDeviceFormatProperties(pd, x, &q);
        s += std::format(" {}:{:x}", (int)x, (u32)q.optimalTilingFeatures);
    }
    Report("Formats (VkFormat:optimal tiling features):" + s);
    const auto ext = DeviceExtensions(pd);
    s.clear();
    for (const auto& x : ext) s += std::format(" {}", x.extensionName + (std::strncmp(x.extensionName, "VK_", 3) ? 0 : 3));
    Report(std::format("Extensions ({}), fragment interlock used {}, sized index binding {}:{}", ext.size(), blend_interlock, BindIndex2 != nullptr, s));
    Report(std::format(
        "ShaderPath: {} (forced {}), cache_control {}, gpl_fast_link {}, unrestricted_topology {}, depth_bounds {}, storage_read_nofmt {}, eds3[neg1to1 {} clamp {} samples {} a2c {} blend_en {} blend_eq {} mask {}]",
        ShaderPathName(caps.path), caps.forced, caps.cache_control, caps.gpl_fast_link, caps.unrestricted_topology,
        caps.depth_bounds, caps.storage_image_read_without_format,
        caps.depth_clip_negative_one_to_one, caps.depth_clamp_enable, caps.rasterization_samples,
        caps.alpha_to_coverage_enable, caps.color_blend_enable, caps.color_blend_equation, caps.color_write_mask));
}

void Renderer::Impl::TempLost(const char* where, VkResult r) {  // TEMP fault
    if (r != VK_ERROR_DEVICE_LOST || temp_lost.exchange(true)) return;
    std::scoped_lock lock(note_lock);
    auto note = [&](std::string s) { stats.notes.push_back("TEMP fault: " + std::move(s)); };
    note(std::format("{}: device lost at frame {}, mirror at {:#x}", where, frame, mirror_addr));
    if (temp_get_fault) {
        VkDeviceFaultCountsEXT n{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
        temp_get_fault(dev, &n, nullptr);
        std::vector<VkDeviceFaultAddressInfoEXT> addr(n.addressInfoCount);
        std::vector<VkDeviceFaultVendorInfoEXT> vendor(n.vendorInfoCount);
        n.vendorBinarySize = 0;
        VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT, nullptr, {}, addr.data(), vendor.data(), nullptr};
        note(std::format("fault info result {} '{}', {} addresses, {} vendor", (int)temp_get_fault(dev, &n, &info), info.description, addr.size(), vendor.size()));
        for (auto& a : addr) {
            const u64 g = a.reportedAddress - mirror_addr;
            note(std::format("address type {} {:#x} precision {:#x}{}", (int)a.addressType, a.reportedAddress, a.addressPrecision,
                             a.reportedAddress >= mirror_addr && g < mirror_size ? std::format(" = guest {:#x}, page bound {}", g, bound[g / page / 64] >> (g / page % 64) & 1) : ""));
            // Every binding over the faulting range, then the nearest ones below and above it, with the resource still holding the handle.
            auto name = [&](u64 h) -> std::string {
                auto is = [&](auto x) { return x && (u64)x == h; };
                if (is(mirror)) return "mirror";
                for (auto m : chunks)
                    if (is(m)) return "mirror chunk";
                const std::pair<const Buffer*, const char*> bufs[]{{&tables_host, "tables_host"}, {&tables, "tables"}, {&rb, "rb"}, {&scratch, "scratch"},
                                                                   {&flight.tables_host, "flight tables_host"}, {&flight.tables, "flight tables"},
                                                                   {&flight.rb, "flight rb"}, {&flight.scratch, "flight scratch"}, {&probe_buf, "probe"}, {&vertex_probe, "vertex probe"}};
                for (auto [b, n] : bufs)
                    if (is(b->buf) || is(b->mem)) return std::format("{} size {:#x}", n, b->size);
                for (auto [v, n] : {std::pair{&frame_temp, "temp"}, {&flight.temp, "flight temp"}, {&stage, "stage"}, {&flight.stage, "flight stage"}})
                    for (auto& b : *v)
                        if (is(b.buf) || is(b.mem)) return std::format("{} size {:#x}", n, b.size);
                for (auto& t : dummies)
                    if (is(t.img) || is(t.mem)) return "dummy";
                for (auto& [k, t] : targets)
                    if (is(t.t.img) || is(t.t.mem)) return std::format("target {:#x} {}x{} format {} samples {} usage {:#x}", t.base, t.w, t.h, (int)t.fmt, t.samples, t.t.usage);
                for (auto& [k, x] : textures)
                    if (is(x.t.img) || is(x.t.mem)) return std::format("texture {:#x} {}x{}x{} format {} tile {} storage {}", x.base, x.w, x.h, x.layers, (int)x.fmt, x.tile, x.storage);
                std::scoped_lock temp_l(temp_alloc_lock);
                for (auto it = temp_allocs.rbegin(); it != temp_allocs.rend(); ++it)
                    if (it->handle == h)
                        return std::format("freed buffer '{}' size {:#x}, made at frame {} slot {}, freed at frame {} (frame in flight busy {})", it->tag, it->size,
                                           it->made, it->slot, it->freed, it->busy);
                return "(no live resource)";
            };
            const u64 lo = a.reportedAddress & ~(std::max<u64>(a.addressPrecision, 1) - 1), hi = lo + std::max<u64>(a.addressPrecision, 1);
            std::scoped_lock bl(temp_bind_lock);
            const TempBinding *below = nullptr, *above = nullptr;
            u32 shown = 0;
            for (const TempBinding& b : temp_binds) {
                if (b.base < hi && b.base + b.size > lo && shown++ < 24)
                    note(std::format("binding over it: #{} {} at frame {} type {} flags {:#x} [{:#x}, {:#x}) handle {:#x}, thread {} tid {} in '{}' {:.1f} ms before this: {}; by '{}'",
                                     &b - temp_binds.data(), b.bound ? "bind" : "UNBIND", b.frame, b.type, b.flags, b.base, b.base + b.size, b.handle, b.thread, b.tid, b.call,
                                     TempBindMs() - b.ms, name(b.handle), b.who));
                if (b.bound && b.base + b.size <= lo && (!below || b.base + b.size > below->base + below->size)) below = &b;
                if (b.bound && b.base >= hi && (!above || b.base < above->base)) above = &b;
            }
            for (const TempBinding* b : {below, above})
                if (b)
                    note(std::format("nearest {}: #{} at frame {} type {} [{:#x}, {:#x}) handle {:#x}: {}", b == below ? "below" : "above", b - temp_binds.data(), b->frame, b->type,
                                     b->base, b->base + b->size, b->handle, name(b->handle)));
            note(std::format("{} bindings recorded; this (GPU) thread {} tid {}", temp_binds.size(), std::hash<std::thread::id>{}(std::this_thread::get_id()) % 100000, Gow3::ThreadId()));
        }
        for (auto& v : vendor) note(std::format("vendor '{}' code {:#x} data {:#x}", v.description, v.vendorFaultCode, v.vendorFaultData));
    }
    if (temp_get_checks)
        for (VkQueue q : {queue, xfer, rbq}) {
            if (!q) continue;
            u32 c = 0;
            temp_get_checks(q, &c, nullptr);
            std::vector<VkCheckpointDataNV> d(c, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
            temp_get_checks(q, &c, d.data());
            for (auto& x : d)
                note(std::format("queue {} stage {:#x} checkpoint {:#x}", q == queue ? "main" : q == xfer ? "xfer" : "readback", (u32)x.stage, (u64)x.pCheckpointMarker));
        }
}
} // namespace Gpu
