// SPDX-License-Identifier: GPL-2.0-or-later
#include "present/presenter.h"
#include "present/presentation_timing.h"
#include "present/splash.h"
#include <SDL3/SDL.h>
#include <backends/imgui_impl_vulkan.h>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>
#include "orbis/common/thread.h"
#include "ui/imgui_core.h"
#include "gpu/gpu_thread.h"
#include "gpu/vk_gpu.h"
#include "main/host.h"
#include "gpu/device.h"
#include "gpu/shaders/gcn_translator.h"
#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

namespace Gow3::Presenter {
namespace {
using Vulkan::Check;
struct Scaled { VkImage image{}; VkDeviceMemory memory{}; VkImageView view{}; VkExtent2D size{}; VkFormat format{}; bool in_general{}; };
struct Buffer { VkBuffer buffer{}; VkDeviceMemory memory{}; VkDeviceSize size{}; };
struct CmaaState { Scaled edges, heads, output; std::array<Buffer, 5> buffers; };
struct Frame { VkCommandBuffer cmd; VkFence fence; VkSemaphore acquired; VkImageView game{}, linear{}; VkImage game_image{}; VkFormat game_format{}; std::vector<Scaled> scaled; CmaaState cmaa; bool probed{}; };
SDL_Window* window;
VkDevice device;
VkQueue queue;
VkSurfaceFormatKHR format;
VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
VkSwapchainKHR swapchain{};
VkFence fifo_available{};
VkExtent2D extent{};
std::vector<VkImage> images;
std::vector<VkImageView> views;
std::vector<VkSemaphore> rendered; // per swapchain image: reused only after that image is acquired again
VkCommandPool pool;
std::array<Frame, 2> frames;
u32 frame_index;
std::mutex flip_mutex;
std::condition_variable_any flip_cv;
bool flipped;
s64 flip_time{};
std::atomic<PresentationTiming*> timing{};
u64 present_id{};
bool reflex_requested{};
#ifdef _WIN32
// Borderless and Exclusive display modes (VK_EXT_full_screen_exclusive, application controlled): exclusive access is held while
// the window has the input focus and released when it loses it, so Alt+Tab and overlay windows get the composed desktop back
// (composed, the NVIDIA DXGI layer copies every frame). Windowed never goes exclusive (the NVIDIA native present method
// otherwise makes a full-screen window exclusive by itself).
// Lost (taken away by the driver, e.g. on Alt+Tab): NVIDIA then fails every acquire with it until it is taken again, so no
// frames are drawn meanwhile. Refused: composed for the rest of the run (retrying it left the screen black).
bool exclusive_mode{}, exclusive_held{}, exclusive_lost{};
PFN_vkAcquireFullScreenExclusiveModeEXT acquire_exclusive;
PFN_vkReleaseFullScreenExclusiveModeEXT release_exclusive;
#endif
u64 flipped_frame{};
std::atomic<u64> rendered_frame{};
std::atomic<float> output_gamma{1.f};  // sceVideoOutAdjustColor: the game's brightness option
thread_local u64 render_frame{};
u64 display; // guest address of the flipped display buffer
bool display_bgra; // its memory order (VideoOut A8R8G8B8), held as is by the renderer's RGBA image
std::jthread thread;
// The scaling pass: the display buffer's image is sampled through a view and drawn as a full-screen triangle into the
// letterboxed area of the swapchain image (no blit, no copies, one barrier).
VkSampler sampler;
VkDescriptorSetLayout set_layout;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;
VkPipeline probe_pipeline;
VkQueryPool probe_queries;
VkDescriptorSetLayout easu_set;
VkPipelineLayout easu_layout;
VkPipeline easu_pipeline;
VkPipeline lanczos_h_pipeline;
VkDescriptorSetLayout cmaa_set;
VkPipelineLayout cmaa_layout;
std::array<VkPipeline, 4> cmaa_pipelines;
ScalingFilter scaling;
AntiAliasing anti_aliasing;
PFN_vkCmdPushDescriptorSetKHR push_descriptor;
constexpr u32 easu_spv[]{
#include "easu_spv.inc"
};
constexpr u32 cmaa0_spv[]{
#include "cmaa2_0_spv.inc"
};
constexpr u32 cmaa1_spv[]{
#include "cmaa2_1_spv.inc"
};
constexpr u32 cmaa2_spv[]{
#include "cmaa2_2_spv.inc"
};
constexpr u32 cmaa3_spv[]{
#include "cmaa2_3_spv.inc"
};

u32 DeviceMemoryType(u32 bits) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(Vulkan::Get().physical, &props);
    for (u32 i = 0; i < props.memoryTypeCount; ++i)
        if ((bits >> i & 1) && (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return i;
    throw std::runtime_error("No device-local memory type for presentation filter");
}

void FreeScaled(Scaled& s) {
    if (s.view) vkDestroyImageView(device, s.view, nullptr);
    if (s.image) vkDestroyImage(device, s.image, nullptr);
    if (s.memory) vkFreeMemory(device, s.memory, nullptr);
    s = {};
}

void ResizeScaled(Scaled& s, VkExtent2D size, VkFormat image_format = VK_FORMAT_R8G8B8A8_UNORM) {
    if (s.view && s.size.width == size.width && s.size.height == size.height && s.format == image_format) return;
    FreeScaled(s);
    const VkImageCreateInfo info{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
        .format = image_format, .extent = {size.width, size.height, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    Check(vkCreateImage(device, &info, nullptr, &s.image), "Create FSR image");
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, s.image, &req);
    const VkMemoryAllocateInfo allocation{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = DeviceMemoryType(req.memoryTypeBits)};
    Check(vkAllocateMemory(device, &allocation, nullptr, &s.memory), "Allocate FSR image");
    Check(vkBindImageMemory(device, s.image, s.memory, 0), "Bind FSR image");
    const VkImageViewCreateInfo view{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = s.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = info.format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    Check(vkCreateImageView(device, &view, nullptr, &s.view), "Create FSR image view");
    s.size = size, s.format = image_format;
}

// True when the buffer was (re)created.
bool ResizeBuffer(Buffer& b, VkDeviceSize size) {
    if (b.buffer && b.size == size) return false;
    if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device, b.memory, nullptr);
    b = {};
    const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
    Check(vkCreateBuffer(device, &info, nullptr, &b.buffer), "Create CMAA2 buffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    const VkMemoryAllocateInfo allocation{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
        .memoryTypeIndex = DeviceMemoryType(req.memoryTypeBits)};
    Check(vkAllocateMemory(device, &allocation, nullptr, &b.memory), "Allocate CMAA2 buffer");
    Check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "Bind CMAA2 buffer");
    b.size = size;
    return true;
}

void ComputeBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 dst) {
    const VkMemoryBarrier2 memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, .dstStageMask = dst,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                         ((dst & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) ? VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT : 0)};
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &memory};
    vkCmdPipelineBarrier2(cmd, &dependency);
}

// CMAA2 bindings: 0 the linear input, 1 the edges, 5 the list heads and 8 the output are images, the rest the working buffers.
constexpr VkDescriptorType CmaaType(u32 binding) {
    return binding == 0 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : binding == 1 || binding == 5 || binding == 8 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                                                                         : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

// CMAA2 in linear color without copies: the passes read the display buffer's sRGB view (decoded on fetch), EdgesColor2x2CS
// writes every pixel re-encoded into the output and DeferredColorApply2x2CS blends the edge pixels there in place.
VkImageView ApplyCmaa(Frame& frame, VkExtent2D size) {
    auto& c = frame.cmaa;
    ResizeScaled(c.output, size);
    ResizeScaled(c.edges, {(size.width + 1) / 2, size.height}, VK_FORMAT_R8_UINT);
    ResizeScaled(c.heads, {(size.width + 1) / 2, (size.height + 1) / 2}, VK_FORMAT_R32_UINT);
    const VkDeviceSize pixels = VkDeviceSize(size.width) * size.height;
    const VkDeviceSize sizes[]{std::max<VkDeviceSize>(4, pixels), std::max<VkDeviceSize>(4, (pixels + 3) / 6 * 4),
                               std::max<VkDeviceSize>(8, pixels / 2 * 8), 64, 16};
    bool fresh = false; // a new control buffer: zeroed once, ComputeDispatchArgsCS resets its counters after each use
    for (size_t i = 0; i < c.buffers.size(); ++i) fresh |= ResizeBuffer(c.buffers[i], sizes[i]) && i == 3;
    if (fresh) vkCmdFillBuffer(frame.cmd, c.buffers[3].buffer, 0, 64, 0);
    VkImageMemoryBarrier2 transitions[3]{};
    u32 transition_count = 0;
    Scaled* images[]{&c.output, &c.edges, &c.heads};
    for (Scaled* img : images) {
        if (img->in_general) continue;
        img->in_general = true;
        transitions[transition_count++] = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = img->image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    }
    const VkMemoryBarrier2 filled{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};
    if (transition_count || fresh) {
        const VkDependencyInfo before{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = fresh, .pMemoryBarriers = &filled,
                                      .imageMemoryBarrierCount = transition_count, .pImageMemoryBarriers = transitions};
        vkCmdPipelineBarrier2(frame.cmd, &before);
    }
    // One layout for the four passes: the set is pushed once and stays bound across their pipelines.
    VkDescriptorImageInfo image_info[9]{};
    image_info[0] = {VK_NULL_HANDLE, frame.linear, VK_IMAGE_LAYOUT_GENERAL};
    image_info[1] = {VK_NULL_HANDLE, c.edges.view, VK_IMAGE_LAYOUT_GENERAL};
    image_info[5] = {VK_NULL_HANDLE, c.heads.view, VK_IMAGE_LAYOUT_GENERAL};
    image_info[8] = {VK_NULL_HANDLE, c.output.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo buffer_info[9]{};
    VkWriteDescriptorSet writes[9]{};
    for (u32 i = 0, index = 0; i < 9; ++i) {
        const VkDescriptorType type = CmaaType(i);
        const bool image = type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (!image) buffer_info[i] = {c.buffers[index++].buffer, 0, VK_WHOLE_SIZE};
        writes[i] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i, .descriptorCount = 1, .descriptorType = type,
                     .pImageInfo = image ? &image_info[i] : nullptr, .pBufferInfo = image ? nullptr : &buffer_info[i]};
    }
    push_descriptor(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cmaa_layout, 0, 9, writes);
    // x == 0: the indirect arguments ComputeDispatchArgsCS wrote.
    const auto dispatch = [&](u32 pass, u32 x, u32 y, VkPipelineStageFlags2 next = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) {
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cmaa_pipelines[pass]);
        if (x) vkCmdDispatch(frame.cmd, x, y, 1);
        else vkCmdDispatchIndirect(frame.cmd, c.buffers[4].buffer, 0);
        ComputeBarrier(frame.cmd, next);
    };
    constexpr VkPipelineStageFlags2 args = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    dispatch(0, (size.width + 27) / 28, (size.height + 27) / 28);
    dispatch(1, 2, 1, args);
    dispatch(2, 0, 0);
    dispatch(1, 1, 2, args);
    dispatch(3, 0, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
    return c.output.view;
}

VkImageView Upscale(Frame& frame, VkImageView source, VkExtent2D input, VkExtent2D output) {
    u32 stage = 0;
    for (; input.width < output.width || input.height < output.height; ++stage) {
        if (stage == frame.scaled.size()) frame.scaled.emplace_back();
        const VkExtent2D next{std::min(output.width, input.width * 2), std::min(output.height, input.height * 2)};
        auto& image = frame.scaled[stage];
        ResizeScaled(image, next);
        if (!image.in_general) {
            image.in_general = true;
            const VkImageMemoryBarrier2 transition{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = image.image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            const VkDependencyInfo before{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &transition};
            vkCmdPipelineBarrier2(frame.cmd, &before);
        }
        const VkDescriptorImageInfo desc[2]{{sampler, source, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_GENERAL}};
        const VkWriteDescriptorSet writes[2]{
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &desc[0]},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &desc[1]}};
        const float x = float(input.width), y = float(input.height), ox = float(next.width), oy = float(next.height);
        const float constants[16]{x / ox, y / oy, .5f * x / ox - .5f, .5f * y / oy - .5f,
                                  1.f / x, 1.f / y, 1.f / x, -1.f / y, -1.f / x, 2.f / y, 1.f / x, 2.f / y, 0.f, 4.f / y, 0.f, 0.f};
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, easu_pipeline);
        push_descriptor(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, easu_layout, 0, 2, writes);
        vkCmdPushConstants(frame.cmd, easu_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), constants);
        vkCmdDispatch(frame.cmd, (next.width + 15) / 16, (next.height + 15) / 16, 1);
        const bool more = next.width < output.width || next.height < output.height;
        const VkMemoryBarrier2 memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask = more ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
        const VkDependencyInfo after{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &memory};
        vkCmdPipelineBarrier2(frame.cmd, &after);
        source = image.view, input = next;
    }
    for (size_t i = stage; i < frame.scaled.size(); ++i) FreeScaled(frame.scaled[i]);
    frame.scaled.resize(stage);
    return source;
}

VkImageView DownscaleX(Frame& frame, VkImageView source, VkExtent2D input, u32 out_width) {
    if (frame.scaled.empty()) frame.scaled.emplace_back();
    for (size_t i = 1; i < frame.scaled.size(); ++i) FreeScaled(frame.scaled[i]);
    frame.scaled.resize(1);
    auto& image = frame.scaled[0];
    ResizeScaled(image, {out_width, input.height}, VK_FORMAT_R16G16B16A16_SFLOAT);
    if (!image.in_general) {
        image.in_general = true;
        const VkImageMemoryBarrier2 transition{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = image.image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        const VkDependencyInfo before{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &transition};
        vkCmdPipelineBarrier2(frame.cmd, &before);
    }
    const VkDescriptorImageInfo desc[2]{{sampler, source, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_GENERAL}};
    const VkWriteDescriptorSet writes[2]{
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &desc[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &desc[1]}};
    const int params[3]{int(input.width), int(out_width), int(input.height)};
    vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lanczos_h_pipeline);
    push_descriptor(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, easu_layout, 0, 2, writes);
    vkCmdPushConstants(frame.cmd, easu_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), params);
    vkCmdDispatch(frame.cmd, (out_width + 15) / 16, (input.height + 15) / 16, 1);
    const VkMemoryBarrier2 memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    const VkDependencyInfo after{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &memory};
    vkCmdPipelineBarrier2(frame.cmd, &after);
    return image.view;
}

// The sRGB (srgb) or UNORM sibling of an 8-bit format, or UNDEFINED. sRGB display buffers are sampled through the UNORM view,
// so the scaling keeps the encoded bytes; CMAA2 reads the sRGB view, linear on fetch.
VkFormat Sibling(VkFormat f, bool srgb) {
    constexpr VkFormat pairs[][2]{{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB}, {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB},
                                  {VK_FORMAT_A8B8G8R8_UNORM_PACK32, VK_FORMAT_A8B8G8R8_SRGB_PACK32}};
    for (const auto& p : pairs)
        if (f == p[!srgb]) return p[srgb];
    return VK_FORMAT_UNDEFINED;
}

void DestroySwapchainViews() {
    for (auto view : views) vkDestroyImageView(device, view, nullptr);
    for (auto semaphore : rendered) vkDestroySemaphore(device, semaphore, nullptr);
    views.clear(); rendered.clear();
}

// False while the window has no area (minimized).
bool CreateSwapchain() {
    const auto& vk = Vulkan::Get();
    VkSurfaceCapabilitiesKHR caps{};
    Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical, vk.surface, &caps), "Query surface");
    int width{}, height{};
    SDL_GetWindowSizeInPixels(window, &width, &height);
    extent = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{
        std::clamp(u32(width), caps.minImageExtent.width, caps.maxImageExtent.width),
        std::clamp(u32(height), caps.minImageExtent.height, caps.maxImageExtent.height)};
    if (!extent.width || !extent.height) return false;
    {
        std::scoped_lock lock(Vulkan::QueueMutex());
        vkDeviceWaitIdle(device);
    }
    if (auto* t = timing.load(std::memory_order_acquire)) t->Detach();
    DestroySwapchainViews();
#ifdef _WIN32
    if (std::exchange(exclusive_held, false)) release_exclusive(device, swapchain);
    VkSurfaceFullScreenExclusiveWin32InfoEXT monitor{.sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT,
        .hmonitor = exclusive_mode ? MonitorFromWindow(static_cast<HWND>(SDL_GetPointerProperty(SDL_GetWindowProperties(window),
                                         SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr)), MONITOR_DEFAULTTONEAREST) : nullptr};
    const VkSurfaceFullScreenExclusiveInfoEXT exclusive{.sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT,
        .pNext = exclusive_mode ? &monitor : nullptr,
        .fullScreenExclusive = exclusive_mode ? VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT : VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT};
    const void* chain = vk.full_screen_exclusive ? &exclusive : nullptr;
#else
    const void* chain = nullptr;
#endif
    const VkSwapchainLatencyCreateInfoNV latency{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV, .pNext = chain, .latencyModeEnable = VK_TRUE};
    if (vk.reflex && reflex_requested) chain = &latency;
    const VkSwapchainCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .pNext = chain, .surface = vk.surface,
        .minImageCount = std::min(std::max(caps.minImageCount, 3u), caps.maxImageCount ? caps.maxImageCount : 3u),
        .imageFormat = format.format, .imageColorSpace = format.colorSpace, .imageExtent = extent, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform, .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode, .clipped = VK_TRUE, .oldSwapchain = swapchain};
    VkSwapchainKHR created{};
    Check(vkCreateSwapchainKHR(device, &info, nullptr, &created), "Create swapchain");
    if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = created;
    u32 count{};
    vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr);
    images.resize(count);
    vkGetSwapchainImagesKHR(device, swapchain, &count, images.data());
    for (auto image : images) {
        const VkImageViewCreateInfo view_info{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format.format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        Check(vkCreateImageView(device, &view_info, nullptr, &views.emplace_back()), "Create swapchain view");
        const VkSemaphoreCreateInfo semaphore_info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        Check(vkCreateSemaphore(device, &semaphore_info, nullptr, &rendered.emplace_back()), "Create semaphore");
    }
    present_id = 0;
    if (auto* t = timing.load(std::memory_order_acquire)) t->Attach(swapchain);
    return true;
}

#ifdef _WIN32
// Exclusive mode: taken while the window is full screen with the input focus, given back when it loses either. Refused: exclusive
// mode ends, and the swapchain is remade without it when the driver next fails it (NVIDIA: on the next acquire).
void HoldExclusive() {
    const SDL_WindowFlags flags = SDL_GetWindowFlags(window);
    const bool want = (flags & SDL_WINDOW_INPUT_FOCUS) && (flags & SDL_WINDOW_FULLSCREEN);
    if (want == exclusive_held) return;
    if (!want) return release_exclusive(device, swapchain), exclusive_held = false, Log("Presentation: exclusive full screen released");
    const VkResult result = acquire_exclusive(device, swapchain);
    exclusive_mode = exclusive_held = result == VK_SUCCESS;
    exclusive_lost = false;  // frames again
    Log(exclusive_held ? "Presentation: exclusive full screen acquired"
                       : "Presentation: exclusive full screen refused (" + std::to_string(int(result)) + "); composed from now on");
}

// The swapchain must be remade: out of date, or its exclusive access taken away (taken again as soon as the window has the focus;
// a retry delay here was pushed on by every failing frame, so it never came: black screen after Alt+Tab). Out of exclusive mode
// (refused) it is only remade, without exclusive access.
bool Remake(VkResult result) {
    if (result != VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT) return result == VK_ERROR_OUT_OF_DATE_KHR;
    if (exclusive_mode && !std::exchange(exclusive_lost, true)) Log("Presentation: exclusive full screen lost");
    exclusive_held = false;
    return true;
}
#else
bool Remake(VkResult result) { return result == VK_ERROR_OUT_OF_DATE_KHR; }
#endif

// The flipped display buffer: a view of its image the pass can sample (UNORM sibling of an sRGB format; a BGRA buffer in an
// RGBA image is read as BGRA, byte for byte) and the area of the swapchain it fills, letterboxed with its aspect kept.
// False if the renderer has not drawn it.
bool Show(Frame& frame, VkRect2D& area, VkExtent2D& input, u64 address, bool bgra) {
    Gpu::Surface game{};
    if (!GpuThread::DisplaySurface(address, game)) return false;
    input = {game.w, game.h};
    const bool swap = bgra && (game.format == VK_FORMAT_R8G8B8A8_SRGB || game.format == VK_FORMAT_R8G8B8A8_UNORM);
    const VkFormat unorm = Sibling(game.format, false);
    const VkFormat view_format = swap ? VK_FORMAT_B8G8R8A8_UNORM : unorm ? unorm : game.format;
    if (!frame.game || frame.game_image != game.image || frame.game_format != view_format) {
        for (VkImageView* old : {&frame.game, &frame.linear})
            if (*old) vkDestroyImageView(device, std::exchange(*old, VkImageView{}), nullptr);
        VkImageViewCreateInfo view{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = game.image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = view_format, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        Check(vkCreateImageView(device, &view, nullptr, &frame.game), "Create display buffer view");
        // Sampled only: the target can be a storage image, which an sRGB format cannot be. No sibling (not 8-bit): no CMAA2.
        const VkImageViewUsageCreateInfo sampled{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
        view.pNext = &sampled, view.format = Sibling(view_format, true);
        if (anti_aliasing == AntiAliasing::Cmaa2 && view.format)
            Check(vkCreateImageView(device, &view, nullptr, &frame.linear), "Create display buffer sRGB view");
        frame.game_image = game.image;
        frame.game_format = view_format;
    }
    const float scale = std::min(float(extent.width) / float(game.w), float(extent.height) / float(game.h));
    const int w = std::max(1, int(float(game.w) * scale)), h = std::max(1, int(float(game.h) * scale));
    area = {{(int(extent.width) - w) / 2, (int(extent.height) - h) / 2}, {u32(w), u32(h)}};
    return true;
}

void Draw(s64 latch, u64 address, bool bgra, u64 guest_frame) {
    if (!swapchain && !CreateSwapchain()) return;
#ifdef _WIN32
    if (exclusive_lost) return;
#endif
    const u32 slot = frame_index;
    auto& frame = frames[slot];
    frame_index ^= 1;
    const auto t_start = std::chrono::steady_clock::now();  // frame recorder: the fence and acquire waits, then the present
    vkWaitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
    if (frame.probed) {
        u64 samples{};
        if (vkGetQueryPoolResults(device, probe_queries, slot, 1, sizeof(samples), &samples, sizeof(samples), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && samples)
            Splash::NonBlackFrame();
        frame.probed = false;
    }
    u32 index{};
    auto* t = timing.load(std::memory_order_acquire);
    const bool take_latest = latch && fifo_available && t->DisplaySlowerThanGuest();
    const auto acquired = t->Acquire(frame.acquired, take_latest ? fifo_available : VK_NULL_HANDLE, &index);
    if (Remake(acquired)) { CreateSwapchain(); return; }
    Check(acquired, "Acquire swapchain image");
    if (take_latest) {
        // Acquire can return before the image is usable. Wait here so the shared GPU queue stays free.
        Check(vkWaitForFences(device, 1, &fifo_available, VK_TRUE, UINT64_MAX), "Wait for FIFO image");
        Check(vkResetFences(device, 1, &fifo_available), "Reset FIFO image fence");
        std::scoped_lock lock(flip_mutex);
        if (std::exchange(flipped, false)) {
            latch = flip_time;
            address = display;
            bgra = display_bgra;
            guest_frame = flipped_frame;
        }
    }
    const auto t_acquired = std::chrono::steady_clock::now();
    vkResetFences(device, 1, &frame.fence);
    vkResetCommandBuffer(frame.cmd, 0);
    const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(frame.cmd, &begin);
    VkRect2D area{};
    VkExtent2D input{};
    const bool shown = Show(frame, area, input, address, bgra);
    const bool probe = shown && probe_pipeline && Splash::NeedsFrameProbe();
    if (probe) vkCmdResetQueryPool(frame.cmd, probe_queries, slot, 1);
    // One barrier: the swapchain image (acquired at the color output stage) to a render target, the renderer's earlier
    // writes to the display buffer (same queue) visible to the pass's texture reads, and the previous use of this frame's
    // CMAA2 state (storage) ordered before this one.
    constexpr VkPipelineStageFlags2 written = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    const VkMemoryBarrier2 memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = written,
        .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};
    const VkImageMemoryBarrier2 target{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .image = images[index],
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    const VkDependencyInfo before{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = shown, .pMemoryBarriers = &memory,
                                  .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &target};
    vkCmdPipelineBarrier2(frame.cmd, &before);
    VkImageView source = frame.game;
    if (shown && frame.linear) source = ApplyCmaa(frame, input);
    if (shown && scaling == ScalingFilter::Fsr1 && area.extent.width >= input.width && area.extent.height >= input.height &&
        (area.extent.width > input.width || area.extent.height > input.height)) {
        source = Upscale(frame, source, input, area.extent);
    } else if (shown && scaling == ScalingFilter::Fsr1 && area.extent.width < input.width) {
        source = DownscaleX(frame, source, input, area.extent.width);
    } else if (!frame.scaled.empty()) {
        for (auto& s : frame.scaled) FreeScaled(s);
        frame.scaled.clear();
    }
    // The bars are cleared; a buffer that fills the window overwrites every pixel.
    const bool covers = shown && area.extent.width == extent.width && area.extent.height == extent.height;
    const VkRenderingAttachmentInfo color{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = views[index],
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = covers ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = {.color = {{0, 0, 0, 1}}}};
    const VkRenderingInfo rendering{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, extent},
                                    .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &color};
    vkCmdBeginRendering(frame.cmd, &rendering);
    if (shown) {
        const VkViewport viewport{float(area.offset.x), float(area.offset.y), float(area.extent.width), float(area.extent.height), 0.f, 1.f};
        const VkDescriptorImageInfo image{sampler, source, VK_IMAGE_LAYOUT_GENERAL};
        const VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
                                         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image};
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        push_descriptor(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &write);
        const float exponent = output_gamma.load(std::memory_order_relaxed);
        vkCmdPushConstants(frame.cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(exponent), &exponent);
        vkCmdSetViewport(frame.cmd, 0, 1, &viewport);
        vkCmdSetScissor(frame.cmd, 0, 1, &area);
        vkCmdDraw(frame.cmd, 3, 1, 0, 0);
        if (probe) {
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, probe_pipeline);
            vkCmdBeginQuery(frame.cmd, probe_queries, slot, 0);
            vkCmdDraw(frame.cmd, 3, 1, 0, 0);
            vkCmdEndQuery(frame.cmd, probe_queries, slot);
            frame.probed = true;
        }
    }
    {
        std::scoped_lock lock(Vulkan::QueueMutex());  // dialog font uploads use the queue
        ImGui::Core::Render(frame.cmd, queue);
    }
    vkCmdEndRendering(frame.cmd);
    const VkImageMemoryBarrier2 present_barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, .image = images[index],
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    const VkDependencyInfo after{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &present_barrier};
    vkCmdPipelineBarrier2(frame.cmd, &after);
    vkEndCommandBuffer(frame.cmd);
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1,
        .pWaitSemaphores = &frame.acquired, .pWaitDstStageMask = &wait_stage, .commandBufferCount = 1,
        .pCommandBuffers = &frame.cmd, .signalSemaphoreCount = 1, .pSignalSemaphores = &rendered[index]};
    const u64 id = t->ReflexActive() ? (guest_frame > present_id ? guest_frame : 0) : present_id + 1;
    if (id) present_id = id;
    const VkPresentIdKHR identity{.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR, .swapchainCount = 1, .pPresentIds = &id};
    const VkPresentInfoKHR present{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .pNext = Vulkan::Get().present_wait ? &identity : nullptr, .waitSemaphoreCount = 1,
        .pWaitSemaphores = &rendered[index], .swapchainCount = 1, .pSwapchains = &swapchain, .pImageIndices = &index};
    const VkResult presented = t->SubmitAndPresent(queue, submit, frame.fence, present, id, latch, Vulkan::QueueMutex());
    GpuThread::RecordPresent(t_start, t_acquired);
    if (presented == VK_SUBOPTIMAL_KHR || Remake(presented)) CreateSwapchain();
    else Check(presented, "Present");
}

VkPipeline CreateCompute(std::span<const u32> code, VkPipelineLayout layout) {
    const VkShaderModuleCreateInfo module_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size_bytes(), .pCode = code.data()};
    VkShaderModule module{};
    Check(vkCreateShaderModule(device, &module_info, nullptr, &module), "Create post-effect shader");
    const VkPipelineShaderStageCreateInfo stage{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"};
    const VkComputePipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stage, .layout = layout};
    VkPipeline result{};
    const VkResult status = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &result);
    vkDestroyShaderModule(device, module, nullptr);
    Check(status, "Create post-effect pipeline");
    return result;
}

// The pass: a full-screen triangle (no vertex buffers) and one texture read, compiled once at startup.
void CreatePass(const Config& config) {
    static constexpr const char* vertex = R"(#version 450
layout(location = 0) out vec2 uv;
void main() {
    uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
})";
    static constexpr const char* lanczos = R"(
float weight(float x) {
    x = abs(x);
    if (x < 0.00001) return 1.0;
    if (x >= 2.0) return 0.0;
    float p = 3.141592653589793 * x;
    return sin(p) * sin(p * 0.5) / (p * p * 0.5);
}
vec4 filtered(vec2 uv) {
    ivec2 size = textureSize(game, 0);
    float ratio = max(fwidth(uv.y) * float(size.y), 1.0);
    if (ratio <= 1.0001) return texture(game, uv);
    float pos = uv.y * float(size.y) - 0.5;
    int center = int(floor(pos));
    int radius = int(ceil(2.0 * ratio));
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int y = -radius; y <= radius; ++y) {
        float w = weight((float(center + y) - pos) / ratio);
        float vy = (float(clamp(center + y, 0, size.y - 1)) + 0.5) / float(size.y);
        sum += texture(game, vec2(uv.x, vy)) * w;
        total += w;
    }
    return sum / total;
})";
    static constexpr const char* lanczos_h_compute = R"(#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) uniform sampler2D src_tex;
layout(set = 0, binding = 1, rgba16f) writeonly uniform image2D dst_img;
layout(push_constant) uniform Params { int in_w; int out_w; int height; };
float weight(float x) {
    x = abs(x);
    if (x < 0.00001) return 1.0;
    if (x >= 2.0) return 0.0;
    float p = 3.141592653589793 * x;
    return sin(p) * sin(p * 0.5) / (p * p * 0.5);
}
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= out_w || p.y >= height) return;
    float ratio = float(in_w) / float(out_w);
    float pos = (float(p.x) + 0.5) * ratio - 0.5;
    int center = int(floor(pos));
    int radius = int(ceil(2.0 * ratio));
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int x = -radius; x <= radius; ++x) {
        float w = weight((float(center + x) - pos) / ratio);
        sum += texelFetch(src_tex, ivec2(clamp(center + x, 0, in_w - 1), p.y), 0) * w;
        total += w;
    }
    imageStore(dst_img, p, sum / total);
})";
    const std::string fragment = std::string(R"(#version 450
layout(set = 0, binding = 0) uniform sampler2D game;
layout(push_constant) uniform Output { float gamma; };
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;)") + (config.scaling_filter == ScalingFilter::Fsr1 ? lanczos : "\nvec4 filtered(vec2 uv) { return texture(game, uv); }") + R"(
void main() { color = vec4(pow(max(filtered(uv).rgb, 0.0), vec3(gamma)), 1.0); }
)";
    Gcn::Compiler compiler;
    std::string error;
    VkShaderModule modules[2]{};
    const char* sources[2]{vertex, fragment.c_str()};
    for (int k = 0; k < 2; ++k) {
        const auto spirv = compiler.Spirv(sources[k], k ? shaderc_fragment_shader : shaderc_vertex_shader, error);
        if (spirv.empty()) throw std::runtime_error("Presentation shader: " + error);
        const VkShaderModuleCreateInfo info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spirv.size() * 4, .pCode = spirv.data()};
        Check(vkCreateShaderModule(device, &info, nullptr, &modules[k]), "Create presentation shader");
    }
    const VkFilter filter = VK_FILTER_LINEAR;
    const VkSamplerCreateInfo sampler_info{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = filter, .minFilter = filter,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    Check(vkCreateSampler(device, &sampler_info, nullptr, &sampler), "Create presentation sampler");
    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    const VkDescriptorSetLayoutCreateInfo set_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 1, .pBindings = &binding};
    Check(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &set_layout), "Create presentation descriptor layout");
    // VideoOut gamma on the encoded output values: out = in ^ gamma (the PS4 curve is not documented; 0.8 brightens, as the game expects).
    const VkPushConstantRange output_constants{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};
    const VkPipelineLayoutCreateInfo layout_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &output_constants};
    Check(vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout), "Create presentation pipeline layout");
    push_descriptor = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));
    if (!push_descriptor) throw std::runtime_error("vkCmdPushDescriptorSetKHR is missing");
    if (config.scaling_filter == ScalingFilter::Fsr1) {
        const VkDescriptorSetLayoutBinding bindings[2]{
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT}};
        const VkDescriptorSetLayoutCreateInfo set{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 2, .pBindings = bindings};
        Check(vkCreateDescriptorSetLayout(device, &set, nullptr, &easu_set), "Create FSR descriptor layout");
        const VkPushConstantRange constants{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
        const VkPipelineLayoutCreateInfo layout{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &easu_set, .pushConstantRangeCount = 1, .pPushConstantRanges = &constants};
        Check(vkCreatePipelineLayout(device, &layout, nullptr, &easu_layout), "Create FSR pipeline layout");
        easu_pipeline = CreateCompute(easu_spv, easu_layout);
        const auto lh_spirv = compiler.Spirv(lanczos_h_compute, shaderc_compute_shader, error);
        if (lh_spirv.empty()) throw std::runtime_error("Presentation Lanczos-H shader: " + error);
        lanczos_h_pipeline = CreateCompute(lh_spirv, easu_layout);
    }
    if (config.anti_aliasing == AntiAliasing::Cmaa2) {
        std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
        for (u32 i = 0; i < bindings.size(); ++i) bindings[i] = {i, CmaaType(i), 1, VK_SHADER_STAGE_COMPUTE_BIT};
        const VkDescriptorSetLayoutCreateInfo set{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = u32(bindings.size()), .pBindings = bindings.data()};
        Check(vkCreateDescriptorSetLayout(device, &set, nullptr, &cmaa_set), "Create CMAA2 descriptor layout");
        const VkPipelineLayoutCreateInfo layout{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &cmaa_set};
        Check(vkCreatePipelineLayout(device, &layout, nullptr, &cmaa_layout), "Create CMAA2 pipeline layout");
        const std::span<const u32> code[]{cmaa0_spv, cmaa1_spv, cmaa2_spv, cmaa3_spv};
        for (size_t pass = 0; pass < cmaa_pipelines.size(); ++pass) cmaa_pipelines[pass] = CreateCompute(code[pass], cmaa_layout);
    }
    const VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, modules[0], "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, modules[1], "main", nullptr}};
    const VkPipelineVertexInputStateCreateInfo vertex_input{.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    const VkPipelineInputAssemblyStateCreateInfo assembly{.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                          .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    const VkPipelineViewportStateCreateInfo viewport{.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    const VkPipelineRasterizationStateCreateInfo raster{.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                        .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.f};
    const VkPipelineMultisampleStateCreateInfo multisample{.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                           .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    const VkPipelineColorBlendAttachmentState attachment{.colorWriteMask = 0xF};
    const VkPipelineColorBlendStateCreateInfo blend{.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &attachment};
    const VkDynamicState dynamic_states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic{.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dynamic_states};
    const VkPipelineRenderingCreateInfo rendering{.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &format.format};
    const VkGraphicsPipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering, .stageCount = 2, .pStages = stages,
        .pVertexInputState = &vertex_input, .pInputAssemblyState = &assembly, .pViewportState = &viewport, .pRasterizationState = &raster,
        .pMultisampleState = &multisample, .pColorBlendState = &blend, .pDynamicState = &dynamic, .layout = pipeline_layout};
    Check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline), "Create presentation pipeline");
    if (!config.skip_intro) {
        // The probe writes its attachment, blended to what is there (source x 0 + destination x 1): a fragment shader that
        // writes nothing, with no depth attachment, may not run at all (Intel's Windows driver), and none of its fragments
        // is discarded or counted.
        const std::string probe_fragment = R"(#version 450
layout(set = 0, binding = 0) uniform sampler2D game;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    if (all(equal(texture(game, uv).rgb, vec3(0.0)))) discard;
    color = vec4(0.0);
}
)";
        const auto spirv = compiler.Spirv(probe_fragment, shaderc_fragment_shader, error);
        if (spirv.empty()) throw std::runtime_error("Splash frame probe shader: " + error);
        const VkShaderModuleCreateInfo module_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spirv.size() * 4, .pCode = spirv.data()};
        VkShaderModule fragment_module{};
        Check(vkCreateShaderModule(device, &module_info, nullptr, &fragment_module), "Create splash frame probe shader");
        VkPipelineShaderStageCreateInfo probe_stages[]{stages[0], stages[1]};
        probe_stages[1].module = fragment_module;
        const VkPipelineColorBlendAttachmentState unchanged{.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE, .colorBlendOp = VK_BLEND_OP_ADD, .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xF};
        VkPipelineColorBlendStateCreateInfo probe_blend = blend;
        probe_blend.pAttachments = &unchanged;
        VkGraphicsPipelineCreateInfo probe_info = info;
        probe_info.pStages = probe_stages;
        probe_info.pColorBlendState = &probe_blend;
        const VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &probe_info, nullptr, &probe_pipeline);
        vkDestroyShaderModule(device, fragment_module, nullptr);
        Check(result, "Create splash frame probe pipeline");
        const VkQueryPoolCreateInfo query_info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION, .queryCount = u32(frames.size())};
        Check(vkCreateQueryPool(device, &query_info, nullptr, &probe_queries), "Create splash frame queries");
    }
    for (auto module : modules) vkDestroyShaderModule(device, module, nullptr);
}

void Run(const std::stop_token& token) {
    Common::SetCurrentThreadName("GOW3 presentation");
    auto last_movie_present = std::chrono::steady_clock::time_point{};
    Splash::Priority priority;  // the splash's frames keep up with the shader build
    while (!token.stop_requested()) {
        priority.Update();
#ifdef _WIN32
        if (exclusive_mode && swapchain) HoldExclusive();  // also while no frames come (16 ms wakes)
#endif
        s64 latch{};
        u64 address{};
        bool bgra{};
        u64 guest_frame{};
        {
            std::unique_lock lock(flip_mutex);
            flip_cv.wait_for(lock, token, std::chrono::milliseconds(16), [] { return flipped; });
            if (std::exchange(flipped, false)) latch = flip_time, guest_frame = flipped_frame;
            else if (!ImGui::Core::MustKeepDrawing()) continue;
            address = display; bgra = display_bgra;
        }
        if (GpuThread::CinematicActive()) {
            const auto now = std::chrono::steady_clock::now();
            const auto deadline = last_movie_present + std::chrono::nanoseconds(FramePeriodNs(60));
            if (last_movie_present != std::chrono::steady_clock::time_point{} && deadline > now)
                Common::AccurateSleep(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now), nullptr, false);
            last_movie_present = std::chrono::steady_clock::now();
        } else last_movie_present = {};
        try { Draw(GpuThread::CinematicActive() ? 0 : latch, address, bgra, guest_frame); } catch (const std::exception& error) { Log(std::string("Presentation: ") + error.what()); return; }
    }
}
} // namespace

s64 TakeVblankCorrection() {
    auto* t = timing.load(std::memory_order_acquire);
    return t ? t->TakeCorrection() : 0;
}

void UpdateDisplay() {
    if (auto* t = timing.load(std::memory_order_acquire)) t->UpdateDisplay(window);
}

void Flipped(u64 address, bool bgra, u64 frame_id) {
    { std::scoped_lock lock(flip_mutex); flipped = true; display = address; display_bgra = bgra; flip_time = PresentationTiming::Now(); flipped_frame = frame_id; }
    flip_cv.notify_one();
}

void SetGamma(float value) { output_gamma.store(value, std::memory_order_relaxed); }

void BeforeInput() { if (auto* t = timing.load(std::memory_order_acquire)) t->BeginInput(); }
u64 EndSimulation() { if (auto* t = timing.load(std::memory_order_acquire)) return t->EndSimulation(); return 0; }
void BeginRender(u64 id) {
    render_frame = id;
    if (auto* t = timing.load(std::memory_order_acquire)) t->Marker(id, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
}
void EndRender(u64 id) {
    rendered_frame.store(id, std::memory_order_release);
    if (auto* t = timing.load(std::memory_order_acquire)) t->Marker(id, VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
}
u64 FrameForFlip() { return render_frame ? render_frame : rendered_frame.load(std::memory_order_acquire); }

void Initialize(SDL_Window* sdl_window, const Config& config) {
    window = sdl_window;
    scaling = config.scaling_filter;
    anti_aliasing = config.anti_aliasing;
    const auto& vk = Vulkan::Get();
    device = vk.device;
    queue = vk.queue;
    reflex_requested = config.nvidia_reflex;
    if (reflex_requested && !vk.reflex) Log("NVIDIA Reflex unavailable; using normal presentation pacing");
    timing.store(new PresentationTiming(device, window, FramePeriodNs(config.fps), vk.present_wait, vk.reflex && reflex_requested), std::memory_order_release);
    u32 format_count{};
    vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical, vk.surface, &format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical, vk.surface, &format_count, formats.data());
    if (formats.empty()) throw std::runtime_error("The window surface has no formats");
    const auto preferred = std::ranges::find_if(formats, [](const auto& f) { return f.format == VK_FORMAT_B8G8R8A8_UNORM; });
    format = preferred != formats.end() ? *preferred : formats[0];
    const VkPresentModeKHR wanted = config.vsync == VSync::Mailbox ? VK_PRESENT_MODE_MAILBOX_KHR
                                  : config.vsync == VSync::Off ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR;
    u32 mode_count{};
    vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical, vk.surface, &mode_count, nullptr);
    std::vector<VkPresentModeKHR> modes(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical, vk.surface, &mode_count, modes.data());
    if (std::ranges::find(modes, wanted) != modes.end()) present_mode = wanted;
#ifdef _WIN32
    // Mailbox presents composed: exclusive full screen with it was refused (-3), which left the screen black.
    const bool full_screen = vk.full_screen_exclusive && config.display_mode != DisplayMode::Windowed;
    exclusive_mode = full_screen && present_mode != VK_PRESENT_MODE_MAILBOX_KHR;
    if (exclusive_mode) {
        acquire_exclusive = reinterpret_cast<PFN_vkAcquireFullScreenExclusiveModeEXT>(vkGetDeviceProcAddr(device, "vkAcquireFullScreenExclusiveModeEXT"));
        release_exclusive = reinterpret_cast<PFN_vkReleaseFullScreenExclusiveModeEXT>(vkGetDeviceProcAddr(device, "vkReleaseFullScreenExclusiveModeEXT"));
        exclusive_mode = acquire_exclusive && release_exclusive;
    }
    if (full_screen && present_mode == VK_PRESENT_MODE_MAILBOX_KHR) Log("Presentation: Mailbox presents composed, without exclusive full screen");
#endif
    if (present_mode == VK_PRESENT_MODE_FIFO_KHR) {
        const VkFenceCreateInfo fence{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        Check(vkCreateFence(device, &fence, nullptr, &fifo_available), "Create FIFO image fence");
    }

    const VkCommandPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = vk.family};
    Check(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "Create command pool");
    for (auto& frame : frames) {
        const VkCommandBufferAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        Check(vkAllocateCommandBuffers(device, &allocate, &frame.cmd), "Allocate command buffer");
        const VkFenceCreateInfo fence{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
        Check(vkCreateFence(device, &fence, nullptr, &frame.fence), "Create fence");
        const VkSemaphoreCreateInfo semaphore{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        Check(vkCreateSemaphore(device, &semaphore, nullptr, &frame.acquired), "Create semaphore");
    }

    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = vk.instance; info.PhysicalDevice = vk.physical; info.Device = device;
    info.QueueFamily = vk.family; info.Queue = queue;
    info.DescriptorPoolSize = 512; // one set per texture: the save list alone shows a thumbnail for every save
    info.MinImageCount = 2; info.ImageCount = 3;
    info.UseDynamicRendering = true;
    auto& rendering = info.PipelineInfoMain.PipelineRenderingCreateInfo;
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1; rendering.pColorAttachmentFormats = &format.format;
    info.CheckVkResultFn = [](VkResult result) { Check(result, "Dialog Vulkan operation"); };
    if (!ImGui_ImplVulkan_Init(&info)) throw std::runtime_error("Initialize dialog renderer");
    CreatePass(config);
    thread = std::jthread(Run);
}
} // namespace Gow3::Presenter
