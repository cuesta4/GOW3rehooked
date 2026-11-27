// SPDX-License-Identifier: GPL-2.0-or-later
#include "gpu/vk_gpu.h"
#include "gpu/device.h"
#include "gpu/textures.h"
#include "gpu/renderer_impl.h"
#include "gpu/shaders/shader_recipes.h"
#include "gpu/shaders/gcn_internal.h"
#include "main/cpu.h"
#include "patches/extras.h"
#include "debug/cpu_probe.h"  // TEMP blocks sampling
#include <limits>

namespace Gpu {
namespace {
// ---- Draw-state & shadow-projection helpers ----

// SDK clear: a clip-space rectangle without vertex fetches, and four uniform color components packed to half floats.
constexpr u32 kClearPs[]{0xBEEB03FF, 7, 0xC2800500, 0xBF8C007F, 0x7E000200, 0x7E020201, 0x7E040202,
                         0x7E060203, 0x5E000300, 0x5E020702, 0xF8001C0F, 0x00000100, 0xBF810000};

// The shadow projection VS passes -clip.xy with w = 1 for this transform.
bool ShadowQuad(u64 key, const Gcn::Input& in, std::span<const u32> table) {
    static const u64 program = PieceHash::UsesAes()
        ? ProgramKey(Gcn::Stage::Vertex, 0xd2b827215e1535ffull, 0x9b6d3e622fbdec47ull, 0x18, false, false)
        : ProgramKey(Gcn::Stage::Vertex, 0x99146c4de28c2c12ull, 0x3fa078268c502bbcull, 0x18, false, false);
    constexpr u32 transform[]{0x3f800000, 0, 0, 0, 0, 0x3f800000, 0, 0, 0, 0, 0, 0, 0, 0, 0xbf800000, 0x3f800000};
    return key == program && in.rsrc1 == 0xc00c1 && in.code.size() == 56 && in.fetch.size() == 6 && table.size() >= 20 &&
           std::equal(std::begin(transform), std::end(transform), table.begin() + 4);
}

bool ShadowProjection(u64 key, const Gcn::Input& in, std::span<const u32> table, std::span<const u32* const> desc) {
    static const u64 program = ProgramKey(Gcn::Stage::Pixel, PieceHash::UsesAes() ? 0xd6d546646a8fc599ull : 0x06901769a0c140efull,
                                          0, 0x20, false, false);
    constexpr u32 depth_sampler[]{0x12, 0, 0x01000000, 0};
    constexpr u32 atlas_sampler[]{0xb0b6, 0, 0x02500000, 0x80000000};
    // The scene depth is R32 float, sampled at LOD 0, and the atlas uses fixed LOD 0 PCF.
    return key == program && in.rsrc1 == 0xc0101 && in.code.size() == 88 && table.size() == 44 &&
           table[40] && table[41] && table[42] && table[43] &&
           desc.size() == 4 && desc[0] && desc[1] && desc[2] && desc[3] &&
           (desc[0][1] >> 20 & 0x3ff) == (4 | 7 << 6) && desc[0][3] >> 28 == 9 && (desc[0][3] & 7) == 4 &&
           std::equal(std::begin(depth_sampler), std::end(depth_sampler), desc[1]) &&
           std::equal(std::begin(atlas_sampler), std::end(atlas_sampler), desc[3]);
}

VkRect2D ShadowScissor(VkRect2D sc, const VkViewport& vp, std::span<const u32> table) {
    struct Point { double x, y; };
    struct Plane { double x, y, c, error, inv_z; };
    std::array<double, 24> p;
    for (size_t i = 0; i < p.size(); ++i) {
        p[i] = F(table[16 + i]);
        if (!std::isfinite(p[i])) return sc;
    }
    constexpr double epsilon = 32 * std::numeric_limits<float>::epsilon();
    std::array<Plane, 4> planes;
    for (size_t i = 0; i < 2; ++i) {
        if (p[16 + i] <= 0) return sc;
        const double limit = 1 / p[16 + i];
        const double x = p[i], y = p[4 + i], a = -p[8 + i] * p[20], b = -p[12 + i] * p[21];
        const double c = p[8 + i] * p[22], d = p[12 + i] * p[23];
        const double slope_err = epsilon * (1 + std::abs(x) + std::abs(y) + std::abs(a) + std::abs(b));
        const double offset_err = epsilon * (1 + std::abs(c) + std::abs(d) + limit);
        // The atlas coordinate is z * (x * ndc.x + y * ndc.y + a + b) + c + d.
        planes[i * 2] = {x, y, a + b, slope_err, c + d + offset_err};
        planes[i * 2 + 1] = {-x, -y, -a - b, slope_err, limit - c - d + offset_err};
    }
    if (std::ranges::all_of(planes, [](Plane p) { return p.inv_z >= 0; })) return sc;  // z = 0 can contribute everywhere

    Point lo{1, 1}, hi{-1, -1};
    Point buffers[2][16];
    for (double sign : {1.0, -1.0}) {
        Point* polygon = buffers[0];
        Point* next = buffers[1];
        constexpr Point screen[]{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        std::copy(std::begin(screen), std::end(screen), polygon);
        size_t count = 4;
        auto clip = [&](double x, double y, double c) {
            if (!count) return;
            size_t n = 0;
            Point a = polygon[count - 1];
            double da = x * a.x + y * a.y + c;
            for (size_t k = 0; k < count; ++k) {
                const Point b = polygon[k];
                const double db = x * b.x + y * b.y + c;
                if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
                    const double t = da / (da - db);
                    next[n++] = {std::lerp(a.x, b.x, t), std::lerp(a.y, b.y, t)};
                }
                if (db >= 0) next[n++] = b;
                a = b, da = db;
            }
            std::swap(polygon, next);
            count = n;
        };
        // Eliminate 1/abs(z): every upper bound must allow zero and every lower bound.
        for (Plane upper : planes) {
            if (upper.inv_z > 0) continue;
            const double uc = sign * upper.c + upper.error;
            clip(sign * upper.x, sign * upper.y, uc);
            if (upper.inv_z == 0) continue;
            for (Plane lower : planes) {
                if (lower.inv_z <= 0) continue;
                const double lc = sign * lower.c + lower.error;
                clip(sign * (upper.x * lower.inv_z - lower.x * upper.inv_z),
                     sign * (upper.y * lower.inv_z - lower.y * upper.inv_z), uc * lower.inv_z - lc * upper.inv_z);
            }
        }
        for (size_t k = 0; k < count; ++k) {
            const Point p = polygon[k];
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y)};
        }
    }
    if (lo.x > hi.x || lo.y > hi.y) return sc;
    auto axis = [](double a, double b, double origin, double span, int offset, u32 extent) {
        a = origin + (a + 1) * span / 2, b = origin + (b + 1) * span / 2;
        const double end = double(offset) + extent;
        // Round outward and keep the quads at the projection edge intact.
        const int first = int(std::clamp(std::floor(std::min(a, b)) - 2, double(offset), end));
        const int last = int(std::clamp(std::ceil(std::max(a, b)) + 2, double(first), end));
        return std::pair{first, u32(last - first)};
    };
    const auto [x, w] = axis(lo.x, hi.x, vp.x, vp.width, sc.offset.x, sc.extent.width);
    const auto [y, h] = axis(lo.y, hi.y, vp.y, vp.height, sc.offset.y, sc.extent.height);
    return {{x, y}, {w, h}};
}

// Early tests must not suppress guest side effects or change discarded/depth-exporting fragments.
bool BlendShaderSafe(std::span<const u32> code, u32 rsrc2, u32 layout) {
    const u64 hash = ::Gpu::PieceHash::ShaderCode(code.data(), code.size_bytes());
    static std::mutex mutex;
    static std::map<std::tuple<u64, u32, u32>, bool> checked;
    std::scoped_lock lock(mutex);
    const auto key = std::tuple{hash, rsrc2, layout & 0xff};
    if (auto it = checked.find(key); it != checked.end()) return it->second;
    Gcn::Input in;
    in.code = code, in.rsrc2 = rsrc2;
    const Gcn::Analysis a = Gcn::Analyze(in);
    bool safe = a.error.empty() && a.program && !a.stored && !a.program->exec_dynamic && a.program->labels.empty() &&
                std::none_of(a.slots.begin(), a.slots.end(), [](const Gcn::Slot& s) { return s.kind == Gcn::SlotKind::Storage; });
    u32 exports = 0;
    if (safe) {
        const auto& insts = a.program->insts;
        for (size_t k = 0; k < insts.size(); ++k) {
            const auto& i = insts[k];
            if (i.enc == Gcn::DS || i.enc == Gcn::MUBUF || i.enc == Gcn::MTBUF || (i.enc == Gcn::SOPP && i.op == 1 && k + 1 != insts.size())) safe = false;
            if (i.enc == Gcn::VINTRP && i.op != 2) {
                static constexpr u32 sizes[]{2, 2, 2, 3, 2, 2, 2, 1};
                const u32 source = i.w0 & 0xff;
                for (u32 g = 0, base = 0; g < 8; ++g)
                    if (layout >> g & 1) {
                        if (source >= base && source < base + sizes[g] && (g == 0 || g == 4)) safe = false;
                        base += sizes[g];
                    }
            }
            if (i.enc == Gcn::EXP) safe &= (i.w0 >> 4 & 0x3f) == 0, ++exports;
        }
        safe &= exports == 1;
    }
    return checked.emplace(key, safe).first->second;
}

VkStencilOp StencilOp(u32 f) {
    static constexpr VkStencilOp t[]{VK_STENCIL_OP_KEEP, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE,
                                     VK_STENCIL_OP_INCREMENT_AND_CLAMP, VK_STENCIL_OP_DECREMENT_AND_CLAMP, VK_STENCIL_OP_INVERT,
                                     VK_STENCIL_OP_INCREMENT_AND_WRAP, VK_STENCIL_OP_DECREMENT_AND_WRAP};
    return f < std::size(t) ? t[f] : VK_STENCIL_OP_KEEP;
}
} // namespace

// ---- Thread-local per-recorder state, setup & public Renderer lifecycle ----

thread_local Renderer::Impl::DynState Renderer::Impl::dyn;
thread_local Hazards Renderer::Impl::hz;
thread_local Renderer::Impl::Prepared Renderer::Impl::prep;
thread_local Renderer::Impl::Memo Renderer::Impl::memo;


bool Renderer::Impl::Setup(std::string& error) {
    LoadDeviceFunctions();

    const auto set_msaa = [&](const VkPhysicalDeviceLimits& l) {
        const VkSampleCountFlags m = l.framebufferColorSampleCounts & l.framebufferDepthSampleCounts &
                                     l.framebufferStencilSampleCounts & l.sampledImageColorSampleCounts &
                                     l.sampledImageDepthSampleCounts;
        msaa_samples = m ? m : VK_SAMPLE_COUNT_1_BIT;
    };
    if (caps.path == ShaderPath::Objects) {  // a driver binary only loads on the GPU and driver that made it, with the bindless layout it was made against
        VkPhysicalDeviceShaderObjectPropertiesEXT so{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &so};
        vkGetPhysicalDeviceProperties2(pd, &p2);
        set_msaa(p2.properties.limits);
        struct { u8 uuid[VK_UUID_SIZE]; u32 version, vendor, device, driver, layout[3]; } id{};
        std::memcpy(id.uuid, so.shaderBinaryUUID, VK_UUID_SIZE);
        id.version = so.shaderBinaryVersion, id.vendor = p2.properties.vendorID, id.device = p2.properties.deviceID, id.driver = p2.properties.driverVersion;
        id.layout[0] = kTextures, id.layout[1] = kSamplers, id.layout[2] = kStorage;
        binary_seed = ::Gpu::PieceHash::DriverKey(&id, sizeof(id));
    } else {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(pd, &p);
        set_msaa(p.limits);
        struct { u8 uuid[VK_UUID_SIZE]; u32 path, vendor, device, driver, layout[3]; } id{};
        std::memcpy(id.uuid, p.pipelineCacheUUID, VK_UUID_SIZE);
        id.path = u32(caps.path), id.vendor = p.vendorID, id.device = p.deviceID, id.driver = p.driverVersion;
        id.layout[0] = kTextures, id.layout[1] = kSamplers, id.layout[2] = kStorage;
        binary_seed = ::Gpu::PieceHash::DriverKey(&id, sizeof(id));
    }
    ClearState();
    const auto fp = QueueFamilies(pd);
    copy_family = CopyFamily(fp);  // CreateDevice asks for the copy engine's queues
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
    vkCreateCommandPool(dev, &pci, nullptr, &cmd_pool);
    if (copy_family != ~0u) pci.queueFamilyIndex = copy_family, vkCreateCommandPool(dev, &pci, nullptr, &copy_pool), pci.queueFamilyIndex = family;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (VkCommandBuffer* c : {&primary, &copy_cmd, &flight.primary, &flight.copy_cmd}) vkAllocateCommandBuffers(dev, &cai, c);
    if (copy_pool) cai.commandPool = copy_pool;
    for (VkCommandBuffer* c : {&xfer_cmd, &flight.xfer_cmd, &send_cmd, &flight.send_cmd, &rb_cmd, &flight.rb_cmd}) vkAllocateCommandBuffers(dev, &cai, c);
    vkCreateFence(dev, &fci, nullptr, &fence), vkCreateFence(dev, &fci, nullptr, &flight.fence);
    const VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (VkSemaphore* s : {&copied, &flight.copied, &rendered, &flight.rendered, &bound_sem, &flight.bound_sem}) vkCreateSemaphore(dev, &semi, nullptr, s);
    if (copy_family != ~0u) {
        vkGetDeviceQueue(dev, copy_family, 0, &xfer);
        if (fp[copy_family].queueCount > 1) vkGetDeviceQueue(dev, copy_family, 1, &rbq);
    } else if (fp[family].queueCount > 1) {
        vkGetDeviceQueue(dev, family, 1, &xfer);
    } else {
        xfer = queue, xfer_lock = queue_lock;
    }
    // Recording threads, the submitting one included (Cpu::plan; 1: serial recording).
    recorders.resize(Gow3::Cpu::plan.recorders);
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    for (auto& w : recorders)
        for (auto& p : w.pool) vkCreateCommandPool(dev, &pci, nullptr, &p);
    for (u32 k = 0; k + 1 < recorders.size(); ++k)
        threads.emplace_back([this, k] {
#ifdef _WIN32
            SetThreadDescription(GetCurrentThread(), std::format(L"GOW3 recorder {}", k).c_str());
#else
            Common::SetCurrentThreadName(std::format("GOW3 recorder {}", k).c_str());
#endif
            Gow3::Cpu::Place(Gow3::Cpu::Role::Recorder, k);
            Gow3::CpuProbe::Register(2 + k);  // TEMP
            for (u64 seen = 0;;) {
                {
                    std::unique_lock lock(pool_lock);
                    pool_cv.wait(lock, [&] { return pool_stop || pool_gen != seen; });
                    if (pool_stop) return;
                    seen = pool_gen;
                }
                RenderProbe::Bind(k);
                (*pool_job)(k);
                std::scoped_lock lock(pool_lock);
                if (--pool_busy == 0) pool_cv.notify_all();
            }
        });
    {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(pd, &pp);
        tick_ms = pp.limits.timestampPeriod * 1e-6;
    }
    if (prof) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_TIMESTAMP, 2 * kProfQueries};
        vkCreateQueryPool(dev, &qi, nullptr, &prof_pool);
    }
    GOW3_DIAG {  // per frame slot GPU timestamps of the copy engine, upload and render work
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_TIMESTAMP, 12};
        vkCreateQueryPool(dev, &qi, nullptr, &temp_qp);
        vkResetQueryPool(dev, temp_qp, 0, 12);
    }
    const VkQueryPoolCreateInfo oqi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_OCCLUSION, 2 * kQueries};
    vkCreateQueryPool(dev, &oqi, nullptr, &query_pool);
    cond = MakeBuffer(2 * kQueries * 4, VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, "cond");
    if (!cond.buf) return error = "occlusion query buffer allocation failed", false;

    if (!SetupMirror(error)) return false;
    if (!SetupTables()) return error = "draw table buffer allocation failed", false;
    if (!SetupTextures()) return error = "dummy texture allocation failed", false;
    VkPushConstantRange pcr{VK_SHADER_STAGE_ALL, 0, sizeof(Push)};
    const VkDescriptorSetLayout layouts[2]{set_layout, table_layout};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 2, layouts, 1, &pcr};
    if (const VkResult e = vkCreatePipelineLayout(dev, &plci, nullptr, &layout); e != VK_SUCCESS) Report(std::format("vkCreatePipelineLayout {}", (int)e));
    if (caps.path != ShaderPath::Objects) {
        LoadPipelineCache();
    }
    if (caps.path == ShaderPath::Gpl) SetupGplGlobals();
    if (!SetupClearRect(error)) return false;
    GOW3_DIAG TempDeviceReport(fp);
    ShaderCache::Instance().SetPrepare([this](u64 key, u64 binary, u32 stage) { return Prepare(key, binary, stage); });
    if (ShaderCache::Instance().Enabled() && ShaderCache::Instance().Kind() == CacheKind::Aot) AotBuild();  // the game waits for the build
    if (caps.path != ShaderPath::Objects && ShaderCache::Instance().Enabled()) {
        ShaderCache::Instance().SetWarmPipelines([this] { WarmPipelines(); });
        ShaderCache::Instance().SetPipelineCacheFlusher([this](bool exit) {
            if (!exit) return;  // the flushes in play leave the pipeline cache alone
            SavePipelineCache();
            TempPlWrite();  // TEMP: the exit's flush
        });
    }
    return true;
}

Renderer::Renderer() : d(std::make_unique<Impl>()) {}
Renderer::~Renderer() {
    if (!d->dev) return;
    {
        std::scoped_lock lock(d->pool_lock);
        d->pool_stop = true;
    }
    d->pool_cv.notify_all();
    for (auto& t : d->threads) t.join();
    vkDeviceWaitIdle(d->dev);
    for (int k = 0; k < 3; ++k) {  // the evicted textures not destroyed yet
        if (d->reaping.valid()) d->reaping.wait();
        d->Reap();
    }
    for (auto& w : d->recorders)
        for (auto p : w.pool) vkDestroyCommandPool(d->dev, p, nullptr);
    d->StopCompilers();
    ShaderCache::Instance().ClearRendererCallbacks();
    if (d->caps.path != ShaderPath::Objects) d->SavePipelineCache();
    ShaderCache::Instance().Finish();
    Recipes::Instance().Report();
    for (auto& [k, f] : d->compiling_pipelines) (void)f.get();
    d->compiling_pipelines.clear();
    for (auto& [k, f] : d->compiling) {
        const ShaderRef r = f.get().Ref(k);
        auto it = d->shaders.find(k);
        if (it == d->shaders.end() || it->second != r) d->DestroyShaderRef(r);
    }
    for (auto& [k, s] : d->shaders) d->DestroyShaderRef(s);
    for (auto& [k, s] : d->rect_gs) d->DestroyShaderRef(s);
    d->DestroyShaderRef(d->clear_rect_ref);
    for (auto& [k, p] : d->pipelines) {
        if (p.main) vkDestroyPipeline(d->dev, p.main, nullptr);
        if (p.square) vkDestroyPipeline(d->dev, p.square, nullptr);
    }
    for (VkPipeline p : d->retired_pipelines) vkDestroyPipeline(d->dev, p, nullptr);
    d->retired_pipelines.clear();
    for (auto p : d->gpl_vi)
        if (p) vkDestroyPipeline(d->dev, p, nullptr);
    for (auto& [k, p] : d->gpl_prerast)
        if (p) vkDestroyPipeline(d->dev, p, nullptr);
    for (auto& [k, p] : d->gpl_frag_msaa)
        if (p) vkDestroyPipeline(d->dev, p, nullptr);
    for (auto& [k, p] : d->gpl_foi)
        if (p) vkDestroyPipeline(d->dev, p, nullptr);
    if (d->q_stat) vkDestroyQueryPool(d->dev, d->q_stat, nullptr), vkDestroyQueryPool(d->dev, d->q_occ, nullptr);
    d->DestroyTextures();
    for (int k = 0; k < 2; ++k, d->SwapFrame()) {
        for (auto* v : {&d->stage, &d->stage_dev, &d->frame_temp})
            for (auto& b : *v) d->FreeBuffer(b);
        d->FreeBuffer(d->tables_host), d->FreeBuffer(d->tables), d->FreeBuffer(d->rb), d->FreeBuffer(d->rb_dev), d->FreeBuffer(d->scratch);
        vkDestroyFence(d->dev, d->fence, nullptr), vkDestroySemaphore(d->dev, d->copied, nullptr), vkDestroySemaphore(d->dev, d->rendered, nullptr),
            vkDestroySemaphore(d->dev, d->bound_sem, nullptr);
    }
    if (d->copy_pool) vkDestroyCommandPool(d->dev, d->copy_pool, nullptr);
    for (auto* b : {&d->probe_buf, &d->vertex_probe, &d->index_probe}) d->FreeBuffer(*b);
    if (d->index_probe_vs && d->DestroyShader) d->DestroyShader(d->dev, d->index_probe_vs, nullptr);
    if (d->index_probe_mod) vkDestroyShaderModule(d->dev, d->index_probe_mod, nullptr);
    for (auto& [k, p] : d->index_probe_pls)
        if (p) vkDestroyPipeline(d->dev, p, nullptr);
    d->DestroyMirror();
    if (d->pipeline_cache) vkDestroyPipelineCache(d->dev, d->pipeline_cache, nullptr);
    vkDestroyPipelineLayout(d->dev, d->layout, nullptr);
    if (d->prof_pool) vkDestroyQueryPool(d->dev, d->prof_pool, nullptr);
    if (d->temp_qp) vkDestroyQueryPool(d->dev, d->temp_qp, nullptr);
    if (d->query_pool) vkDestroyQueryPool(d->dev, d->query_pool, nullptr), d->FreeBuffer(d->cond);
    vkDestroyCommandPool(d->dev, d->cmd_pool, nullptr);
    if (!d->owned) return;
    vkDestroyDevice(d->dev, nullptr);
    if (d->messenger) ((PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(d->inst, "vkDestroyDebugUtilsMessengerEXT"))(d->inst, d->messenger, nullptr);
    vkDestroyInstance(d->inst, nullptr);
}

bool Renderer::Init(const External& e, std::string& error) {
    d->owned = false;
    d->inst = e.instance, d->pd = e.physical, d->dev = e.device, d->queue = e.queue, d->family = e.family, d->queue_lock = e.queue_lock;
    return d->Setup(error);
}
void Renderer::SetTracker(std::function<bool(std::uint64_t, std::uint64_t)> track) { d->track = std::move(track); }
void Renderer::SetReadback(std::function<std::uint64_t(std::uint64_t, const void*, std::uint64_t)> readback) { d->readback = std::move(readback); }
void Renderer::SetQueries(std::uint64_t lo, std::uint64_t hi) { d->query_lo = lo, d->query_hi = hi; }
void Renderer::SetAnisotropy(unsigned level) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(d->pd, &properties);
    d->anisotropy = std::min(float(level), properties.limits.maxSamplerAnisotropy);
}
void Renderer::SetMotionBlurFrameTime(float seconds) {
    d->motion_blur_rate = seconds >= 0.00001f && std::isfinite(seconds) ? std::max(1.f, 1.f / (59.94f * seconds)) : 1.f;
}
const std::vector<const std::uint32_t*>* Renderer::Effects() const { return d->effects_whole ? &d->effects : nullptr; }
Stats& Renderer::GetStats() {
    GOW3_DIAG {
        constexpr float mb = 1.f / (1 << 20);
        auto& s = d->stats;
        {
            std::scoped_lock lock(d->image_lock);
            s.vram_mb[1] = float(d->image_blocks.size() * d->image_block_size) * mb, s.vram_mb[2] = float(d->image_free[0] + d->image_free[1]) * mb;
            s.vram_mb[5] = float(std::count_if(d->image_blocks.begin(), d->image_blocks.end(), [](auto& b) { return b.lasting; }) * d->image_block_size) * mb;
            s.vram_mb[6] = float(std::count_if(d->image_blocks.begin(), d->image_blocks.end(), [](auto& b) { return b.draining; }) * d->image_block_size) * mb;
        }
        s.vram_mb[0] = float(d->mirror_chunks.load(std::memory_order_relaxed)) * mb;
        s.vram_mb[3] = float(d->dedicated_images.load(std::memory_order_relaxed)) * mb;
        s.vram_mb[4] = float(d->device_buffers.load(std::memory_order_relaxed)) * mb;
        s.vram_mb[7] = float(d->host_buffers.load(std::memory_order_relaxed)) * mb;
        s.vram_textures = u32(d->textures.size());
    }
    return d->stats;
}

// ---- Occlusion queries and predication ----

// One end of an occlusion query the CPU reads, outside any rendering pass (both ends then sit in one command buffer and
// no pass instance); Submit hands the counts on.
void Renderer::Impl::Query(u64 a) {
    FlushStores();
    EndRendering(false);
    if (!(a & 8) && !query_open && query_blocks.size() < kRays)
        vkCmdBeginQuery(cmd, query_pool, slot * kQueries + (u32)query_blocks.size(), VK_QUERY_CONTROL_PRECISE_BIT), query_blocks.push_back(a), query_open = true;
    else if (a & 8 && query_open)
        vkCmdEndQuery(cmd, query_pool, slot * kQueries + (u32)query_blocks.size() - 1), query_open = false;
}

// A mesh query's dump. The begin numbers it in walk order (the walk maps its block to it) and leaves it to the next draw,
// which begins it inside its pass; the end, where `record` (this thread records the range it lies in), closes it, or
// ends an empty one when no draw began it: the copy waits for every query it covers.
void Renderer::Impl::MeshQuery(const u32* b, bool record) {
    const u64 a = ((u64)b[1] | (u64)(b[2] & 0xFFFF) << 32) & ~7ull;
    if (!(a & 8)) {
        query_next = query_seq < kQueries ? query_seq++ : ~0u;
        if (!in_worker && query_next != ~0u) mesh[a] = query_next;
        return;
    }
    if (record && query_next != ~0u && query_on == ~0u) vkCmdBeginQuery(cmd, query_pool, slot * kQueries + query_next, 0), query_on = query_next;
    if (record) EndQuery();
    query_next = ~0u;
}

// The walk, at a draw: when it is predicated on a query `cond` lacks, the queries numbered since the last copy go there
// first (CopyQueries, before it in its command buffer).
std::pair<u32, u32> Renderer::Impl::CopyRange() {
    const auto q = pred ? mesh.find(pred) : mesh.end();
    return q == mesh.end() || q->second < mesh_copied ? std::pair<u32, u32>{} : std::pair{std::exchange(mesh_copied, query_seq), query_seq};
}
void Renderer::Impl::CopyQueries(std::pair<u32, u32> range) {
    EndRendering(false);
    if (block_now) block_now->started = true;  // its first command is no rendering begin
    vkCmdCopyQueryPoolResults(cmd, query_pool, slot * kQueries + range.first, range.second - range.first, cond.buf, (slot * kQueries + range.first) * 4ull, 4,
                              VK_QUERY_RESULT_WAIT_BIT);
    StageBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT, VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT);
}

// ---- Draw and dispatch recording ----

void Renderer::Impl::Draw(u32 op, const u32* b, const CpuRead& cpu) {
    RenderProbe::DrawScope draw_time;
    FlushStores();
    const u32 prim = uc[0x242], color_control = ctx[0x202], mode = color_control >> 4 & 7, depth_control = ctx[0x200];
    if (stats.trace)
        stats.trace_lines.push_back(std::format("op {:x} prim {:x} n {} mode {} cb0 {:x} pitch {:x} slice {:x} view {:x} info {:x} attr {:x} col_fmt {:x} "
                                          "mask {:x} z {:x} zinfo {:x} zsize {:x} dc {:x} vte {:x} vp {} {} {} {} {} {} scr {:x} {:x} gen {:x} {:x} "
                                          "vps {:x} {:x} win {:x} {:x} {:x} clip {:x} su {:x} sinfo {:x} sctl {:x} sref {:x} {:x} rc {:x} blend {:x} "
                                          "bc {} {} {} {} a2m {:x} vs {:x} ps {:x}",
                                          op, prim, op == 0x2D ? b[0] : op == 0x27 ? b[3] : b[2], mode, ctx[0x318], ctx[0x319], ctx[0x31A], ctx[0x31B],
                                          ctx[0x31C], ctx[0x31D], ctx[0x1C5], ctx[0x08E], ctx[0x014], ctx[0x010], ctx[0x016], depth_control, ctx[0x206],
                                          F(ctx[0x10F]), F(ctx[0x110]), F(ctx[0x111]), F(ctx[0x112]), F(ctx[0x113]), F(ctx[0x114]), ctx[0x00C], ctx[0x00D],
                                          ctx[0x090], ctx[0x091], ctx[0x094], ctx[0x095], ctx[0x080], ctx[0x081], ctx[0x082], ctx[0x204], ctx[0x205],
                                          ctx[0x011], ctx[0x10B], ctx[0x10C], ctx[0x10D], ctx[0x000], ctx[0x1E0], F(ctx[0x105]), F(ctx[0x106]),
                                          F(ctx[0x107]), F(ctx[0x108]), ctx[0x2DC],
                                          (u64)(sh[0x48] | (u64)sh[0x49] << 32) << 8, (u64)(sh[0x08] | (u64)sh[0x09] << 32) << 8));
    if (draw_seq++ > stats.stop) return;
    draw_vs_addr = (u64)(sh[0x48] | (u64)sh[0x49] << 32) << 8;
    draw_ps_addr = (u64)(sh[0x08] | (u64)sh[0x09] << 32) << 8;
    draw_rts.clear();
    if (mode > 1) return Count(stats.skipped), Note(std::format("CB_COLOR_CONTROL mode {} skipped", mode));
    const bool rect = prim == 0x11, rect_vs = RectVs(op, b, cpu);
    VkPrimitiveTopology topo;
    switch (prim) {
    case 1: topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
    case 2: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
    case 3: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
    case 4: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
    case 0x11: topo = rect_vs ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
    case 5: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
    case 6: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
    default: return Count(stats.skipped), Note(std::format("primitive type {:#x} skipped", prim));
    }

    // Attachments.
    const Target* color[8]{};
    VkImageView color_view[8]{};
    u32 rw = ~0u, rh = ~0u, col_format = ctx[0x1C5], target_mask = ctx[0x08E], color_count = 0;
    for (u32 k = 0; k < 8 && mode == 1; ++k) {
        const u32* cb = &ctx[0x318 + k * 15];
        const VkFormat fmt = ColorFormat(cb[4]);
        if (!cb[0] || fmt == VK_FORMAT_UNDEFINED || !(col_format >> (4 * k) & 0xF) || !(target_mask >> (4 * k) & 0xF)) continue;
        const u32 pitch = ((cb[1] & 0x7FF) + 1) * 8, ph = ((cb[2] & 0x3FFFFF) + 1) * 64 / pitch;
        color[k] = GetTarget((u64)cb[0] << 8, pitch, ph, fmt, false, false, cb_extent[k], 1u << (cb[5] >> 12 & 7), cb[5] & 0x1F);  // ATTRIB NUM_SAMPLES, TILE_MODE_INDEX
        if (!color[k]) return Count(stats.skipped), Note("render target allocation failed");
        // Written only when it changes: the target is shared by every recording thread.
        if (const bool bgra = AmdGpu::MrtSwizzle(cb[4] >> 11 & 3, AmdGpu::DataFormat(cb[4] >> 2 & 0x1F)).Map(0) == 2;
            std::atomic_ref(color[k]->bgra).load(std::memory_order_relaxed) != bgra)
            std::atomic_ref(color[k]->bgra).store(bgra, std::memory_order_relaxed);
        color_view[k] = fmt == color[k]->fmt ? color[k]->t.view : TargetAttachment(*color[k], fmt);
        rw = std::min(rw, color[k]->w), rh = std::min(rh, color[k]->h), color_count = k + 1;
    }
    u32 samples = 0;
    // DB_DEPTH_CONTROL: STENCIL_ENABLE 0, Z_ENABLE 1, Z_WRITE_ENABLE 2, DEPTH_BOUNDS_ENABLE 3, ZFUNC 4-6.
    const bool z_test = depth_control >> 1 & 1, z_write = depth_control >> 2 & 1, stencil = depth_control & 1, z_bounds = caps.depth_bounds && (depth_control >> 3 & 1);
    const u32 z_fmt = ctx[0x010] & 3, s_fmt = ctx[0x011] & 1;
    const Target* depth = nullptr;
    const Target* clear_apart = nullptr;  // depth cleared in its own pass: its sample count differs from the color's
    // DB_RENDER_CONTROL depth/stencil clear: the clear value lands under the draw; color is still written by the PS.
    const bool db_clear = ctx[0x000] & 3;
    const u32 st_ctl = ctx[0x10B], st_ref_f = ctx[0x10C], st_ref_b = (depth_control >> 7 & 1) ? ctx[0x10D] : st_ref_f;
    const u32 st_fn_f = depth_control >> 8 & 7, st_fn_b = (depth_control >> 7 & 1) ? (depth_control >> 20 & 7) : st_fn_f;
    const auto st_op_writes = [](u32 fn, u32 op_fail, u32 op_zpass, u32 op_zfail, u32 wmask) {
        return (wmask & 0xFF) != 0 && ((fn != 7 && op_fail != 0) || (fn != 0 && (op_zpass != 0 || op_zfail != 0)));
    };
    const bool st_can_reject = stencil && s_fmt && (st_fn_f != 7 || ((depth_control >> 7 & 1) && st_fn_b != 7));
    const bool st_can_write = stencil && s_fmt &&
                              (st_op_writes(st_fn_f, st_ctl & 0xF, st_ctl >> 4 & 0xF, st_ctl >> 8 & 0xF, st_ref_f >> 16) ||
                               ((depth_control >> 7 & 1) && st_op_writes(st_fn_b, st_ctl >> 12 & 0xF, st_ctl >> 16 & 0xF, st_ctl >> 20 & 0xF, st_ref_b >> 16)));
    const bool stencil_eff = caps.path == ShaderPath::Objects ? stencil : (st_can_reject || st_can_write);
    const bool z_effect = (z_test && (depth_control >> 4 & 7) != 7) || z_write || stencil_eff || z_bounds;  // test ALWAYS: no test
    for (const Target* t : color)
        if (t && (samples = samples ? samples : t->samples) != t->samples) return Count(stats.skipped), Note("color targets with mixed sample counts skipped");
    if (z_fmt && (z_effect || db_clear)) {
        const u32 size = ctx[0x016], w = ((size & 0x7FF) + 1) * 8, h = ((size >> 11 & 0x7FF) + 1) * 8;
        const VkFormat fmt = z_fmt == 1 ? VK_FORMAT_D16_UNORM : s_fmt ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
        depth = GetTarget((u64)ctx[0x014] << 8, w, h, fmt, true, s_fmt, db_extent, 1u << (ctx[0x010] >> 2 & 3), ctx[0x010] >> 20 & 7);  // Z_INFO NUM_SAMPLES, TILE_MODE_INDEX
        if (!depth) return Count(stats.skipped), Note("depth target allocation failed");
        if (samples && (depth->samples != samples || (!z_effect && db_clear && (depth->w < rw || depth->h < rh)))) {
            if (depth->samples != samples && z_effect) return Count(stats.skipped), Note("color/depth sample counts differ, skipped");
            clear_apart = depth, depth = nullptr;
        } else {
            rw = std::min(rw, depth->w), rh = std::min(rh, depth->h), samples = depth->samples;
        }
    }
    if (rw == ~0u) return Count(stats.skipped), Note("draw without attachments skipped");

    // Restrict early tests to shaders without guest side effects, depth exports or dynamic discard.
    const u64 ps_addr = draw_ps_addr;
    const u32 bc0 = ctx[0x1E0], info0 = ctx[0x31C], mask0 = color[0] ? AmdGpu::MrtSwizzle(info0 >> 11 & 3, AmdGpu::DataFormat(info0 >> 2 & 0x1F)).ApplyMask(target_mask & 15) : 0;
    const bool sep0 = bc0 >> 29 & 1;
    const u32 as0 = sep0 ? bc0 >> 16 & 31 : bc0 & 31, af0 = sep0 ? bc0 >> 21 & 7 : bc0 >> 5 & 7, ad0 = sep0 ? bc0 >> 24 & 31 : bc0 >> 8 & 31;
    u32 blend_mode = 0, blend_element = 0;
    if (blend_interlock && mode == 1 && color[0] && ColorFormat(info0) == VK_FORMAT_R8G8B8A8_SRGB &&
        (color[0]->t.usage & VK_IMAGE_USAGE_STORAGE_BIT) && mask0 == 8 && (bc0 >> 30 & 1) && (af0 == 2 || af0 == 3) &&
        (as0 == 2 || as0 == 4) && (ad0 == 8 || ad0 == 6) && samples <= 32 && !z_write && !stencil && !db_clear &&
        !(ctx[0x2DC] & 1) && !(ctx[0x203] & 0x41) && !((ctx[0x1B4] ? ctx[0x1B4] : ctx[0x1B3]) & (1u << 14)) &&
        query_next == ~0u && query_on == ~0u && !query_open &&
        std::none_of(color + 1, color + 8, [](const Target* t) { return t != nullptr; })) {
        const u32* header = (const u32*)cpu(ps_addr, 8);
        const u64 bytes = header && header[0] == 0xBEEB03FF ? ((u64)header[1] + 1) * 8 : 0;
        const void* code = bytes && bytes < (1u << 20) ? cpu(ps_addr, bytes) : nullptr;
        if (code && BlendShaderSafe({(const u32*)code, size_t(bytes / 4)}, sh[0x0B], ctx[0x1B4] ? ctx[0x1B4] : ctx[0x1B3])) {
            blend_element = TargetBlendElement(*color[0]);
            if (blend_element) blend_mode = samples | (af0 == 3 ? 0x100u : 0u);
        }
    }

    const Target* clear_z = clear_apart ? clear_apart : depth;
    VkImageAspectFlags clear_pending = clear_z ? (ctx[0x000] & 1 ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u) |
                                               (ctx[0x000] & 2 && clear_z->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u) : 0;
    bool full_depth_clear = false;
    bool color_clear_pending = false;
    VkClearColorValue clear_color{};
    auto depth_attachments = [&](const Target* z, VkImageAspectFlags clear) {
        std::array<VkRenderingAttachmentInfo, 2> a{};
        for (u32 k = 0; k < a.size(); ++k) {
            a[k].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            if (z) a[k].imageView = z->t.view, a[k].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            a[k].loadOp = clear & (VK_IMAGE_ASPECT_DEPTH_BIT << k) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            a[k].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            a[k].clearValue.depthStencil = {F(ctx[0x00B]), ctx[0x00A] & 0xFF};
        }
        return a;
    };
    // Called once the draw's textures are resolved: a texture uploaded again mid-frame ends the rendering first.
    size_t vs_rts = 0;  // the first of `draw_rts` are the vertex shader's
    auto use = [&] {  // the targets this draw samples and stores to
        for (size_t k = 0; k < draw_rts.size(); ++k) {
            const auto [img, store] = draw_rts[k];
            if (Add(hz.r, img); store) Add(pass_w, img);
            if (!hz.fenced) Add(hz.in_r, img);
            if (k < vs_rts && !hz.wide) Add(hz.vs_in, img);
        }
    };
    // Its vertex shader samples a target written before a barrier that left the vertex stages out (`all`: or since the
    // last barrier, or stores to one).
    auto vs_pending = [&](bool all) {
        return std::any_of(draw_rts.begin(), draw_rts.begin() + vs_rts, [&](auto& u) { return In(hz.wv, u.first) || (all && (u.second || In(hz.w, u.first))); });
    };
    auto begin = [&] {
        const bool same = rendering && std::equal(color_view, color_view + 8, bound_color) && depth == bound_depth;
        // A full barrier only where this draw depends on the passes since the last one: it samples a target they
        // wrote, or stores to one they sampled; a target they sampled or rendered to only needs the attachment barrier.
        // The open pass counts too: the tonemap samples the scene the bloom draw before it added to (tile-sized stale
        // squares in the bloom without the barrier).
        auto conflict = [&] {
            return vs_pending(false) || std::any_of(draw_rts.begin(), draw_rts.end(), [&](auto& u) {
                       return In(hz.w, u.first) || (u.second ? !In(pass_w, u.first) && In(hz.r, u.first) : In(pass_w, u.first));
                   });
        };
        if (same && !conflict()) return use();
        EndRendering(false);
        bool raw = conflict(), war = false, waw = false;
        for (const Target* t : color)
            if (t) war |= In(hz.r, t->t.img), waw |= In(hz.w, t->t.img);
        if (depth) war |= In(hz.r, depth->t.img), waw |= In(hz.w, depth->t.img);
        if (raw) {
            Barrier(cmd, !vs_pending(true));
        } else if (war || waw) {
            if (block_now) block_now->started = true;
            PassBarrier(cmd, 0, war);
            if (war) {
                for (const Target* t : color)
                    if (t) std::erase(hz.r, t->t.img);
                if (depth) std::erase(hz.r, depth->t.img);
            }
        }
        VkRenderingAttachmentInfo ca[8]{};
        for (u32 k = 0; k < color_count; ++k) {
            ca[k].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            if (color[k]) ca[k].imageView = color_view[k], ca[k].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            ca[k].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, ca[k].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            if (k == 0 && color_clear_pending) ca[k].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, ca[k].clearValue.color = clear_color;
        }
        const VkImageAspectFlags load_clear = depth && full_depth_clear ? clear_pending : 0;
        const auto da = depth_attachments(depth, load_clear);
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {rw, rh}}, ri.layerCount = 1, ri.colorAttachmentCount = color_count, ri.pColorAttachments = color_count ? ca : nullptr;
        ri.pDepthAttachment = depth ? &da[0] : nullptr;
        ri.pStencilAttachment = depth && depth->stencil ? &da[1] : nullptr;
        if (prof)
            prof_pass = ProfBegin(std::format("pass c0 {:x} {}x{} z {:x} s{}", color[0] ? color[0]->base : 0, rw, rh, depth ? depth->base : 0, samples)),
            prof_draws = stats.draws;
        if (block_now && !block_now->started) {
            block_now->started = block_now->clean = true, block_now->head.depth = depth;
            std::copy(color_view, color_view + 8, block_now->head.color);
        }
        if (temp_label_begin)
            TempLabel(cmd, std::format("pass c0 {:x} {}x{} z {:x}", color[0] ? color[0]->base : 0, rw, rh, depth ? depth->base : 0).c_str());
        TempCheck(cmd, 1, (color[0] ? color[0]->base : depth ? depth->base : 0) >> 8);
        vkCmdBeginRendering(cmd, &ri);
        clear_pending &= ~load_clear;  // a later texture upload may reopen this pass without clearing again
        color_clear_pending = false;
        std::copy(color_view, color_view + 8, bound_color), bound_depth = depth, rendering = true;
        pass_w.clear();
        for (const Target* t : color)
            if (t) Add(pass_w, t->t.img);
        if (depth) Add(pass_w, depth->t.img);
        use();
    };

    // Viewport: PA_CL_VPORT_* (identity when VTE disables scale/offset). Z = z_ndc * zs + zo, with z_ndc in [0, 1] (DX clip
    // space) or [-1, 1] (PA_CL_CLIP_CNTL DX_CLIP_SPACE_DEF = 0, Vulkan negativeOneToOne).
    const u32 vte = ctx[0x206], clip = ctx[0x204];
    const bool gl_clip = !(clip >> 19 & 1);
    const float xs = vte & 1 ? F(ctx[0x10F]) : 1.0f, xo = vte & 2 ? F(ctx[0x110]) : 0.0f, ys = vte & 4 ? F(ctx[0x111]) : 1.0f,
                yo = vte & 8 ? F(ctx[0x112]) : 0.0f, zs = vte & 16 ? F(ctx[0x113]) : 1.0f, zo = vte & 32 ? F(ctx[0x114]) : 0.0f;
    const VkViewport vp{xo - xs, yo - ys, 2 * xs, 2 * ys, std::clamp(gl_clip ? zo - zs : zo, 0.0f, 1.0f), std::clamp(zo + zs, 0.0f, 1.0f)};
    if (vp.width <= 0 || vp.height == 0) return Count(stats.skipped), Note("degenerate viewport skipped");
    auto tl = [&](u32 r) { return std::pair<int, int>((int)(r & 0x7FFF), (int)(r >> 16 & 0x7FFF)); };
    auto [x0, y0] = tl(ctx[0x00C]);
    auto [x1, y1] = tl(ctx[0x00D]);
    for (u32 r : {0x081u, 0x090u, 0x094u}) {  // window, generic, viewport 0 scissors
        auto [a, bb] = tl(ctx[r]);
        auto [c, d] = tl(ctx[r + 1]);
        x0 = std::max(x0, a), y0 = std::max(y0, bb), x1 = std::min(x1, c), y1 = std::min(y1, d);
    }
    x1 = std::min<int>(x1, rw), y1 = std::min<int>(y1, rh);
    const VkRect2D sc{{x0, y0}, {(u32)std::max(0, x1 - x0), (u32)std::max(0, y1 - y0)}};
    full_depth_clear = clear_z && sc.offset.x == 0 && sc.offset.y == 0 && sc.extent.width >= clear_z->w && sc.extent.height >= clear_z->h;

    if (db_clear) {
        auto clear = [&](const Target* z, VkRect2D r) {
            r.extent.width = std::min<u32>(r.extent.width, z->w - std::min<u32>(z->w, r.offset.x));
            r.extent.height = std::min<u32>(r.extent.height, z->h - std::min<u32>(z->h, r.offset.y));
            VkClearAttachment ca{clear_pending};
            ca.clearValue.depthStencil = {F(ctx[0x00B]), ctx[0x00A] & 0xFF};
            VkClearRect cr{r, 0, 1};
            if (ca.aspectMask && r.extent.width && r.extent.height) vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
            clear_pending = 0;
        };
        if (clear_apart) {
            EndRendering();
            if (block_now) block_now->started = true;
            const auto da = depth_attachments(clear_apart, full_depth_clear ? clear_pending : 0);
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {clear_apart->w, clear_apart->h}}, ri.layerCount = 1, ri.pDepthAttachment = &da[0];
            ri.pStencilAttachment = clear_apart->stencil ? &da[1] : nullptr;
            vkCmdBeginRendering(cmd, &ri);
            if (!full_depth_clear) clear(clear_apart, sc);
            clear_pending = 0;
            vkCmdEndRendering(cmd);
            Add(hz.w, clear_apart->t.img);
        }
        if (depth) {
            begin();
            if (clear_pending) clear(depth, sc);
        }
        bool writes_color = false;
        for (u32 k = 0; k < 8; ++k) writes_color |= color[k] && target_mask >> (4 * k) & 0xF;
        if (!writes_color || mode != 1 || !(sh[0x08] | sh[0x09])) return void(Count(stats.draws));
    }

    // Programs and their draw tables.
    ps_table.clear();
    u64 vs_mask = 0, unused = 0;
    waiting = false;
    read_prog = (u64)(sh[0x48] | (u64)sh[0x49] << 32) << 8;
    const ShaderRef vs = Shader(Gcn::Stage::Vertex, (u64)(sh[0x48] | (u64)sh[0x49] << 32) << 8, &sh[0x4C], cpu, sh[0x4A], sh[0x4B], vs_mask, rect_vs, vs_table);
    const bool clear_vs = rect_vs && ClearVs(prep.in.code);
    const bool shadow_quad = vs && prim == 6 && op == 0x2D && b[0] == 4 && instances <= 1 && mask0 == VK_COLOR_COMPONENT_B_BIT &&
                             !stats.trace && !boom_on && ShadowQuad(prep.pkey, prep.in, vs_table);
    vs_rts = draw_rts.size();
    // Depth-only passes still run the PS when it exports Z or kills (DB_SHADER_CONTROL Z_EXPORT / KILL_ENABLE).
    ShaderRef ps = ps_addr && (mode == 1 || ctx[0x203] & 0x41)
                       ? Shader(Gcn::Stage::Pixel, ps_addr, &sh[0x0C], cpu, sh[0x0A], sh[0x0B], unused, rect, ps_table, blend_mode)
                       : ShaderRef{};
    if (blend_mode && std::any_of(draw_rts.begin(), draw_rts.end(), [&](const auto& use) { return use.first == color[0]->t.img; })) {
        blend_mode = 0;
        draw_rts.resize(vs_rts);
        ps = Shader(Gcn::Stage::Pixel, ps_addr, &sh[0x0C], cpu, sh[0x0A], sh[0x0B], unused, rect, ps_table);
    }
    if (blend_mode) {
        ps_table.push_back(blend_element);
        draw_rts.push_back({color[0]->t.img, true});
        color_view[0] = VK_NULL_HANDLE;  // written by the PS through its storage view, not as an attachment
    }
    const ShaderRef gs = rect && !rect_vs ? RectGs(vs_mask) : ShaderRef{};
    // No PS program: the draw only touches depth/stencil (color writes masked below).
    if (!vs || (ps_addr && mode == 1 && !ps) || (rect && !rect_vs && !gs)) {
        stores.clear(), Count(stats.skipped);
        if (!waiting) Note(std::format("draw without {} skipped", !vs ? "VS" : rect && !rect_vs && !gs ? "rect GS" : "PS"));
        return;
    }
    // Only full, unconditional SDK clears. Zero and one survive the half-float export exactly on every driver.
    const bool native_clear = clear_vs && ps && mode == 1 && color[0] && color[0]->w == rw && color[0]->h == rh &&
                              (target_mask & 15) == 15 && (col_format & 15) == 4 &&
                              (ColorFormat(info0) == VK_FORMAT_R8G8B8A8_UNORM || ColorFormat(info0) == VK_FORMAT_R8G8B8A8_SRGB) &&
                              std::none_of(color + 1, color + 8, [](const Target* t) { return t != nullptr; }) && !(bc0 >> 30 & 1) &&
                              (!depth || db_clear || !z_effect) && !(ctx[0x2DC] & 1) && instances <= 1 && !pred &&
                              query_next == ~0u && query_on == ~0u && !query_open && !boom_on &&
                              vp.x == 0 && vp.width == rw && ((vp.y == 0 && vp.height == rh) || (vp.y == rh && vp.height == -float(rh))) &&
                              sc.offset.x == 0 && sc.offset.y == 0 && sc.extent.width == rw && sc.extent.height == rh &&
                              prep.in.code.size() >= std::size(kClearPs) && std::equal(std::begin(kClearPs), std::end(kClearPs), prep.in.code.begin()) &&
                              ps_table.size() == 4 && std::all_of(ps_table.begin(), ps_table.end(), [](u32 v) { return v == 0 || v == 0x3F800000; });
    if (native_clear) {
        const auto swz = AmdGpu::MrtSwizzle(info0 >> 11 & 3, AmdGpu::DataFormat(info0 >> 2 & 0x1F));
        for (u32 k = 0; k < 4; ++k) clear_color.float32[k] = F(ps_table[swz.Map(k)]);
        color_clear_pending = true;
        begin();
        if (color_clear_pending) {
            const VkClearAttachment ca{VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = clear_color}};
            const VkClearRect cr{sc, 0, 1};
            vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
        }
        if ((ctx[0x31D] & 0x1F) == 8) WriteBackTarget(*color[0], ((ctx[0x319] & 0x7FF) + 1) * 8, sc);
        WriteBack(), Count(stats.draws);
        return;
    }
    BlendState& bs = blend_memo;
    if (color_count) {
        u32 key[18]{bool(ps) | color_count << 9, target_mask};
        for (u32 k = 0; k < color_count; ++k) key[0] |= u32(color[k] != nullptr) << (k + 1), key[2 + k] = ctx[0x1E0 + k], key[10 + k] = ctx[0x31C + k * 15];
        if (std::memcmp(bs.key, key, sizeof(key))) {
            std::memcpy(bs.key, key, sizeof(key)), bs.square = false, bs.square_color_mask = 0, bs.square_alpha_mask = 0, bs.unsupported = 0;
            for (u32 k = 0; k < color_count; ++k) {
                const u32 bc = ctx[0x1E0 + k], info = ctx[0x31C + k * 15], nt = info >> 8 & 7;  // CB_COLOR_INFO NUMBER_TYPE: 0 UNORM, 6 SRGB
                const bool sep = bc >> 29 & 1;
                u32 cs = bc & 0x1F, cf = bc >> 5 & 7, cd = bc >> 8 & 0x1F, as = sep ? bc >> 16 & 31 : cs, af = sep ? bc >> 21 & 7 : cf,
                    ad = sep ? bc >> 24 & 31 : cd;
                bs.blend[k] = color[k] && bc >> 30 & 1;
                // CB_TARGET_MASK names exported components: the export swizzle (COMP_SWAP) moves them to image channels.
                bs.mask[k] = color[k] && ps ? AmdGpu::MrtSwizzle(info >> 11 & 3, AmdGpu::DataFormat(info >> 2 & 0x1F)).ApplyMask(target_mask >> (4 * k) & 0xF) : 0;
                const MinMax rc = RewriteMinMax(cs, cf, cd, nt == 0 || nt == 6, false), ra = RewriteMinMax(as, af, ad, nt == 0 || nt == 6, true);
                const bool wc = bs.blend[k] && bs.mask[k] & 7, wa = bs.blend[k] && bs.mask[k] & 8;
                bs.unsupported += (wc && rc == MinMax::Unsupported) || (wa && ra == MinMax::Unsupported);
                const bool sc_c = wc && rc == MinMax::Squared, sc_a = wa && ra == MinMax::Squared;
                bs.square |= sc_c || sc_a;
                if (sc_c) bs.square_color_mask |= u8(1u << k);
                if (sc_a) bs.square_alpha_mask |= u8(1u << k);
                bs.eq[k] = {BlendFactor(cs), BlendFactor(cd), BlendOp(cf), BlendFactor(as), BlendFactor(ad), BlendOp(af)};
                bs.square_blend[k] = bs.mask[k] != 0;  // second draw: dst * dst on squared channels, dst elsewhere
                bs.square_eq[k] = {VK_BLEND_FACTOR_ZERO, sc_c ? VK_BLEND_FACTOR_DST_COLOR : VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD,
                                   VK_BLEND_FACTOR_ZERO, sc_a ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD};
            }
        }
        for (u32 n = bs.unsupported; n; --n) Note("scaled MIN/MAX blend not representable");
    }
    FixedPipelineState raw_state{};
    CompiledPipeline pl{};
    if (caps.path != ShaderPath::Objects) {
        raw_state.color_count = u8(color_count);
        for (u32 k = 0; k < color_count; ++k) {
            raw_state.color_formats[k] = color_view[k] ? ColorFormat(ctx[0x318 + k * 15 + 4]) : VK_FORMAT_UNDEFINED;
            raw_state.blend[k] = {u8(bs.blend[k] ? 1 : 0), u8(bs.mask[k]),
                                  u8(bs.eq[k].srcColorBlendFactor), u8(bs.eq[k].dstColorBlendFactor), u8(bs.eq[k].colorBlendOp),
                                  u8(bs.eq[k].srcAlphaBlendFactor), u8(bs.eq[k].dstAlphaBlendFactor), u8(bs.eq[k].alphaBlendOp)};
        }
        raw_state.depth_format = depth ? depth->fmt : VK_FORMAT_UNDEFINED;
        raw_state.stencil_format = (depth && depth->stencil) ? depth->fmt : VK_FORMAT_UNDEFINED;
        raw_state.samples = u8(std::max(1u, samples));
        raw_state.topology_class = (rect && !rect_vs) ? 2
                                   : topo == VK_PRIMITIVE_TOPOLOGY_POINT_LIST ? 0
                                   : (topo == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || topo == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP) ? 1
                                   : 2;
        raw_state.flags = (gl_clip ? FixedPipelineState::kNegOne : 0) |
                          ((clip >> 26 & 3) ? FixedPipelineState::kDepthClamp : 0) |
                          ((ctx[0x2DC] & 1) ? FixedPipelineState::kAlphaToCoverage : 0) |
                          (ps ? FixedPipelineState::kHasPs : 0) |
                          (gs ? FixedPipelineState::kHasGs : 0);
        const bool want_square = color_count && !blend_mode && !(z_write || stencil_eff);
        raw_state.square_color_mask = want_square ? bs.square_color_mask : 0;
        raw_state.square_alpha_mask = want_square ? bs.square_alpha_mask : 0;
        pl = GetGraphicsPipeline(vs, ps, gs, raw_state, cpu);
        if (!pl.main) {
            stores.clear(), Count(stats.skipped);
            if (!waiting) Note("draw without pipeline skipped");
            return;
        }
    }
    begin();
    auto put = [&](const std::vector<u32>& t) {
        const u64 n = t.size() * 4, at = TableAt(n, table_align);
        if (at != ~0ull) std::memcpy(TablePtr() + at, t.data(), n);
        return (u32)at;
    };
    if (stats.trace && vertex_probe.buf) {  // the probe programs write it in every traced draw: the last one shared past the end
        const u64 at = vertex_probe.addr + std::min<u64>(probes.size(), kProbes - 1) * kVertexProbe;
        vs_table.push_back((u32)at), vs_table.push_back((u32)(at >> 32));
    } else if (boom_on) {  // this draw's exploded-vertex record (the last one shared past the end)
        const u32 k = std::min(boom_next.fetch_add(1, std::memory_order_relaxed), kBoomDraws - 1);
        const u64 at = boom_buf.addr + ((u64)slot * kBoomDraws + k) * kBoomRec;
        vs_table.push_back((u32)at), vs_table.push_back((u32)(at >> 32));
        boom_meta[slot][k] = {(u64)(sh[0x48] | (u64)sh[0x49] << 32) << 8, ps_addr, (op == 0x2D ? b[0] : op == 0x27 ? b[3] : b[2]) * std::max(1u, instances)};
        std::copy_n(&sh[0x4C], 16, boom_meta[slot][k].user);
        BoomUser(&sh[0x4C], cpu);
    }
    const u32 offsets[2]{vs_table.empty() ? 0 : put(vs_table), ps_table.empty() ? 0 : put(ps_table)};
    if (offsets[0] == ~0u || offsets[1] == ~0u) return stores.clear(), Count(stats.skipped), Note("draw tables full");
    RenderProbe::Scope state_time(RenderProbe::State);
    RenderProbe::Cmd(RenderProbe::Tables, 2);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &table_sets[slot], 2, offsets);
    Push push{mirror_addr};
    std::copy_n(&sh[0x4C], 16, push.user[0]), std::copy_n(&sh[0x0C], 16, push.user[1]);
    // Push constants last through the command buffer: only the dword span that differs from its last push (~4 of 34).
    constexpr u32 kPushWords = sizeof(Push) / 4;
    u64 diff = (1ull << kPushWords) - 1;
    if (dyn.pushed) {
        const u32 *now = (const u32*)&push, *was = (const u32*)&dyn.push;
        diff = 0;
        for (u32 k = 0; k < kPushWords; ++k) diff |= u64(now[k] != was[k]) << k;
    }
    if (diff) {
        const u32 lo = std::countr_zero(diff), hi = 64 - std::countl_zero(diff);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL, lo * 4, (hi - lo) * 4, (const u32*)&push + lo), dyn.push = push, dyn.pushed = true;
    }
    // Dynamic state: each group is emitted only when it differs from what this command buffer last recorded.
    auto changed_n = [](auto* last, const auto* now, size_t n) {
        if (std::memcmp(last, now, n * sizeof(*now)) == 0) return false;
        std::memcpy(last, now, n * sizeof(*now));
        return true;
    };
    auto changed = [&](auto& last, const auto& now) { return changed_n(&last, &now, 1); };
    const bool first = !dyn.valid;
    decltype(dyn) prof_before;  // profile: state groups this draw changes
    if (prof >= 2) prof_before = dyn;
    if (first) {  // fixed for every draw
        RenderProbe::Cmd(RenderProbe::Fixed, caps.path == ShaderPath::Objects ? 4 : 1);
        dyn.valid = true;
        vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
        if (caps.path == ShaderPath::Objects) {
            SetPolygonMode(cmd, VK_POLYGON_MODE_FILL);
            SetLogicOpEnable(cmd, VK_FALSE);
            SetVertexInput(cmd, 0, nullptr, 0, nullptr);
        }
    }
    if (caps.path == ShaderPath::Objects) {
        const VkShaderStageFlagBits stages[]{VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_GEOMETRY_BIT, VK_SHADER_STAGE_FRAGMENT_BIT,
                                             VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT};
        const VkShaderEXT bound[]{vs.obj, gs.obj, ps.obj, VK_NULL_HANDLE, VK_NULL_HANDLE};
        u32 lo = 0, hi = first ? 5 : 3;
        while (!first && lo < hi && dyn.shaders[lo] == bound[lo]) ++lo;
        while (!first && hi > lo && dyn.shaders[hi - 1] == bound[hi - 1]) --hi;
        if (lo < hi) RenderProbe::Cmd(RenderProbe::Shaders), BindShaders(cmd, hi - lo, stages + lo, bound + lo), std::copy(bound + lo, bound + hi, dyn.shaders + lo);
    } else {
        if (dyn.pipeline != pl.main || first)
            RenderProbe::Cmd(RenderProbe::Shaders), dyn.pipeline = pl.main, vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.main);
    }
    if (SetNegativeOneToOne && (dyn.neg_one != gl_clip || first))
        RenderProbe::Cmd(RenderProbe::Clip), dyn.neg_one = gl_clip, SetNegativeOneToOne(cmd, gl_clip);
    if (changed(dyn.vp, vp) || first) RenderProbe::Cmd(RenderProbe::Viewport), vkCmdSetViewportWithCount(cmd, 1, &vp);

    // Raster, depth/stencil, blend: all dynamic.
    const u32 su = ctx[0x205];
    const VkCullModeFlags cull = rect ? VK_CULL_MODE_NONE : (su & 1 ? VK_CULL_MODE_FRONT_BIT : 0) | (su & 2 ? VK_CULL_MODE_BACK_BIT : 0);
    // DB_ALPHA_TO_MASK ALPHA_TO_MASK_ENABLE: cutouts (grilles, foliage) on MSAA targets. ZCLIP_NEAR/FAR_DISABLE: clamp
    // instead of clip. POLY_OFFSET_FRONT_ENABLE: PA_SU_POLY_OFFSET_FRONT_SCALE (1/16 units per slope) and _OFFSET,
    // clamped by _CLAMP.
    const u32 raster[8]{samples, ctx[0x2DC] & 1, (clip >> 26 & 3) != 0, cull, u32(su & 4 ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE),
                        (su >> 11 & 3) != 0, (u32)topo, (topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || topo == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP) && ctx[0x2A5] & 1};
    if (const u32 old_samples = dyn.raster[0]; changed(dyn.raster, raster) || first) {
        RenderProbe::Cmd(RenderProbe::Raster, 7);
        if (raster[0] != old_samples || first) {
            if (SetSamples) RenderProbe::Cmd(RenderProbe::Raster), SetSamples(cmd, VkSampleCountFlagBits(samples));
            if (SetSampleMask) {
                RenderProbe::Cmd(RenderProbe::Raster);
                const VkSampleMask sample_mask = ~0u;
                SetSampleMask(cmd, VkSampleCountFlagBits(samples), &sample_mask);
            }
        }
        if (SetAlphaToCoverage) SetAlphaToCoverage(cmd, raster[1]);
        if (SetDepthClamp) SetDepthClamp(cmd, raster[2]);
        vkCmdSetCullMode(cmd, raster[3]);
        vkCmdSetFrontFace(cmd, VkFrontFace(raster[4]));
        vkCmdSetDepthBiasEnable(cmd, raster[5]);
        vkCmdSetPrimitiveTopology(cmd, topo);
        vkCmdSetPrimitiveRestartEnable(cmd, raster[7]);
    }
    const float bias[3]{F(ctx[0x2E1]), F(ctx[0x2DF]), F(ctx[0x2E0]) / 16.0f};
    if (changed(dyn.bias, bias) || first) RenderProbe::Cmd(RenderProbe::Bias), vkCmdSetDepthBias(cmd, bias[0], bias[1], bias[2]);
    // A clearing draw leaves the clear value.
    const u32 dz[5]{depth && z_test && !db_clear, depth && z_write && !db_clear, depth_control >> 4 & 7, depth && depth->stencil && stencil_eff && !db_clear,
                    depth && z_bounds && !db_clear};
    if (changed(dyn.depth, dz) || first) {
        RenderProbe::Cmd(RenderProbe::Depth, 5);
        vkCmdSetDepthTestEnable(cmd, dz[0]);
        vkCmdSetDepthWriteEnable(cmd, dz[1]);
        vkCmdSetDepthCompareOp(cmd, (VkCompareOp)dz[2]);
        vkCmdSetStencilTestEnable(cmd, dz[3]);
        if (caps.path == ShaderPath::Objects || caps.depth_bounds) vkCmdSetDepthBoundsTestEnable(cmd, dz[4]);
    }
    if (dz[4]) {
        const float bmin = std::clamp(F(ctx[0x008]), 0.0f, 1.0f), bmax = std::clamp(F(ctx[0x009]), bmin, 1.0f);
        const float bounds[2]{bmin, bmax};
        if (changed(dyn.bounds, bounds) || first) RenderProbe::Cmd(RenderProbe::Bounds), vkCmdSetDepthBounds(cmd, bounds[0], bounds[1]);
    }
    const u32 sc_ctl = ctx[0x10B], ref = ctx[0x10C], back = depth_control >> 7 & 1, rb = back ? ctx[0x10D] : ref;
    const u32 st[14]{sc_ctl & 0xF, sc_ctl >> 4 & 0xF, sc_ctl >> 8 & 0xF, depth_control >> 8 & 7, (back ? sc_ctl >> 12 : sc_ctl) & 0xF,
                     (back ? sc_ctl >> 16 : sc_ctl >> 4) & 0xF, (back ? sc_ctl >> 20 : sc_ctl >> 8) & 0xF,
                     back ? depth_control >> 20 & 7 : depth_control >> 8 & 7, ref & 0xFF, ref >> 8 & 0xFF, ref >> 16 & 0xFF, rb & 0xFF, rb >> 8 & 0xFF,
                     rb >> 16 & 0xFF};
    VkRect2D draw_sc = sc;
    // The projection writes 1 outside the atlas rect, an identity for unscaled MIN on UNORM.
    if (shadow_quad && ps && !blend_mode && col_format == 4 && (bc0 >> 30 & 1) && (bc0 & 0x1fff) == 0x141 &&
        ColorFormat(info0) == VK_FORMAT_R8G8B8A8_UNORM && std::none_of(color + 1, color + 8, [](const Target* t) { return t != nullptr; }) &&
        !db_clear && !z_write && (!stencil || (!st[10] && !st[13])) && !raster[1] && !pred &&
        query_next == ~0u && query_on == ~0u && !query_open && ctx[0x1B3] == 2 && ctx[0x1B4] == 2 && ctx[0x191] == 0 &&
        vp.x == 0 && vp.y == rh && vp.width == rw && vp.height == -float(rh) &&
        draw_rts.size() == vs_rts + 2 && ShadowProjection(prep.pkey, prep.in, ps_table, prep.desc))
        draw_sc = ShadowScissor(sc, vp, ps_table);
    if (changed(dyn.sc, draw_sc) || first) RenderProbe::Cmd(RenderProbe::Scissor), vkCmdSetScissorWithCount(cmd, 1, &draw_sc);
    if (changed(dyn.stencil, st) || first) {
        RenderProbe::Cmd(RenderProbe::Stencil, 8);
        vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_BIT, StencilOp(st[0]), StencilOp(st[1]), StencilOp(st[2]), (VkCompareOp)st[3]);
        vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_BACK_BIT, StencilOp(st[4]), StencilOp(st[5]), StencilOp(st[6]), (VkCompareOp)st[7]);
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_BIT, st[8]);
        vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_BIT, st[9]);
        vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_BIT, st[10]);
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_BACK_BIT, st[11]);
        vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_BACK_BIT, st[12]);
        vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_BACK_BIT, st[13]);
    }
    const Target* probe_target = blend_mode ? color[0] : nullptr;
    if (color_count) {
        for (u32 k = 0; k < color_count && !probe_target; ++k) probe_target = bs.mask[k] ? color[k] : nullptr;
        const bool first_blend = dyn.blend_count == 0, more = color_count > dyn.blend_count;
        if (more) dyn.blend_count = color_count;
        if (SetBlendEnable && (changed_n(dyn.blend, bs.blend, color_count) || more))
            RenderProbe::Cmd(RenderProbe::Blend), SetBlendEnable(cmd, 0, color_count, bs.blend);
        if (SetBlendEquation && (changed_n(dyn.eq, bs.eq, color_count) || more))
            RenderProbe::Cmd(RenderProbe::Blend), SetBlendEquation(cmd, 0, color_count, bs.eq);
        if (SetWriteMask && (changed_n(dyn.mask, bs.mask, color_count) || more))
            RenderProbe::Cmd(RenderProbe::Blend), SetWriteMask(cmd, 0, color_count, bs.mask);
        const float constants[4]{F(ctx[0x105]), F(ctx[0x106]), F(ctx[0x107]), F(ctx[0x108])};
        if (changed(dyn.constants, constants) || first_blend) RenderProbe::Cmd(RenderProbe::Blend), vkCmdSetBlendConstants(cmd, constants);
    }

    state_time.End();
    const u32 inst = std::max(1u, instances);
    u32 first_index = 0;
    if (op != 0x2D) {
        // A window runs from four bytes past a 4 GB multiple to the next one (4 GB - 4 bytes, aligned for both index types).
        // ANV, AMD and Intel's Windows driver keep a binding's byte count in 32 bits; without VK_KHR_maintenance5 the binding
        // runs to the mirror's end, a multiple of 4 GB, so they see (-offset) mod 4 GB bytes: from the window's start, the
        // window's size, the same one maintenance5 binds; from a 4 GB multiple, none (every index reads 0).
        // Indices outside one window are bound at their own address: whole with maintenance5, up to the next multiple without.
        const u64 addr = op == 0x27 ? ((u64)b[1] | (u64)(b[2] & 0xFFFF) << 32) : index_base + (u64)b[1] * index_size;
        const u64 bytes = (u64)(op == 0x27 ? b[3] : b[2]) * index_size;
        // Indices past the mirror's end (above 16 GB when the driver could not reserve 32) cannot be bound.
        if (addr + bytes > mirror_size) return stores.clear(), Count(stats.skipped), Note("indices beyond the mirror skipped");
        const u64 line = addr & ~(kIndexWindow - 1);  // the 4 GB multiple at or below the indices
        const bool inside = addr >= line + 4 && addr + bytes <= line + kIndexWindow;
        const u64 window = inside ? line + 4 : addr;
        if (!BindIndex2 && (addr == line || addr + bytes > line + kIndexWindow)) Note("indices at or across a 4 GB multiple without VK_KHR_maintenance5");
        Track(addr, addr + bytes), NoteRead(addr, addr + bytes);
        const VkIndexType type = index_size == 4 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
        auto bind_index = [&] {
            RenderProbe::Cmd(RenderProbe::Index);
            if (BindIndex2) BindIndex2(cmd, mirror, window, std::min(mirror_size - window, kIndexWindow - 4), type);
            else vkCmdBindIndexBuffer(cmd, mirror, window, type);
        };
        if (dyn.index != window || dyn.index_type != type) dyn.index = window, dyn.index_type = type, bind_index();
        first_index = u32((addr - window) / index_size);
        if (stats.trace) {  // TEMP index probe: what the CPU reads at the index address, then the draw once per way of binding
            const u32 n = op == 0x27 ? b[3] : b[2], cut = raster[7] ? (index_size == 4 ? ~0u : 0xFFFFu) : 0, base = ctx[0x102];
            const u8* at = (const u8*)cpu(addr, bytes);
            u32 lo = ~0u, hi = 0;
            std::string head;
            for (u32 k = 0; at && k < n; ++k) {
                const u32 v = index_size == 4 ? ((const u32*)at)[k] : ((const u16*)at)[k];
                if (k < 6) head += std::format(" {}", v);
                if (!raster[7] || v != cut) lo = std::min(lo, v), hi = std::max(hi, v);
            }
            stats.trace_lines.back() += std::format(" | I {:x} x{} first {} base {} cpu {}{}", addr, index_size, first_index, (int)base,
                                                    lo <= hi ? std::format("[{}, {}] head", lo, hi) : "unreadable", head);
            if (index_probe.buf && index_probes.size() < kProbes) {
                if (caps.path == ShaderPath::Objects) {
                    if (index_probe_vs) {
                        const VkShaderStageFlagBits stages[]{VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_GEOMETRY_BIT, VK_SHADER_STAGE_FRAGMENT_BIT,
                                                             VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT};
                        const VkShaderEXT bound[]{vs.obj, gs.obj, ps.obj, VK_NULL_HANDLE, VK_NULL_HANDLE};
                        const u64 rec = index_probe.addr + index_probes.size() * kIndexProbe;
                        index_probes.push_back({stats.trace_lines.size() - 1, lo <= hi ? lo + base : 1u, lo <= hi ? hi + base : 0u});
                        const VkShaderEXT only[]{index_probe_vs, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
                        BindShaders(cmd, 5, stages, only);
                        vkCmdSetRasterizerDiscardEnable(cmd, VK_TRUE);
                        auto shadow = [&](u32 way, u32 first) {
                            const u64 to = rec + way * 12;
                            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL, 0, 8, &to);
                            vkCmdDrawIndexed(cmd, n, 1, first, (int)base, 0);
                        };
                        vkCmdBindIndexBuffer(cmd, mirror, addr, type), shadow(1, 0);
                        vkCmdBindIndexBuffer(cmd, mirror, window, type), shadow(0, first_index);
                        if (BindIndex2) bind_index(), shadow(2, first_index);  // last, either way: the draw's own binding stays
                        vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
                        BindShaders(cmd, 5, stages, bound);
                        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL, 0, 8, &push);
                    }
                } else if (const VkPipeline probe_pl = IndexProbePipeline(raw_state)) {
                    const u64 rec = index_probe.addr + index_probes.size() * kIndexProbe;
                    index_probes.push_back({stats.trace_lines.size() - 1, lo <= hi ? lo + base : 1u, lo <= hi ? hi + base : 0u});
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, probe_pl);
                    vkCmdSetRasterizerDiscardEnable(cmd, VK_TRUE);
                    auto shadow = [&](u32 way, u32 first) {
                        const u64 to = rec + way * 12;
                        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL, 0, 8, &to);
                        vkCmdDrawIndexed(cmd, n, 1, first, (int)base, 0);
                    };
                    vkCmdBindIndexBuffer(cmd, mirror, addr, type), shadow(1, 0);
                    vkCmdBindIndexBuffer(cmd, mirror, window, type), shadow(0, first_index);
                    if (BindIndex2) bind_index(), shadow(2, first_index);
                    vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.main);
                    dyn.pipeline = pl.main;
                    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL, 0, 8, &push);
                }
            }
        }
    }
    auto issue = [&] {
        RenderProbe::Scope issue_time(RenderProbe::Issue);
        RenderProbe::Cmd(RenderProbe::DrawCall);
        if (op == 0x2D) vkCmdDraw(cmd, rect_vs ? 4 : b[0], inst, 0, 0);
        else vkCmdDrawIndexed(cmd, op == 0x27 ? b[3] : b[2], inst, first_index, (int)ctx[0x102], 0);
    };
    // The mesh query its dump left for this draw begins here, inside the pass; a predicated draw runs on its query's count.
    if (query_next != ~0u) EndQuery(), vkCmdBeginQuery(cmd, query_pool, slot * kQueries + query_next, 0), query_on = std::exchange(query_next, ~0u);
    const auto pred_q = pred ? mesh.find(pred) : mesh.end();
    if (pred_q != mesh.end()) {
        const VkConditionalRenderingBeginInfoEXT ci{VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT, nullptr, cond.buf, (slot * kQueries + pred_q->second) * 4ull};
        BeginConditional(cmd, &ci);
    }
    // Pipeline statistics of every traced draw; samples passed and the texel (which ends the pass) only outside the frame's own
    // occlusion queries: one at a time, begun and ended in one pass.
    const bool probe = stats.trace && probes.size() < kProbes, probe_px = probe && !query_open && query_on == ~0u;
    if (probe) vkCmdBeginQuery(cmd, q_stat, (u32)probes.size(), 0);
    if (probe_px) vkCmdBeginQuery(cmd, q_occ, (u32)probes.size(), VK_QUERY_CONTROL_PRECISE_BIT);
    u32 prof_draw = ~0u;
    if (prof >= 2) {
        std::string ch;
#define PROF_CH(f) if (std::memcmp(&prof_before.f, &dyn.f, sizeof(dyn.f))) ch += #f ",";
        PROF_CH(pipeline) PROF_CH(shaders) PROF_CH(vp) PROF_CH(sc) PROF_CH(raster) PROF_CH(bias) PROF_CH(depth) PROF_CH(stencil) PROF_CH(blend) PROF_CH(eq) PROF_CH(mask)
        PROF_CH(constants) PROF_CH(index)
#undef PROF_CH
        prof_draw = ProfBegin(std::format("draw pass {} vs {:x} ps {:x} n {} inst {} s{} ch {}", prof_pass / 2, (u64)(sh[0x48] | (u64)sh[0x49] << 32) << 8, ps_addr,
                                          op == 0x2D ? b[0] : op == 0x27 ? b[3] : b[2], inst, samples, ch.empty() ? "-" : ch));
    }
    issue();
    if (prof >= 2) ProfEnd(prof_draw);
    // The squaring draw repeats every side effect of the first one: only when depth and stencil stay unwritten.
    if (blend_mode) Note("scaled MIN/MAX blend via fragment interlock");
    else if (color_count && bs.square && (z_write || stencil_eff)) Note("scaled MIN/MAX blend not squared (depth/stencil write)");
    else if (color_count && bs.square) {
        if (caps.path == ShaderPath::Objects || (SetBlendEnable && SetBlendEquation)) {
            changed_n(dyn.blend, bs.square_blend, color_count), changed_n(dyn.eq, bs.square_eq, color_count);
            RenderProbe::Cmd(RenderProbe::Blend, 2);
            SetBlendEnable(cmd, 0, color_count, bs.square_blend);
            SetBlendEquation(cmd, 0, color_count, bs.square_eq);
            issue();
            Note("scaled MIN/MAX blend squared by a second draw");
        } else if (pl.square) {
            RenderProbe::Cmd(RenderProbe::Blend);
            dyn.pipeline = pl.square;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.square);
            if (SetBlendEnable && changed_n(dyn.blend, bs.square_blend, color_count)) SetBlendEnable(cmd, 0, color_count, bs.square_blend);
            if (SetBlendEquation && changed_n(dyn.eq, bs.square_eq, color_count)) SetBlendEquation(cmd, 0, color_count, bs.square_eq);
            issue();
            Note("scaled MIN/MAX blend squared by a second draw");
        } else {
            Note("scaled MIN/MAX square pipeline missing");
        }
    }
    if (pred_q != mesh.end()) EndConditional(cmd);
    // Viewport ∩ scissor: the texels this draw can write.
    const int px0 = std::max(sc.offset.x, (int)std::min(vp.x, vp.x + vp.width)), py0 = std::max(sc.offset.y, (int)std::min(vp.y, vp.y + vp.height));
    const int px1 = std::min((int)(sc.offset.x + sc.extent.width), (int)std::max(vp.x, vp.x + vp.width));
    const int py1 = std::min((int)(sc.offset.y + sc.extent.height), (int)std::max(vp.y, vp.y + vp.height));
    const VkRect2D area = px1 > px0 && py1 > py0 ? VkRect2D{{px0, py0}, {(u32)(px1 - px0), (u32)(py1 - py0)}} : sc;
    if (probe_px) vkCmdEndQuery(cmd, q_occ, (u32)probes.size());
    if (probe) vkCmdEndQuery(cmd, q_stat, (u32)probes.size());
    if (probe_px) ProbeColor(probe_target, area);
    else if (probe) probes.push_back({stats.trace_lines.size() - 1, 0, 0, false});
    if (blend_mode) EndRendering();
    for (u32 k = 0; k < color_count; ++k)  // the blend memo holds this draw's masks up to color_count only
        if (bs.mask[k] && (ctx[0x318 + k * 15 + 5] & 0x1F) == 8) WriteBackTarget(*color[k], ((ctx[0x319 + k * 15] & 0x7FF) + 1) * 8, area);
    WriteBack();
    Count(stats.draws);
}

// DISPATCH_DIRECT: the CS program with COMPUTE_USER_DATA, outside any rendering; consecutive independent dispatches share
// one write-back barrier pair.
void Renderer::Impl::Dispatch(const u32* b, const CpuRead& cpu) {
    const u64 addr = (u64)(sh[0x20C] | (u64)sh[0x20D] << 32) << 8;
    if (stats.trace) stats.trace_lines.push_back(std::format("dispatch {} {} {} threads {} {} {} cs {:x}", b[0], b[1], b[2], sh[0x207], sh[0x208], sh[0x209], addr));
    if (draw_seq++ > stats.stop) return;
    draw_rts.clear(), reads.clear(), twins.clear();
    if (!Identify(Gcn::Stage::Compute, addr, &sh[0x240], cpu, sh[0x212], sh[0x213], false, prep))
        return stores.clear(), Count(stats.skipped), Note("dispatch without CS skipped");
    if (!stats.trace && prep.linear_copy && TryLinearCopyDispatch(b, cpu)) return;
    std::vector<u32> table;
    u64 unused = 0;
    dispatching = true;
    const ShaderRef cs = Shader(Gcn::Stage::Compute, addr, &sh[0x240], cpu, sh[0x212], sh[0x213], unused, false, table, 0, true);
    dispatching = false;
    if (!cs) return stores.clear(), Count(stats.skipped), Note("dispatch without CS skipped");
    const u64 n = table.size() * 4, at = TableAt(n, table_align);
    if (at == ~0ull) return stores.clear(), Count(stats.skipped), Note("draw tables full");
    if (n) std::memcpy(TablePtr() + at, table.data(), n);
    if (std::any_of(draw_rts.begin(), draw_rts.end(), [&](auto& u) { return In(hz.w, u.first) || (u.second && In(hz.r, u.first)); })) FlushStores();
    EndRendering();
    // Buffers the program stores to (format stores): their memory is mapped for the GPU and counts as a GPU write (images over it
    // load it again, the CPU reads it back). Everything before the dispatch finishes first, everything after waits for it.
    const u64 stored = prep.stored;
    if (stored) {
        Barrier(cmd);
        for (u32 k = 0; k < prep.fdesc.size(); ++k) {
            const u32* v = stored >> k & 1 ? prep.fdesc[k] : nullptr;
            if (!v) continue;
            const u64 lo = (u64)(v[1] & 0xFFFF) << 32 | v[0], stride = v[1] >> 16 & 0x3FFF, hi = lo + std::min<u64>(stride ? stride * v[2] : v[2], 1ull << 30);
            if (hi <= lo || hi > mirror_size) continue;
            BindPages(lo, hi);
            (in_worker ? own_writes : gpu_writes).push_back({lo, hi, std::atomic_ref(write_seq).fetch_add(1) + 1, nullptr});
        }
    }
    // Targets stored to through their twin (TargetTexture): the twin holds the target while the program runs (it may load
    // what it does not store), then the target takes it back.
    std::sort(twins.begin(), twins.end()), twins.erase(std::unique(twins.begin(), twins.end()), twins.end());
    auto twin_copies = [&](bool back) {
        if (!back || !stored) Barrier(cmd);
        for (const Target* t : twins) {
            const VkImageCopy ic{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {t->w, t->h, 1}};
            vkCmdCopyImage(cmd, back ? t->storage_copy.img : t->t.img, VK_IMAGE_LAYOUT_GENERAL, back ? t->t.img : t->storage_copy.img,
                           VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
            if (back) Add(hz.w, t->t.img);
        }
        if (!back) Barrier(cmd);
    };
    if (!twins.empty()) twin_copies(false);
    for (auto [img, store] : draw_rts) Add(store ? hz.w : hz.r, img);
    Push push{mirror_addr};
    std::copy_n(&sh[0x240], 16, push.user[0]);
    const VkDescriptorSet sets[2]{set, table_sets[slot]};
    const u32 offsets[2]{(u32)at, 0};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 2, offsets);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
    dyn.push = push, dyn.pushed = true;  // shared with the graphics draws
    if (caps.path == ShaderPath::Objects) {
        const VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
        BindShaders(cmd, 1, &stage, &cs.obj);
    } else {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cs.pl);
    }
    vkCmdDispatch(cmd, b[0], b[1], b[2]);
    if (stored) Barrier(cmd);
    if (!twins.empty()) twin_copies(true);
    pend_stores.insert(pend_stores.end(), stores.begin(), stores.end()), stores.clear();
    pend_reads.insert(pend_reads.end(), reads.begin(), reads.end());
    disp_pending = true;
    Count(stats.dispatches);
}

// ---- Parallel recording, retirement and submission ----

VkCommandBuffer Renderer::Impl::Secondary(u32 rec) {
    Recorder& w = recorders[rec];
    if (w.used[slot] == w.cbs[slot].size()) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, w.pool[slot], VK_COMMAND_BUFFER_LEVEL_SECONDARY, 1};
        vkAllocateCommandBuffers(dev, &ai, &w.cbs[slot].emplace_back());
    }
    const VkCommandBuffer c = w.cbs[slot][w.used[slot]++];
    VkCommandBufferInheritanceInfo ii{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, &ii};
    vkBeginCommandBuffer(c, &bi);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
    dyn = {}, rendering = false, hz.Reset(), pass_w.clear();
    return c;
}

// Takes the parallel blocks not yet taken, any order: each from its registers, seeing the GPU writes before its cut.
void Renderer::Impl::RecordBlocks(u32 rec) {
    std::unordered_map<const GuestTex*, u64> synced;
    in_worker = true, synced_here = &synced;
    for (size_t k; (k = next_block++) < blocks.size();) {
        Block& b = blocks[k];
        if (b.serial) continue;
        const auto t = std::chrono::steady_clock::now();
        cmd = Secondary(rec), LoadRegs(block_regs[b.regs]), horizon = b.horizon, synced.clear(), block_now = &b, block_discovered = false;
        if (temp_label_begin) TempLabel(cmd, std::format("block {} draws {}", k, b.draws).c_str());
        TempCheck(cmd, 2, k);
        if (b.copy.second) CopyQueries(b.copy);
        Walk(chain_now, b.from, b.to, [&](u32 op, const u32* p, size_t si, size_t i) {
            if (op == 0x27 || op == 0x2D || op == 0x35) {
                if (!std::binary_search(skipped_draws.begin(), skipped_draws.end(), Pos{si, i}))
                    walk_seg = si, walk_pos = i, Draw(op, p, *cpu_now);
            } else if (op == 0x46 && MeshDump(p)) MeshQuery(p, true);
        });
        // A pass still open ends without its barrier: Submit orders it against the next block.
        if (rendering) b.open = true, b.tail.depth = bound_depth, std::copy(bound_color, bound_color + 8, b.tail.color);
        EndRendering(false);
        SaveHazards(b);
        block_now = nullptr;
        TempLabelEnd(cmd);
        vkEndCommandBuffer(cmd);
        b.cb = cmd;
        b.us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
        if (RenderProbe::current) {
            auto& p = *RenderProbe::current;
            ++p.blocks, p.block_ns += u64(b.us * 1000);
#ifdef _WIN32
            const auto cpu = GetCurrentProcessorNumber();
#else
            const auto cpu = static_cast<unsigned>(sched_getcpu());
#endif
            if (cpu < 64) p.cpu_mask |= 1ull << cpu;
        }
        if (!own_writes.empty()) b.writes = std::move(own_writes), own_writes.clear();
    }
    AddCounts();
    in_worker = false, synced_here = nullptr, horizon = ~size_t(0);
}

void Renderer::Impl::Retire() {
    if (!flight.busy) {
        DrainRetiredElements();
        return;
    }
    const auto temp_t0 = DiagNow();
    temp_call = "fence", TempLost("fence", vkWaitForFences(dev, 1, &flight.fence, VK_TRUE, ~0ull)), temp_call = nullptr;
    GOW3_DIAG temp_fence_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - temp_t0).count();
    vkResetFences(dev, 1, &flight.fence);
    flight.busy = false;
    DrainRetiredElements();
    query_counts.assign(flight.queries, 0);  // what the next Submit writes into its query blocks
    if (flight.queries)
        vkGetQueryPoolResults(dev, query_pool, flight.slot * kQueries, flight.queries, flight.queries * 8, query_counts.data(), 8, VK_QUERY_RESULT_64_BIT);
    // The retired frame's exploded vertices and their loads {pc, address / 4, value, who wrote the address (1 this frame's
    // GPU, 2 the next frame's GPU, 4 this frame's mirror upload, 8 the next frame's), the bytes this frame's upload and
    // the next frame's put there}.
    const auto staged = [](const std::vector<Pending>& v, const std::vector<Buffer>& st, u64 a, u32& out) {  // the last upload of a
        for (size_t i = v.size(); i--;)
            if (a - v[i].addr < v[i].size && v[i].chunk < st.size()) return std::memcpy(&out, st[v[i].chunk].ptr + v[i].src + (a - v[i].addr), 4), true;
        return false;
    };
    const auto gpu_wrote = [](const std::vector<VkBufferCopy>& v, u64 a) { return std::any_of(v.begin(), v.end(), [&](const VkBufferCopy& c) { return a - c.srcOffset < c.size; }); };
    for (u32 k = 0, s = flight.slot; boom_on && k < boom_used[s]; ++k)
        if (u32* b = (u32*)(boom_buf.ptr + ((u64)s * kBoomDraws + k) * kBoomRec); b[0]) {
            const float* f = (const float*)(b + 4);
            std::vector<u32> loads;
            for (u32 j = 0; j < std::min(b[2], kBoomLoads); ++j) {
                const u64 a = (u64)b[8 + kBoomLoads + j] << 2;
                u32 mine = 0, next = 0;
                const bool by_mine = staged(boom_up[s], flight.stage, a, mine), by_next = staged(boom_up[slot], stage, a, next);
                loads.insert(loads.end(), {b[8 + j], b[8 + kBoomLoads + j], b[8 + 2 * kBoomLoads + j],
                                           u32(gpu_wrote(flight.back, a) | gpu_wrote(back, a) << 1 | by_mine << 2 | by_next << 3), mine, next});
            }
            const BoomDraw& m = boom_meta[s][k];
            std::string bank = " user";  // the vertex stage's user data, then the constant bank's nonzero dwords (index=value)
            for (u32 v : m.user) bank += std::format(" {:x}", v);
            bank += "; consts";
            for (u32 j = 0; j < kBoomConsts; ++j)
                if (const u32 v = b[8 + 3 * kBoomLoads + j]) bank += std::format(" {}={:x}", j, v);
            stats.boom.push_back({m.vs ^ m.ps * 0x9E3779B97F4A7C15ull ^ (u64)m.vertices << 1,
                                  std::format("Exploded vertices: frame {} vs {:x} ps {:x}, {} vertices drawn: {} exploded, first #{} {}({:g} {:g} {:g} {:g}), {} loads{}",
                                              flight.frame, m.vs, m.ps, m.vertices, b[0], b[1], b[3] ? std::format("param {} ", b[3] - 1) : "", f[0], f[1], f[2], f[3], b[2], bank),
                                  std::move(loads)});
            std::memset(b, 0, kBoomRec);
        }
    for (const VkBufferCopy& c : flight.back) {
        const u8* p = flight.rb.ptr + c.dstOffset;
        // The mirror already holds these bytes: the same guest bytes offered again are not uploaded.
        for (u64 o = 0; o < c.size;) {
            const u64 n = std::min<u64>(0x1000 - ((c.srcOffset + o) & 0xFFF), c.size - o);
            WrittenSlot(c.srcOffset + o, n) = readback(c.srcOffset + o, p + o, n);
            InvalidateBlocks(c.srcOffset + o);
            o += n;
        }
    }
    flight.back.clear();
    if (u64 q[6]; temp_qp && vkGetQueryPoolResults(dev, temp_qp, flight.slot * 6, 6, sizeof(q), q, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
        for (int k = 0; k < 4; ++k) stats.temp_ms[8 + k] = (q[k + 1 + (k > 0)] - q[k + (k > 0)]) * tick_ms;
    if (prof) {
        std::vector<u64> ts(3 + flight.prof_n);
        vkGetQueryPoolResults(dev, prof_pool, flight.prof_base, (u32)ts.size(), ts.size() * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT);
        std::fprintf(stderr, "PROF gpu copy %.3f ms render %.3f ms\n", (ts[1] - ts[0]) * tick_ms, (ts[2] - ts[1]) * tick_ms);
        static u32 retired = 0;
        if (++retired % 20 == 0)
            for (u32 i = 0; i < flight.prof_n; i += 2)
                std::fprintf(stderr, "PROF %.4f ms %s\n", (ts[4 + i] - ts[3 + i]) * tick_ms, flight.prof_label[i / 2].c_str());
    }
}

bool Renderer::Retired() {
    if (d->flight.busy && vkGetFenceStatus(d->dev, d->flight.fence) == VK_SUCCESS)
        d->boom_up[d->slot].clear(), d->Retire();  // the next frame's uploads (BOOM's "next"): not staged yet
    return !d->flight.busy;
}

void Renderer::Submit(std::span<const Segment> chain, const CpuRead& cpu, std::span<const std::pair<u64, u64>> deferred,
                      const std::function<void()>& staged) {
    Impl& r = *d;
    r.skip_motion_blur = Gow3::Extras::Enabled(Gow3::Extras::Toggle::RemoveMotionBlur);
    r.skip_dof_spartan = Gow3::Extras::Enabled(Gow3::Extras::Toggle::RemoveDofSpartan);
    r.mb_seen_blur = false;
    r.skipped_draws.clear();
    r.orb_hash_cache.clear();
    // The deferred spans out to whole pages (the game's arenas are 128-byte aligned), merged where they meet, as far as the
    // recording threads' bitmaps reach (any other span is staged whole after recording).
    std::vector<std::pair<u64, u64>> spans;
    for (auto [lo, hi] : deferred) spans.push_back({lo & ~0xFFFull, (hi + 0xFFF) & ~0xFFFull});
    std::sort(spans.begin(), spans.end());
    r.deferred_n = 0;
    for (u64 word = 0; auto [lo, hi] : spans) {
        Impl::Deferred* s = r.deferred_n && lo <= r.deferred[r.deferred_n - 1].hi ? &r.deferred[r.deferred_n - 1] : nullptr;
        const Impl::Deferred next = s ? Impl::Deferred{s->lo, std::max(s->hi, hi), s->word} : Impl::Deferred{lo, hi, word};
        const u64 end = next.word + ((next.hi - next.lo + (64 << 9) - 1) >> 15);
        if (end > Impl::kDeferredWords || (!s && r.deferred_n == std::size(r.deferred))) continue;
        (s ? *s : r.deferred[r.deferred_n++]) = next, word = end;
    }
    r.marked_lo = r.deferred_n ? r.deferred[0].lo : 0, r.marked_hi = r.deferred_n ? r.deferred[r.deferred_n - 1].hi : 0;
    const auto t_begin = DiagNow();  // the clocks of this function are diagnostics: none is read outside a debug run
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkResetCommandBuffer(r.primary, 0);
    vkBeginCommandBuffer(r.primary, &bi);
    r.cmd = r.primary, r.tables_used = 0, r.draw_seq = 0, r.dyn = {}, r.lin_syncs.clear(), r.stage_sends.clear(), Impl::hz.Reset(), Impl::pass_w.clear();
    for (auto& w : r.recorders) vkResetCommandPool(r.dev, w.pool[r.slot], 0), w.used[r.slot] = 0;
    ++r.frame;
    GOW3_DIAG Impl::temp_bars[0] = Impl::temp_bars[1] = 0;
    GOW3_DIAG temp_bind_frame.store(r.frame, std::memory_order_relaxed);
    GOW3_DIAG {  // fault diagnostics: the tool layers' own large buffers, as they come and go
        static size_t seen = 0;
        std::vector<std::string> made;
        {
            std::scoped_lock bl(temp_bind_lock);
            for (; seen < temp_binds.size(); ++seen)
                if (const TempBinding& b = temp_binds[seen]; b.who.find("WarpViz") != std::string::npos)
                    made.push_back(std::format("TEMP layer buffer {} type {} [{:#x}, {:#x}) size {:#x} at frame {} tid {} by '{}'", b.bound ? "bind" : "UNBIND", b.type,
                                               b.base, b.base + b.size, b.size, b.frame, b.tid, b.who));
        }
        for (const auto& s : made) r.Note(s);
    }
    if (r.temp_qp) vkResetQueryPool(r.dev, r.temp_qp, r.slot * 6, 6);  // GPU timestamps: the pool exists only in a debug run
    vkResetQueryPool(r.dev, r.query_pool, r.slot * Impl::kQueries, Impl::kQueries), r.query_blocks.clear();
    r.mesh.clear(), r.mesh_copied = Impl::query_seq = Impl::kRays, Impl::query_next = Impl::query_on = ~0u, Impl::pred = 0;
    auto temp_ts = [&](VkCommandBuffer c, u32 k) { if (r.temp_qp) vkCmdWriteTimestamp2(c, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, r.temp_qp, r.slot * 6 + k); };
    // Guest ranges the CPU wrote since the last frame, and the last frame's GPU writes: processed before eviction/regrow
    // so pending lazy writes are known and materialized rather than dropped.
    r.dirty.clear();
    for (auto& p : r.pending) r.dirty.push_back({p.addr, p.addr + p.size});
    r.dirty.insert(r.dirty.end(), deferred.begin(), deferred.end());  // staged after recording: as written by the CPU
    if (Impl::temp_copylog) std::fprintf(stderr, "COPY frame %u\n", r.frame);
    for (auto& w : r.gpu_writes)  // left in their image: the newest per image and range
        if (w.lazy) std::erase_if(r.lazy_writes, [&](const Impl::GpuWrite& x) { return x.by == w.by && x.lo == w.lo; }), r.lazy_writes.push_back(w);
    r.last_writes.swap(r.gpu_writes), r.gpu_writes.clear(), r.frame_seq = r.write_seq;
    std::sort(r.dirty.begin(), r.dirty.end());
    size_t merged = 0;  // disjoint ranges: Texture() looks only at the last one starting below its end
    for (auto& d : r.dirty)
        if (merged && d.first <= r.dirty[merged - 1].second) r.dirty[merged - 1].second = std::max(r.dirty[merged - 1].second, d.second);
        else r.dirty[merged++] = d;
    r.dirty.resize(merged);
    for (auto [lo, hi] : r.dirty)
        for (u64 p = lo >> 12, end = std::min((hi + 0xFFF) >> 12, Impl::kPieces); p < end; ++p) r.staged_table.get()[p] = r.frame;
    for (const auto& w : r.last_writes)
        if (!w.lazy && !w.by)
            for (u64 p = w.lo >> 12, end = std::min((w.hi + 0xFFF) >> 12, Impl::kPieces); p < end; ++p) r.staged_table.get()[p] = r.frame;
    // After an overflow, the longest age that frees an eighth of the full array (age 1 keeps only what the frame that
    // overflowed used: Rage of Sparta's grab passes overflowed the storage array every ~50 frames, and age 1 re-uploaded
    // the ~240 textures used every few frames at once). Their images go later (Reap). Under memory pressure, textures unused
    // for a few frames go now: a failed allocation skips its texture, upload or copy (stale or garbage images).
    if (const u32 pressed = r.PressureAge(); r.frame % 64 == 0 || r.evict_soon || pressed)
        Impl::Stalled(r.stall_evicts, r.stall_evict_ns, [&] {
            if (!r.evict_soon) return r.EvictTextures(pressed ? pressed : 64);
            const u32 cap = r.evict_binding >= kStorageBinding ? kStorage : kTextures;
            if (r.free_elements[r.evict_binding].size() < cap / 8)
                for (u32 age : {16u, 4u, 1u})
                    if (r.EvictTextures(age), r.free_elements[r.evict_binding].size() >= cap / 8) break;
            if (pressed) r.EvictTextures(pressed);
        }), r.evict_soon = false;
    if (r.frame % (r.Pressed() ? 32 : 128) == 0) Impl::Stalled(r.stall_evicts, r.stall_evict_ns, [&] { r.CompactImageBlocks(); });
    if (r.regrow) Impl::Stalled(r.stall_evicts, r.stall_evict_ns, [&] { r.Regrow(); });
    if (r.stats.trace) r.BeginTraceProbes();
    // Targets whose memory the CPU rewrote, and single-sampled ones a texture's storage write went over: textures again
    // until drawn again (`rendered` ~0u). A base leaves `target_at` with its last drawn target.
    if (!r.target_at.empty() && (!r.dirty.empty() || !r.last_writes.empty())) {
        bool stale = false;
        std::erase_if(r.target_at, [&](u64 base) {
            const auto d = std::upper_bound(r.dirty.begin(), r.dirty.end(), std::pair{base + 1, 0ull});
            const bool cpu = d != r.dirty.begin() && std::prev(d)->second > base;
            if (!cpu && std::none_of(r.last_writes.begin(), r.last_writes.end(), [&](const Impl::GpuWrite& w) { return w.by && w.lo <= base && w.hi > base; }))
                return false;
            bool drawn = false;
            r.TargetsAt(base, [&](Impl::Target& t) {
                if (cpu || t.samples == 1) stale |= std::atomic_ref(t.rendered).exchange(~0u, std::memory_order_relaxed) != ~0u;
                else drawn |= std::atomic_ref(t.rendered).load(std::memory_order_relaxed) != ~0u;
            });
            return !drawn;
        });
        if (stale) std::atomic_ref(r.epoch).fetch_add(1, std::memory_order_release);
    }
    r.Send();
    if (Impl::boom_on) {
        r.boom_snap.clear();
        for (const auto& p : r.pending) r.boom_snap.push_back({p.addr, p.addr + p.size, r.stage[p.chunk].ptr + p.src});
        std::sort(r.boom_snap.begin(), r.boom_snap.end(), [](const auto& x, const auto& y) { return x.lo < y.lo; });
    }
    // Draw tables and texture uploads are collected while recording; mirror upload, table copy and texture uploads run
    // first, in their own command buffer.
    std::map<u32, std::pair<u32, std::string>> unhandled;
    r.walk_chain = chain, r.discovered = false, r.effects.clear(), r.effects_whole = true;
    const auto temp_setup = DiagNow();
    const Impl::Pos end{chain.size(), 0};
    std::chrono::steady_clock::time_point temp_walked{}, temp_recorded{};
    if (r.threads.empty() || r.stats.trace || r.prof) {  // serial: everything in the primary
        vkCmdBindDescriptorSets(r.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.layout, 0, 1, &r.set, 0, nullptr);
        r.Walk(chain, {0, 0}, end, [&](u32 op, const u32* b, size_t si, size_t i) {
            if (Impl::Effect(op)) r.Collect(op, b);
            switch (op) {
            case 0x27:
            case 0x2D:
            case 0x35:
                if (r.ShouldSkipMotionBlur(cpu)) {
                    r.Count(r.stats.skipped);
                    break;
                }
                if (const auto c = r.CopyRange(); c.second) r.CopyQueries(c);
                r.RegisterStencil();
                r.walk_seg = si, r.walk_pos = i, r.Draw(op, b, cpu);
                break;
            case 0x15: r.walk_seg = si, r.walk_pos = i, r.Dispatch(b, cpu); break;
            case 0x46:
                if (const u64 q = r.QueryBlock(b)) {
                    r.Query(q);
                    break;
                }
                if (r.MeshDump(b)) {
                    r.MeshQuery(b, true);
                    break;
                }
                [[fallthrough]];
            default:
                if (r.stats.trace) {  // packets not executed: count, and where memory writers aim
                    auto& [count, dsts] = unhandled[op];
                    ++count;
                    if (dsts.size() < 160 && (op == 0x50 || op == 0x37))
                        dsts += std::format(" {:x}", op == 0x50 ? (u64)b[3] | (u64)(b[4] & 0xFFFF) << 32 : (u64)b[1] | (u64)(b[2] & 0xFFFF) << 32);
                    if (dsts.size() < 400 && op == 0x3C)  // where the waits sit (next draw) and on what
                        dsts += std::format(" [before draw {} {} {:x} fn {} ref {:x} mask {:x}]", r.stats.trace_lines.size(), b[0] >> 4 & 1 ? "mem" : "reg",
                                            (u64)(b[1] & ~3u) | (u64)b[2] << 32, b[0] & 7, b[3], b[4]);
                }
                break;
            }
        });
        if (r.query_open) r.Query(8);
        r.FlushStores(), r.EndRendering(), Impl::Barrier(r.cmd);  // the next frame starts after this one
    } else {
        // Parallel: this walk applies the registers, records itself dispatches, draws to linear targets (the GPU writes
        // that order what follows) and the queries the CPU reads with their draws (both ends in one command buffer), and
        // cuts the draws between them into blocks of about a quarter of a thread's share of the last frame's cost (64
        // draws without one), preferably where the attachments change; the blocks are recorded by every thread, this one
        // included.
        r.blocks.clear(), r.chain_now = chain, r.cpu_now = &cpu;
        const u32 last = (u32)r.recorders.size() - 1;
        const double share = r.last_total > 0 ? r.last_total / (r.recorders.size() * 4) : 64;
        u64 parallel = 0, pass = ~0ull;
        u32 nregs = 0;  // parallel blocks opened
        size_t ci = 0;
        double cost = 0;
        auto end_serial = [&] { r.FlushStores(), r.EndRendering(false), Impl::SaveHazards(r.blocks.back()), r.TempLabelEnd(r.cmd), vkEndCommandBuffer(r.cmd); };
        auto open = [&](Impl::Pos at, bool serial) {
            if (!r.blocks.empty()) {
                r.blocks.back().to = at;
                if (r.blocks.back().serial) end_serial();
            }
            Impl::Block& b = r.blocks.emplace_back();
            b.from = at, b.to = end, b.serial = serial, b.horizon = r.gpu_writes.size();
            if (serial) b.cb = r.cmd = r.Secondary(last), r.TempLabel(r.cmd, "serial"), r.TempCheck(r.cmd, 3, r.blocks.size() - 1);
            else {
                if (b.regs = nregs++; b.regs == r.block_regs.size()) r.block_regs.emplace_back();
                Impl::SaveRegs(r.block_regs[b.regs]);
            }
        };
        auto linear = [&] {  // a written color target with CB tile index 8 (WriteBackTarget)
            for (u32 k = 0; k < 8 && (r.ctx[0x202] >> 4 & 7) == 1; ++k)
                if (r.ctx[0x318 + k * 15] && r.ctx[0x1C5] >> (4 * k) & 0xF && r.ctx[0x08E] >> (4 * k) & 0xF && (r.ctx[0x318 + k * 15 + 5] & 0x1F) == 8)
                    return true;
            return false;
        };
        auto storage_draw = [&] {
            if (!r.has_storage_draw_progs.load(std::memory_order_relaxed)) return false;
            const u64 vs_a = (u64)(r.sh[0x48] | (u64)r.sh[0x49] << 32) << 8;
            const u64 ps_a = (u64)(r.sh[0x08] | (u64)r.sh[0x09] << 32) << 8;
            return r.storage_draw_progs.contains(ps_a ^ (vs_a * 0x9E3779B97F4A7C15ull));
        };
        r.Walk(chain, {0, 0}, end, [&](u32 op, const u32* b, size_t si, size_t i) {
            if (Impl::Effect(op)) return r.Collect(op, b);
            const bool draw = op == 0x27 || op == 0x2D || op == 0x35;
            if (draw && r.ShouldSkipMotionBlur(cpu)) {
                r.skipped_draws.push_back({si, i});
                r.Count(r.stats.skipped);
                return;
            }
            if (draw) r.RegisterStencil();
            const u64 query = op == 0x46 ? r.QueryBlock(b) : 0;
            if (op == 0x46 && r.MeshDump(b)) return r.MeshQuery(b, !r.blocks.empty() && r.blocks.back().serial);
            if (!draw && op != 0x15 && !query) return;
            const auto copy = draw ? r.CopyRange() : std::pair<u32, u32>{};  // a predicated draw's queries, before it
            if (op == 0x15 || query || r.query_open || linear() || storage_draw()) {
                if (r.blocks.empty() || !r.blocks.back().serial) open({si, i}, true);
                if (copy.second) r.CopyQueries(copy);
                r.walk_seg = si, r.walk_pos = i;
                return query ? r.Query(query) : op == 0x15 ? r.Dispatch(b, cpu) : r.Draw(op, b, cpu);
            }
            while (ci + 1 < r.last_cost.size() && r.last_cost[ci + 1].first <= parallel) ++ci;
            const u64 key = (u64)r.ctx[0x318] << 32 | r.ctx[0x014];
            if (const Impl::Block* cur = r.blocks.empty() ? nullptr : &r.blocks.back();
                !cur || cur->serial || copy.second || (cost >= share && key != pass) || cost >= 2 * share)
                open({si, i}, false), r.blocks.back().first = parallel, r.blocks.back().copy = copy, cost = 0;
            cost += r.last_cost.empty() ? 1.0 : r.last_cost[ci].second;
            pass = key, ++r.blocks.back().draws, ++parallel;
        });
        if (r.query_open) r.Query(8);
        if (!r.blocks.empty() && r.blocks.back().serial) end_serial();
        temp_walked = DiagNow();
        const auto regs = std::make_unique<Impl::Regs>();  // the walk's registers carry over to the next frame
        Impl::SaveRegs(*regs);
        r.next_block = 0;
        Gow3::CpuProbe::BlocksBegin();  // TEMP
        r.Parallel([&](u32 k) { r.RecordBlocks(k); });
        Gow3::CpuProbe::BlocksEnd();
        temp_recorded = DiagNow();
        Impl::LoadRegs(*regs);
        r.last_cost.clear(), r.last_total = 0;
        for (auto& b : r.blocks)
            if (!b.serial && b.draws) r.last_cost.push_back({b.first, b.us / b.draws}), r.last_total += b.us;
        r.cmd = r.primary;
        // As recorded serially: a block continuing the pass the block before left open gets the barrier between draws of
        // one pass, anything else the barrier of a pass end (blocks that recorded nothing pass it on).
        std::vector<VkCommandBuffer> cbs;
        auto execute = [&] { cbs.empty() ? void() : vkCmdExecuteCommands(r.primary, (u32)cbs.size(), cbs.data()), cbs.clear(); };
        const Impl::Block* left_open = nullptr;
        // Lazy pass ends: targets written / sampled since the primary's last full barrier, blocks included; written before a
        // barrier that left the vertex stages out, since the last one that did not.
        std::vector<VkImage> pend_w, pend_r, pend_wv;
        // Mid-frame uploads the blocks left (Resync): once per image and version, before the first block that needs it.
        std::unordered_map<Impl::GuestTex*, u64> uploaded;
        std::vector<Impl::GuestTex*> up;
        std::unordered_set<Impl::GuestTex*> vs_up;  // uploaded ones a vertex shader samples: their upload ends with a full barrier
        std::unordered_set<u64> materialized;       // lazy writes written back (by seq)
        for (auto& b : r.blocks) vs_up.insert(b.resync_vs.begin(), b.resync_vs.end()), b.resync_vs.clear();
        for (auto& b : r.blocks) {
            for (auto [g, need] : b.resync)
                if (u64& have = uploaded[g]; have < need) have = need, up.push_back(g);
            if (!up.empty()) {
                execute(), left_open = nullptr;
                for (VkImage x : pend_w) Impl::Add(pend_wv, x);
                pend_w.clear(), pend_r.clear();
                std::scoped_lock res(r.res_lock);
                r.TempLabel(r.primary, "resync");
                Impl::Barrier(r.primary, true);    // the blocks before may still sample them
                for (auto& [w, learn] : b.lazy)
                    if (materialized.insert(w.seq).second) r.Materialize(r.primary, w, learn);
                r.Upload(r.primary, up, nullptr, 0, std::none_of(up.begin(), up.end(), [&](auto* g) { return vs_up.contains(g); }));
                r.TempLabelEnd(r.primary);
            }
            if (!b.cb) continue;
            // What the block did before its first full barrier against what is pending: a full barrier for a written
            // target sampled, the attachment barrier for one written again or overwritten after shader reads.
            // Its vertex shaders sampling a target written before: a barrier for every stage.
            const bool vs = Impl::Overlap(b.vs_in, pend_w) || Impl::Overlap(b.vs_in, pend_wv);
            const bool raw = vs || (b.all && (!pend_w.empty() || !pend_r.empty())) || Impl::Overlap(b.in_r, pend_w);
            const bool war = Impl::Overlap(b.in_w, pend_r);
            if (raw || war || Impl::Overlap(b.in_w, pend_w)) {
                execute();
                if (raw) {
                    const bool frag = !vs;
                    Impl::Barrier(r.primary, frag);
                    if (frag)
                        for (VkImage x : pend_w) Impl::Add(pend_wv, x);
                    else pend_wv.clear();
                    pend_w.clear(), pend_r.clear();
                } else {
                    Impl::PassBarrier(r.primary, !war && left_open && b.clean && b.head == left_open->tail ? VK_DEPENDENCY_BY_REGION_BIT : 0, war);
                    if (war)
                        for (VkImage x : b.in_w) std::erase(pend_r, x);
                }
            }
            if (b.wide) pend_wv.clear();
            else if (b.fenced)
                for (VkImage x : pend_w) Impl::Add(pend_wv, x);
            if (b.fenced) pend_w.clear(), pend_r.clear();
            for (VkImage x : b.out_wv) Impl::Add(pend_wv, x);
            for (VkImage x : b.out_w) Impl::Add(pend_w, x);
            for (VkImage x : b.out_r) Impl::Add(pend_r, x);
            if (b.serial || b.started) left_open = nullptr;
            cbs.push_back(b.cb);
            if (b.open) left_open = &b;
        }
        execute();
        Impl::Barrier(r.primary);  // the next frame starts after this one (its prologue fences only its own uploads)
        for (auto& b : r.blocks)
            if (!b.writes.empty()) r.gpu_writes.insert(r.gpu_writes.end(), b.writes.begin(), b.writes.end());
    }
    r.FlushLinearSyncs();
    for (auto& [op, u] : unhandled)  // once per opcode (first frame that has it)
        if (const std::string tag = std::format("PM4 op {:#x} ", op);
            std::none_of(r.stats.notes.begin(), r.stats.notes.end(), [&](const std::string& n) { return n.starts_with(tag); }))
            r.stats.notes.push_back(std::format("{}not executed x{}{}", tag, u.first, u.second));
    // The deferred spans, now that every Track of the frame is done: the blocks its work read, hashed and staged on the
    // recording threads (the copy engine sends them below). The caller may let them change from then on.
    if (!deferred.empty()) {
        static std::vector<u64> map(Impl::kDeferredWords);  // the threads' bitmaps merged (GPU thread)
        {
            std::scoped_lock l(Impl::deferred_lock);
            for (u64* m : Impl::deferred_maps)
                for (u64 k = 0; k < Impl::kDeferredWords; ++k) map[k] |= m[k], m[k] = 0;
        }
        std::vector<WriteRange> ranges;
        for (const Impl::Deferred& s : std::span(r.deferred, r.deferred_n))
            ranges.push_back({s.lo, reinterpret_cast<const void*>(s.lo), s.hi - s.lo, false, true, map.data() + s.word});
        for (auto [lo, hi] : deferred) {
            const auto* s = std::find_if(r.deferred, r.deferred + r.deferred_n, [&](const Impl::Deferred& s) { return s.lo <= lo && hi <= s.hi; });
            if (s == r.deferred + r.deferred_n) {
                ranges.push_back({lo, reinterpret_cast<const void*>(lo), hi - lo, false, true});
                continue;
            }
            u64 read = 0;  // the span's blocks some work read
            for (u64 k = s->word * 64 + ((lo - s->lo) >> 9), end = s->word * 64 + ((hi - s->lo + 511) >> 9), n; k < end; k += n)
                n = std::min(64 - (k & 63), end - k), read += std::popcount(map[k >> 6] >> (k & 63) & (n == 64 ? ~0ull : (1ull << n) - 1));
            RenderProbe::ExcludeArena(lo, hi - std::min(read * Gpu::PieceHash::kMemoryBlock, hi - lo));  // the bytes no work read
        }
        r.deferred_n = 0, r.marked_hi = 0;
        Write(ranges);
        std::fill(map.begin(), map.end(), 0);
    }
    if (staged) staged();
    temp_ts(r.primary, 4);
    const u64 rb_total = r.RecordReadback(bi);
    const bool rb_async = r.rbq && rb_total;
    if (r.prof) vkCmdWriteTimestamp2(r.primary, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, r.prof_pool, r.prof_base + 2);
    temp_ts(r.primary, 5), r.TempCheck(r.primary, 7, 0);
    vkEndCommandBuffer(r.primary);
    // What recording staged (the deferred spans, newly tracked pages) goes to VRAM on the copy engine too, before the tables.
    const bool need_xfer = r.use_stage_dev || !r.direct_tables;
    if (need_xfer) {
        vkResetCommandBuffer(r.xfer_cmd, 0);
        vkBeginCommandBuffer(r.xfer_cmd, &bi);
        temp_ts(r.xfer_cmd, 0), r.TempLabel(r.xfer_cmd, "table copy"), r.TempCheck(r.xfer_cmd, 6, 0);
        r.RecordSend(r.xfer_cmd);
    }
    // The mirror upload once the frame in flight is done with the mirror (the game rewrites its arenas while that frame
    // still reads them), then the texture uploads, which read it (their tiling info goes with the tables, on the second queue).
    vkResetCommandBuffer(r.copy_cmd, 0);
    vkBeginCommandBuffer(r.copy_cmd, &bi);
    if (!need_xfer) temp_ts(r.copy_cmd, 0), temp_ts(r.copy_cmd, 1);
    temp_ts(r.copy_cmd, 2), r.TempLabel(r.copy_cmd, "texture uploads");
    GOW3_DIAG r.stats.temp_ms[13] = (double)r.uploads.size();
    r.TempCheck(r.copy_cmd, 4, r.uploads.size());
    if (r.prof)
        vkCmdResetQueryPool(r.copy_cmd, r.prof_pool, r.prof_base, Impl::kProfQueries),
            vkCmdWriteTimestamp2(r.copy_cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, r.prof_pool, r.prof_base);
    const u64 temp_up = r.stats.uploaded;
    if (!r.pending.empty()) {
        Impl::StageBarrier(r.copy_cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        r.Flush(r.copy_cmd);
        Impl::StageBarrier(r.copy_cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                           VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    } else  // the frame before's readback copies (after its last barrier) still read the mirror this frame writes
        Impl::StageBarrier(r.copy_cmd, VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
    GOW3_DIAG r.stats.temp_ms[12] = (r.stats.uploaded - temp_up) / 1048576.0;
    for (auto& [w, learn] : r.prologue_lazy) {  // earlier writes left in their image that these uploads read
        if (std::erase_if(r.lazy_writes, [&](const Impl::GpuWrite& x) { return x.by == w.by && x.lo == w.lo; }))
            r.Materialize(r.copy_cmd, w, learn);
        else if (learn && w.by && w.by->surf)
            w.by->surf->foreign = true;
    }
    r.prologue_lazy.clear();
    for (auto [img, aspect] : r.fresh) {
        Impl::ToGeneral(r.copy_cmd, img, aspect);
        const VkImageSubresourceRange range{aspect, 0, 1, 0, 1};
        const VkClearDepthStencilValue z{1.0f, 0};
        const VkClearColorValue c{};
        if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) vkCmdClearDepthStencilImage(r.copy_cmd, img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &range);
        else vkCmdClearColorImage(r.copy_cmd, img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &range);
    }
    r.Upload(r.copy_cmd, r.uploads, &r.scratch, r.scratch_reserve);
    if (!r.fresh.empty())  // Upload may record nothing (tables full): its closing barrier is no fence for the clears
        Impl::StageBarrier(r.copy_cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT),
            r.fresh.clear();
    if (r.prof) vkCmdWriteTimestamp2(r.copy_cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, r.prof_pool, r.prof_base + 1);
    r.TempLabelEnd(r.copy_cmd), temp_ts(r.copy_cmd, 3);
    vkEndCommandBuffer(r.copy_cmd);
    if (need_xfer) {
        if (!r.direct_tables && r.tables_used) {
            const VkBufferCopy tc{0, 0, std::min(r.tables_used, r.table_bytes)};  // TableAt counts the pieces it refused too
            vkCmdCopyBuffer(r.xfer_cmd, r.tables_host.buf, r.tables.buf, 1, &tc);
        }
        r.TempLabelEnd(r.xfer_cmd), temp_ts(r.xfer_cmd, 1);
        vkEndCommandBuffer(r.xfer_cmd);
    }
    const auto temp_tail = DiagNow();
    if (need_xfer) {
        const VkSubmitInfo xi{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &r.xfer_cmd, 1, &r.copied};
        std::scoped_lock lock(*r.xfer_lock);
        temp_call = "xfer submit", r.TempLost("xfer submit", vkQueueSubmit(r.xfer, 1, &xi, VK_NULL_HANDLE)), temp_call = nullptr;
    }
    // This frame goes to the GPU before the wait for the frame before (the frame in flight, which records nothing more),
    // so the GPU goes on to it without a gap. The frame's fence: the readback's, which waits for the frame.
    const auto t_recorded = DiagNow();
    r.temp_fence_ms = 0;
    const auto temp_r0 = DiagNow();
    const VkCommandBuffer both[]{r.copy_cmd, r.primary};
    const bool has_binds = !r.pending_binds.empty();
    VkPipelineStageFlags wait_stages[2]{};
    VkSemaphore waits[2]{};
    u32 wait_count = 0;
    if (need_xfer) {
        waits[wait_count] = r.copied;
        wait_stages[wait_count++] =
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    if (has_binds) {
        waits[wait_count] = r.bound_sem;
        wait_stages[wait_count++] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, wait_count, waits, wait_stages, 2, both, rb_async ? 1u : 0u, &r.rendered};
    {
        std::scoped_lock lock(*r.queue_lock);
        if (has_binds) {
            const VkSparseBufferMemoryBindInfo bi{r.mirror, (u32)r.pending_binds.size(), r.pending_binds.data()};
            VkBindSparseInfo bsi{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
            bsi.bufferBindCount = 1, bsi.pBufferBinds = &bi;
            bsi.signalSemaphoreCount = 1, bsi.pSignalSemaphores = &r.bound_sem;
            Impl::Stalled(r.stall_binds, r.stall_bind_ns, [&] {
                temp_call = "bind sparse", r.TempLost("bind sparse", vkQueueBindSparse(r.queue, 1, &bsi, VK_NULL_HANDLE)), temp_call = nullptr;  // TEMP fault
            });
            r.pending_binds.clear();
        }
        temp_call = "submit", r.TempLost("submit", vkQueueSubmit(r.queue, 1, &si, rb_async ? VK_NULL_HANDLE : r.fence)), temp_call = nullptr;
    }
    if (rb_async) {
        const VkPipelineStageFlags rb_wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        const VkSubmitInfo ri{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 1, &r.rendered, &rb_wait_stage, 1, &r.rb_cmd};
        temp_call = "readback submit", r.TempLost("readback submit", vkQueueSubmit(r.rbq, 1, &ri, r.fence)), temp_call = nullptr;
    }
    const auto temp_submitted = DiagNow();
    r.Retire();
    r.Reap();
    const auto t_retired = DiagNow();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const double retire_ms = ms(t_recorded, temp_r0) + ms(temp_submitted, t_retired);
    GOW3_DIAG {
        if (temp_walked.time_since_epoch().count()) {
            const double phases[8]{ms(t_begin, temp_setup), ms(temp_setup, temp_walked), ms(temp_walked, temp_recorded), ms(temp_recorded, temp_tail),
                                   ms(temp_tail, t_recorded), r.temp_fence_ms, retire_ms - r.temp_fence_ms, ms(temp_r0, temp_submitted)};
            std::copy_n(phases, 8, r.stats.temp_ms);
        }
        r.stats.temp_barriers[0] = Impl::temp_bars[0], r.stats.temp_barriers[1] = Impl::temp_bars[1];
    }
    if (Impl::temp_barsites && r.frame % 60 == 0) {
        std::scoped_lock l(Impl::temp_site_lock);
        for (auto [line, n] : Impl::temp_sites) std::fprintf(stderr, "BARSITE line %u: %.1f/frame\n", line, n / 60.0);
        Impl::temp_sites.clear();
    }
    r.boom_used[r.slot] = std::min(r.boom_next.exchange(0, std::memory_order_relaxed), Impl::kBoomDraws);
    if (Impl::boom_on) {  // this frame's reads, skipped ranges and torn snapshots (its recording threads are done), for FrameDump
        std::scoped_lock lock(r.tear_lock, Impl::read_lock);
        r.stats.frame = r.frame;
        for (auto* v : Impl::read_lists) r.stats.reads.insert(r.stats.reads.end(), v->begin(), v->end()), v->clear();
        for (auto* v : Impl::gpu_read_lists) r.stats.gpu_reads.insert(r.stats.gpu_reads.end(), v->begin(), v->end()), v->clear();  // TEMP coverage
        r.stats.tears.insert(r.stats.tears.end(), r.tears.begin(), r.tears.end()), r.tears.clear();
        r.stats.skips.insert(r.stats.skips.end(), r.skips.begin(), r.skips.end()), r.skips.clear();
    }
    constexpr auto relaxed = std::memory_order_relaxed;
    r.stats.compiles = r.stall_compiles.exchange(0, relaxed), r.stats.compile_waits = r.stall_waits.exchange(0, relaxed);
    r.stats.binds = r.stall_binds.exchange(0, relaxed), r.stats.evicts = r.stall_evicts.exchange(0, relaxed);
    r.stats.compile_wait_ms = float(r.stall_wait_ns.exchange(0, relaxed) * 1e-6), r.stats.bind_ms = float(r.stall_bind_ns.exchange(0, relaxed) * 1e-6);
    r.stats.evict_ms = float(r.stall_evict_ns.exchange(0, relaxed) * 1e-6);
    r.stats.reaped = r.reaped.exchange(0, relaxed), r.stats.reap_max_ms = float(r.reap_max_ns.exchange(0, relaxed) * 1e-6);
    {
        std::scoped_lock lock(r.compiled_lock);
        r.stats.compiled.clear(), r.stats.compiled.swap(r.compiled);
    }
    r.NoteFrameBuffers();
    r.SwapFrame();
    r.flight.busy = true, r.flight.queries = (u32)r.query_blocks.size(), r.flight.frame = r.frame;
    r.stage_at = 0, r.stage_used = 0;  // the retired frame's staging chunks
    r.TrimFrameBuffers();
    GOW3_DIAG r.stats.record_ms = ms(t_begin, t_recorded), r.stats.gpu_ms = retire_ms;
    if (r.stats.trace) r.Retire();  // probes below, and the dump reads the targets
    // This frame's query blocks get the last retired frame's counts, in order: the CPU reads them two frames later, past
    // the label after this Submit (this frame's own counts arrive only with the next one, after that read may have run).
    for (size_t k = 0; r.readback && k < std::min(r.query_blocks.size(), r.query_counts.size()); ++k) {
        const u64 block[16]{0, r.query_counts[k]};  // 8 {begin, end} pairs: one count
        r.readback(r.query_blocks[k], block, sizeof(block));
    }
    if (!r.probes.empty() || !r.index_probes.empty()) r.CollectTraceProbes();
    GOW3_DIAG {
        std::scoped_lock lock(r.note_lock);
        r.stats.report.insert(r.stats.report.end(), std::make_move_iterator(r.report.begin()), std::make_move_iterator(r.report.end()));
        r.report.clear();
    }
}

// ---- Diagnostics: exploded-vertex capture & traced probes ----

// Debug (boom_on), per draw, on the vertex stage's user data: (a) each 64-bit pointer whose 16 bytes differ between the frame's staged
// snapshot and guest memory now (the game wrote after the snapshot); (b) each V# of 0x40-byte records (a bone palette: three rows and
// a translation of 3 floats every 16 bytes): the records that are not finite, have a row far from unit length or a huge translation,
// and, with two such V#s (the velocity pass reads this and the previous frame's palette), the records whose translations are far
// apart. Logged when a program's counts change and once per frame while it has offenders, with the first offender.
void Renderer::Impl::BoomUser(const u32* user, const CpuRead& cpu) {
    struct Last {
        u32 bad[2]{}, apart = 0;
        u64 frame = 0;
    };
    static std::mutex lock;
    static std::map<std::pair<u64, u32>, Last> last;
    static std::atomic<u32> late_logged{0};
    for (u32 i = 0; i < 16; i += 2) {
        const u64 p = user[i] | (u64)user[i + 1] << 32;
        if (p < 0x10000 || (p & 3)) continue;
        const u32* now = static_cast<const u32*>(cpu(p, 16));
        if (!now) continue;
        const auto at = std::upper_bound(boom_snap.begin(), boom_snap.end(), p, [](u64 a, const BoomSnap& s) { return a < s.lo; });
        if (at == boom_snap.begin() || p + 16 > std::prev(at)->hi) continue;
        const u32* staged = reinterpret_cast<const u32*>(std::prev(at)->p + (p - std::prev(at)->lo));
        if (std::memcmp(staged, now, 16) && late_logged.fetch_add(1) < 64)
            Note(std::format("BOOM user: frame {} vs {:x} s{}:s{} -> {:x} WRITTEN AFTER THE SNAPSHOT: now {:08x} {:08x} {:08x} {:08x} staged {:08x} {:08x} {:08x} {:08x}", frame,
                             read_prog, i, i + 1, p, now[0], now[1], now[2], now[3], staged[0], staged[1], staged[2], staged[3]));
    }
    const float* pal[2]{};
    u32 count[2]{}, bad[2]{}, first_bad[2]{~0u, ~0u}, regs[2]{}, n_pal = 0;
    u64 bases[2]{};
    for (u32 g = 4; g < 16 && n_pal < 2; g += 4) {
        const u32 stride = user[g + 1] >> 16 & 0x3FFF, n = std::min(user[g + 2], 512u);
        if (stride != 0x40 || n < 2) continue;
        const u64 base = user[g] | (u64)(user[g + 1] & 0xFFFF) << 32;
        const float* m = static_cast<const float*>(cpu(base, n * 0x40ull));
        if (!m) continue;
        const u32 k = n_pal++;
        pal[k] = m, count[k] = n, regs[k] = g, bases[k] = base;
        for (u32 r = 0; r < n; ++r) {
            const float* f = m + r * 16;
            bool ok = true;
            for (u32 row = 0; row < 3; ++row) {
                const float len2 = f[row * 4] * f[row * 4] + f[row * 4 + 1] * f[row * 4 + 1] + f[row * 4 + 2] * f[row * 4 + 2];
                ok &= len2 > 1e-4f && len2 < 1e4f;  // a row of a bone matrix is not near zero (scaled bones are 0.08 long); NaN fails both
            }
            ok &= std::fabs(f[12]) < 1e5f && std::fabs(f[13]) < 1e5f && std::fabs(f[14]) < 1e5f;
            if (!ok && !bad[k]++) first_bad[k] = r;
        }
    }
    if (!n_pal) return;
    u32 apart = 0, first_apart = ~0u;
    float apart_d = 0;
    if (n_pal == 2)
        for (u32 r = 0; r < std::min(count[0], count[1]); ++r) {
            const float* a = pal[0] + r * 16 + 12;
            const float* b = pal[1] + r * 16 + 12;
            const float d = std::fabs(a[0] - b[0]) + std::fabs(a[1] - b[1]) + std::fabs(a[2] - b[2]);
            if (!(d < 50.f) && !apart++) first_apart = r, apart_d = d;
        }
    bool log;
    {
        std::scoped_lock l(lock);
        Last& e = last[{read_prog, user[4]}];
        log = e.bad[0] != bad[0] || e.bad[1] != bad[1] || e.apart != apart || ((bad[0] || bad[1] || apart) && e.frame != frame);  // changes, and each frame with offenders
        e.bad[0] = bad[0], e.bad[1] = bad[1], e.apart = apart, e.frame = frame;
    }
    if (!log) return;
    std::string s = std::format("BOOM bones: frame {} vs {:x}: {} palettes", frame, read_prog, n_pal);
    for (u32 k = 0; k < n_pal; ++k) {
        s += std::format("; s{} V# {:x} x{}: {} bad", regs[k], bases[k], count[k], bad[k]);
        if (bad[k]) {
            const float* f = pal[k] + first_bad[k] * 16;
            s += std::format(" (first #{}:", first_bad[k]);
            for (u32 q = 0; q < 16; ++q) s += std::format(" {:g}", f[q]);
            s += ")";
        }
    }
    if (apart) s += std::format("; {} records far apart between the palettes, first #{} by {:g}", apart, first_apart, apart_d);
    Note(s);
}

void Renderer::Impl::BeginTraceProbes() {
    Retire();
    if (!q_stat) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_PIPELINE_STATISTICS, kProbes, kStats};
        vkCreateQueryPool(dev, &qi, nullptr, &q_stat);
        qi.queryType = VK_QUERY_TYPE_OCCLUSION, qi.pipelineStatistics = 0;
        vkCreateQueryPool(dev, &qi, nullptr, &q_occ);
        probe_buf = MakeBuffer(kProbes * kProbeBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        vertex_probe = MakeBuffer(kProbes * kVertexProbe, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
        // TEMP index probe: the record's address comes in the first push constant (the mirror's, in a guest program).
        index_probe = MakeBuffer(kProbes * kIndexProbe, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
        std::string err;
        const auto spirv = compiler.Spirv(R"(#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
layout(buffer_reference, scalar) buffer Rec { uint n, lo, hi; };
layout(push_constant, scalar) uniform Push { Rec rec; };
void main() {
    atomicAdd(rec.n, 1u);
    atomicMin(rec.lo, uint(gl_VertexIndex));
    atomicMax(rec.hi, uint(gl_VertexIndex));
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
)", shaderc_vertex_shader, err);
        if (caps.path == ShaderPath::Objects) {
            index_probe_vs = spirv.empty() ? VK_NULL_HANDLE : Create(spirv, VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT);
            if (!index_probe_vs) Report("Index probe: no vertex shader " + err.substr(0, 300));
        } else {
            index_probe_mod = spirv.empty() ? VK_NULL_HANDLE : CreateShaderModule(spirv);
            if (!index_probe_mod) Report("Index probe: no vertex shader module " + err.substr(0, 300));
        }
    }
    if (index_probe.ptr)
        for (u32* p = (u32*)index_probe.ptr, *end = p + kProbes * kIndexProbe / 4; p < end; p += 3) p[0] = 0, p[1] = ~0u, p[2] = 0;
    index_probes.clear(), traced_tex.clear();
    if (vertex_probe.ptr) std::memset(vertex_probe.ptr, 0, kProbes * kVertexProbe);
    vkResetQueryPool(dev, q_stat, 0, kProbes), vkResetQueryPool(dev, q_occ, 0, kProbes);
}

// After a traced draw: one texel of `t` (multisampled targets resolved) into the probe buffer: the scissor centre, or with
// GOW3_PROBE_XY=<x>:<y> (fractions of the target) the 16x16 texels from there, hashed: the draws that write that spot.
void Renderer::Impl::ProbeColor(const Target* t, const VkRect2D& sc) {
    const u64 at_buf = probes.size() * kProbeBytes;
    probes.push_back({stats.trace_lines.size() - 1, t && sc.extent.width && sc.extent.height ? std::min(16u, TexelBytes(t->fmt)) : 0});
    if (!probes.back().bytes || !probe_buf.buf) return;
    EndRendering();
    static const float* xy = [] {
        static float f[2];
        char* y;
        const char* e = DebugOption("GOW3_PROBE_XY");
        if (!e) return (float*)nullptr;
        f[0] = std::clamp(std::strtof(e, &y), 0.0f, 1.0f), f[1] = std::clamp(*y ? std::strtof(y + 1, nullptr) : 0.5f, 0.0f, 1.0f);
        return f;
    }();
    const VkOffset3D at = xy ? VkOffset3D{std::min(int(xy[0] * t->w), (int)t->w - 1), std::min(int(xy[1] * t->h), (int)t->h - 1), 0}
                             : VkOffset3D{sc.offset.x + (int)sc.extent.width / 2, sc.offset.y + (int)sc.extent.height / 2, 0};
    const VkExtent3D ext = xy ? VkExtent3D{std::min(16u, t->w - at.x), std::min(16u, t->h - at.y), 1} : VkExtent3D{1, 1, 1};
    if (xy) probes.back().rect = ext.width * ext.height * probes.back().bytes;
    VkImage src = t->t.img;
    VkOffset3D from = at;
    if (t->samples > 1) {
        Tex& one = probe_img[t->fmt];
        if (!one.img) {
            one = MakeImage(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, t->fmt, 16, 16, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, true);
            if (!one.img) return;
            ToGeneral(cmd, one.img, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        const VkImageResolve rs{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, at, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, ext};
        vkCmdResolveImage(cmd, t->t.img, VK_IMAGE_LAYOUT_GENERAL, one.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rs);
        Barrier(cmd);
        src = one.img, from = {};
    }
    const VkBufferImageCopy bc{at_buf, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, from, ext};
    vkCmdCopyImageToBuffer(cmd, src, VK_IMAGE_LAYOUT_GENERAL, probe_buf.buf, 1, &bc);
    Barrier(cmd);
}

void Renderer::Impl::CollectTraceProbes() {
    if (const u32 n = (u32)probes.size()) {
        // prims, VS invocations, clipping in/out, FS invocations; samples passing depth/stencil; texel (hex, high byte first).
        // The frame has retired: no wait (a draw inside an occlusion query of the frame never began its own).
        std::vector<u64> st((size_t)n * 5), occ(n);
        vkGetQueryPoolResults(dev, q_stat, 0, n, st.size() * 8, st.data(), 40, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        vkGetQueryPoolResults(dev, q_occ, 0, n, occ.size() * 8, occ.data(), 8, VK_QUERY_RESULT_64_BIT);
        for (u32 i = 0; i < n; ++i) {
            const u64* s = &st[(size_t)i * 5];
            std::string& l = stats.trace_lines[probes[i].line];
            l += std::format(" || prims {} vs {} clip {}/{} fs {} px {}", s[0], s[1], s[2], s[3], s[4], probes[i].px ? std::to_string(occ[i]) : "-");
            if (probe_buf.ptr) {
                const u8* texels = probe_buf.ptr + (size_t)i * kProbeBytes;
                if (probes[i].bytes) l += " c ";
                for (u32 k = probes[i].bytes; k--;) l += std::format("{:02x}", texels[k]);
                if (probes[i].rect) l += std::format(" h {:08x}", (u32)::Gpu::PieceHash::DebugHash(texels, probes[i].rect));
            }
            if (!vertex_probe.ptr) continue;
            const float* vp = (const float*)(vertex_probe.ptr + (size_t)i * kVertexProbe);
            const u32* vu = (const u32*)vp;
            if (!vu[4]) continue;
            l += std::format(" pos {:g} {:g} {:g} {:g} @{}", vp[0], vp[1], vp[2], vp[3], vu[5]);
            if (vu[6]) l += std::format(" BOOM {} @{} ({:g} {:g} {:g} {:g})", vu[6], vu[7], vp[68], vp[69], vp[70], vp[71]);  // exploded vertices
            if (static const char* want = DebugOption("GOW3_PROBE_DRAW"); want && std::strtoull(want, nullptr, 10) == probes[i].line)
                for (u32 k = 0; k < 64; ++k) l += std::format("{}v{}={:x}/{:g}", k % 8 ? " " : "\n  ", k, vu[8 + k], vp[8 + k]);
        }
        probes.clear();  // read once: the dump takes the lines after this Submit
    }
    if (!index_probes.empty()) {  // TEMP index probe: per way {invocations, lowest, highest}; a way that differs from the CPU's range is marked
        if (!index_probe.ptr) {
            index_probes.clear();
            return;
        }
        static constexpr const char* kWay[]{"window", "exact", "sized"};
        u32 readable = 0, match[3]{}, none[3]{};
        for (size_t i = 0; i < index_probes.size(); ++i) {
            const IndexProbe& p = index_probes[i];
            const u32* g = (const u32*)(index_probe.ptr + i * kIndexProbe);
            std::string& l = stats.trace_lines[p.line];
            l += " || gpu index";
            readable += p.lo <= p.hi;
            for (u32 w = 0; w < (BindIndex2 ? 3u : 2u); ++w, g += 3) {
                const bool same = g[0] && g[1] == p.lo && g[2] == p.hi;
                match[w] += same && p.lo <= p.hi, none[w] += !g[0];
                l += g[0] ? std::format(" {} {}x [{}, {}]{}", kWay[w], g[0], g[1], g[2], same || p.lo > p.hi ? "" : " DIFFERS") : std::format(" {} none", kWay[w]);
            }
        }
        std::string s;
        for (u32 w = 0; w < (BindIndex2 ? 3u : 2u); ++w) s += std::format("; {} binding equals the CPU's range in {}, ran no vertex in {}", kWay[w], match[w], none[w]);
        Report(std::format("Index probe: frame {}, {} indexed draws, {} with readable indices{}", frame, index_probes.size(), readable, s));
        index_probes.clear();
    }
}
} // namespace Gpu
