// SPDX-License-Identifier: GPL-2.0-or-later
// JIT strategy: asynchronous discovery (Discover) and parallel compilation on draw misses.
#include "gpu/shaders/shader_jit.h"
#include "gpu/shaders/shader_cache.h"
#include "gpu/renderer_impl.h"
#include "gpu/textures.h"
#include "main/cpu.h"
#include "orbis/common/thread.h"

namespace Gpu {

Compiled Renderer::Impl::Compile(Gcn::Stage stage, u64 prog, u64 key, Gcn::Input in, std::vector<u32> code, std::vector<u32> fetch,
                                 std::vector<Gcn::SlotSpec> spec) {
    static constexpr VkShaderStageFlagBits vk_stage[]{VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_COMPUTE_BIT};
    const VkShaderStageFlags next = stage == Gcn::Stage::Vertex ? VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT : 0;
    ShaderCache& cache = ShaderCache::Instance();
    Compiled c;
    auto make_stage = [&](std::span<const u32> words) {
        if (caps.path == ShaderPath::Objects) {
            c.s = Create({words.begin(), words.end()}, vk_stage[(int)stage], next);
        } else if (stage == Gcn::Stage::Compute) {
            c.pl = CreateComputePipeline(words);
        } else {
            c.mod = CreateShaderModule(words);
            if (caps.path == ShaderPath::Gpl && c.mod) {
                if (stage == Gcn::Stage::Vertex) c.pl = CreateGplPreRaster(c.mod, VK_NULL_HANDLE, !caps.depth_clip_negative_one_to_one, false);
                else if (stage == Gcn::Stage::Pixel) c.pl = CreateGplFragShader(c.mod, 1);
            }
        }
    };
    // Debug flags: the probe variant (traced frames only) neither reads nor writes the cache; the exploded-vertex one is cached
    // under its own key (ProgramKey's boom bit), every debug run has it. One they only observe (TEMP_STATS) translates again.
    static const bool temp_stats = DebugDeep("GOW3_TEMP_STATS");  // driver statistics of each program
    const bool altered = in.probe;
    if (!(altered || temp_stats)) {  // translated by an earlier run (or by the AOT build)
        if (caps.path == ShaderPath::Objects) {
            const u64 content = cache.ContentOf(key);
            if (const auto bin = content ? cache.FindBinary(BinaryKey(content)) : nullptr) {  // the driver's own code for this GPU: no compile at all
                c.s = CreateFrom(VK_SHADER_CODE_TYPE_BINARY_EXT, bin->data.data(), bin->data.size(), vk_stage[(int)stage], next), c.out_mask = bin->out_mask;
                if (c.s) return c;
            }
        }
        if (const auto hit = cache.Find(key)) {
            make_stage(hit->spirv), c.out_mask = hit->out_mask;
            if (c.Ready(caps.path, stage)) return c;
        }
    }
    in.code = code, in.fetch = fetch;
    std::vector<u32> spirv;
    u64 mask = 0;
    if (Gcn::EmitSpirv(Gcn::Analyze(in), in, spec, spirv, mask, c.err)) {
        make_stage(spirv), c.out_mask = mask;  // exported parameters, for the rect list geometry shader
        if (temp_stats && !spirv.empty() && stage != Gcn::Stage::Compute) {
            std::string e2;
            static const std::vector<u32> vs0 = compiler.Spirv("#version 460\nvoid main() { gl_Position = vec4(0.0); }\n", shaderc_vertex_shader, e2);
            VkShaderModule m[2]{};
            const std::vector<u32>* code[2]{stage == Gcn::Stage::Vertex ? &spirv : &vs0, &spirv};
            const u32 nst = stage == Gcn::Stage::Vertex ? 1 : 2;
            VkPipelineShaderStageCreateInfo st[2]{};
            for (u32 k = 0; k < nst; ++k) {
                VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, code[k]->size() * 4, code[k]->data()};
                vkCreateShaderModule(dev, &mi, nullptr, &m[k]);
                st[k] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, k ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT, m[k], "main"};
            }
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
            VkPipelineColorBlendAttachmentState cba[8]{};
            for (auto& a : cba) a.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 8, cba};
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            const VkDynamicState dyns[]{VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT};
            VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dyns};
            VkFormat fmts[8];
            std::fill(fmts, fmts + 8, VK_FORMAT_R32G32B32A32_SFLOAT);
            VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, nst == 2 ? 8u : 0u, fmts, VK_FORMAT_D32_SFLOAT};
            VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR, nst, st, &vi, &ia,
                                            nullptr, &vps, &rs, &ms, &ds, nst == 2 ? &cb : nullptr, &dsi, layout};
            VkPipeline pl = VK_NULL_HANDLE;
            std::string line = std::format("TEMP stats {} {:x}:", stage == Gcn::Stage::Vertex ? "vs" : "ps", prog);
            if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gi, nullptr, &pl) == VK_SUCCESS) {
                auto get_exec = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutablePropertiesKHR");
                auto get_stat = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableStatisticsKHR");
                VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, nullptr, pl};
                u32 ne = 0;
                get_exec(dev, &pi, &ne, nullptr);
                std::vector<VkPipelineExecutablePropertiesKHR> ep(ne, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
                get_exec(dev, &pi, &ne, ep.data());
                for (u32 x = 0; x < ne; ++x) {
                    if (!(ep[x].stages & vk_stage[(int)stage])) continue;
                    VkPipelineExecutableInfoKHR xi{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, nullptr, pl, x};
                    u32 ns = 0;
                    get_stat(dev, &xi, &ns, nullptr);
                    std::vector<VkPipelineExecutableStatisticKHR> sv(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
                    get_stat(dev, &xi, &ns, sv.data());
                    for (auto& s : sv)
                        line += std::format(" [{}]={}", s.name, s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR ? s.value.f64 : (double)s.value.u64);
                }
                vkDestroyPipeline(dev, pl, nullptr);
            } else {
                line += " pipeline failed";
            }
            for (auto mm : m)
                if (mm) vkDestroyShaderModule(dev, mm, nullptr);
            std::fprintf(stderr, "%s\n", line.c_str());
        }
    }
    if (c.Ready(caps.path, stage) && !altered) cache.Record(key, (u32)stage, c.out_mask, spirv);
    return c;
}

u32 Renderer::Impl::Prepare(u64 key, u64 binary, u32 stage) {
    static constexpr VkShaderStageFlagBits vk_stage[]{VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_COMPUTE_BIT};
    if (stage >= std::size(vk_stage)) return 2;
    {
        std::shared_lock lock(prog_lock);
        if (shaders.contains(key) || compiling.contains(key)) return 0;
    }
    const bool vertex = stage == u32(Gcn::Stage::Vertex);
    if (caps.path != ShaderPath::Objects) {
        const auto hit = ShaderCache::Instance().Find(key);
        if (!hit || hit->spirv.empty()) return 2;
        ShaderRef ref{key};
        if (stage == u32(Gcn::Stage::Compute)) {
            ref.pl = CreateComputePipeline(hit->spirv, true);
            if (!ref.pl) ref.pl = CreateComputePipeline(hit->spirv, false);
        } else {
            ref.mod = CreateShaderModule(hit->spirv);
            if (caps.path == ShaderPath::Gpl && ref.mod)
                ref.pl = vertex ? CreateGplPreRaster(ref.mod, VK_NULL_HANDLE, !caps.depth_clip_negative_one_to_one, false) : CreateGplFragShader(ref.mod, 1);
        }
        if (!ref) return 2;
        std::unique_lock lock(prog_lock);
        if (shaders.contains(key)) return lock.unlock(), DestroyShaderRef(ref), 0;
        if (vertex) vs_outputs[key] = hit->out_mask;
        shaders[key] = ref;
        return 1;
    }
    const auto bin = ShaderCache::Instance().FindBinary(binary);
    const VkShaderEXT s = bin ? CreateFrom(VK_SHADER_CODE_TYPE_BINARY_EXT, bin->data.data(), bin->data.size(), vk_stage[stage],
                                           vertex ? VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT : 0)
                              : VK_NULL_HANDLE;
    if (!s) return 2;
    std::unique_lock lock(prog_lock);
    if (shaders.contains(key)) return lock.unlock(), DestroyShader(dev, s, nullptr), 0;  // a draw made it meanwhile
    if (vertex) vs_outputs[key] = bin->out_mask;
    shaders[key] = {key, s, VK_NULL_HANDLE, VK_NULL_HANDLE};
    return 1;
}

u64 Renderer::Impl::Enqueue(Gcn::Stage stage, u64 prog, const u32* user, const CpuRead& cpu, u32 rsrc1, u32 rsrc2, bool rect, bool urgent) {
    Prepared p;
    if (!prog || !Identify(stage, prog, user, cpu, rsrc1, rsrc2, rect, p)) return 0;
    if (HasClearRectVs() && p.in.rect && ClearVs(p.in.code)) return kClearRectVsKey;
    Specify(stage, p);
    ShaderCache& cache = ShaderCache::Instance();
    {
        std::shared_lock lock(prog_lock);
        if (shaders.contains(p.key) || compiling.contains(p.key)) return p.key;
    }
    const u64 content = cache.ContentOf(p.key);
    const bool built = caps.path == ShaderPath::Objects ? (content && cache.ContainsBinary(BinaryKey(content))) : (content != 0);  // a short load, which a draw waits for
    if (urgent && !built) return p.key;
    std::unique_lock lock(prog_lock);
    if (!shaders.contains(p.key) && !compiling.contains(p.key)) compiling.emplace(p.key, Async(stage, prog, p, urgent, built));
    return p.key;
}

std::shared_future<Compiled> Renderer::Impl::Async(Gcn::Stage stage, u64 prog, const Prepared& p, bool miss, bool waited) {
    stall_compiles.fetch_add(1, std::memory_order_relaxed);
    // What makes it a new variant: the parts of its key (Prepare, Shader) against its program's last variant. The PS state
    // runs from ps_input_ena to mrt_map in Gcn::Input.
    const Gcn::Input& pin = p.in;
    Variant now{::Gpu::PieceHash::ShaderCode(pin.code.data(), pin.code.size_bytes()), ::Gpu::PieceHash::ShaderCode(pin.fetch.data(), pin.fetch.size_bytes()),
                ::Gpu::PieceHash::ShaderVariant(&pin.ps_input_ena, size_t((const char*)pin.threads - (const char*)&pin.ps_input_ena)), ::Gpu::PieceHash::ShaderVariant(pin.threads, sizeof(pin.threads)),
                pin.rsrc1, pin.rsrc2, p.rect, p.spec};
    Stats::Compile rec{prog, p.key, (u32)stage, miss ? 0u : 1u << 31, ~0u};
    auto pack = [](const Gcn::SlotSpec& s, u32* w) {
        w[0] = u32(s.dim) | u32(s.cls) << 4 | s.unnormalized << 8 | s.degamma << 9 | s.srgb << 10 | s.vfmt << 11 | u32(s.dfmt) << 12 | u32(s.nfmt) << 16 | u32(s.dst_sel) << 20;
        w[1] = s.part;
    };
    if (auto [it, first] = variants.try_emplace(prog | (u64)stage); first) {
        rec.changed |= 1, it->second = std::move(now);
    } else {
        Variant& v = it->second;
        rec.changed |= (v.code != now.code) << 1 | (v.fetch != now.fetch) << 2 | (v.rsrc1 != now.rsrc1) << 3 | (v.rsrc2 != now.rsrc2) << 4 |
                       (v.rect != now.rect) << 5 | (v.spec.size() != now.spec.size()) << 6 | (v.ps != now.ps) << 8 | (v.threads != now.threads) << 9;
        for (size_t k = 0; k < std::min(v.spec.size(), now.spec.size()); ++k)
            if (Gcn::Pack(v.spec[k]) != Gcn::Pack(now.spec[k])) {
                rec.changed |= 128, rec.slot = u32(k), pack(v.spec[k], rec.was), pack(now.spec[k], rec.now);
                break;
            }
        v = std::move(now);
    }
    ShaderCache::Instance().Queued();
    auto task = std::make_shared<std::packaged_task<Compiled()>>([this, stage, prog, key = p.key, rec, in = p.in, waited,
                                                                   code = std::vector<u32>(p.in.code.begin(), p.in.code.end()),
                                                                   fetch = std::vector<u32>(p.in.fetch.begin(), p.in.fetch.end()), spec = p.spec]() mutable {
#ifdef _WIN32
        if (waited) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);  // the worker sets its own again for the next task
#else
        if (waited) Common::SetCurrentThreadPriority(Common::ThreadPriority::Normal);
#endif
        const auto t = std::chrono::steady_clock::now();
        Compiled c = Compile(stage, prog, key, in, std::move(code), std::move(fetch), std::move(spec));
        ShaderCache::Instance().Completed();
        rec.us = u32(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count());
        std::scoped_lock lock(compiled_lock);
        compiled.push_back(rec);
        return c;
    });
    auto future = task->get_future().share();
    Queue([task] { (*task)(); }, miss ? 0 : 1);
    return future;
}

void Renderer::Impl::Queue(std::function<void()> task, u32 priority) {
    {
        std::scoped_lock lock(task_lock);
        if (compilers.empty())
            for (unsigned i = 0; i < Gow3::Cpu::plan.shader_threads; ++i)  // idle workers sleep
                compilers.emplace_back([this] {
                    for (;;) {
                        std::function<void()> next;
                        {
                            std::unique_lock lock(task_lock);
                            task_cv.wait(lock, [&] { return task_stop || !tasks[0].empty() || !tasks[1].empty(); });
                            auto& queue = tasks[0].empty() ? tasks[1] : tasks[0];
                            if (queue.empty()) return;  // stopping, nothing left
                            next = std::move(queue.front()), queue.pop_front();
                        }
                        // Sync: the game waits for the shader. Async: it keeps running, and its threads come first.
#ifdef _WIN32
                        SetThreadPriority(GetCurrentThread(), Jit::mode == Jit::Mode::Async ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_NORMAL);
#else
                        Common::SetCurrentThreadPriority(Jit::mode == Jit::Mode::Async ? Common::ThreadPriority::Low : Common::ThreadPriority::Normal);
#endif
                        next();
                    }
                });
        tasks[priority].push_back(std::move(task));
    }
    task_cv.notify_one();
}

// Runs what is queued, then ends the workers.
void Renderer::Impl::StopCompilers() {
    {
        std::scoped_lock lock(task_lock);
        task_stop = true;
    }
    task_cv.notify_all();
    for (auto& t : compilers) t.join();
    compilers.clear();
}

ShaderRef Renderer::Impl::ResolveCachedModule(Gcn::Stage stage, u64 key) {
    std::shared_future<Compiled> pending;
    {
        std::shared_lock lock(prog_lock);
        if (auto it = shaders.find(key); it != shaders.end()) return it->second;
        if (auto it = compiling.find(key); it != compiling.end()) pending = it->second;
    }
    if (pending.valid() && pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const Compiled c = pending.get();
        if (!c.Ready(caps.path, stage)) return {};
        std::unique_lock lock(prog_lock);
        if (auto it = shaders.find(key); it != shaders.end()) {
            if (auto cit = compiling.find(key); cit != compiling.end()) {
                compiling.erase(cit);
                const ShaderRef cr = c.Ref(key);
                if (cr && cr != it->second) DestroyShaderRef(cr);
            }
            return it->second;
        }
        compiling.erase(key);
        if (stage == Gcn::Stage::Vertex) vs_outputs[key] = c.out_mask;
        Count(stats.shaders);
        return shaders[key] = c.Ref(key);
    }
    const auto hit = ShaderCache::Instance().Find(key);
    if (!hit || hit->spirv.empty()) return {};
    const VkShaderModule mod = CreateShaderModule(hit->spirv);
    if (!mod) return {};
    std::unique_lock lock(prog_lock);
    auto [it, inserted] = shaders.try_emplace(key, ShaderRef{key, VK_NULL_HANDLE, mod, VK_NULL_HANDLE});
    if (!inserted) {
        vkDestroyShaderModule(dev, mod, nullptr);
        return it->second;
    }
    if (stage == Gcn::Stage::Vertex) vs_outputs[key] = hit->out_mask;
    Count(stats.shaders);
    return it->second;
}

std::shared_future<CompiledPipeline> Renderer::Impl::AsyncPipeline(u64 h, ShaderRef vs, ShaderRef ps, ShaderRef gs,
                                                                   FixedPipelineState state, bool probe, bool urgent) {
    stall_compiles.fetch_add(1, std::memory_order_relaxed);
    ShaderCache::Instance().PipelineQueued();
    auto task = std::make_shared<std::packaged_task<CompiledPipeline()>>(
        [this, h, vs, ps, gs, state, probe, urgent]() mutable {
            {
                std::unique_lock lock(prog_lock);
                if (auto it = pipelines.find(h); it != pipelines.end()) {
                    const CompiledPipeline existing = it->second;
                    compiling_pipelines.erase(h);
                    lock.unlock();
                    ShaderCache::Instance().PipelineCompleted();
                    return existing;
                }
            }
            if (!vs.mod) vs = (vs.key == kClearRectVsKey) ? clear_rect_ref : ResolveCachedModule(Gcn::Stage::Vertex, vs.key);
            const bool has_ps = (state.flags & FixedPipelineState::kHasPs) != 0;
            if (has_ps && !ps.mod) ps = ResolveCachedModule(Gcn::Stage::Pixel, ps.key);
            const bool has_gs = (state.flags & FixedPipelineState::kHasGs) != 0;
            if (has_gs && !gs.mod) gs = RectGs(gs.key);
            if (!vs.mod || (has_ps && !ps.mod) || (has_gs && !gs.mod)) {
                {
                    std::unique_lock lock(prog_lock);
                    compiling_pipelines.erase(h);
                }
                ShaderCache::Instance().PipelineCompleted();
                return CompiledPipeline{};
            }
            CompiledPipeline pl{};
            const auto t = std::chrono::steady_clock::now();
            PlOrigin origin = urgent ? kPlHit : kPlAheadHit;
            if (caps.path == ShaderPath::Gpl && has_ps) {
                const VkPipeline vi = gpl_vi[std::min<u8>(state.topology_class, 2)];
                const VkPipeline prerast = GetGplPreRaster(vs, gs, state);
                const VkPipeline fs = GetGplFragShader(ps, state);
                const VkPipeline foi = GetGplFragOutput(state, false);
                const bool need_sq = state.NeedsSquarePipeline(caps);
                const VkPipeline foi_sq = need_sq ? GetGplFragOutput(state, true) : VK_NULL_HANDLE;
                if (vi && prerast && fs && foi && (!need_sq || foi_sq)) {
                    const VkPipeline libs[4]{vi, prerast, fs, foi};
                    if (probe) pl = LinkGplPair(libs, foi_sq, GplLink::CachedLto);
                    if (!pl.main) pl = LinkGplPair(libs, foi_sq, GplLink::Unoptimized), origin = urgent ? kPlGplLink : kPlAhead;
                }
            }
            if (!pl.main) {
                if (caps.cache_control && probe) pl = CreateMonolithicPipeline(vs, ps, gs, state, true);
                if (!pl.main) pl = CreateMonolithicPipeline(vs, ps, gs, state, false), origin = urgent ? kPlMiss : kPlAhead;
            }
            const u32 us = u32(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count());
            CompiledPipeline ret = pl;
            {
                std::unique_lock lock(prog_lock);
                auto [it, inserted] = pipelines.try_emplace(h, pl);
                if (!inserted) {
                    if (pl.main) vkDestroyPipeline(dev, pl.main, nullptr);
                    if (pl.square) vkDestroyPipeline(dev, pl.square, nullptr);
                    ret = it->second;
                } else if (temp_pipelog && pl.main) {
                    TempPlOrigin(h, origin, us);
                }
                compiling_pipelines.erase(h);
            }
            ShaderCache::Instance().PipelineCompleted();
            return ret;
        });
    auto future = task->get_future().share();
    Queue([task] { (*task)(); }, urgent ? 0 : 1);
    return future;
}

bool Renderer::Impl::ExtractDiscoverPipelineState(bool rect, bool rect_vs, bool has_ps, FixedPipelineState& out) const {
    const u32 prim = uc[0x242], color_control = ctx[0x202], mode = color_control >> 4 & 7, depth_control = ctx[0x200];
    if (mode > 1) return false;
    u32 topo_class = 2;
    switch (prim) {
    case 1: topo_class = 0; break;
    case 2:
    case 3: topo_class = 1; break;
    case 4:
    case 5:
    case 6:
    case 0x11: topo_class = 2; break;
    default: return false;
    }
    const u32 vte = ctx[0x206], clip = ctx[0x204];
    const float xs = vte & 1 ? F(ctx[0x10F]) : 1.0f, ys = vte & 4 ? F(ctx[0x111]) : 1.0f;
    if (2.0f * xs <= 0.0f || 2.0f * ys == 0.0f) return false;

    const u32 col_format = ctx[0x1C5], target_mask = ctx[0x08E];
    bool has_color[8]{};
    VkFormat color_fmt[8]{};
    u32 color_count = 0, samples = 0;
    for (u32 k = 0; k < 8 && mode == 1; ++k) {
        const u32* cb = &ctx[0x318 + k * 15];
        const VkFormat fmt = ColorFormat(cb[4]);
        if (!cb[0] || fmt == VK_FORMAT_UNDEFINED || !(col_format >> (4 * k) & 0xF) || !(target_mask >> (4 * k) & 0xF)) continue;
        const u32 s = ClampSamples(1u << (cb[5] >> 12 & 7));
        if (samples && samples != s) return false;
        samples = s;
        has_color[k] = true;
        color_fmt[k] = fmt;
        color_count = k + 1;
    }

    const bool z_test = depth_control >> 1 & 1, z_write = depth_control >> 2 & 1, stencil = depth_control & 1, z_bounds = caps.depth_bounds && (depth_control >> 3 & 1);
    const u32 z_fmt = ctx[0x010] & 3, s_fmt = ctx[0x011] & 1;
    const bool db_clear = (ctx[0x000] & 3) != 0;
    const u32 st_ctl = ctx[0x10B], st_ref_f = ctx[0x10C], st_ref_b = (depth_control >> 7 & 1) ? ctx[0x10D] : st_ref_f;
    const u32 st_fn_f = depth_control >> 8 & 7, st_fn_b = (depth_control >> 7 & 1) ? (depth_control >> 20 & 7) : st_fn_f;
    const auto st_op_writes = [](u32 fn, u32 op_fail, u32 op_zpass, u32 op_zfail, u32 wmask) {
        return (wmask & 0xFF) != 0 && ((fn != 7 && op_fail != 0) || (fn != 0 && (op_zpass != 0 || op_zfail != 0)));
    };
    const bool st_can_reject = stencil && s_fmt && (st_fn_f != 7 || ((depth_control >> 7 & 1) && st_fn_b != 7));
    const bool st_can_write = stencil && s_fmt &&
                              (st_op_writes(st_fn_f, st_ctl & 0xF, st_ctl >> 4 & 0xF, st_ctl >> 8 & 0xF, st_ref_f >> 16) ||
                               ((depth_control >> 7 & 1) && st_op_writes(st_fn_b, st_ctl >> 12 & 0xF, st_ctl >> 16 & 0xF, st_ctl >> 20 & 0xF, st_ref_b >> 16)));
    const bool stencil_eff = st_can_reject || st_can_write;
    const bool z_effect = (z_test && (depth_control >> 4 & 7) != 7) || z_write || stencil_eff || z_bounds;
    VkFormat depth_fmt = VK_FORMAT_UNDEFINED, stencil_fmt = VK_FORMAT_UNDEFINED;
    if (z_fmt && (z_effect || db_clear)) {
        const VkFormat fmt = z_fmt == 1 ? VK_FORMAT_D16_UNORM : s_fmt ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
        const u32 zs = ClampSamples(1u << (ctx[0x010] >> 2 & 3));
        if (samples && zs != samples) {
            if (z_effect) return false;
        } else {
            depth_fmt = fmt;
            if (s_fmt) stencil_fmt = fmt;
            if (!samples) samples = zs;
        }
    }
    if (!color_count && depth_fmt == VK_FORMAT_UNDEFINED) return false;
    if (db_clear && (!color_count || mode != 1 || !(sh[0x08] | sh[0x09]))) return false;

    const u32 bc0 = ctx[0x1E0], info0 = ctx[0x31C];
    const u32 mask0 = has_color[0] ? AmdGpu::MrtSwizzle(info0 >> 11 & 3, AmdGpu::DataFormat(info0 >> 2 & 0x1F)).ApplyMask(target_mask & 15) : 0;
    const bool sep0 = bc0 >> 29 & 1;
    const u32 as0 = sep0 ? bc0 >> 16 & 31 : bc0 & 31, af0 = sep0 ? bc0 >> 21 & 7 : bc0 >> 5 & 7, ad0 = sep0 ? bc0 >> 24 & 31 : bc0 >> 8 & 31;
    if (blend_interlock && mode == 1 && has_color[0] && color_fmt[0] == VK_FORMAT_R8G8B8A8_SRGB &&
        mask0 == 8 && (bc0 >> 30 & 1) && (af0 == 2 || af0 == 3) &&
        (as0 == 2 || as0 == 4) && (ad0 == 8 || ad0 == 6) && samples <= 32 && !z_write && !stencil && !db_clear &&
        !(ctx[0x2DC] & 1) && !(ctx[0x203] & 0x41) && !((ctx[0x1B4] ? ctx[0x1B4] : ctx[0x1B3]) & (1u << 14)) &&
        std::none_of(has_color + 1, has_color + 8, [](bool b) { return b; }))
        return false;

    out = {};
    out.color_count = u8(color_count);
    const bool want_square = color_count && !(z_write || stencil_eff);
    for (u32 k = 0; k < color_count; ++k) {
        out.color_formats[k] = color_fmt[k];
        const u32 bc = ctx[0x1E0 + k], info = ctx[0x31C + k * 15], nt = info >> 8 & 7;
        const bool sep = bc >> 29 & 1;
        u32 cs = bc & 0x1F, cf = bc >> 5 & 7, cd = bc >> 8 & 0x1F;
        u32 as = sep ? bc >> 16 & 31 : cs, af = sep ? bc >> 21 & 7 : cf, ad = sep ? bc >> 24 & 31 : cd;
        const bool blend_on = has_color[k] && (bc >> 30 & 1);
        const u32 write_mask = (has_color[k] && has_ps)
                                   ? AmdGpu::MrtSwizzle(info >> 11 & 3, AmdGpu::DataFormat(info >> 2 & 0x1F)).ApplyMask(target_mask >> (4 * k) & 0xF)
                                   : 0;
        const MinMax rc = RewriteMinMax(cs, cf, cd, nt == 0 || nt == 6, false);
        const MinMax ra = RewriteMinMax(as, af, ad, nt == 0 || nt == 6, true);
        const bool wc = blend_on && (write_mask & 7), wa = blend_on && (write_mask & 8);
        if (want_square) {
            if (wc && rc == MinMax::Squared) out.square_color_mask |= u8(1u << k);
            if (wa && ra == MinMax::Squared) out.square_alpha_mask |= u8(1u << k);
        }
        out.blend[k] = {u8(blend_on ? 1 : 0), u8(write_mask),
                        u8(BlendFactor(cs)), u8(BlendFactor(cd)), u8(BlendOp(cf)),
                        u8(BlendFactor(as)), u8(BlendFactor(ad)), u8(BlendOp(af))};
    }
    out.depth_format = depth_fmt;
    out.stencil_format = stencil_fmt;
    out.samples = u8(std::max(1u, samples));
    out.topology_class = u8(topo_class);
    out.flags = ((!(clip >> 19 & 1)) ? FixedPipelineState::kNegOne : 0) |
                ((clip >> 26 & 3) ? FixedPipelineState::kDepthClamp : 0) |
                ((ctx[0x2DC] & 1) ? FixedPipelineState::kAlphaToCoverage : 0) |
                (has_ps ? FixedPipelineState::kHasPs : 0) |
                ((rect && !rect_vs) ? FixedPipelineState::kHasGs : 0);
    return true;
}

void Renderer::Impl::EnqueuePipeline(u64 vs_key, u64 ps_key, bool has_gs, const FixedPipelineState& raw_state) {
    if (!vs_key || caps.path == ShaderPath::Objects || (caps.path == ShaderPath::Gpl && caps.gpl_fast_link && ps_key)) return;
    ShaderRef vs{}, ps{}, gs{};
    u64 gs_mask = 0;
    {
        std::shared_lock lock(prog_lock);
        if (vs_key == kClearRectVsKey) {
            vs = clear_rect_ref;
        } else if (auto it = shaders.find(vs_key); it != shaders.end()) {
            if (!it->second.mod) return;
            vs = it->second;
        }
        if (ps_key) {
            if (auto it = shaders.find(ps_key); it != shaders.end()) {
                if (!it->second.mod) return;
                ps = it->second;
            }
        }
        if (has_gs) {
            if (vs_key == kClearRectVsKey) {
                gs_mask = 1;
            } else if (auto oit = vs_outputs.find(vs_key); oit != vs_outputs.end()) {
                gs_mask = oit->second;
            } else {
                return;
            }
            if (auto git = rect_gs.find(gs_mask); git != rect_gs.end()) {
                if (!git->second.mod) return;
                gs = git->second;
            }
        }
    }
    if (!vs.mod && !ShaderCache::Instance().Contains(vs_key)) return;
    if (ps_key && !ps.mod && !ShaderCache::Instance().Contains(ps_key)) return;
    vs.key = vs_key;
    ps.key = ps_key;
    gs.key = has_gs ? gs_mask : 0ull;

    GraphicsPipelineKey gkey{vs_key, ps_key, gs.key, raw_state};
    gkey.state.Canonicalize(caps);
    const u64 h = gkey.Hash();
    {
        std::shared_lock lock(prog_lock);
        if (pipelines.contains(h) || compiling_pipelines.contains(h)) return;
    }
    std::unique_lock lock(prog_lock);
    if (!pipelines.contains(h) && !compiling_pipelines.contains(h))
        compiling_pipelines.emplace(h, AsyncPipeline(h, vs, ps, gs, gkey.state, true, false));
}

// The rest of the frame from the packet being executed, registers only: the programs its draws and dispatches select
// (as Draw and Dispatch do) go to the compile threads. The registers are restored afterwards.
void Renderer::Impl::Discover(const CpuRead& cpu, Pos to, bool urgent) {
    const auto saved = std::make_unique<Regs>();
    SaveRegs(*saved);
    Walk(walk_chain, {walk_seg, walk_pos}, to, [&](u32 op, const u32* b, size_t, size_t) {
        if (op == 0x27 || op == 0x2D || op == 0x35) {
            const bool rect = uc[0x242] == 0x11, rect_vs = RectVs(op, b, cpu);
            const u32 mode = ctx[0x202] >> 4 & 7;
            const u64 vs_addr = (u64)(sh[0x48] | (u64)sh[0x49] << 32) << 8;
            const u64 ps_addr = (u64)(sh[0x08] | (u64)sh[0x09] << 32) << 8;
            const u64 vs_key = Enqueue(Gcn::Stage::Vertex, vs_addr, &sh[0x4C], cpu, sh[0x4A], sh[0x4B], rect_vs, urgent);
            const bool want_ps = ps_addr && (mode == 1 || (ctx[0x203] & 0x41));
            const u64 ps_key = want_ps ? Enqueue(Gcn::Stage::Pixel, ps_addr, &sh[0x0C], cpu, sh[0x0A], sh[0x0B], rect, urgent) : 0;
            if ((caps.path == ShaderPath::Monolithic || (caps.path == ShaderPath::Gpl && (!ps_key || !caps.gpl_fast_link))) &&
                vs_key && !(want_ps && mode == 1 && !ps_key)) {
                FixedPipelineState raw_state{};
                if (ExtractDiscoverPipelineState(rect, rect_vs, ps_key != 0, raw_state))
                    EnqueuePipeline(vs_key, ps_key, rect && !rect_vs, raw_state);
            }
        } else if (op == 0x15) {
            Enqueue(Gcn::Stage::Compute, (u64)(sh[0x20C] | (u64)sh[0x20D] << 32) << 8, &sh[0x240], cpu, sh[0x212], sh[0x213], false, urgent);
        }
    });
    LoadRegs(*saved);
}
} // namespace Gpu
