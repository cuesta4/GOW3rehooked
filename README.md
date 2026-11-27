<h1 align="center">GOW3rehooked</h1>

<p align="center"><i>“Your pawn has failed you, Gaia. Perhaps you should have chosen the other one!”</i></p>

![Title menu with the added Continue, Rehooked, Controls and Extras, and Quit Game entries](assets/screenshots/title-menu.png)

GOW3rehooked lets you play God of War III Remastered on Windows and Linux PCs. The same developer who created the original GOW3 build for shadPS4 made it with a lot of love, care and obsession for performance.

No game code or assets are included. You need your own copy of the game.

<p align="center">
  <br>
  <a href="https://github.com/cuesta4/GOW3rehooked/releases/latest">
    <img src="assets/download-button.svg" alt="Download latest release" width="286" height="58">
  </a>
  <br><br>
</p>

> [!IMPORTANT]
> GOW3rehooked is independent from shadPS4. Please don't take questions or bug reports about it to the shadPS4 developers. Use the [Issues](../../issues) page of this repository.

<br>

[Requirements](#requirements) | [Installation](#installation) | [Performance](#performance) | [Features](#features) | [Settings](#settings) | [Troubleshooting](#troubleshooting) | [Reporting bugs](#reporting-bugs) | [Credits](#credits) | [License](#license)

<br>

## Requirements

| Component | Requirement |
| --- | --- |
| Operating system | Windows 10 or 11, or Linux x86-64 |
| Intel CPU | 4th-generation Core (Haswell) or newer for the standard build. |
| AMD CPU | Ryzen 1000 or newer for the standard build. |
| AMD GPU | Radeon RX 5300 (RDNA 1) or newer. Vega and Polaris: experimental. |
| NVIDIA GPU | GeForce GTX 950 or newer. |
| Intel GPU | UHD Graphics: experimental. Arc and Xe: untested. |
| Memory | 16 GB of RAM recommended |
| Game | God of War III Remastered, European (CUSA01715) or American (CUSA01623) release, updated to version 1.02 |

The standard build requires an x86-64-v3 CPU (including AVX2, BMI1/2, FMA, F16C, LZCNT and MOVBE). Six cores or more are recommended.

The experimental AVX1 build may work on some CPUs older than 4th-generation Intel Core or AMD Ryzen 1000. The same GPU requirements apply. Feel free to [report any issues](#reporting-bugs).

The renderer requires Vulkan 1.3 with depth clip control, push descriptors, conditional rendering, sparse buffer residency and min/max sampler filtering, and uses shader objects where the driver has them. Experimental fallbacks support older GPUs without them (AMD Vega/Polaris, Intel UHD Graphics; Arc/Xe untested, but should also work). Please update to the latest drivers. On Linux, Radeon also needs `VK_KHR_maintenance5`.

## Installation

Only use the AVX1 build if your CPU doesn't meet the requirements for the standard build.

### Windows

1. Download `GOW3rehooked-Windows-x86_64.zip` (standard) or `GOW3rehooked-Windows-x86_64-AVX1.zip` (AVX1) from the [Releases](../../releases/latest) page.
2. Extract the ZIP into the game folder, next to `eboot.bin`:

   ```text
   God of War III\
   ├── eboot.bin
   ├── sce_module\
   ├── sce_sys\
   ├── EXEC\ ...
   ├── gow3.exe
   ├── gow3_debug.cmd
   ├── licenses\
   └── _rehooked_\      the shader recipes; everything else is created on first launch
   ```

3. Run `gow3.exe`. The first few launches may show the [shader preparation screen](#shaders-prepared-in-advance).
4. Choose your settings under Rehooked in the title menu.

Windows releases check for updates at startup. Choose **Yes** to install the same CPU variant and reopen the game, or **No** to skip that release. To see the offer again, delete `_rehooked_/settings/skipped-release.txt`. The previous `gow3.exe` is kept as `gow3.update.old`.

### Linux

1. Download `GOW3rehooked-Linux-x86_64.zip` (standard) or `GOW3rehooked-Linux-x86_64-AVX1.zip` (AVX1) from the [Releases](../../releases/latest) page.
2. Extract the ZIP into the game folder, next to `eboot.bin`.
3. Run `./gow3.AppImage`.

On startup, release AppImages offer to install the latest version for your CPU variant. Choose **Yes** to update both AppImages and restart, or **No** to skip that release. The previous game AppImage is kept as `gow3.AppImage.old`. To see a skipped release again, delete `_rehooked_/settings/skipped-release.txt` from the game folder.

If your extractor dropped the executable bits, run `chmod +x gow3*.AppImage`. On systems without FUSE, run `APPIMAGE_EXTRACT_AND_RUN=1 ./gow3.AppImage`. For [diagnostics](#reporting-bugs), use `./gow3_debug.AppImage` with the same prefix if needed.

For a source build, use Clang and LLD, a C++23 standard library such as libstdc++ 14, CMake, Ninja, NASM, Vulkan headers 1.4.328 or newer, `glslc`, shaderc, glslang, SPIRV-Tools, and libunwind. The [workflow](.github/workflows/windows.yml) lists the Ubuntu 24.04 packages. Run `bash build-fast.sh --release`, then `build-fast/Linux/Release/bin/gow3 --game-dir /path/to/game`. Add `--ivy-bridge` to build for AVX1.

### Important files

The game stores its settings, saves and diagnostics in `_rehooked_`:

| Path | Contents |
| --- | --- |
| `settings/config.cfg` | Your Rehooked settings. See [Tools](#tools) for the in-game reset. |
| `saves/` | Your saves |
| `shader_cache/` | Translated shaders, and the shader recipes from the ZIP, which grow as you play |
| `gow3.log` | A log, which grows with every launch |
| `gow3-<date>-<time>.zip` | Diagnostics of one session run with the debug launcher, split into `.part1.zip`, `.part2.zip` and so on past 24 MB, see [Reporting bugs](#reporting-bugs) |

To update, extract the new ZIP over the old files and let it replace them. Your settings, saves and translated shaders are not in the ZIP and stay as they are. To uninstall, delete `gow3.exe` and `gow3_debug.cmd` (or the two AppImages) and `licenses`, and `_rehooked_` as well if you don't want to keep your saves.

## Performance

Performance is the main goal of the project. Both screenshots show the same scene on a Ryzen 5 5600 with an RTX 3060. At 4K the graphics card sets the frame rate. At 480p it no longer does, so the second number shows what the processor can deliver.

<table>
  <tr>
    <td width="50%"><img src="assets/screenshots/performance-4k.png" alt="4K gameplay running at 150 FPS"></td>
    <td width="50%"><img src="assets/screenshots/performance-480p.png" alt="480p gameplay running at 403 FPS"></td>
  </tr>
  <tr>
    <td align="center">4K: 150 FPS, 6.5 ms</td>
    <td align="center">480p: 403 FPS, 2.6 ms</td>
  </tr>
</table>

## Features

### Resolution and frame rate

The game can render at 480p, 720p, 1080p (the PS4 original), 1440p, 1800p or 4K. The higher, the heavier. The frame rate can be 30, 40, 60, 120, 240 or 480 FPS, with deltatime.

### Image quality

The game's default is 2x MSAA and trilinear texture filtering. You can go up to 4x MSAA or turn it off for better performance. For textures, you can use 4x to 16x anisotropic filtering in place of the original trilinear.

From the title menu, open **Rehooked → Graphics → Post Effects** to choose CMAA2 (better) or Fast Adaptive AA (faster).

When the game's resolution and your monitor's differ, the image needs to be resized. You can use bilinear scaling (faster) or FSR 1/Lanczos 2 (better).

### Shaders prepared in advance

A shader is a small program the graphics card runs to draw a surface or an effect. The game's shaders were written for the PS4's GPU, so each one has to be translated before your graphics card can run it. Emulators usually translate a shader the first time the game needs it, in the middle of gameplay, and that pause is the familiar shader stutter.

GOW3rehooked translates the game's shaders before gameplay instead, behind a progress screen at startup. This mode is called AOT (ahead of time) and is the default. The work is spread over up to four launches, so the first one isn't a long wait, and the results are saved for the next launches. The ZIP brings recipes for every shader variant met in a complete playthrough, and the first launch prepares all of them before the rest. On the experimental fallbacks for older GPUs, the same screen also prepares the graphics pipelines, a longer step on weak GPUs.

JIT (just in time) translates a shader when the game first needs it. With AOT it stays active as a safety net for any shader the preparation didn't catch, so nothing goes missing. It can also be used alone, with no preparation screen. Holding Esc, Circle (PlayStation controllers) or B (other gamepads) on the preparation screen skips it, and the game then restarts once with JIT alone.

On GPUs that support fast pipeline linking, the renderer builds new graphics pipelines quickly during play. It records the pipelines it encounters so that the next AOT preparation after restarting can optimize them.

Both modes are described under [Settings](#settings).

<p align="center"><img src="assets/screenshots/shader-preparation.png" alt="Shader preparation screen with a progress bar and a hold-to-skip prompt" width="80%"></p>

### Controls and Extras

You can play with a gamepad or with keyboard and mouse. Open **Controls and Extras** from the title menu or from the game's Options menu during play.

- [**Tools**](#tools) schedules shader compilation, imports saves and restores the default Rehooked settings.
- **Controls** remaps the gamepad and the keyboard and mouse.
- **Cheats & Patches** holds the controller's vibration strength, cheats and visual patches. They apply as soon as you change them and stay as you left them for the next launch.
- **Help** explains each Rehooked setting.

On the keyboard, press **W** or **↑** to move up and **S** or **↓** to move down. Use **Page Up** and **Page Down** to switch tabs, **Enter** or **Space** to activate the selected option, and **Esc** to go back.

The added screens show PlayStation symbols for a DualShock 3, DualShock 4 or DualSense, and Xbox-style buttons for other gamepads.

| Option | What it does |
| --- | --- |
| Vibration | Strength of the controller's vibration, from 0% (off) to **100%** (the game's own). |
| Infinite Health | Kratos's health stays full. |
| Infinite Magic | The magic meter stays full. |
| Infinite Item Meter | The item meter stays full: the Bow of Apollo, the Head of Helios and the Boots of Hermes never run dry. |
| Infinite Rage of Sparta | The Rage of Sparta meter stays full. |
| Max Red Orbs | Red orbs stay at the maximum, so every upgrade can be bought. |
| Infinite Double Jumps | Kratos can jump again in the air as many times as you like. |
| Unlock Chaos Difficulty | Chaos can be chosen in New Game without finishing the game first. |
| Automatic QTE | Quick-time prompts answer themselves: single presses, repeated presses, holds and stick rotations. |
| Remove Motion Blur | Removes the blur on the camera and on fast-moving objects. |
| Remove Depth of Field | Removes the background blur of cutscenes and close camera shots. |

Every cheat and patch is off by default.

Thanks to the people involved in developing, researching and porting these cheats and patches: **SpartanPort** and **Celogamez**.

<table>
  <tr>
    <td width="33%"><img src="assets/screenshots/controls.png" alt="Controls tab with Xbox-style gamepad prompts and keyboard and mouse bindings"></td>
    <td width="33%"><img src="assets/screenshots/cheats-patches.png" alt="Cheats & Patches tab with controller vibration, cheats and visual patches"></td>
    <td width="33%"><img src="assets/screenshots/help.png" alt="Help tab explaining the Resolution setting"></td>
  </tr>
  <tr>
    <td align="center">Controls</td>
    <td align="center">Cheats &amp; Patches</td>
    <td align="center">Help</td>
  </tr>
</table>

#### Tools

| Option | What it does |
| --- | --- |
| Compile shaders | Choose **Schedule compilation** to start an AOT shader preparation pass on the next launch, including when your saved shader mode is JIT. |
| Import saves | Choose **Choose source** to import saves from shadPS4 or SpartanPort. Open Controls and Extras from the title menu to use this option. |
| Reset settings | Choose **Reset to defaults** and confirm to restart with the default Rehooked settings, including vibration, cheats and patches. You can resume from your existing saves. |

### Menus and saves

Continue appears above New Game once you have a save and directly loads the latest. The game originally has 32 manual save slots. GOW3rehooked raises that to 64. The save and load screens let you duplicate or delete saves. Open Save Folder opens the selected save in your file manager.

On first launch, GOW3rehooked offers to import existing saves from **shadPS4** or **SpartanPort**. It recognizes both American (`CUSA01623`) and European (`CUSA01715`) save folders. To import later, open **Controls and Extras → Tools → Import saves** from the title menu.

If the automatic search doesn't find your saves, extract any ZIP you have and choose the `CUSA…` folder, its parent, or a save slot directly inside it. Select the source that made the saves. SpartanPort exports pair each slot with a `<slot>.sce_sys` folder.

If you import a different autosave, the previous one is kept as `AUTOSAVE - OLD` with a numbered suffix if needed. Other saves with the same slot name are skipped. Look for `Save import:` in `_rehooked_/gow3.log` for rejection reasons.

The Rehooked menus, save and load dialogs, and startup screens follow the game's text language. Set it in the game's Options menu.

The opening logos can be skipped, and so can any video or cutscene, by pressing Cross.

<table>
  <tr>
    <td width="50%"><img src="assets/screenshots/load-game.png" alt="Load Game screen with thumbnails, details of the selected save and the Open Save Folder prompt"></td>
    <td width="50%"><img src="assets/screenshots/save-game.png" alt="Save Game screen with Create a New Save selected"></td>
  </tr>
  <tr>
    <td align="center">Load Game</td>
    <td align="center">Save Game</td>
  </tr>
</table>

## Settings

Defaults are in bold.

### Display

| Option | Choices | What it does |
| --- | --- | --- |
| Resolution | 480p, 720p, **1080p**, 1440p, 1800p, 4K | The resolution the game renders at. 1080p is the original. Higher looks sharper and needs more GPU power and video memory. |
| Frame Rate | 30, 40, **60**, 120, 240, 480 FPS | The game keeps its normal speed at every rate. Higher rates need more CPU and GPU power. 40 FPS suits a 120 Hz or 240 Hz display, or VRR. |
| Display Mode | Windowed, **Borderless**, Exclusive | Windowed: a resizable window. Borderless: full screen at the desktop's resolution and refresh rate. Exclusive: full screen, switching the display to the game's resolution and frame rate, or the closest it supports. |
| VRR | **Off**, On | For G-SYNC and FreeSync displays. The display runs at its highest refresh rate and follows the game, so drops below the selected FPS stay smooth. Full screen only. |
| VSync | **FIFO**, Mailbox, Off | FIFO: no tearing, more input lag. Mailbox: no tearing, less lag, more GPU work. Off: least lag, may tear. |
| NVIDIA Reflex | **Off**, On | Lowers input lag when the GPU is the limit, at a small cost in FPS and frame pacing. Requires a compatible NVIDIA GPU and driver. |

### Graphics

| Option | Choices | What it does |
| --- | --- | --- |
| MSAA | Off, **2x**, 4x | Smooths jagged edges. 2x is the original. 4x is smoother and needs more GPU power and video memory. |
| Texture Filtering | **Trilinear**, 4x, 8x, 16x | Trilinear is the original. 4x to 16x anisotropic keeps textures sharp at steep angles, at a small GPU cost. |
| Post Effects | **Off**, CMAA2, Fast Adaptive AA | Smooths jagged edges after the image is drawn, before scaling it to the display. |
| Scaler | **Bilinear**, FSR 1 / Lanczos 2 | How the image is resized to your display. Bilinear is the fastest. FSR 1 (enlarging) and Lanczos 2 (shrinking) keep more detail. |

### Other

| Option | Choices | What it does |
| --- | --- | --- |
| Skip Opening Logos | **Off**, On | Starts the game at the title menu. |
| Skip Videos with X | Off, **On** | Press Cross to skip any video or cutscene. |
| Shader Mechanism | JIT, **AOT** | AOT prepares the shaders before you play, so stutter is rare. JIT translates each shader when it first appears, which can stutter. |
| Shader Recompilation | Sync, **Async** | How shaders are translated during gameplay. Async: no stutter, but an object can appear a moment late. Sync: nothing appears late, but a hitch is possible. |
| Lazy AOT | Off, **On** | On: the first shader preparation is split over up to four launches. Off: it is done in one. |

<table>
  <tr>
    <td width="33%"><img src="assets/screenshots/settings-display.png" alt="Display settings page"></td>
    <td width="33%"><img src="assets/screenshots/settings-graphics.png" alt="Graphics settings page"></td>
    <td width="33%"><img src="assets/screenshots/settings-other.png" alt="Other settings page"></td>
  </tr>
  <tr>
    <td align="center">Display</td>
    <td align="center">Graphics</td>
    <td align="center">Other</td>
  </tr>
</table>

## Troubleshooting

- “EBOOT.BIN NOT FOUND” at startup means the launcher did not find the game folder. Put `gow3.exe` or `gow3.AppImage` next to `eboot.bin`.
- “YOUR GAME IS COMPRESSED” means the folder holds a packaged game. Extract it first.
- “YOUR GAME IS NOT VERSION 1.02” means the game needs its 1.02 update, or is not the European (CUSA01715) or American (CUSA01623) release.
- If the game crashed before reaching the title menu, the next launch offers to reset your settings. Answer Yes if the crashes began after you changed one.
- To restore defaults, open **Controls and Extras → Tools → Reset settings**. If you can't reach the menu, delete `_rehooked_/settings/config.cfg`.
- Windows Security reporting `gow3.exe` as `Trojan:Win32/Wacatac.B!ml` is a false positive that hit the small launchers of v0.1.5 to v0.2.1. Install the latest release, which no longer has them.

## Reporting bugs

Report problems on the [Issues](../../issues) page, after checking that yours isn't already there. The bug report form requires `gow3.log`, and a report that also comes with the debug dump is much easier to act on.

1. Start the game with `gow3_debug.cmd` (or `gow3.exe --debug`) on Windows or `gow3_debug.AppImage` on Linux. The game runs slower with diagnostics turned on.
2. Reproduce the problem. If it is visual, press F9 while it is on screen. The game pauses for a moment while it records that frame, and **SAVING DEBUG DUMP** shows in the bottom-left corner. Wait until it says **DEBUG DUMP SAVED** before you close the game, or the dump loses that frame's images.
3. Quit from the title menu, or close the window.
4. Open `_rehooked_` and find `gow3.log` and the newest `gow3-<date>-<time>.zip`, or all its `.part1.zip`, `.part2.zip`… files when the dump is larger than 24 MB (a crash may leave only `gow3.log`).
5. Open an issue with those files attached, and describe:
   - what happened and what you expected;
   - where in the game it happens (a save from `_rehooked_/saves` made just before the problem helps);
   - your settings;
   - your CPU, GPU, driver version and operating system version.

If you are asked for a more detailed capture, add `--debug=2` to the debug launcher. It is slower than the default debug mode.

```powershell
& "C:\your\game\path\gow3_debug.cmd" --debug=2
```

```sh
./gow3_debug.AppImage --debug=2
```

## Credits

- The [shadPS4](https://github.com/shadps4-emu/shadPS4) team, for the PS4 kernel, system library and loader work the system layer is built on.
- kvicken, for the original resolution patches, extended here.
- Third-party components: [SDL3](https://github.com/libsdl-org/SDL), [Dear ImGui](https://github.com/ocornut/imgui), [FFmpeg](https://ffmpeg.org), [AMD FidelityFX Super Resolution 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR), [Intel CMAA2](https://github.com/GameTechDev/CMAA2), [Fast Adaptive AA](https://github.com/grebord/Fast-Adaptive-AA), [shaderc](https://github.com/google/shaderc) with [glslang](https://github.com/KhronosGroup/glslang) and [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools), [Zydis](https://github.com/zyantific/zydis), [Xbyak](https://github.com/herumi/xbyak), [xxHash](https://github.com/Cyan4973/xxHash), [Zstandard](https://github.com/facebook/zstd), [fmt](https://github.com/fmtlib/fmt), [magic_enum](https://github.com/Neargye/magic_enum), [stb](https://github.com/nothings/stb), [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9) and the [Cinzel](https://github.com/NDISCOVER/Cinzel) typeface.

## License

GOW3rehooked is licensed under the GNU General Public License v2.0 or later. Third-party components keep their own licenses, and their texts are in [`licenses/`](licenses). The release ZIP carries the same folder, with each component's copyright notice in `THIRD-PARTY-NOTICES.txt`.

God of War is a trademark of Sony Interactive Entertainment. This project is not affiliated with or endorsed by Sony Interactive Entertainment or Santa Monica Studio.

The software is provided as is, without warranty of any kind. There is no roadmap, and no commitment to fix bugs, add features or release updates.
