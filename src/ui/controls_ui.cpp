// SPDX-License-Identifier: GPL-2.0-or-later
#include "controls_ui.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "ui/gow3_ui_theme.h"
#include "ui/imgui_core.h"
#include "ui/imgui_layer.h"
#include "ui/imgui_std.h"
#include "ui/translation.h"
#include "input/input_handler.h"
#include "main/host.h"
#include "patches/extras.h"
#include "present/window.h"

namespace Gow3::ControlsUi {
namespace {
using Glyph = UI::PadGlyph;
using UI::Fade;
using UI::Tr;
namespace Theme = UI::Theme;
using namespace ImGui;

// The name of a resting row: under the focused card's white and the gold titles, over the muted reading text.
constexpr ImU32 RestText = IM_COL32(190, 181, 163, 255);

// What the game reads: each action is a PlayStation output of the input config. `fallback` is the controller input bound to
// it by default (empty for the analog directions, which the stick itself drives). The keyboard keys of an output are its
// lines of the config.
struct Action { const char* name; const char* output; const char* fallback; Glyph glyph; };
constexpr Action actions[]{
    {"Light Attack", "square", "square", Glyph::Square},
    {"Heavy Attack", "triangle", "triangle", Glyph::Triangle},
    {"Jump", "cross", "cross", Glyph::Cross},
    {"Grab", "circle", "circle", Glyph::Circle},
    {"Block", "l1", "l1", Glyph::L1},
    {"Interact", "r1", "r1", Glyph::R1},
    {"Prepare Item", "l2", "l2", Glyph::L2},
    {"Magic", "r2", "r2", Glyph::R2},
    {"Pause", "touchpad_right", "back", Glyph::Touchpad},
    {"Power Up Menu", "options", "options", Glyph::Options},
    {"L3", "l3", "l3", Glyph::L3},
    {"R3", "r3", "r3", Glyph::R3},
    {"Left Analog Up", "axis_left_y_minus", "", Glyph::StickUp},
    {"Left Analog Down", "axis_left_y_plus", "", Glyph::StickDown},
    {"Left Analog Left", "axis_left_x_minus", "", Glyph::StickLeft},
    {"Left Analog Right", "axis_left_x_plus", "", Glyph::StickRight},
    {"Right Analog Up", "axis_right_y_minus", "", Glyph::StickUp},
    {"Right Analog Down", "axis_right_y_plus", "", Glyph::StickDown},
    {"Right Analog Left", "axis_right_x_minus", "", Glyph::StickLeft},
    {"Right Analog Right", "axis_right_x_plus", "", Glyph::StickRight},
    {"D-Pad Up", "pad_up", "pad_up", Glyph::DpadUp},
    {"D-Pad Down", "pad_down", "pad_down", Glyph::DpadDown},
    {"D-Pad Left", "pad_left", "pad_left", Glyph::DpadLeft},
    {"D-Pad Right", "pad_right", "pad_right", Glyph::DpadRight},
};
constexpr int action_count = int(std::size(actions));

// The controller inputs a binding can take, by their name in the input config, and the key ImGui reports them as. Xbox and
// PlayStation controllers both arrive as these PlayStation buttons.
struct PadInput { std::string_view token; Glyph glyph; ImGuiKey key; };
constexpr PadInput pad_inputs[]{
    {"cross", Glyph::Cross, ImGuiKey_GamepadFaceDown}, {"circle", Glyph::Circle, ImGuiKey_GamepadFaceRight},
    {"square", Glyph::Square, ImGuiKey_GamepadFaceLeft}, {"triangle", Glyph::Triangle, ImGuiKey_GamepadFaceUp},
    {"l1", Glyph::L1, ImGuiKey_GamepadL1}, {"r1", Glyph::R1, ImGuiKey_GamepadR1},
    {"l2", Glyph::L2, ImGuiKey_GamepadL2}, {"r2", Glyph::R2, ImGuiKey_GamepadR2},
    {"l3", Glyph::L3, ImGuiKey_GamepadL3}, {"r3", Glyph::R3, ImGuiKey_GamepadR3},
    {"options", Glyph::Options, ImGuiKey_GamepadStart}, {"back", Glyph::Touchpad, ImGuiKey_GamepadBack},
    {"pad_up", Glyph::DpadUp, ImGuiKey_GamepadDpadUp}, {"pad_down", Glyph::DpadDown, ImGuiKey_GamepadDpadDown},
    {"pad_left", Glyph::DpadLeft, ImGuiKey_GamepadDpadLeft}, {"pad_right", Glyph::DpadRight, ImGuiKey_GamepadDpadRight},
};

// The keyboard keys and mouse buttons a binding can take, by their name in the input config, the key (or mouse button, when
// there is no key) ImGui reports them as and the label of their cap.
struct KeyInput { std::string token; ImGuiKey key; std::string label; int mouse = -1; };
const std::vector<KeyInput>& KeyInputs() {
    static const std::vector<KeyInput> keys = [] {
        std::vector<KeyInput> list;
        list.reserve(106);
        for (int i = 0; i < 26; ++i) list.push_back({std::string(1, char('a' + i)), ImGuiKey(ImGuiKey_A + i), std::string(1, char('A' + i))});
        for (int i = 0; i < 10; ++i) {
            list.push_back({std::to_string(i), ImGuiKey(ImGuiKey_0 + i), std::to_string(i)});
            list.push_back({"kp" + std::to_string(i), ImGuiKey(ImGuiKey_Keypad0 + i), "NUM " + std::to_string(i)});
        }
        for (int i = 1; i <= 12; ++i) list.push_back({"f" + std::to_string(i), ImGuiKey(ImGuiKey_F1 + i - 1), "F" + std::to_string(i)});
        list.insert(list.end(), {
            {"space", ImGuiKey_Space, "SPACE"}, {"enter", ImGuiKey_Enter, "ENTER"}, {"tab", ImGuiKey_Tab, "TAB"},
            {"backspace", ImGuiKey_Backspace, "BACKSPACE"}, {"delete", ImGuiKey_Delete, "DELETE"}, {"insert", ImGuiKey_Insert, "INSERT"},
            {"home", ImGuiKey_Home, "HOME"}, {"end", ImGuiKey_End, "END"}, {"pgup", ImGuiKey_PageUp, "PAGE UP"}, {"pgdown", ImGuiKey_PageDown, "PAGE DOWN"},
            {"up", ImGuiKey_UpArrow, "UP"}, {"down", ImGuiKey_DownArrow, "DOWN"}, {"left", ImGuiKey_LeftArrow, "LEFT"}, {"right", ImGuiKey_RightArrow, "RIGHT"},
            {"lshift", ImGuiKey_LeftShift, "L SHIFT"}, {"rshift", ImGuiKey_RightShift, "R SHIFT"}, {"lctrl", ImGuiKey_LeftCtrl, "L CTRL"},
            {"rctrl", ImGuiKey_RightCtrl, "R CTRL"}, {"lalt", ImGuiKey_LeftAlt, "L ALT"}, {"ralt", ImGuiKey_RightAlt, "R ALT"},
            {"lwin", ImGuiKey_LeftSuper, "L WIN"}, {"rwin", ImGuiKey_RightSuper, "R WIN"}, {"capslock", ImGuiKey_CapsLock, "CAPS LOCK"},
            {"printscreen", ImGuiKey_PrintScreen, "PRINT SCR"}, {"scrolllock", ImGuiKey_ScrollLock, "SCROLL LOCK"}, {"pausebreak", ImGuiKey_Pause, "PAUSE"},
            {"grave", ImGuiKey_GraveAccent, "`"}, {"minus", ImGuiKey_Minus, "-"}, {"equals", ImGuiKey_Equal, "="}, {"lbracket", ImGuiKey_LeftBracket, "["},
            {"rbracket", ImGuiKey_RightBracket, "]"}, {"backslash", ImGuiKey_Backslash, "\\"}, {"semicolon", ImGuiKey_Semicolon, ";"},
            {"apostrophe", ImGuiKey_Apostrophe, "'"}, {"comma", ImGuiKey_Comma, ","}, {"period", ImGuiKey_Period, "."}, {"slash", ImGuiKey_Slash, "/"},
            {"kpperiod", ImGuiKey_KeypadDecimal, "NUM ."}, {"kpslash", ImGuiKey_KeypadDivide, "NUM /"}, {"kpasterisk", ImGuiKey_KeypadMultiply, "NUM *"},
            {"kpminus", ImGuiKey_KeypadSubtract, "NUM -"}, {"kpplus", ImGuiKey_KeypadAdd, "NUM +"}, {"kpenter", ImGuiKey_KeypadEnter, "NUM ENTER"},
            {"kpequals", ImGuiKey_KeypadEqual, "NUM ="},
            {"leftbutton", ImGuiKey_None, "MOUSE 1", ImGuiMouseButton_Left}, {"rightbutton", ImGuiKey_None, "MOUSE 2", ImGuiMouseButton_Right},
            {"middlebutton", ImGuiKey_None, "MOUSE 3", ImGuiMouseButton_Middle}, {"sidebuttonback", ImGuiKey_None, "MOUSE 4", 3}});
        return list;
    }();
    return keys;
}

// The options of the Rehooked menu, by page.
struct Help { const char* section; const char* name; const char* text; };
constexpr Help help[]{
    {"Display", "Resolution", "The resolution the game renders at. 1080p is the original. Higher looks sharper and needs more GPU power and video memory."},
    {"Display", "Frame Rate", "The game keeps its normal speed at every rate. Higher rates need more CPU and GPU power. 40 FPS suits a 120 Hz or 240 Hz display, or VRR."},
    {"Display", "Display Mode", "Windowed: a resizable window. Borderless: full screen at the desktop's resolution and refresh rate. Exclusive: full screen, switching the display to the game's resolution and frame rate, or the closest it supports."},
    {"Display", "VRR", "For G-SYNC and FreeSync displays. The display runs at its highest refresh rate and follows the game, so drops below the selected FPS stay smooth. Full screen only."},
    {"Display", "VSync", "FIFO: no tearing, more input lag. Mailbox: no tearing, less lag, more GPU work. Off: least lag, may tear."},
    {"Display", "NVIDIA Reflex", "Lowers input lag when the GPU is the limit, at a small cost in FPS and frame pacing. Requires a compatible NVIDIA GPU and driver."},
    {"Graphics", "MSAA", "Smooths jagged edges. 2x is the original. 4x is smoother and needs more GPU power and video memory."},
    {"Graphics", "Texture Filtering", "Trilinear is the original. 4x to 16x anisotropic keeps textures sharp at steep angles, at a small GPU cost."},
    {"Graphics", "Post Effects", "CMAA2 and Fast Adaptive AA smooth jagged edges after rendering."},
    {"Graphics", "Scaler", "How the image is resized to your display. Bilinear is the fastest. FSR 1 (enlarging) and Lanczos 2 (shrinking) keep more detail."},
    {"Other", "Skip Opening Logos", "Starts the game at the title menu."},
    {"Other", "Skip Videos with X", "Press Cross to skip any video or cutscene."},
    {"Other", "Shader Mechanism", "AOT prepares the shaders before you play, so stutter is rare. JIT translates each shader when it first appears, which can stutter."},
    {"Other", "Shader Recompilation", "How shaders are translated during gameplay. Async: no stutter, but an object can appear a moment late. Sync: nothing appears late, but a hitch is possible."},
    {"Other", "Lazy AOT", "On: the first shader preparation is split over up to four launches. Off: it is done in one."},
};
constexpr int help_count = int(std::size(help));

struct ExtraItem {
    const char* section;
    const char* name;
    bool slider;
    Extras::Toggle toggle;
    const char* meta_key;
    const char* meta_val;
    const char* p1;
    const char* p2;
    const char* note;
};
constexpr ExtraItem extras[]{
    {
        "Controller", "Vibration", true, Extras::Toggle::Count, nullptr, nullptr,
        "Strength of the controller's vibration. 100% is the game's own.",
        "0% turns it off.",
        nullptr,
    },
    {
        "Cheats", "Infinite Health", false, Extras::Toggle::InfiniteHealth, "Author", "Celogamez",
        "Kratos's health stays full, whatever hits him. Enemies are not affected.",
        nullptr,
        "Turning it off does not take back health that was already refilled.",
    },
    {
        "Cheats", "Infinite Magic", false, Extras::Toggle::InfiniteMagic, "Author", "Celogamez",
        "The magic meter stays full, so magic attacks can be used back to back.",
        nullptr,
        nullptr,
    },
    {
        "Cheats", "Infinite Item Meter", false, Extras::Toggle::InfiniteItem, "Author", "Celogamez",
        "The item meter stays full: the Bow of Apollo, the Head of Helios and the Boots of Hermes never run dry.",
        nullptr,
        nullptr,
    },
    {
        "Cheats", "Infinite Rage of Sparta", false, Extras::Toggle::InfiniteRage, "Author", "Celogamez",
        "The Rage of Sparta meter stays full, so the Blade of Olympus can be drawn at any time.",
        nullptr,
        nullptr,
    },
    {
        "Cheats", "Max Red Orbs", false, Extras::Toggle::MaxRedOrbs, "Author", "Celogamez",
        "Red orbs stay at the maximum, so every weapon and magic upgrade can be bought.",
        nullptr,
        "Turning it off keeps the orbs you have at that moment.",
    },
    {
        "Cheats", "Infinite Double Jumps", false, Extras::Toggle::InfiniteDoubleJumps, "Author", "SpartanPort",
        "Kratos can jump again in the air as many times as you like, without landing.",
        nullptr,
        "Some gaps can be flown over. Turn it off if a scripted jump behaves oddly.",
    },
    {
        "Cheats", "Unlock Chaos Difficulty", false, Extras::Toggle::UnlockChaos, "Author", "SpartanPort",
        "Chaos, the hardest difficulty, can be chosen in New Game without finishing the game first.",
        nullptr,
        "Turn it on before New Game. It does not change the difficulty of an existing save.",
    },
    {
        "Cheats", "Automatic QTE", false, Extras::Toggle::AutoQte, "Author", "SpartanPort",
        "Quick-time prompts answer themselves: single presses, repeated presses, holds and stick rotations.",
        nullptr,
        nullptr,
    },
    {
        "Visual", "Remove Motion Blur", false, Extras::Toggle::RemoveMotionBlur, "Research", "SpartanPort",
        "Removes the blur the game adds to the camera and to fast-moving objects, so the image stays sharp in motion.",
        "Slightly lighter on the GPU.",
        nullptr,
    },
    {
        "Visual", "Remove Depth of Field", false, Extras::Toggle::RemoveDofSpartan, "Research", "SpartanPort",
        "Removes the background blur of cutscenes and close camera shots at the renderer level, keeping the whole picture sharp.",
        nullptr,
        nullptr,
    },
};
constexpr int extra_count = int(std::size(extras));

std::string Upper(std::string_view text) {
    std::string upper(text);
    for (char& c : upper) c = char(std::toupper(static_cast<unsigned char>(c)));
    return upper;
}

constexpr float kCard = 50.0f; // a card's height at 1080p, before the focus grows it

// Text with the soft shadow of the titles.
void ShadowText(ImDrawList* list, ImFont* font, float size, ImVec2 pos, ImU32 color, const char* text) {
    list->AddText(font, size, pos + ImVec2{1.5f, 1.5f}, IM_COL32(0, 0, 0, 220), text);
    list->AddText(font, size, pos, color, text);
}

// Draws the text wrapped to the width from `pos` down; returns the y below the last line.
float WrapText(ImDrawList* list, ImFont* font, float size, ImVec2 pos, float width, ImU32 color, const char* text) {
    const char* end = text + std::strlen(text);
    for (const char* s = text; s < end;) {
        const char* next = font->CalcWordWrapPosition(size, s, end, width);
        if (next <= s) next = s + 1;
        const char* trim = next;
        while (trim > s && (trim[-1] == ' ' || trim[-1] == '\n')) --trim;
        list->AddText(font, size, pos, color, s, trim);
        pos.y += size * 1.4f;
        for (s = next; s < end && (*s == ' ' || *s == '\n'); ++s) {}
    }
    return pos.y;
}

// A key's cap centered at c, the arrow keys with arrows, at most `max_w` wide; a mouse button as the mouse with it lit; an
// empty token is "none".
void DrawKey(ImDrawList* list, ImVec2 c, float h, float max_w, const std::string& token, ImFont* font) {
    const auto& keys = KeyInputs();
    const auto key = std::ranges::find(keys, token, &KeyInput::token);
    if (key != keys.end() && key->mouse >= 0) return UI::DrawMouseGlyph(list, c, h * 1.1f, key->mouse);
    if (token.empty()) {
        const auto* none = Tr("None");
        const ImVec2 extent = font->CalcTextSizeA(h * 0.5f, FLT_MAX, 0.0f, none);
        return list->AddText(font, h * 0.5f, c - extent * 0.5f, Theme::TextMuted, none);
    }
    const ImVec2 dir = token == "up" ? ImVec2{0, -1} : token == "down" ? ImVec2{0, 1} : token == "left" ? ImVec2{-1, 0} : token == "right" ? ImVec2{1, 0} : ImVec2{0, 0};
    const auto unknown = key == keys.end() ? Upper(token) : std::string{};
    UI::DrawKeyCap(list, c, h, key != keys.end() ? Tr(key->label.c_str()) : unknown.c_str(), font, max_w, dir);
}

bool AnyPadButtonHeld() {
    for (int key = ImGuiKey_GamepadStart; key <= ImGuiKey_GamepadR3; ++key)
        if (IsKeyDown(ImGuiKey(key))) return true;
    return false;
}

// Whether a pad button, a key the list navigates with or a mouse button is down.
bool AnyInputHeld() {
    for (const ImGuiKey key : {ImGuiKey_UpArrow, ImGuiKey_DownArrow, ImGuiKey_LeftArrow, ImGuiKey_RightArrow, ImGuiKey_Enter, ImGuiKey_KeypadEnter,
                               ImGuiKey_W, ImGuiKey_A, ImGuiKey_S, ImGuiKey_D, ImGuiKey_Space, ImGuiKey_Escape, ImGuiKey_Tab})
        if (IsKeyDown(key)) return true;
    for (int button = 0; button < 5; ++button)
        if (IsMouseDown(button)) return true;
    return AnyPadButtonHeld();
}

class Screen final : public ImGui::Layer {
public:
    void Open(bool from_title) {
        if (shown.exchange(true)) return;
        import_available = from_title;
        ImGui::Core::AcquireGamepadInputCapture(); // the game does not see the pad while this is up
        restart = true;
        AddLayer(this);
    }
    void Draw() override;

private:
    enum Tab { ToolsTab, ControlsTab, CheatsTab, HelpTab, TabCount };
    enum class Act { None, Close, PrevTab, NextTab, Reassign, Reset, ToggleExtra, AdjustExtra, ResetExtras, Compile, Import, ResetSettings };
    struct Tool { const char* name; const char* text; const char* button; Act action; };
    static constexpr Tool tools[]{
        {"Compile shaders", "Forces shader precompilation on the next restart to prevent stutters.", "Schedule compilation", Act::Compile},
        {"Import saves", "Import saves from SpartanPort or shadPS4 to continue playing here.", "Choose source", Act::Import},
        {"Reset settings", "Resets Rehooked settings. Saves are preserved.", "Reset to defaults", Act::ResetSettings},
    };
    struct Layout { float cqw, pad_x, right_edge, pad_y, grid_top, grid_h, left_w, right_x, right_w, footer_y; };
    struct Hint { UI::PS4Symbol symbol; const char* label; Act act; bool dpad_horiz = false; };

    void Start();
    void MoveKeyboardFocus();
    void DrawHeader(const Layout& L, ImDrawList* list, ImFont* font);
    Act DrawTools(const Layout& L, ImDrawList* list, ImFont* font);
    void DrawControls(const Layout& L, ImDrawList* list, ImFont* font);
    void DrawCheats(const Layout& L, ImDrawList* list, ImFont* font);
    void DrawHelp(const Layout& L, ImDrawList* list, ImFont* font);
    void BeginList(const char* id, const Layout& L);
    void EndList();
    float BeginRow(int id, float height, ImRect& bb);
    void Heading(const char* text, ImFont* font, bool spaced, const char* badge = nullptr);
    void ControlRow(int index, ImFont* font);
    void CheatsRow(int index, ImFont* font);
    void HelpRow(int index, ImFont* font);
    void DrawCard(ImFont* font, const ImRect& bb, float right, float t, const std::string& name, bool disabled = false);
    float DrawPaneTitle(const Layout& L, ImDrawList* list, ImFont* font, const char* title);
    Act DrawFooter(const Layout& L, ImDrawList* list, ImFont* font);
    void Bind(int index, std::string_view token);
    void BindKey(int index, const std::string& token);
    void Reset(int index);
    bool ToolAvailable(Act action) const;

    // Where a row puts its card, arrow and the two input boxes (gamepad, then keyboard), from the row's right edge.
    struct Columns { float card_right, pad_min, pad_max, key_min, key_max, glyph_h; };
    Columns ColumnsOf(const ImRect& bb) const {
        Columns c;
        c.glyph_h = kCard * S * 0.52f;
        const float pad_w = UI::PadGlyphWidth(Glyph::Touchpad, c.glyph_h) + 22.0f * S, key_w = pad_w * 1.15f;
        c.key_max = bb.Max.x, c.key_min = c.key_max - key_w, c.pad_max = c.key_min - 10.0f * S, c.pad_min = c.pad_max - pad_w;
        c.card_right = c.pad_min - 36.0f * S; // the arrow sits between the card and the gamepad box
        return c;
    }

    std::atomic<bool> shown{}, restart{};
    bool import_available{}; // set before publishing the layer
    // Only the drawing thread from here on.
    bool closing{}, armed{}, first{}, pin{}, swallow{}, extras_dirty{}, dragging_slider{}; // swallow: the input that ended a capture is still held
    bool compile_scheduled{};
    Act tool_error{Act::None};
    int tab{ToolsTab}, focus[TabCount]{}, scroll_to{-1}, activated{-1}, capturing{-1};
    float S{1.0f};
    ImRect tab_rects[TabCount]{};
    std::string bound[action_count];            // the controller input of each action; empty is the default
    std::vector<std::string> keys[action_count]; // its keyboard keys, in the order of the config
};
Screen screen;

void Screen::Start() {
    closing = false, armed = false, first = true, pin = true, swallow = false, extras_dirty = false, dragging_slider = false;
    tab = ToolsTab, scroll_to = -1, activated = -1, capturing = -1;
    std::ranges::fill(focus, 0);
    tool_error = Act::None;
    for (int i = 0; i < action_count; ++i) {
        bound[i] = Input::ControllerBinding(actions[i].output);
        keys[i] = Input::KeyboardBindings(actions[i].output);
    }
    // Older configs put the keyboard's pause on the touchpad's left half, which the game does not read: it moves to the right.
    const auto pause = std::ranges::find_if(actions, [](const Action& a) { return !std::strcmp(a.output, "touchpad_right"); });
    if (const auto old = Input::KeyboardBindings("touchpad_left"); keys[pause - std::begin(actions)].empty() && !old.empty()) {
        keys[pause - std::begin(actions)] = old;
        Input::SetKeyboardBindings("touchpad_right", old);
        Input::SetKeyboardBindings("touchpad_left", {});
    }
}

void Screen::Bind(int index, std::string_view token) {
    bound[index] = token;
    Input::SetControllerBinding(actions[index].output, token);
}

// A key another action has goes to it from this one's first key (or it is left without), so the two never fire together.
void Screen::BindKey(int index, const std::string& token) {
    const std::string previous = keys[index].empty() ? std::string{} : keys[index].front();
    for (int k = 0; k < action_count; ++k) {
        const auto at = std::ranges::find(keys[k], token);
        if (k == index || at == keys[k].end()) continue;
        if (previous.empty() || std::ranges::find(keys[k], previous) != keys[k].end()) keys[k].erase(at);
        else *at = previous;
        Input::SetKeyboardBindings(actions[k].output, keys[k]);
    }
    keys[index] = {token};
    Input::SetKeyboardBindings(actions[index].output, keys[index]);
}

// Both columns of the action go back to what a fresh config has.
void Screen::Reset(int index) {
    Bind(index, actions[index].fallback);
    keys[index] = Input::DefaultKeyboardBindings(actions[index].output);
    Input::SetKeyboardBindings(actions[index].output, keys[index]);
}

void Screen::MoveKeyboardFocus() {
    if (!armed || capturing >= 0 || swallow || dragging_slider) return;
    const ImGuiIO& io = GetIO();
    if (io.KeyCtrl || io.KeyAlt || io.KeySuper || !IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) return;
    const bool up = IsKeyPressed(ImGuiKey_UpArrow, true) || IsKeyPressed(ImGuiKey_W, true);
    const bool down = IsKeyPressed(ImGuiKey_DownArrow, true) || IsKeyPressed(ImGuiKey_S, true);
    // Consume ImGui's move while the list handles the same arrow, including between repeats.
    if (up || down || IsKeyDown(ImGuiKey_UpArrow) || IsKeyDown(ImGuiKey_DownArrow)) NavMoveRequestCancel();
    if (up == down) return;
    constexpr int row_counts[]{int(std::size(tools)), action_count, extra_count, help_count};
    const int next = std::clamp(focus[tab] + (up ? -1 : 1), 0, row_counts[tab] - 1);
    if (next != focus[tab]) focus[tab] = next, pin = true;
}

// The scroll area of a list: the rows stop short of its scrollbar.
void Screen::BeginList(const char* id, const Layout& L) {
    SetCursorPos({L.pad_x, L.grid_top});
    PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{16.0f * S, 8.0f * S});
    PushStyleVar(ImGuiStyleVar_ScrollbarSize, std::max(10.0f, 0.85f * L.cqw));
    PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 2.0f);
    PushStyleColor(ImGuiCol_ScrollbarBg, IM_COL32(20, 12, 10, 200));
    PushStyleColor(ImGuiCol_ScrollbarGrab, IM_COL32(131, 102, 57, 230));
    PushStyleColor(ImGuiCol_ScrollbarGrabHovered, IM_COL32(177, 140, 73, 245));
    PushStyleColor(ImGuiCol_ScrollbarGrabActive, Theme::GoldBronze);
    BeginChild(id, {L.left_w, L.grid_h}, ImGuiChildFlags_NavFlattened, ImGuiWindowFlags_NoBackground);
    // The rows' ids belong to this child. While a button is being captured the focus stays where it was, whatever the pad does.
    if (pin || capturing >= 0 || swallow) {
        SetItemCurrentNavFocus(GetID(focus[tab]));
        if (pin) scroll_to = focus[tab];
        pin = false;
    } else if (tab == CheatsTab && (IsKeyDown(ImGuiKey_LeftArrow) || IsKeyDown(ImGuiKey_RightArrow) ||
                                     IsKeyDown(ImGuiKey_GamepadDpadLeft) || IsKeyDown(ImGuiKey_GamepadDpadRight))) {
        NavMoveRequestCancel();
    }
}

void Screen::EndList() {
    EndChild();
    PopStyleColor(4);
    PopStyleVar(3);
}

// Takes the room of a row; returns how much the focus has grown it (0 to 1), or -1 when it is out of sight.
float Screen::BeginRow(int id, float height, ImRect& bb) {
    auto& ctx = *GetCurrentContext();
    auto& window = *ctx.CurrentWindow;
    const ImGuiID item = GetID(id);
    bb = ImRect{window.DC.CursorPos, window.DC.CursorPos + ImVec2{GetContentRegionAvail().x - 16.0f * S, height}};
    ItemSize(bb.GetSize());
    if (scroll_to == id) {
        ScrollToRect(&window, bb, ImGuiScrollFlags_KeepVisibleEdgeY | ImGuiScrollFlags_NoScrollParent);
        scroll_to = -1;
    }
    if (!ItemAdd(bb, item)) return -1.0f;

    bool hovered = false, held = false;
    const bool pressed = ButtonBehavior(bb, item, &hovered, &held), free = capturing < 0 && !swallow && !dragging_slider, nav_here = ctx.NavId == item;
    // The pointer takes the focus only when it moves, and the gamepad's focus follows it, so neither fights the other.
    if (free && hovered && (ctx.IO.MouseDelta.x != 0.0f || ctx.IO.MouseDelta.y != 0.0f)) {
        if (!nav_here) SetFocusID(item, &window);
        focus[tab] = id;
    } else if (free && nav_here) {
        focus[tab] = id;
    }
    if (pressed && free) { // a click or Cross on the row in focus; a click on another only moves the focus to it
        if (nav_here || focus[tab] == id) activated = id;
        focus[tab] = id;
    }
    const float target = focus[tab] == id ? 1.0f : 0.0f;
    float& amount = *GetStateStorage()->GetFloatRef(item, target);
    amount += (target - amount) * std::min(1.0f, ctx.IO.DeltaTime * 16.0f);
    if (std::abs(target - amount) < 0.01f) amount = target;
    return amount;
}

// The title of a group of rows, outside their cards; every group after the first has room above to set it apart.
void Screen::Heading(const char* text, ImFont* font, bool spaced, const char* badge) {
    const ImVec2 pos = GetCurrentWindow()->DC.CursorPos;
    const float width = GetContentRegionAvail().x - 16.0f * S, size = 19.0f * S, top = spaced ? 41.0f * S : 10.0f * S;
    ItemSize(ImVec2{width, top + size * 1.25f + 1.0f * S});
    auto* draw = GetWindowDrawList();
    draw->AddText(font, size, {pos.x + 8.0f * S, pos.y + top}, Theme::GoldBronze, text);
    if (badge && *badge) {
        const float badge_size = 14.0f * S;
        const ImVec2 ext = font->CalcTextSizeA(badge_size, FLT_MAX, 0.0f, badge);
        draw->AddText(font, badge_size, {pos.x + width - 8.0f * S - ext.x, pos.y + top + size * 1.25f - ext.y - 2.0f * S},
                      Theme::GoldMuted, badge);
    }
}

// A row's card, which the focus t grows, from the row's left to `right`, with the name centered in it.
void Screen::DrawCard(ImFont* font, const ImRect& bb, float right, float t, const std::string& name, bool disabled) {
    const float rest = 1.0f - t;
    const ImVec2 min{bb.Min.x + rest * 8.0f * S, bb.Min.y + (6.0f - 4.0f * t) * S};
    const ImVec2 max{right - rest * 8.0f * S, min.y + (kCard + 8.0f * t) * S};
    auto* draw = GetWindowDrawList();
    UI::DrawFocusCard(draw, min, max, t, S);
    float size = (21.0f + 3.0f * t) * S;
    ImVec2 extent = font->CalcTextSizeA(size, FLT_MAX, 0.0f, name.c_str());
    const float max_w = std::max(40.0f * S, (max.x - min.x) - 24.0f * S);
    if (extent.x > max_w) {
        size *= max_w / extent.x;
        extent = font->CalcTextSizeA(size, FLT_MAX, 0.0f, name.c_str());
    }
    const ImVec2 at{(min.x + max.x - extent.x) * 0.5f, (min.y + max.y - extent.y) * 0.5f};
    if (t > 0.0f) draw->AddText(font, size, at + ImVec2{1.0f, 1.0f}, Fade(IM_COL32(0, 0, 0, 200), t), name.c_str());
    const ImU32 base_col = disabled ? IM_COL32(109, 101, 90, 255) : RestText;
    const ImVec4 focus_col = disabled ? ColorConvertU32ToFloat4(IM_COL32(158, 148, 134, 255)) : ImVec4{1, 1, 1, 1};
    draw->AddText(font, size, at, ColorConvertFloat4ToU32(ImLerp(ColorConvertU32ToFloat4(base_col), focus_col, t)), name.c_str());
}

// The pane at the right: its title over a gold line; returns where its text starts.
float Screen::DrawPaneTitle(const Layout& L, ImDrawList* list, ImFont* font, const char* title) {
    const float base_size = std::max(18.0f, 1.45f * L.cqw);
    float size = base_size;
    if (const ImVec2 extent = font->CalcTextSizeA(size, FLT_MAX, 0.0f, title); extent.x > L.right_w) {
        size *= L.right_w / extent.x;
    }
    const float line = L.grid_top + base_size + 0.8f * L.cqw;
    ShadowText(list, font, size, {L.right_x, L.grid_top + (base_size - size) * 0.5f}, Theme::GoldLight, title);
    list->AddLine({L.right_x, line}, {L.right_x + L.right_w, line}, IM_COL32(201, 158, 75, 55), 1.0f);
    return line + 1.2f * L.cqw;
}

// An action, its arrow and the two boxes of the inputs it takes: the gamepad's and the keyboard's, each its own column.
void Screen::ControlRow(int index, ImFont* font) {
    const float base = kCard * S;
    ImRect bb;
    const float t = BeginRow(index, base + 12.0f * S, bb);
    if (t < 0.0f) return;
    auto* draw = GetWindowDrawList();

    const Columns c = ColumnsOf(bb);
    DrawCard(font, bb, c.card_right, t, Tr(actions[index].name));

    // The arrow stays centered between the card's edge and the gamepad box; the focus grows the card and pushes it to the right.
    const float top = bb.Min.y + 6.0f * S, mid_y = top + base * 0.5f, edge = c.card_right - (1.0f - t) * 8.0f * S;
    const float len = 18.0f * S, head = 8.0f * S, x1 = (edge + c.pad_min + len) * 0.5f, x0 = x1 - len;
    for (const auto [color, alpha] : {std::pair{Theme::GoldMuted, 1.0f - t}, std::pair{Theme::GoldLight, t}}) {
        draw->AddLine({x0, mid_y}, {x1 - head * 0.5f, mid_y}, Fade(color, alpha), 2.2f * S);
        draw->AddTriangleFilled({x1, mid_y}, {x1 - head, mid_y - head * 0.62f}, {x1 - head, mid_y + head * 0.62f}, Fade(color, alpha));
    }

    const float pulse = 0.5f + 0.5f * std::sin(static_cast<float>(GetTime()) * 7.5f);
    for (const bool pad : {true, false}) { // waiting for an input, both boxes pulse
        const ImVec2 box_min{pad ? c.pad_min : c.key_min, top}, box_max{pad ? c.pad_max : c.key_max, top + base}, center = (box_min + box_max) * 0.5f;
        if (capturing == index) {
            draw->AddRectFilled(box_min, box_max, Fade(Theme::GoldBronze, 0.10f + 0.28f * pulse), 3.0f * S);
            draw->AddRect(box_min, box_max, Fade(Theme::GoldLight, 0.35f + 0.65f * pulse), 3.0f * S, 0, 2.0f);
            continue;
        }
        draw->AddRectFilled(box_min, box_max, IM_COL32(18, 5, 5, 255), 3.0f * S);
        draw->AddRect(box_min, box_max, Fade(Theme::GoldBronze, 0.35f + 0.4f * t), 3.0f * S, 0, 1.2f);
        if (!pad) {
            DrawKey(draw, center, c.glyph_h, box_max.x - box_min.x - 12.0f * S, keys[index].empty() ? std::string{} : keys[index].front(), font);
            continue;
        }
        const std::string_view token = bound[index].empty() ? std::string_view(actions[index].fallback) : std::string_view(bound[index]);
        if (const auto input = std::ranges::find(pad_inputs, token, &PadInput::token); input != std::end(pad_inputs)) {
            UI::DrawPadGlyph(draw, center, c.glyph_h, input->glyph);
        } else if (token.empty()) { // the analog directions the stick drives
            UI::DrawPadGlyph(draw, center, c.glyph_h, actions[index].glyph);
        } else { // an input that has no picture
            const std::string name = Upper(token);
            const ImVec2 extent = font->CalcTextSizeA(15.0f * S, FLT_MAX, 0.0f, name.c_str());
            draw->AddText(font, 15.0f * S, center - extent * 0.5f, Theme::GoldLight, name.c_str());
        }
    }
}

void Screen::CheatsRow(int index, ImFont* font) {
    const float base = kCard * S;
    ImRect bb;
    const float t = BeginRow(index, base + 12.0f * S, bb);
    if (t < 0.0f) return;
    auto* draw = GetWindowDrawList();
    const ExtraItem& item = extras[index];
    const bool avail = item.slider || Extras::Available(item.toggle);

    const float box_w = 214.0f * S;
    const float top = bb.Min.y + 6.0f * S;
    const ImVec2 box_min{bb.Max.x - box_w, top};
    const ImVec2 box_max{bb.Max.x, top + base};
    const float card_right = box_min.x - 22.0f * S;

    std::string label = Tr(item.name);
    if (!avail) label += fmt::format(" ({})", Tr("Unavailable"));
    DrawCard(font, bb, card_right, t, label, !avail);

    draw->AddRectFilled(box_min, box_max, IM_COL32(18, 5, 5, 255), 3.0f * S);
    draw->AddRect(box_min, box_max, Fade(Theme::GoldBronze, avail ? (0.35f + 0.40f * t) : 0.18f), 3.0f * S, 0, 1.2f);

    const ImVec2 mouse = GetIO().MousePos;
    const bool clicked = armed && IsMouseClicked(ImGuiMouseButton_Left);
    const auto in_rect = [&](ImVec2 rmin, ImVec2 rmax) {
        return mouse.x >= rmin.x && mouse.x <= rmax.x && mouse.y >= rmin.y && mouse.y <= rmax.y;
    };

    if (item.slider) {
        const UI::SliderHits hits = UI::DrawDiamondSlider(draw, box_min, box_max, S, t, Extras::Vibration(), font);
        if (clicked) {
            if (in_rect(hits.left_min, hits.left_max)) {
                focus[tab] = index;
                const u32 cur = Extras::Vibration();
                Extras::SetVibration(cur >= 10u ? cur - 10u : 0u);
                extras_dirty = true;
                activated = -1;
            } else if (in_rect(hits.right_min, hits.right_max)) {
                focus[tab] = index;
                Extras::SetVibration(std::min(100u, Extras::Vibration() + 10u));
                extras_dirty = true;
                activated = -1;
            } else if (in_rect({hits.track_min.x - 6.0f * S, hits.track_min.y},
                               {hits.track_max.x + 6.0f * S, hits.track_max.y})) {
                focus[tab] = index;
                dragging_slider = true;
                activated = -1;
            }
        }
        if (dragging_slider && armed && IsMouseDown(ImGuiMouseButton_Left)) {
            NavMoveRequestCancel();
            const float norm = std::clamp((mouse.x - hits.track_min.x) / std::max(1.0f, hits.track_max.x - hits.track_min.x), 0.0f, 1.0f);
            const u32 val = static_cast<u32>(std::lround(norm * 10.0f)) * 10u;
            if (val != Extras::Vibration()) {
                Extras::SetVibration(val);
                extras_dirty = true;
            }
        }
        return;
    }

    const bool on = avail && Extras::Enabled(item.toggle);
    UI::DrawToggleSelector(draw, box_min, box_max, S, t, on, avail, font);
    if (clicked && avail && in_rect(box_min, box_max)) {
        focus[tab] = index;
        Extras::SetEnabled(item.toggle, !on);
        extras_dirty = true;
        activated = -1;
    }
}

void Screen::DrawCheats(const Layout& L, ImDrawList* list, ImFont* font) {
    const auto active_in_section = [&](const char* sec) {
        int n = 0;
        for (int i = 0; i < extra_count; ++i) {
            if (!std::strcmp(extras[i].section, sec) && !extras[i].slider &&
                Extras::Available(extras[i].toggle) && Extras::Enabled(extras[i].toggle)) {
                ++n;
            }
        }
        return n;
    };

    BeginList("##CheatsList", L);
    const char* section = nullptr;
    for (int i = 0; i < extra_count; ++i) {
        if (!section || std::strcmp(section, extras[i].section)) {
            const int active = active_in_section(extras[i].section);
            const std::string badge = active > 0 ? fmt::format(fmt::runtime(Tr("Active: {}")), active) : std::string{};
            Heading(Tr(extras[i].section), font, section != nullptr, badge.empty() ? nullptr : badge.c_str());
            section = extras[i].section;
        }
        CheatsRow(i, font);
    }
    EndList();

    const ExtraItem& entry = extras[std::clamp(focus[CheatsTab], 0, extra_count - 1)];
    float y = DrawPaneTitle(L, list, font, Tr(entry.name));
    const float body = std::max(17.0f, 1.24f * L.cqw);
    auto* text_font = UI::g_font_body ? UI::g_font_body : font;
    y = WrapText(list, text_font, body, {L.right_x, y}, L.right_w, Theme::TextMuted, Tr(entry.p1));
    if (entry.p2) {
        y = WrapText(list, text_font, body, {L.right_x, y + body * 0.55f}, L.right_w, Theme::TextMuted, Tr(entry.p2));
    }
    if (entry.meta_key && entry.meta_val) {
        y += body * 0.55f;
        const float key_size = std::max(12.0f, 0.8f * L.cqw);
        const float val_size = std::max(14.0f, 1.05f * L.cqw);
        const float row_pad = std::max(6.0f, 0.46f * L.cqw);
        const ImU32 sep_col = IM_COL32(201, 158, 75, 31);
        list->AddLine({L.right_x, y}, {L.right_x + L.right_w, y}, sep_col, 1.0f);
        y += row_pad;
        list->AddText(font, key_size, {L.right_x, y + (val_size * 1.2f - key_size) * 0.5f}, Theme::GoldBronze, Tr(entry.meta_key));
        const float val_x = L.right_x + 9.6f * L.cqw;
        const float val_end = WrapText(list, text_font, val_size, {val_x, y}, std::max(60.0f, L.right_w - 9.6f * L.cqw),
                                       IM_COL32(207, 196, 176, 255), entry.meta_val);
        y = std::max(y + key_size * 1.4f, val_end - val_size * 0.25f) + row_pad;
        list->AddLine({L.right_x, y}, {L.right_x + L.right_w, y}, sep_col, 1.0f);
    }
    if (entry.note) {
        y += body * 0.65f;
        const float label_size = std::max(12.0f, 0.8f * L.cqw);
        const float note_size = std::max(15.0f, 1.1f * L.cqw);
        const float indent = 0.9f * L.cqw;
        const float note_top = y;
        list->AddText(font, label_size, {L.right_x + indent, y}, Theme::GoldBronze, Tr("Note"));
        y += label_size * 1.35f;
        y = WrapText(list, text_font, note_size, {L.right_x + indent, y}, std::max(60.0f, L.right_w - indent),
                     IM_COL32(180, 168, 146, 255), Tr(entry.note));
        list->AddLine({L.right_x, note_top}, {L.right_x, y - note_size * 0.35f}, Fade(Theme::GoldBronze, 0.55f),
                      std::max(1.8f, 0.12f * L.cqw));
    }
    y += body * 0.6f;
    WrapText(list, text_font, body, {L.right_x, y}, L.right_w, Theme::TextMuted,
             Tr("Cheats and patches apply as soon as you change them, and stay as you left them for the next launch."));
}

void Screen::HelpRow(int index, ImFont* font) {
    ImRect bb;
    const float t = BeginRow(index, (kCard + 12.0f) * S, bb);
    if (t >= 0.0f) DrawCard(font, bb, bb.Max.x, t, Tr(help[index].name));
}

bool Screen::ToolAvailable(Act action) const {
    return (action != Act::Import || import_available) && (action != Act::Compile || !compile_scheduled);
}

Screen::Act Screen::DrawTools(const Layout& L, ImDrawList* list, ImFont* font) {
    BeginList("##ToolsList", L);
    for (int i = 0; i < int(std::size(tools)); ++i) {
        ImRect row;
        const float t = BeginRow(i, (kCard + 12.0f) * S, row);
        if (t >= 0.0f) DrawCard(font, row, row.Max.x, t, Tr(tools[i].name));
    }
    EndList();

    const Tool& tool = tools[focus[ToolsTab]];
    const bool available = ToolAvailable(tool.action);
    auto* text_font = UI::g_font_body ? UI::g_font_body : font;
    const float body = std::max(17.0f, 1.24f * L.cqw);
    float y = DrawPaneTitle(L, list, font, Tr(tool.name));
    y = WrapText(list, text_font, body, {L.right_x, y}, L.right_w, Theme::TextMuted, Tr(tool.text));

    const float gap = body * 0.8f;
    y += gap;
    list->AddLine({L.right_x, y}, {L.right_edge, y}, Fade(Theme::GoldBronze, 0.23f), S);
    y += gap;

    const char* label = Tr(tool.action == Act::Compile && compile_scheduled ? "Compilation scheduled" : tool.button);
    float size = std::max(16.0f, 1.15f * L.cqw);
    const float padding = 13.0f * S, height = 2.3f * size;
    const float width = font->CalcTextSizeA(size, FLT_MAX, 0.0f, label).x;
    const ImRect button{{L.right_x, y}, {L.right_x + std::min(L.right_w, width + padding * 2.0f), y + height}};
    const ImGuiID id = GetID("##ToolAction");
    bool hovered = false, held = false, pressed = false;
    if (available && ItemAdd(button, id, nullptr, ImGuiItemFlags_NoNav))
        pressed = ButtonBehavior(button, id, &hovered, &held, ImGuiButtonFlags_NoNavFocus);
    list->AddRectFilled(button.Min, button.Max, available ? (hovered || held ? Theme::BloodMid : Theme::BloodDeep) : Theme::BloodTrenchMid, 2.0f * S);
    list->AddRect(button.Min, button.Max, Fade(Theme::GoldBronze, available ? 0.75f : 0.32f), 2.0f * S);
    if (width > button.GetWidth() - padding * 2.0f) size *= std::max(1.0f, button.GetWidth() - padding * 2.0f) / width;
    list->AddText(font, size, {button.Min.x + padding, button.Min.y + (height - size) * 0.5f},
                  available ? Theme::GoldLight : Theme::TextMuted, label);

    const char* note = tool.action == Act::Import && !available ? "Available from the title menu." :
                       tool_error == tool.action ? "Could not complete the action. Try again." : nullptr;
    if (note) WrapText(list, text_font, body, {L.right_x, button.Max.y + gap}, L.right_w, Theme::TextMuted, Tr(note));
    return pressed && armed ? tool.action : Act::None;
}

void Screen::DrawControls(const Layout& L, ImDrawList* list, ImFont* font) {
    BeginList("##ControlsList", L);
    { // the titles of the two columns, over their boxes
        const ImVec2 pos = GetCurrentWindow()->DC.CursorPos;
        const float width = GetContentRegionAvail().x - 16.0f * S, size = 18.0f * S;
        ItemSize(ImVec2{width, size * 1.25f + 6.0f * S});
        const Columns c = ColumnsOf(ImRect{pos, pos + ImVec2{width, 0.0f}});
        auto* draw = GetWindowDrawList();
        const struct { const char* name; float min, max; } columns[]{{Tr("Gamepad"), c.pad_min, c.pad_max}, {Tr("Mouse & keys"), c.key_min, c.key_max}};
        for (const auto& column : columns) {
            const float text_width = font->CalcTextSizeA(size, FLT_MAX, 0.0f, column.name).x;
            const float fit = std::min(1.0f, (column.max - column.min - 4.0f * S) / text_width);
            draw->AddText(font, size * fit, {(column.min + column.max - text_width * fit) * 0.5f, pos.y},
                          Theme::GoldBronze, column.name);
        }
    }
    for (int i = 0; i < action_count; ++i) ControlRow(i, font);
    EndList();

    const float body = std::max(17.0f, 1.24f * L.cqw);
    auto* text_font = UI::g_font_body ? UI::g_font_body : font;
    const float y = DrawPaneTitle(L, list, font, Tr("Reassign controls"));
    WrapText(list, text_font, body, {L.right_x, y}, L.right_w, Theme::TextMuted,
             Tr("Select an action, then press the controller/keyboard/mouse key you want to map."));
}

void Screen::DrawHelp(const Layout& L, ImDrawList* list, ImFont* font) {
    BeginList("##HelpList", L);
    const char* section = nullptr;
    for (int i = 0; i < help_count; ++i) {
        if (!section || std::strcmp(section, help[i].section)) Heading(Tr(help[i].section), font, section != nullptr);
        section = help[i].section;
        HelpRow(i, font);
    }
    EndList();

    const Help& entry = help[std::clamp(focus[HelpTab], 0, help_count - 1)];
    float y = DrawPaneTitle(L, list, font, Tr(entry.name));
    const float body = std::max(17.0f, 1.24f * L.cqw);
    auto* text_font = UI::g_font_body ? UI::g_font_body : font;
    y = WrapText(list, text_font, body, {L.right_x, y}, L.right_w, Theme::TextMuted, Tr(entry.text)) + body * 0.6f;
    WrapText(list, text_font, body, {L.right_x, y}, L.right_w, Theme::TextMuted, Tr("Settings are applied automatically when you leave the menu, and the game restarts."));
}

// Omega, title and tabs, with shoulder buttons or Page Up and Page Down.
void Screen::DrawHeader(const Layout& L, ImDrawList* list, ImFont* font) {
    const float omega_r = std::max(16.0f, 1.4f * L.cqw), cy = L.pad_y + omega_r;
    const float pulse = 0.5f + 0.5f * std::sin(static_cast<float>(GetTime()) * 2.1f);
    UI::DrawOmegaSymbol(list, {L.pad_x + omega_r, cy}, omega_r, Theme::GoldBronze, pulse, true);

    const float size = std::max(22.0f, 2.0f * L.cqw), x = L.pad_x + omega_r * 2.0f + 1.2f * L.cqw;
    const auto* title = Tr("Controls and Extras");

    const char* names[TabCount]{Tr("Tools"), Tr("Controls"), Tr("Cheats & patches"), Tr("Help")};
    const float base_tab_size = std::max(16.0f, 1.5f * L.cqw);
    const float base_glyph_h = base_tab_size * 1.3f, base_cap_h = base_glyph_h * 0.85f;
    // The shoulder buttons that switch the tabs, or Page Up and Page Down.
    const bool pad = ImGui::Core::GamepadPrompts();
    const char* page_up = Tr("PG UP");
    const char* page_down = Tr("PG DN");
    const float base_shoulder = pad ? UI::PadGlyphWidth(Glyph::L1, base_glyph_h)
        : std::max(UI::KeyCapWidth(page_up, base_cap_h, font), UI::KeyCapWidth(page_down, base_cap_h, font));
    float widths[TabCount]{};
    const float base_gap = base_tab_size * 0.75f, base_between = base_tab_size * 1.2f;
    float nav_width = base_shoulder * 2.0f + base_gap * 2.0f + base_between * (TabCount - 1);
    for (int i = 0; i < TabCount; ++i) {
        widths[i] = font->CalcTextSizeA(base_tab_size, FLT_MAX, 0.0f, names[i]).x;
        nav_width += widths[i];
    }
    const float title_width = font->CalcTextSizeA(size, FLT_MAX, 0.0f, title).x;
    const float scale = std::min(1.0f, std::max(1.0f, L.right_edge - x) / (title_width + base_gap + nav_width));
    const float tab_size = base_tab_size * scale, glyph_h = base_glyph_h * scale, cap_h = base_cap_h * scale;
    const float gap = base_gap * scale, between = base_between * scale, shoulder = base_shoulder * scale;
    const auto shoulder_glyph = [&](float x, bool next) {
        if (pad) UI::DrawPadGlyph(list, {x, cy}, glyph_h, next ? Glyph::R1 : Glyph::L1);
        else UI::DrawKeyCap(list, {x, cy}, cap_h, next ? page_down : page_up, font);
    };
    float at = L.right_edge - nav_width * scale;
    const float fitted = size * scale;
    ShadowText(list, font, fitted, {x, cy - fitted * 0.5f}, Theme::GoldLight, title);
    shoulder_glyph(at + shoulder * 0.5f, false);
    at += shoulder + gap;
    for (int i = 0; i < TabCount; ++i) {
        const bool active = tab == i;
        if (active) ShadowText(list, font, tab_size, {at, cy - tab_size * 0.5f}, Theme::GoldLight, names[i]);
        else list->AddText(font, tab_size, {at, cy - tab_size * 0.5f}, Theme::TextMuted, names[i]);
        const float width = widths[i] * scale;
        if (active) list->AddLine({at, cy + tab_size * 0.62f}, {at + width, cy + tab_size * 0.62f}, Theme::GoldBronze, 2.0f);
        tab_rects[i] = ImRect{{at, cy - tab_size * 0.7f}, {at + width, cy + tab_size * 0.8f}};
        at += width + between;
    }
    shoulder_glyph(at - between + gap + shoulder * 0.5f, true);

    const float meander_y = L.pad_y + 3.4f * L.cqw;
    UI::DrawGreekMeander(list, {L.pad_x, meander_y}, L.right_edge - L.pad_x, std::max(10.0f, 1.15f * L.cqw), Theme::GoldBronze, 1.8f);
}

// The buttons that work now, from the right edge inwards; while a button is awaited, what to press instead. Returns the
// action of the one clicked.
Screen::Act Screen::DrawFooter(const Layout& L, ImDrawList* list, ImFont* font) {
    const float radius = std::max(11.0f, 0.95f * L.cqw), size = std::max(15.0f, 1.06f * L.cqw);
    if (capturing >= 0) {
        const float pulse = 0.5f + 0.5f * std::sin(static_cast<float>(GetTime()) * 7.5f);
        const std::string message = fmt::format(fmt::runtime(Tr("Press a button or key for {} · Esc cancels")), Tr(actions[capturing].name));
        list->AddText(font, size, {L.pad_x, L.footer_y - size * 0.55f}, Fade(Theme::GoldLight, 0.55f + 0.45f * pulse), message.c_str());
        return Act::None;
    }
    Hint hints[3];
    int count = 0;
    if (tab == ControlsTab) {
        hints[count++] = {UI::PS4Symbol::Cross, Tr("Reassign"), Act::Reassign};
        hints[count++] = {UI::PS4Symbol::Triangle, Tr("Reset to defaults"), Act::Reset};
    } else if (tab == CheatsTab) {
        const bool slider = extras[std::clamp(focus[CheatsTab], 0, extra_count - 1)].slider;
        if (slider) hints[count++] = {UI::PS4Symbol::Cross, Tr("Adjust"), Act::AdjustExtra, true};
        else hints[count++] = {UI::PS4Symbol::Cross, Tr("Toggle"), Act::ToggleExtra};
        hints[count++] = {UI::PS4Symbol::Triangle, Tr("Reset all"), Act::ResetExtras};
    }
    hints[count++] = {UI::PS4Symbol::Circle, Tr("Back"), Act::Close};

    const bool pad = ImGui::Core::GamepadPrompts();
    if (!pad) {
        const char* keys = "↑ / ↓ · W / S";
        const float height = radius * 1.7f;
        UI::DrawKeyCap(list, {L.pad_x + UI::KeyCapWidth(keys, height, font) * 0.5f, L.footer_y}, height, keys, font);
    }
    const auto glyph_w = [&](const Hint& h) {
        if (h.dpad_horiz) return pad ? radius * 2.0f : UI::KeyCapWidth("← / →", radius * 1.7f, font);
        return UI::HintGlyphWidth(h.symbol, radius, font);
    };
    const auto hint_w = [&](const Hint& h) {
        return glyph_w(h) + (h.label && *h.label ? size * 0.55f + font->CalcTextSizeA(size, FLT_MAX, 0.0f, h.label).x : 0.0f);
    };

    ImRect rects[3];
    float right = L.right_edge;
    for (int i = count - 1; i >= 0; --i) {
        const Hint& h = hints[i];
        const float left = right - hint_w(h);
        if (!h.dpad_horiz) {
            UI::DrawHint(list, left, L.footer_y, radius, h.symbol, h.label, font, size);
        } else {
            const float w = glyph_w(h);
            if (pad) UI::DrawDpadHorizontalGlyph(list, {left + radius, L.footer_y}, radius * 2.0f);
            else UI::DrawKeyCap(list, {left + w * 0.5f, L.footer_y}, radius * 1.7f, "← / →", font);
            if (h.label && *h.label) list->AddText(font, size, {left + w + size * 0.55f, L.footer_y - size * 0.55f}, Theme::GoldLight, h.label);
        }
        rects[i] = ImRect{{left, L.footer_y - radius}, {right, L.footer_y + radius}};
        right = left - size * 0.95f;
    }
    if (armed && IsMouseClicked(ImGuiMouseButton_Left)) {
        for (int i = 0; i < count; ++i)
            if (rects[i].Contains(GetIO().MousePos)) return hints[i].act;
    }
    return Act::None;
}

void Screen::Draw() {
    if (restart.exchange(false)) Start();
    if (closing) { // Keep the input that closed this screen from reaching the menu behind it.
        if (AnyInputHeld()) return;
        closing = false;
        ImGui::Core::ReleaseGamepadInputCapture();
        RemoveLayer(this);
        shown = false;
        return;
    }

    const ImGuiIO& io = GetIO();
    Layout L;
    L.cqw = std::min(io.DisplaySize.x / 100.0f, io.DisplaySize.y / 56.25f);
    S = std::max(0.6f, L.cqw / 19.2f);
    // The screen is laid out in 100 cqw of width; a wider display keeps that block centered.
    const float layout_w = std::min(io.DisplaySize.x, 100.0f * L.cqw), layout_x = (io.DisplaySize.x - layout_w) * 0.5f;
    L.pad_x = layout_x + 5.5f * L.cqw;
    L.right_edge = layout_x + layout_w - 5.5f * L.cqw;
    L.pad_y = 3.5f * L.cqw;
    L.grid_top = L.pad_y + 3.4f * L.cqw + std::max(10.0f, 1.15f * L.cqw) + 1.6f * L.cqw;
    L.grid_h = io.DisplaySize.y - L.grid_top - 4.2f * L.cqw;
    L.left_w = 55.0f * L.cqw;
    L.right_x = L.pad_x + L.left_w + 3.2f * L.cqw;
    L.right_w = L.right_edge - L.right_x;
    L.footer_y = io.DisplaySize.y - 2.8f * L.cqw;

    SetNextWindowPos({0, 0});
    SetNextWindowSize(io.DisplaySize);
    SetNextWindowCollapsed(false);
    if (first || !io.NavActive) SetNextWindowFocus();
    first = false;
    PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    PushStyleColor(ImGuiCol_WindowBg, IM_COL32(0, 0, 0, 0));
    if (!IsMouseDown(ImGuiMouseButton_Left) || tab != CheatsTab) dragging_slider = false;

    Act act = Act::None;
    if (Begin("Controls and Extras##HelpControls", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus)) {
        auto* list = GetWindowDrawList();
        auto* font = UI::g_font_cinzel ? UI::g_font_cinzel : GetFont();
        UI::DrawBasaltBackground(list, {0, 0}, io.DisplaySize);
        MoveKeyboardFocus();
        DrawHeader(L, list, font);
        if (tab == ToolsTab) act = DrawTools(L, list, font);
        else if (tab == ControlsTab) DrawControls(L, list, font);
        else if (tab == CheatsTab) DrawCheats(L, list, font);
        else DrawHelp(L, list, font);
        if (const Act footer = DrawFooter(L, list, font); footer != Act::None) act = footer;
    }
    End();
    PopStyleColor();
    PopStyleVar(2);

    // The press that opened the screen must not answer it: input counts once the pad has been let go.
    if (!armed) {
        armed = !AnyPadButtonHeld() && !IsKeyDown(ImGuiKey_Enter) && !IsKeyDown(ImGuiKey_Space);
        activated = -1;
        return;
    }
    // While an input is awaited, and until what ended it is let go, it belongs to the capture: the list's own navigation, the
    // activation of a row and the close of the screen all stay quiet.
    if (capturing >= 0 || swallow) {
        NavMoveRequestCancel();
        activated = -1;
        if (capturing < 0) {
            swallow = AnyInputHeld();
            return;
        }
        const auto finish = [&] { capturing = -1, swallow = true; };
        if (IsKeyPressed(ImGuiKey_Escape, false)) { // the mouse buttons can be bound, so only Escape cancels
            finish();
            return;
        }
        for (const PadInput& input : pad_inputs) {
            if (!IsKeyPressed(input.key, false)) continue;
            // An input another action has takes the one this action leaves, so the two swap instead of both firing.
            const auto effective = [&](int k) { return bound[k].empty() ? std::string(actions[k].fallback) : bound[k]; };
            const std::string previous = effective(capturing);
            for (int k = 0; k < action_count; ++k)
                if (k != capturing && !previous.empty() && effective(k) == input.token) Bind(k, previous);
            Bind(capturing, input.token);
            finish();
            return;
        }
        for (const KeyInput& input : KeyInputs()) {
            if (input.mouse >= 0 ? !IsMouseClicked(input.mouse) : !IsKeyPressed(input.key, false)) continue;
            BindKey(capturing, input.token);
            finish();
            return;
        }
        return;
    }

    if (IsMouseClicked(ImGuiMouseButton_Left)) {
        for (int i = 0; i < TabCount; ++i)
            if (tab_rects[i].Contains(io.MousePos) && tab != i) tab = i, pin = true, dragging_slider = false;
    }
    const bool confirm = IsKeyPressed(ImGuiKey_GamepadFaceDown, false) || IsKeyPressed(ImGuiKey_Enter, false) ||
                         IsKeyPressed(ImGuiKey_KeypadEnter, false) || IsKeyPressed(ImGuiKey_Space, false);
    if (IsKeyPressed(ImGuiKey_GamepadFaceRight, false) || IsKeyPressed(ImGuiKey_Escape, false)) act = Act::Close; // no repeat: holding Escape after a cancelled capture must not close
    else if (IsKeyPressed(ImGuiKey_GamepadL1, false) || IsKeyPressed(ImGuiKey_PageUp, false)) act = Act::PrevTab;
    else if (IsKeyPressed(ImGuiKey_GamepadR1, false) || IsKeyPressed(ImGuiKey_PageDown, false)) act = Act::NextTab;
    else if (tab == ToolsTab && act == Act::None && confirm) {
        const Act action = tools[focus[ToolsTab]].action;
        if (ToolAvailable(action)) act = action;
    }
    else if (tab == ControlsTab && (IsKeyPressed(ImGuiKey_GamepadFaceUp, false) || IsKeyPressed(ImGuiKey_Insert, false))) act = Act::Reset;
    else if (tab == ControlsTab && act == Act::None && (activated >= 0 || confirm)) act = Act::Reassign;
    else if (tab == CheatsTab && (IsKeyPressed(ImGuiKey_GamepadFaceUp, false) || IsKeyPressed(ImGuiKey_Insert, false))) act = Act::ResetExtras;
    else if (tab == CheatsTab && act == Act::None) {
        const int idx = std::clamp(focus[CheatsTab], 0, extra_count - 1);
        const ExtraItem& focused = extras[idx];
        const bool horiz_held = IsKeyDown(ImGuiKey_LeftArrow) || IsKeyDown(ImGuiKey_RightArrow) ||
                                IsKeyDown(ImGuiKey_GamepadDpadLeft) || IsKeyDown(ImGuiKey_GamepadDpadRight);
        if (horiz_held) NavMoveRequestCancel();
        const bool step_left = IsKeyPressed(ImGuiKey_LeftArrow, focused.slider) || IsKeyPressed(ImGuiKey_A, focused.slider) ||
                               IsKeyPressed(ImGuiKey_GamepadDpadLeft, focused.slider);
        const bool step_right = IsKeyPressed(ImGuiKey_RightArrow, focused.slider) || IsKeyPressed(ImGuiKey_D, focused.slider) ||
                                IsKeyPressed(ImGuiKey_GamepadDpadRight, focused.slider);
        if (focused.slider && (step_left || step_right)) {
            const u32 cur = Extras::Vibration();
            Extras::SetVibration(step_left ? (cur >= 10u ? cur - 10u : 0u) : std::min(100u, cur + 10u));
            extras_dirty = true;
        } else if (!focused.slider && (step_left || step_right)) {
            if (Extras::Available(focused.toggle)) {
                const bool target = step_right;
                if (Extras::Enabled(focused.toggle) != target) {
                    Extras::SetEnabled(focused.toggle, target);
                    extras_dirty = true;
                }
            }
        } else if (!focused.slider && (activated >= 0 || confirm)) {
            act = Act::ToggleExtra;
        }
    }
    activated = -1;

    switch (act) {
    case Act::Close:
        if (extras_dirty) {
            Extras::Save();
            extras_dirty = false;
        }
        closing = true;
        break;
    case Act::PrevTab: case Act::NextTab: {
        const int next = std::clamp(tab + (act == Act::NextTab ? 1 : -1), 0, TabCount - 1);
        if (next != tab) tab = next, pin = true, dragging_slider = false;
        break;
    }
    case Act::Reassign: capturing = focus[ControlsTab]; break;
    case Act::Reset: Reset(focus[ControlsTab]); break;
    case Act::ToggleExtra: {
        const ExtraItem& item = extras[std::clamp(focus[CheatsTab], 0, extra_count - 1)];
        if (!item.slider && Extras::Available(item.toggle)) {
            Extras::SetEnabled(item.toggle, !Extras::Enabled(item.toggle));
            extras_dirty = true;
        }
        break;
    }
    case Act::AdjustExtra: {
        const ExtraItem& item = extras[std::clamp(focus[CheatsTab], 0, extra_count - 1)];
        if (item.slider) {
            const u32 cur = Extras::Vibration();
            Extras::SetVibration(cur >= 100u ? 0u : std::min(100u, cur + 10u));
            extras_dirty = true;
        }
        break;
    }
    case Act::ResetExtras:
        Extras::ResetAll();
        extras_dirty = true;
        break;
    case Act::Compile:
        compile_scheduled = ScheduleAotBuild();
        tool_error = compile_scheduled ? Act::None : act;
        break;
    case Act::Import: case Act::ResetSettings:
        if (act == Act::Import ? Window::RequestSaveImport() : Window::RequestResetSettings()) {
            if (extras_dirty) Extras::Save();
            extras_dirty = false;
            closing = true;
        } else {
            tool_error = act;
        }
        break;
    case Act::None: break;
    }
}
} // namespace

void Open(bool from_title) { screen.Open(from_title); }
} // namespace Gow3::ControlsUi
