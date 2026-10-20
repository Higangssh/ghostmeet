# MeetButler brand

`meetbutler-mascot.png` (1254×1254) is the mascot as the owner supplied it, drawn with
its navy background. It is the source for the app icon; the icons in
`app/src-tauri/icons/` are generated from it, never edited by hand.

Decided 2026-10-20:
- Icon: the whole mascot at 128 px and up (the same composition as PortButler's), the
  mascot zoomed 1.33× at 64 px and below so the face and the tray stay readable.
  Tile inset 4.5 %, corner radius 22.37 % of the inner square.
- Accent: the periwinkle of the call-and-voice glyph on the tray. Not red: red means
  recording.

Siblings: HomeButler has no tile and a cyan site accent; PortButler has a steel
blue-grey tile and a mint site accent. MeetButler's tile is navy.

`meetbutler-mascot-cutout.png` (1086×1448, transparent) is the same mascot without the
background, supplied 2026-10-20; its alpha was snapped to opaque from 240 up, because
the lossy source left the body faintly see-through. Use it wherever the background is
ours — the app's windows, the site, the README — not a tile on top of a tile.
