#!/usr/bin/env swift
// The app's icons, from the mascot. app/src-tauri/icons/ is made by this script only.
//
//     swift app/make-icons.swift
//
// Source: assets/brand/meetbutler-mascot.png (1254×1254, drawn on its own navy).
// Decided 2026-10-20 (assets/brand/README.md):
// - 128 px and up: the whole mascot. 64 px and below: the mascot zoomed 1.33× around
//   the face and the tray (x 640, y 560 in the source), so they stay readable; the whole
//   figure shrinks to a dark blot there.
// - The tile: inset 4.5 % of the side, corner radius 22.37 % of the inner square, the
//   source's own navy behind the mascot (the source is drawn on it, so the tile is the
//   source clipped).
// - A faint white edge (10 %), as PortButler's, so the navy does not melt into a dark
//   Dock.
// - .icns and .ico carry one picture per size, so the small sizes get the zoomed one;
//   `cargo tauri icon` would shrink a single picture for every size.
//
// The tray (menu bar) glyph is drawn here too: the call-and-voice glyph on the tray, one
// colour on transparent, for macOS to tint as a template image; and a second one with a
// dot for while recording.

import AppKit
import CoreGraphics
import Foundation

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let source = root.appendingPathComponent("assets/brand/meetbutler-mascot.png")
let out = root.appendingPathComponent("app/src-tauri/icons")

guard let data = try? Data(contentsOf: source),
      let rep = NSBitmapImageRep(data: data),
      let mascot = rep.cgImage
else {
    FileHandle.standardError.write(Data("no mascot at \(source.path)\n".utf8))
    exit(1)
}
precondition(mascot.width == 1254 && mascot.height == 1254, "the composition below is measured on the 1254² source")

/// Pixel sizes at and below this get the zoomed composition.
let zoomUpTo = 64
let zoom: CGFloat = 1.33
let zoomCentre = CGPoint(x: 640, y: 560) // source pixels, from the top left: face and tray

func crop(forPixels size: Int) -> CGImage {
    guard size <= zoomUpTo else { return mascot }
    let side = CGFloat(mascot.width) / zoom
    let rect = CGRect(x: zoomCentre.x - side / 2, y: zoomCentre.y - side / 2, width: side, height: side)
    precondition(CGRect(x: 0, y: 0, width: mascot.width, height: mascot.height).contains(rect), "zoom runs off the source")
    return mascot.cropping(to: rect.integral)!
}

func context(_ size: Int) -> CGContext {
    let ctx = CGContext(
        data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )!
    ctx.interpolationQuality = .high
    return ctx
}

func png(_ image: CGImage) -> Data {
    let rep = NSBitmapImageRep(cgImage: image)
    return rep.representation(using: .png, properties: [:])!
}

/// The app icon at exactly `size` pixels.
func icon(_ size: Int) -> Data {
    let s = CGFloat(size)
    let ctx = context(size)
    let inset = s * 0.045
    let inner = s - inset * 2
    let tile = CGRect(x: inset, y: inset, width: inner, height: inner)
    let radius = inner * 0.2237
    let shape = CGPath(roundedRect: tile, cornerWidth: radius, cornerHeight: radius, transform: nil)
    ctx.saveGState()
    ctx.addPath(shape)
    ctx.clip()
    ctx.draw(crop(forPixels: size), in: tile)
    ctx.restoreGState()
    // a faint edge, as PortButler's: the navy tile otherwise melts into a dark Dock or menu
    ctx.addPath(shape)
    ctx.setStrokeColor(CGColor(gray: 1, alpha: 0.10))
    ctx.setLineWidth(max(1, s * 0.006))
    ctx.strokePath()
    return png(ctx.makeImage()!)
}

// --- the tray glyph: a camera and voice bars, black on transparent (a template image) ---

func trayGlyph(_ size: Int, recording: Bool) -> Data {
    let s = CGFloat(size)
    let ctx = context(size)
    ctx.setFillColor(CGColor(gray: 0, alpha: 1))
    // camera body and lens wedge, left half
    let body = CGRect(x: s * 0.02, y: s * 0.25, width: s * 0.38, height: s * 0.50)
    ctx.addPath(CGPath(roundedRect: body, cornerWidth: s * 0.09, cornerHeight: s * 0.09, transform: nil))
    ctx.fillPath()
    ctx.move(to: CGPoint(x: s * 0.38, y: s * 0.43))
    ctx.addLine(to: CGPoint(x: s * 0.55, y: s * 0.30))
    ctx.addLine(to: CGPoint(x: s * 0.55, y: s * 0.70))
    ctx.addLine(to: CGPoint(x: s * 0.38, y: s * 0.57))
    ctx.closePath()
    ctx.fillPath()
    // voice bars, right half, tallest in the middle
    let heights: [CGFloat] = [0.30, 0.56, 0.82, 0.52, 0.28]
    let bar = s * 0.07
    for (i, h) in heights.enumerated() {
        let x = s * 0.61 + CGFloat(i) * s * 0.08
        let r = CGRect(x: x, y: s * 0.5 - s * h / 2, width: bar, height: s * h)
        ctx.addPath(CGPath(roundedRect: r, cornerWidth: bar / 2, cornerHeight: bar / 2, transform: nil))
    }
    ctx.fillPath()
    if recording {
        // a filled dot in the top right corner, cut out of what it overlaps so it reads
        let d = s * 0.34
        let dot = CGRect(x: s - d, y: s - d, width: d, height: d)
        ctx.setBlendMode(.clear)
        ctx.fillEllipse(in: dot.insetBy(dx: -s * 0.05, dy: -s * 0.05))
        ctx.setBlendMode(.normal)
        ctx.fillEllipse(in: dot)
    }
    return png(ctx.makeImage()!)
}

// --- .ico: a directory of PNG pictures, one per size ---

func ico(_ sizes: [Int]) -> Data {
    var head = Data([0, 0, 1, 0])
    head.append(contentsOf: withUnsafeBytes(of: UInt16(sizes.count).littleEndian, Array.init))
    var body = Data()
    var offset = 6 + 16 * sizes.count
    for size in sizes {
        let picture = icon(size)
        head.append(UInt8(size >= 256 ? 0 : size)) // 0 means 256
        head.append(UInt8(size >= 256 ? 0 : size))
        head.append(contentsOf: [0, 0]) // no palette, reserved
        head.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian, Array.init)) // planes
        head.append(contentsOf: withUnsafeBytes(of: UInt16(32).littleEndian, Array.init)) // bits
        head.append(contentsOf: withUnsafeBytes(of: UInt32(picture.count).littleEndian, Array.init))
        head.append(contentsOf: withUnsafeBytes(of: UInt32(offset).littleEndian, Array.init))
        body.append(picture)
        offset += picture.count
    }
    return head + body
}

func write(_ data: Data, _ name: String) {
    try! data.write(to: out.appendingPathComponent(name))
    print("wrote \(name)")
}

try! FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
write(icon(32), "32x32.png")
write(icon(128), "128x128.png")
write(icon(256), "128x128@2x.png")
write(icon(512), "icon.png")
write(ico([16, 24, 32, 48, 64, 256]), "icon.ico")
write(trayGlyph(44, recording: false), "tray.png")
write(trayGlyph(44, recording: true), "tray-recording.png")

// .icns through iconutil, from an iconset with every size drawn for itself
let iconset = FileManager.default.temporaryDirectory.appendingPathComponent("MeetButler-\(getpid()).iconset")
try! FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for point in [16, 32, 128, 256, 512] {
    try! icon(point).write(to: iconset.appendingPathComponent("icon_\(point)x\(point).png"))
    try! icon(point * 2).write(to: iconset.appendingPathComponent("icon_\(point)x\(point)@2x.png"))
}
let run = Process()
run.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
run.arguments = ["-c", "icns", iconset.path, "-o", out.appendingPathComponent("icon.icns").path]
try! run.run()
run.waitUntilExit()
precondition(run.terminationStatus == 0, "iconutil failed")
try? FileManager.default.removeItem(at: iconset)
print("wrote icon.icns")
