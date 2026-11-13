//! CPU-only layout accounting using the release's actual owners and model config.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const Sizing = struct {
    var next: u64 = 1 << 30;
    fn alloc(out: *u64, bytes: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(usize, bytes, 256);
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result { return 0; }
    fn zero(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result { return 0; }
};

test "GLM compact snapshots: release layout accounting from real config on both TP2 ranks" {
    const c = try glm.config.parse(t.allocator, @embedFile("release-model-config.json"));
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Sizing.alloc;
    d.api.cuMemFree_v2 = Sizing.free;
    d.api.cuMemsetD8_v2 = Sizing.zero;
    for (0..2) |rank| {
        var w: glm.weights.Weights = undefined;
        w.config = c;
        w.share = try glm.shard.Share.of(&c, @intCast(rank), 2);
        var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
        w.layers = layers[0..c.layers];
        for (w.layers, 0..) |*l, i| l.mixer = switch (c.kind(@intCast(i))) {
            .mla => .{ .mla = undefined }, .kda => .{ .kda = undefined },
        };
        const capacity = glm.native.capacity(1048576);
        const page = glm.state.pageBytesOf(&w, true, false);
        var e: glm.engine.Engine = undefined;
        var seq: @TypeOf(e.cur.*) = undefined;
        seq.st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, capacity,
            .{ .kv = 1 << 44, .pages = glm.pages.tableLen(capacity), .page_bytes = page }, true, false);
        defer seq.st.deinit();
        e.cur = &seq;
        e.c = c;
        e.mtp = false; // actual DFlash2 load uses .draft, excluding target MTP state
        e.drafter = @ptrFromInt(4096); // bytesFormat reads presence only
        e.capacity = capacity;
        const st = &seq.st;
        const bank = st.kda_layers * st.rec_layer;
        const replay = st.rings[0] - st.proj;
        for ([_]usize{ 4096, 8192, 12288, 16384, 28672, 126976 }) |at| {
            const legacy = try glm.snapshots.compact.size(st, &c, capacity, at, null, .legacy);
            const committed = try glm.snapshots.compact.size(st, &c, capacity, at, null, .committed);
            try t.expectEqual(bank + replay, legacy - committed);
            const ring = glm.dflash.ring_bytes;
            const old_charge = try glm.snapshots.pages.rounded(legacy + ring);
            const bank_only_charge = try glm.snapshots.pages.rounded(legacy - bank + ring);
            const charge = try glm.snapshots.pages.rounded(committed + ring);
            try t.expectEqual(old_charge, glm.snapshots.bytesFormat(&e, at, .legacy));
            try t.expectEqual(charge, glm.snapshots.bytesFormat(&e, at, .committed));
            std.debug.print("SIZE,{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d}\n", .{
                rank, at, st.kda_layers, st.heads, legacy, bank, replay, committed,
                ring, old_charge, bank_only_charge, charge,
            });
        }
    }
}
