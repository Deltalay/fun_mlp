const std = @import("std");
const math = std.math;
const Io = std.Io;
pub fn main(init: std.process.Init) !void {
    const alloc = init.arena.allocator();

    // const input_option = MlpOption{
    //     .activation = .none,
    //     .layer_size = 2,
    //     .bias = .zero,
    // };
    const seed: [32]u8 = "12345678901234567890123456789012".*;
    var hidden_layer = try mlp.init(alloc, .{ .activation = .relu, .layer_size = 4, .input_size = 2, .bias = .random }, &seed);
    var output_layer = try mlp.init(alloc, .{ .activation = .softmax, .layer_size = 2, .input_size = 4, .bias = .random }, &seed);
    var input_item: [2]f32 = [2]f32{ 1.0, 0.0 };
    var target_item: [2] f32 = [2]f32{0.0, 1.0};
    _ = hidden_layer.forward(&input_item);
    _ = output_layer.forward(hidden_layer.output);
    std.debug.print("HIDDEN LAYER\n", .{});
    for (0..4) |index| {
        std.debug.print("{}\n", .{hidden_layer.output[index]});
    }

    std.debug.print("OUTPUT LAYER\n", .{});
    for (0..2) |index| {
        std.debug.print("{}\n", .{output_layer.output[index]});
    }
    const loss = output_layer.CELoss(&target_item);
    std.debug.print("CE Loss: {}\n", .{loss});
}
pub const ActivationType = enum { relu, sigmoid, softmax, none };
pub const BiasInit = enum {
    zero,
    random,
};
pub const MlpOption = struct {
    activation: ActivationType = .relu,
    layer_size: u32 = 32,
    input_size: u32 = 16,
    bias: BiasInit = .random,
};
pub const mlp = struct {
    alloc: std.mem.Allocator,
    option: MlpOption = .{ .activation = .relu, .layer_size = 32 },
    weight: []f32,
    bias: []f32,
    output: []f32,
    softmax_x_sum: f32 = 0,
    pub fn init(
        alloc: std.mem.Allocator,
        option: MlpOption,
        seed: *const [32]u8,
    ) !mlp {
        const bias_init: []f32 = try alloc.alloc(f32, option.layer_size);
        defer alloc.free(bias_init);
        const weight_init: []f32 = try alloc.alloc(f32, option.layer_size * option.input_size);
        defer alloc.free(weight_init);
        var random_init = std.Random.DefaultCsprng.init(seed.*);
        const rng = random_init.random();
        if (option.bias == .zero) {
            @memset(bias_init, 0.0);
        } else {
            for (bias_init) |*b| {
                b.* = rng.float(f32);
            }
        }
        for (weight_init) |*w| {
            w.* = rng.float(f32);
        }

        return .{ .alloc = alloc, .option = option, .weight = try alloc.dupe(f32, weight_init), .bias = try alloc.dupe(f32, bias_init), .output = try alloc.alloc(f32, option.layer_size) };
    }
    pub fn forward(self: *mlp, X: []f32) []f32 {
        for (0..self.option.layer_size) |layer| {
            var temp_weight: f32 = 0.0;
            for (0..X.len) |index| {
                const weight_index = layer * X.len + index;

                temp_weight += self.weight[weight_index] * X[index];
            }
            self.output[layer] = temp_weight + self.bias[layer];
        }
        activate(self);
        return self.output;
    }
    // activate the ouput
    pub fn activate(self: *mlp) void {
        switch (self.option.activation) {
            .relu => {
                for (0..self.option.layer_size) |layer| {
                    self.output[layer] = relu(self.output[layer]);
                }
            },
            .softmax => {
                softmax(self.output);
            },
            .sigmoid => {
                for (0..self.option.layer_size) |layer| {
                    self.output[layer] = sigmoid(self.output[layer]);
                }
            },
            else => {},
        }
    }
    pub fn relu(x: f32) f32 {
        return @max(0.0, x);
    }
    pub fn d_relu(x: f32) f32 {
        return if (x > 0) 1 else 0;
    }
    pub fn sigmoid(x: f32) f32 {
        return 1.0 / (1.0 + math.exp(-x));
    }
    pub fn softmax(output: []f32) void {
        var max_value = output[0];

        for (output[1..]) |x| {
            max_value = @max(max_value, x);
        }

        var sum: f32 = 0.0;

        for (output) |*x| {
            x.* = @exp(x.* - max_value);
            sum += x.*;
        }

        for (output) |*x| {
            x.* /= sum;
        }
    }
    pub fn CELoss(self: *mlp, X: []f32) f32 {
        var loss : f32 = 0.0;
        for (0..self.option.layer_size) |index| {
            loss += X[index]*@log(self.output[index]);
        }
        return -loss;
    }
    pub fn backward(self: *mlp, Y: []f32) f32 {
        var dz: []f32 = self.alloc.alloc(f32, self.option.layer_size);
        for (0..self.option.layer_size) |index| {
            dz[index] = self.output[index] - Y[index];
        }
    }

};
