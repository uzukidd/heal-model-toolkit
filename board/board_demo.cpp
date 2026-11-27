#include <hobot/dnn/hb_dnn.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
namespace {

struct DType {
    std::string_view name;
    int32_t sdk_type;
    std::size_t size;
};

// This UCP SDK exposes split NV12 planes as ordinary U8 tensors, not image enums.
constexpr std::array<DType, 5> kTypes{{
    {"int8", HB_DNN_TENSOR_TYPE_S8, 1},
    {"int16", HB_DNN_TENSOR_TYPE_S16, 2},
    {"int32", HB_DNN_TENSOR_TYPE_S32, 4},
    {"uint8", HB_DNN_TENSOR_TYPE_U8, 1},
    {"float32", HB_DNN_TENSOR_TYPE_F32, 4},
}};

struct Spec {
    std::string name;
    DType dtype;
    std::vector<int32_t> shape;
};

struct Manifest {
    std::vector<Spec> inputs;
    std::vector<Spec> outputs;
};

void check(int32_t rc, const char* api) {
    if (rc != 0) {
        throw std::runtime_error(std::string(api) + " failed: " + std::to_string(rc));
    }
}

int32_t parse_integer(std::string_view text, int32_t minimum, std::string_view label) {
    int32_t value{};
    if (!text.empty()) {
        const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
        if (result.ec == std::errc{} && result.ptr == text.data() + text.size() && value >= minimum) {
            return value;
        }
    }
    throw std::runtime_error(std::string(label) + ": invalid integer '" + std::string(text) + "'");
}

bool safe_name(std::string_view name) {
    return !name.empty() && name != "." && name != ".." &&
           name.find_first_of("/\\:") == std::string_view::npos &&
           std::all_of(name.begin(), name.end(), [](unsigned char ch) { return ch >= 32 && ch != 127; });
}

Manifest read_manifest(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path.string());
    Manifest manifest;
    std::unordered_set<std::string> input_names, output_names;
    std::string line;
    for (std::size_t line_number = 1; std::getline(file, line); ++line_number) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto context = path.string() + ":" + std::to_string(line_number);
        std::array<std::string_view, 4> fields{};
        std::string_view rest(line);
        for (std::size_t i = 0; i + 1 < fields.size(); ++i) {
            const auto tab = rest.find('\t');
            if (tab == std::string_view::npos) throw std::runtime_error(context + ": expected 4 TSV fields");
            fields[i] = rest.substr(0, tab);
            rest.remove_prefix(tab + 1);
        }
        if (rest.find('\t') != std::string_view::npos) throw std::runtime_error(context + ": expected 4 TSV fields");
        fields.back() = rest;
        const bool input = fields[0] == "input";
        if ((!input && fields[0] != "output") || (input && !manifest.outputs.empty())) {
            throw std::runtime_error(context + ": all input entries must precede the output entries");
        }
        if (!safe_name(fields[1])) throw std::runtime_error(context + ": invalid tensor filename");
        const auto dtype = std::find_if(kTypes.begin(), kTypes.end(), [&](const auto& type) { return type.name == fields[2]; });
        if (dtype == kTypes.end()) throw std::runtime_error(context + ": unsupported dtype '" + std::string(fields[2]) + "'");
        Spec spec{std::string(fields[1]), *dtype, {}};
        if (!fields[3].empty()) {
            auto dims = fields[3];
            for (;;) {
                const auto comma = dims.find(',');
                if (spec.shape.size() == HB_DNN_TENSOR_MAX_DIMENSIONS) {
                    throw std::runtime_error(context + ": tensor rank exceeds SDK limit");
                }
                spec.shape.push_back(parse_integer(dims.substr(0, comma), 1, context + " dimension"));
                if (comma == std::string_view::npos) break;
                dims.remove_prefix(comma + 1);
            }
        }
        auto& names = input ? input_names : output_names;
        if (!names.insert(spec.name).second) throw std::runtime_error(context + ": duplicate tensor '" + spec.name + "'");
        (input ? manifest.inputs : manifest.outputs).push_back(std::move(spec));
    }
    if (!file.eof() || file.bad()) throw std::runtime_error("Read failed: " + path.string());
    if (manifest.inputs.empty() || manifest.outputs.empty()) {
        throw std::runtime_error(path.string() + ": expected both input and output tensors");
    }
    return manifest;
}

std::string shape_string(const std::vector<int32_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) text += ',';
        text += std::to_string(shape[i]);
    }
    return text + ']';
}

void validate_tensor(const Spec& spec, const hbDNNTensorProperties& p) {
    if (p.tensorType != spec.dtype.sdk_type) {
        throw std::runtime_error(spec.name + ": dtype mismatch (manifest " + std::string(spec.dtype.name) +
                                 ", SDK type ID " + std::to_string(p.tensorType) + ")");
    }
    const auto rank = p.validShape.numDimensions;
    if (rank < 0 || rank > HB_DNN_TENSOR_MAX_DIMENSIONS) {
        throw std::runtime_error(spec.name + ": invalid SDK tensor rank " + std::to_string(rank));
    }
    const auto* dims = p.validShape.dimensionSize;
    if (static_cast<std::size_t>(rank) != spec.shape.size() || !std::equal(spec.shape.begin(), spec.shape.end(), dims)) {
        throw std::runtime_error(spec.name + ": shape mismatch (manifest " + shape_string(spec.shape) +
                                 ", runtime " + shape_string(std::vector<int32_t>(dims, dims + rank)) + ")");
    }
}

struct PackedModel {
    hbDNNPackedHandle_t handle{};
    ~PackedModel() { if (handle) hbDNNRelease(handle); }
};

struct Task {
    hbUCPTaskHandle_t handle{};
    ~Task() { if (handle) hbUCPReleaseTask(handle); }
};

struct Tensors {
    explicit Tensors(std::size_t size) : items(size) {}
    ~Tensors() {
        for (auto& item : items) if (item.sysMem.virAddr) hbUCPFree(&item.sysMem);
    }
    std::vector<hbDNNTensor> items;
};

// UCP leaves pyramid plane strides/allocation to the caller (alignedByteSize=-1).
// Follow the S600 SDK sample: NHWC, separate Y and interleaved UV, common 64-byte row pitch.
std::size_t set_pyramid_layout(const Spec& spec, hbDNNTensorProperties& p,
                               const std::vector<Spec>& specs) {
    const std::string_view name = spec.name;
    const auto suffix = name.rfind('_');
    if (suffix == std::string_view::npos) throw std::runtime_error(spec.name + ": unknown dynamic input layout");
    const auto plane = name.substr(suffix);
    const std::string prefix(name.substr(0, suffix));
    auto find = [&](const std::string& key) -> const Spec* {
        const auto it = std::find_if(specs.begin(), specs.end(), [&](const auto& s) { return s.name == key; });
        return it == specs.end() ? nullptr : &*it;
    };
    const Spec* y = find(prefix + "_y");
    const Spec* uv = find(prefix + "_uv");
    if ((plane != "_y" && plane != "_uv") || !y || !uv || y->shape.size() != 4 || uv->shape.size() != 4 ||
        y->dtype.sdk_type != HB_DNN_TENSOR_TYPE_U8 || uv->dtype.sdk_type != HB_DNN_TENSOR_TYPE_U8 ||
        y->shape[0] != 1 || uv->shape[0] != 1 || y->shape[3] != 1 || uv->shape[3] != 2 ||
        y->shape[1] % 2 || y->shape[2] % 2 || y->shape[1] / 2 != uv->shape[1] || y->shape[2] / 2 != uv->shape[2]) {
        throw std::runtime_error(spec.name + ": expected a matched uint8 NHWC NV12 _y/_uv plane pair");
    }
    constexpr int64_t alignment = 64;
    const int64_t pitch = (static_cast<int64_t>(y->shape[2]) + alignment - 1) / alignment * alignment;
    p.stride[3] = 1;
    p.stride[2] = spec.shape[3];
    p.stride[1] = pitch;
    p.stride[0] = pitch * spec.shape[1];
    if (static_cast<uint64_t>(p.stride[0]) > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(spec.name + ": NV12 allocation size too large");
    }
    return static_cast<std::size_t>(p.stride[0]);
}

std::size_t aligned_size(const hbDNNTensorProperties& p, const Spec& spec) {
    if (p.alignedByteSize <= 0 || static_cast<uint64_t>(p.alignedByteSize) > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(spec.name + ": unsupported runtime allocation size " + std::to_string(p.alignedByteSize));
    }
    return static_cast<std::size_t>(p.alignedByteSize);
}

// Logical C-order traversal using SDK byte strides; copy contiguous suffixes as blocks.
class Layout {
public:
    Layout(const Spec& spec, const hbDNNTensorProperties& properties, std::size_t capacity)
        : capacity(capacity), shape_(spec.shape), outer_rank_(shape_.size()), block_bytes_(spec.dtype.size) {
        validate_tensor(spec, properties);
        auto fail = [&] { throw std::runtime_error(spec.name + ": invalid byte strides or byte size"); };
        if (capacity < spec.dtype.size) fail();
        std::vector<std::size_t> axes;
        packed_bytes = spec.dtype.size;
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            const auto dim = static_cast<std::size_t>(shape_[i]);
            if (packed_bytes > capacity / dim) fail();
            packed_bytes *= dim;
            if (dim == 1) continue;  // A singleton never advances its stride.
            const auto stride = properties.stride[i];
            if (stride <= 0 || static_cast<uint64_t>(stride) > capacity) fail();
            strides_[i] = static_cast<std::size_t>(stride);
            axes.push_back(i);
        }
        if (packed_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) fail();
        std::sort(axes.begin(), axes.end(), [&](auto a, auto b) { return strides_[a] < strides_[b]; });
        std::size_t span = spec.dtype.size;
        for (auto axis : axes) {
            const auto steps = static_cast<std::size_t>(shape_[axis] - 1);
            if (strides_[axis] < span || strides_[axis] > (capacity - span) / steps) fail();
            span += strides_[axis] * steps;
        }
        while (outer_rank_ > 0) {
            const auto axis = outer_rank_ - 1;
            if (shape_[axis] != 1 && strides_[axis] != block_bytes_) break;
            block_bytes_ *= static_cast<std::size_t>(shape_[axis]);
            --outer_rank_;
        }
    }

    template <class Copy>
    void visit(Copy&& copy) const { visit_at(0, 0, copy); }

    const std::size_t capacity;
    std::size_t packed_bytes;

private:
    template <class Copy>
    void visit_at(std::size_t axis, std::size_t offset, Copy& copy) const {
        if (axis == outer_rank_) {
            copy(offset, block_bytes_);
            return;
        }
        for (int32_t i = 0; i < shape_[axis]; ++i) {
            visit_at(axis + 1, offset + static_cast<std::size_t>(i) * strides_[axis], copy);
        }
    }

    const std::vector<int32_t>& shape_;
    std::size_t outer_rank_, block_bytes_;
    std::array<std::size_t, HB_DNN_TENSOR_MAX_DIMENSIONS> strides_{};
};

void allocate(hbDNNTensor& t, std::size_t size) {
    check(hbUCPMallocCached(&t.sysMem, size, 0), "hbUCPMallocCached");
    if (!t.sysMem.virAddr || t.sysMem.memSize < size) throw std::runtime_error("Invalid UCP memory allocation");
}

void read_input(const fs::path& path, hbDNNTensor& t, const Layout& layout) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(layout.packed_bytes)) {
        throw std::runtime_error(path.string() + ": missing or wrong byte size (expected " +
                                 std::to_string(layout.packed_bytes) + ")");
    }
    file.seekg(0);
    std::vector<char> packed(layout.packed_bytes);
    if (!file.read(packed.data(), static_cast<std::streamsize>(packed.size()))) {
        throw std::runtime_error("Read failed: " + path.string());
    }
    allocate(t, layout.capacity);
    auto* dst = static_cast<char*>(t.sysMem.virAddr);
    std::memset(dst, 0, layout.capacity);
    std::size_t cursor = 0;
    layout.visit([&](auto offset, auto size) {
        std::memcpy(dst + offset, packed.data() + cursor, size);
        cursor += size;
    });
    check(hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_CLEAN), "hbUCPMemFlush(input clean)");
}

void write_output(const fs::path& path, const hbDNNTensor& t, const Layout& layout) {
    if (!t.sysMem.virAddr || t.sysMem.memSize < layout.capacity) {
        throw std::runtime_error(path.string() + ": output layout exceeds allocated memory");
    }
    check(hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_INVALIDATE), "hbUCPMemFlush(output invalidate)");
    const auto* src = static_cast<const char*>(t.sysMem.virAddr);
    std::vector<char> packed(layout.packed_bytes);
    std::size_t cursor = 0;
    layout.visit([&](auto offset, auto size) {
        std::memcpy(packed.data() + cursor, src + offset, size);
        cursor += size;
    });
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path, std::ios::binary);
    file.write(packed.data(), static_cast<std::streamsize>(packed.size()));
    file.close();
}

void validate_name(const Spec& spec, const char* name, int32_t index, bool output) {
    if (name && (spec.name == name || (output && ("_output_" + std::to_string(index)) == name))) return;
    throw std::runtime_error(std::string(output ? "Output" : "Input") + " name mismatch at index " +
                             std::to_string(index) + ": expected " + spec.name + ", got " + (name ? name : "<null>"));
}

// Linear interpolation between sorted samples, including for a single timed run.
double percentile(const std::vector<double>& sorted, double fraction) {
    const double at = fraction * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(at);
    const auto hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (at - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

void write_timing(const fs::path& path, const std::vector<double>& times, int32_t warmup) {
    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());
    const auto mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    const auto p50 = percentile(sorted, 0.50);
    const auto p95 = percentile(sorted, 0.95);
    std::cout << "Mean " << mean << " ms | P50 " << p50 << " ms | P95 " << p95
              << " ms | min/max " << sorted.front() << '/' << sorted.back() << " ms\n";
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path);
    file.imbue(std::locale::classic());
    file << std::fixed << std::setprecision(6)
         << "{\n  \"runs\": " << times.size() << ",\n  \"warmup_runs\": " << warmup
         << ",\n  \"scope\": \"hbDNNInferV2 + hbUCPSubmitTask + hbUCPWaitTaskDone\",\n  \"per_run_ms\": [";
    for (std::size_t i = 0; i < times.size(); ++i) {
        if (i) file << ", ";
        file << times[i];
    }
    file << "],\n  \"mean_ms\": " << mean << ",\n  \"p50_ms\": " << p50 << ",\n  \"p95_ms\": " << p95
         << ",\n  \"min_ms\": " << sorted.front() << ",\n  \"max_ms\": " << sorted.back() << "\n}\n";
    file.close();
}

void run(const char* hbm, const fs::path& input_dir, const fs::path& output_dir,
         int32_t runs, int32_t warmup) {
    if (fs::exists(fs::symlink_status(output_dir))) {
        throw std::runtime_error("Output directory already exists: " + output_dir.string());
    }
    const Manifest manifest = read_manifest(input_dir / "tensors.tsv");
    PackedModel model;
    check(hbDNNInitializeFromFiles(&model.handle, &hbm, 1), "hbDNNInitializeFromFiles");
    const char** names = nullptr;
    int32_t model_count = 0;
    check(hbDNNGetModelNameList(&names, &model_count, model.handle), "hbDNNGetModelNameList");
    if (model_count != 1 || !names || !names[0]) throw std::runtime_error("Expected a single-graph HBM");
    hbDNNHandle_t graph{};
    check(hbDNNGetModelHandle(&graph, model.handle, names[0]), "hbDNNGetModelHandle");
    int32_t input_count = 0, output_count = 0;
    check(hbDNNGetInputCount(&input_count, graph), "hbDNNGetInputCount");
    check(hbDNNGetOutputCount(&output_count, graph), "hbDNNGetOutputCount");
    if (input_count < 0 || output_count < 0 || static_cast<std::size_t>(input_count) != manifest.inputs.size() ||
        static_cast<std::size_t>(output_count) != manifest.outputs.size()) {
        throw std::runtime_error("Tensor count mismatch: HBM has " + std::to_string(input_count) + " inputs / " +
                                 std::to_string(output_count) + " outputs; manifest has " +
                                 std::to_string(manifest.inputs.size()) + " inputs / " + std::to_string(manifest.outputs.size()) + " outputs");
    }
    std::cout << "DNN " << hbDNNGetVersion() << " | graph " << names[0] << " | " << input_count
              << " inputs, " << output_count << " outputs\n";

    Tensors inputs(manifest.inputs.size()), outputs(manifest.outputs.size());
    for (int32_t i = 0; i < input_count; ++i) {
        const auto& spec = manifest.inputs[i];
        auto& t = inputs.items[i];
        const char* name = nullptr;
        check(hbDNNGetInputName(&name, graph, i), "hbDNNGetInputName");
        validate_name(spec, name, i, false);
        check(hbDNNGetInputTensorProperties(&t.properties, graph, i), "hbDNNGetInputTensorProperties");
        validate_tensor(spec, t.properties);
    }
    for (int32_t i = 0; i < output_count; ++i) {
        const auto& spec = manifest.outputs[i];
        auto& t = outputs.items[i];
        const char* name = nullptr;
        check(hbDNNGetOutputName(&name, graph, i), "hbDNNGetOutputName");
        validate_name(spec, name, i, true);
        check(hbDNNGetOutputTensorProperties(&t.properties, graph, i), "hbDNNGetOutputTensorProperties");
        validate_tensor(spec, t.properties);
    }
    for (std::size_t i = 0; i < manifest.inputs.size(); ++i) {
        const auto& spec = manifest.inputs[i];
        auto& t = inputs.items[i];
        const auto capacity = t.properties.alignedByteSize == -1
            ? set_pyramid_layout(spec, t.properties, manifest.inputs) : aligned_size(t.properties, spec);
        read_input(input_dir / (spec.name + ".bin"), t, Layout(spec, t.properties, capacity));
    }
    for (std::size_t i = 0; i < manifest.outputs.size(); ++i) {
        const auto& spec = manifest.outputs[i];
        auto& t = outputs.items[i];
        allocate(t, Layout(spec, t.properties, aligned_size(t.properties, spec)).capacity);
    }

    std::vector<hbDNNTensorProperties> final_properties(manifest.outputs.size());
    const auto infer = [&](bool final_run) {
        Task task;
        hbUCPSchedParam schedule{};
        HB_UCP_INITIALIZE_SCHED_PARAM(&schedule);
        const auto start = std::chrono::steady_clock::now();
        check(hbDNNInferV2(&task.handle, outputs.items.data(), inputs.items.data(), graph), "hbDNNInferV2");
        check(hbUCPSubmitTask(task.handle, &schedule), "hbUCPSubmitTask");
        check(hbUCPWaitTaskDone(task.handle, 0), "hbUCPWaitTaskDone");
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (final_run) {
            for (int32_t i = 0; i < output_count; ++i) {
                check(hbDNNGetTaskOutputTensorProperties(&final_properties[i], task.handle, 0, i),
                      "hbDNNGetTaskOutputTensorProperties");
            }
        }
        const auto status = hbUCPReleaseTask(task.handle);
        if (status == 0) task.handle = nullptr;
        check(status, "hbUCPReleaseTask");
        return elapsed;
    };

    std::cout << warmup << " warm-up(s), " << runs << " timed run(s)\n" << std::fixed << std::setprecision(3);
    for (int32_t i = 0; i < warmup; ++i) infer(false);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(runs));
    for (int32_t i = 0; i < runs; ++i) {
        times.push_back(infer(i == runs - 1));
        std::cout << "Run " << i + 1 << '/' << runs << ": " << times.back() << " ms\n";
    }

    // Task metadata can refine the output strides. Confirm all final layouts before writing files.
    std::vector<Layout> final_layouts;
    final_layouts.reserve(manifest.outputs.size());
    for (std::size_t i = 0; i < manifest.outputs.size(); ++i) {
        const auto& spec = manifest.outputs[i];
        const auto& t = outputs.items[i];
        const auto capacity = aligned_size(final_properties[i], spec);
        if (capacity > t.sysMem.memSize) throw std::runtime_error(spec.name + ": task output exceeds its allocation");
        final_layouts.emplace_back(spec, final_properties[i], capacity);
    }
    if (!fs::create_directory(output_dir)) throw std::runtime_error("Cannot create " + output_dir.string());
    for (std::size_t i = 0; i < manifest.outputs.size(); ++i) {
        const auto path = output_dir / (manifest.outputs[i].name + ".bin");
        write_output(path, outputs.items[i], final_layouts[i]);
        std::cout << "Wrote " << path << " (" << final_layouts[i].packed_bytes << " bytes)\n";
    }
    write_timing(output_dir / "timing.json", times, warmup);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " MODEL.hbm INPUT_DIR NEW_OUTPUT_DIR [RUNS] [WARMUP]\n";
        return 2;
    }
    try {
        const auto runs = argc >= 5 ? parse_integer(argv[4], 1, "RUNS") : 1;
        const auto warmup = argc == 6 ? parse_integer(argv[5], 0, "WARMUP") : 0;
        run(argv[1], argv[2], argv[3], runs, warmup);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "board_demo: " << e.what() << '\n';
        return 1;
    }
}
